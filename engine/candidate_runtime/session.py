"""Worker-side callbacks for early validation, bounded training and durable exports.

No Torch dependency: callbacks own the model and must restore its train/eval mode.
All GPU work remains in the candidate's existing process and execution slot.
"""

import os
import tempfile
import time
import uuid
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CandidateRuntimeConfig
from .io import atomic_json, file_hashes, read_json, seal_directory
from .jigsaw import load_contract, predictions, score
from .store import ResultStore


def _runtime_work(method):
    """Inference/checkpoint work consumes the deadline, not the training interval."""

    @wraps(method)
    def measured(self, *args, **kwargs):
        started = time.time()
        try:
            return method(self, *args, **kwargs)
        finally:
            self._runtime_seconds += time.time() - started

    return measured


class CandidateSession:
    @classmethod
    def from_env(cls):
        path = os.environ.get("MLEVOLVE_CANDIDATE_SPEC")
        if not path:
            raise RuntimeError("Candidate runtime is not enabled for this execution")
        return cls(read_json(path))

    def __init__(self, spec):
        self.spec = spec
        self.config = CandidateRuntimeConfig(**spec["config"])
        self.config.validate()
        self.store = ResultStore(spec["workspace"], spec["log_dir"])
        self.node_id = spec["node_id"]
        self.directory = self.store.candidate_dir(self.node_id)
        self.contract_dir = self.store.root / "contract"
        self.contract = load_contract(self.contract_dir)
        self.answers = pd.read_csv(self.contract_dir / "validation.csv", dtype={"id": str})
        self.test_ids = pd.read_csv(self.contract_dir / "test_ids.csv", dtype=str).id.tolist()
        self.train_ids = pd.read_csv(self.contract_dir / "train_ids.csv", dtype=str).id.tolist()
        with np.load(self.contract_dir / "split.npz", allow_pickle=False) as split:
            self.train_indices = split["train"].copy()
            self.validation_indices = split["validation"].copy()
        self.steps = 0
        self.training_started = None
        self.last_validation = None
        self.last_validation_steps = None
        self.last_validation_training_seconds = None
        self.last_export = None
        self._runtime_seconds = 0.0
        self.validation_seconds = 0.0
        self.test_seconds = 0.0
        self.save_seconds = 0.0
        self.export_seconds = None
        self.best = None
        self.published_checkpoint = None
        self.published_snapshot = None
        self._finish_result = None
        self.smoke_done = False
        self.closed = False
        self.predict_validation = self.predict_test = self.save_checkpoint = self.load_checkpoint = None

    def elapsed(self):
        return time.time() - self.spec["started_at"]

    def remaining(self):
        return self.spec["deadline"] - time.time()

    def training_seconds(self):
        if self.training_started is None:
            return 0.0
        return max(0.0, time.time() - self.training_started - self._runtime_seconds)

    @property
    def best_validation_score(self):
        """Best full-validation score, or None before any formal validation."""
        return None if self.best is None else float(self.best["metric"])

    @property
    def best_score(self):
        """Compatibility alias for generated candidates using the shorter name."""
        return self.best_validation_score

    def split(self, train, test):
        """Call before preprocessing; fit transforms ONLY on the returned training frame."""
        if train.id.astype(str).tolist() != self.train_ids or test.id.astype(str).tolist() != self.test_ids:
            raise ValueError("Input rows/order differ from the fixed public-data contract")
        return train.iloc[self.train_indices].copy(), train.iloc[self.validation_indices].copy(), test.copy()

    def bind(self, *, predict_validation, predict_test, save_checkpoint, load_checkpoint):
        """Predict callbacks take positional indices and return probabilities in that order.

        Save/load callbacks take a directory and include model, tokenizer/feature transforms,
        configuration and everything needed for identical inference. They must not train.
        """
        self.predict_validation, self.predict_test = predict_validation, predict_test
        self.save_checkpoint, self.load_checkpoint = save_checkpoint, load_checkpoint

    def start_training(self, train_ids):
        if self.training_started is not None:
            raise RuntimeError("start_training may only be called once")
        expected = [self.train_ids[i] for i in self.train_indices]
        if sorted(map(str, train_ids)) != sorted(expected):
            raise ValueError("Training rows must be exactly the fixed training partition, without validation rows")
        if not all(callable(f) for f in (self.predict_validation, self.predict_test, self.save_checkpoint, self.load_checkpoint)):
            raise RuntimeError("Bind all runtime callbacks before starting training")
        self.training_started = time.time()
        self.store.event(self.node_id, "training_started", elapsed_seconds=self.elapsed())

    def _estimate_prediction(self, callback, count):
        rows = min(self.config.smoke_rows, count)

        def measure(size):
            # Cover the partition rather than calibrating only on its first texts.
            indices = np.linspace(0, count - 1, num=size, dtype=np.int64)
            started = time.time()
            predictions(callback(indices), size)
            return time.time() - started

        warmup = measure(rows)
        # A whole-partition smoke call already measures the real cost. Near the
        # deadline, skip optional calibration calls and retain a provisional estimate.
        if rows == count or self.remaining() <= self.config.finalization_reserve_seconds:
            return warmup * count / rows, {"warmup_seconds": warmup, "sample_rows": rows,
                                           "calibrated": rows == count}
        small = measure(rows)
        large_rows = min(4 * rows, count)
        if self.remaining() <= self.config.finalization_reserve_seconds:
            return small * count / rows, {"warmup_seconds": warmup, "sample_rows": rows,
                                          "small_seconds": small, "calibrated": False}
        large = measure(large_rows)
        # Fit fixed + per-row cost using warmed calls. If noise makes the slope
        # negative or the intercept negative, use the larger warmed batch's rate.
        slope = (large - small) / (large_rows - rows)
        fixed = small - slope * rows
        if slope < 0 or fixed < 0:
            slope, fixed = large / large_rows, 0.0
        estimate = max(large, fixed + slope * count)
        return estimate, {"warmup_seconds": warmup, "sample_rows": rows,
                          "small_seconds": small, "large_rows": large_rows,
                          "large_seconds": large, "fixed_seconds": fixed,
                          "seconds_per_row": slope, "calibrated": True}

    @_runtime_work
    def _smoke(self):
        if self.steps < 1:
            raise RuntimeError("A smoke check needs at least one completed training update")
        timings = {}
        for kind, callback, count in (("validation", self.predict_validation, len(self.answers)),
                                       ("test", self.predict_test, len(self.test_ids))):
            estimate, timings[kind] = self._estimate_prediction(callback, count)
            if kind == "validation":
                self.validation_seconds = estimate
            else:
                self.test_seconds = estimate
        self.smoke_done = True
        self.store.event(self.node_id, "smoke_passed", optimizer_steps=self.steps,
                         estimated_validation_seconds=self.validation_seconds,
                         estimated_test_seconds=self.test_seconds, timings=timings)

    def reserve_seconds(self):
        # Once available, measured export includes reload, verification inference,
        # full test inference and publication I/O, rather than only prediction cost.
        export = (self.export_seconds if self.export_seconds is not None
                  else self.validation_seconds + self.test_seconds)
        estimated = self.validation_seconds + self.save_seconds + export
        return max(self.config.finalization_reserve_seconds,
                   self.config.finalization_safety_factor * estimated)

    def should_stop(self):
        return self.closed or self.remaining() <= self.reserve_seconds()

    def step(self):
        """Call AFTER each real optimizer.step (or completed estimator fit). True means STOP."""
        if self.closed or self.training_started is None:
            raise RuntimeError("Training update outside an active session")
        self.steps += 1
        if not self.smoke_done and self.steps >= self.config.smoke_steps:
            self._smoke()
        if self.should_stop():
            # Small-sample timing is provisional. Measure one real cycle before
            # committing to an early close, unless the minimum reserve is already
            # exhausted. With no previous checkpoint, export reloads this exact
            # current model, so resuming cannot roll weights back under an optimizer.
            if self.best is None and self.remaining() > self.config.finalization_reserve_seconds:
                previous_reserve = self.reserve_seconds()
                self._validate()
                self._export()
                self.store.event(self.node_id, "budget_recalibrated", optimizer_steps=self.steps,
                                 previous_reserve_seconds=previous_reserve,
                                 reserve_seconds=self.reserve_seconds(), remaining_seconds=self.remaining(),
                                 continue_training=not self.should_stop())
            if self.should_stop():
                self.finish(reason="budget_exhausted")
                return True
        now = self.training_seconds()
        interval = max(self.config.validation_interval_seconds, 3 * self.validation_seconds)
        due = ((self.last_validation_training_seconds is None and now >= self.config.first_validation_seconds)
               or (self.last_validation_training_seconds is not None
                   and now - self.last_validation_training_seconds >= interval))
        if due:
            if not self.smoke_done:
                self._smoke()
            improved = self._validate()
            # Only export the current model while continuing training: loading an older
            # checkpoint here would desynchronize model and optimizer state.
            if improved and (self.last_export is None or time.time() - self.last_export >= self.config.export_interval_seconds):
                self._export()
            # Real validation/export may consume the remaining reserve. Finalize
            # at this same update instead of allowing another optimizer step.
            if self.should_stop():
                self.finish(reason="budget_exhausted")
                return True
        return False

    @_runtime_work
    def _validate(self):
        self.last_validation_training_seconds = self.training_seconds()
        started = time.time()
        values = predictions(self.predict_validation(np.arange(len(self.answers))), len(self.answers))
        metric = score(self.answers, values)
        self.validation_seconds = time.time() - started
        self.last_validation = time.time()
        self.last_validation_steps = self.steps
        self.store.event(self.node_id, "validation", metric=metric, optimizer_steps=self.steps,
                         duration_seconds=self.validation_seconds, elapsed_seconds=self.elapsed())
        if self.best is not None and metric <= self.best["metric"]:
            return False
        checkpoints = self.directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        checkpoint_id = uuid.uuid4().hex
        started = time.time()
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=checkpoints) as tmp:
            staging = Path(tmp) / "checkpoint"
            staging.mkdir()
            model_dir = staging / "model"
            model_dir.mkdir()
            self.save_checkpoint(model_dir)
            if not any(p.is_file() and p.stat().st_size > 0 for p in model_dir.rglob("*")):
                raise ValueError("save_checkpoint did not save model/inference state")
            pd.DataFrame({"id": self.answers.id, "prediction": values}).to_csv(staging / "validation.csv", index=False)
            manifest = dict(checkpoint_id=checkpoint_id, contract_id=self.contract["contract_id"],
                            metric=metric, optimizer_steps=self.steps, files=file_hashes(staging))
            atomic_json(staging / "manifest.json", manifest)
            seal_directory(staging, checkpoints / checkpoint_id)
        self.save_seconds = time.time() - started
        self.best = manifest
        self.store.prune(self.node_id, checkpoint_id, self.config.keep_snapshots)
        return True

    @_runtime_work
    def _export(self):
        if self.best is None or self.published_checkpoint == self.best["checkpoint_id"]:
            return
        started = time.time()
        checkpoint_id = self.best["checkpoint_id"]
        checkpoint = self.directory / "checkpoints" / checkpoint_id
        # Re-read the exact saved checkpoint and verify its predictions before publishing.
        self.load_checkpoint(checkpoint / "model")
        expected = pd.read_csv(checkpoint / "validation.csv").prediction.to_numpy()
        actual = predictions(self.predict_validation(np.arange(len(self.answers))), len(self.answers))
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-7):
            raise ValueError("Reloaded checkpoint does not reproduce validation predictions")

        def write_submission(path):
            test_started = time.time()
            values = predictions(self.predict_test(np.arange(len(self.test_ids))), len(self.test_ids))
            pd.DataFrame({"id": self.test_ids, "prediction": values}).to_csv(path, index=False)
            self.test_seconds = time.time() - test_started

        snapshot = self.store.publish(self.node_id, checkpoint_id, write_submission, self.contract,
                                      self.steps, self.elapsed)
        self.last_export = time.time()
        self.published_checkpoint = checkpoint_id
        self.store.prune(self.node_id, checkpoint_id, self.config.keep_snapshots)
        self.published_snapshot = snapshot
        self.export_seconds = time.time() - started
        self.store.event(self.node_id, "export", duration_seconds=self.export_seconds)

    def finish(self, reason="completed"):
        """Publish the best checkpoint and return its stable result dictionary.

        Repeated calls (including after step() stopped for budget) return the same
        result without inference, training or another export. Paths are immutable.
        """
        if reason not in ("completed", "budget_exhausted"):
            raise ValueError("Unknown cooperative completion reason")
        if self.closed:
            return dict(self._finish_result)
        if self.training_started is None or self.steps < 1:
            raise RuntimeError("No trained candidate to finalize")
        if not self.smoke_done:
            self._smoke()
        if self.best is None or (self.last_validation_steps != self.steps
                and self.remaining() > self.reserve_seconds() / self.config.finalization_safety_factor):
            self._validate()
        self._export()
        snapshot = self.published_snapshot
        snapshot_dir = self.directory / "snapshots" / snapshot["snapshot_id"]
        result = dict(version=1, node_id=self.node_id, reason=reason, optimizer_steps=self.steps,
                      finished_at=time.time(), elapsed_seconds=self.elapsed(),
                      checkpoint_id=self.published_checkpoint, snapshot_id=snapshot["snapshot_id"],
                      selected_optimizer_steps=snapshot["optimizer_steps"],
                      best_validation_score=float(snapshot["metric"]), maximize=snapshot["maximize"],
                      submission_path=str(snapshot_dir / "submission.csv"),
                      validation_path=str(snapshot_dir / "validation.csv"))
        atomic_json(self.directory / "worker_finished.json", result)
        self._finish_result = result
        self.closed = True
        self.store.event(self.node_id, "worker_finished", reason=reason, elapsed_seconds=self.elapsed())
        print(f"Final Validation Score: {result['best_validation_score']}", flush=True)
        return dict(result)
