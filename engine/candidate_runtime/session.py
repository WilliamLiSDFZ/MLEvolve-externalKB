"""Worker-side callbacks for early validation, bounded training and durable exports.

No Torch dependency: callbacks own the model and must restore its train/eval mode.
All GPU work remains in the candidate's existing process and execution slot.
"""

import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CandidateRuntimeConfig
from .io import atomic_json, file_hashes, read_json, seal_directory
from .jigsaw import load_contract, predictions, score
from .store import ResultStore


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
        self.last_export = None
        self.validation_seconds = 0.0
        self.test_seconds = 0.0
        self.save_seconds = 0.0
        self.best = None
        self.published_checkpoint = None
        self.smoke_done = False
        self.closed = False
        self.predict_validation = self.predict_test = self.save_checkpoint = self.load_checkpoint = None

    def elapsed(self):
        return time.time() - self.spec["started_at"]

    def remaining(self):
        return self.spec["deadline"] - time.time()

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

    def _smoke(self):
        if self.steps < 1:
            raise RuntimeError("A smoke check needs at least one completed training update")
        for kind, callback, count in (("validation", self.predict_validation, len(self.answers)),
                                       ("test", self.predict_test, len(self.test_ids))):
            rows = min(self.config.smoke_rows, count)
            started = time.time()
            predictions(callback(np.arange(rows)), rows)
            estimate = (time.time() - started) * count / rows
            if kind == "validation":
                self.validation_seconds = estimate
            else:
                self.test_seconds = estimate
        self.smoke_done = True
        self.store.event(self.node_id, "smoke_passed", optimizer_steps=self.steps,
                         estimated_validation_seconds=self.validation_seconds,
                         estimated_test_seconds=self.test_seconds)

    def reserve_seconds(self):
        # Full validation + a reloaded-checkpoint validation + test inference + checkpoint I/O.
        estimated = 2 * self.validation_seconds + self.test_seconds + self.save_seconds
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
            self.finish(reason="budget_exhausted")
            return True
        now = time.time()
        interval = max(self.config.validation_interval_seconds, 3 * self.validation_seconds)
        due = ((self.last_validation is None and now - self.training_started >= self.config.first_validation_seconds)
               or (self.last_validation is not None and now - self.last_validation >= interval))
        if due:
            if not self.smoke_done:
                self._smoke()
            improved = self._validate()
            # Only export the current model while continuing training: loading an older
            # checkpoint here would desynchronize model and optimizer state.
            if improved and (self.last_export is None or time.time() - self.last_export >= self.config.export_interval_seconds):
                self._export()
        return False

    def _validate(self):
        started = time.time()
        values = predictions(self.predict_validation(np.arange(len(self.answers))), len(self.answers))
        metric = score(self.answers, values)
        self.validation_seconds = time.time() - started
        self.last_validation = time.time()
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

        self.store.publish(self.node_id, checkpoint_id, write_submission, self.contract,
                           self.steps, self.elapsed)
        self.last_export = time.time()
        self.published_checkpoint = checkpoint_id
        self.store.event(self.node_id, "export", duration_seconds=time.time() - started)
        self.store.prune(self.node_id, checkpoint_id, self.config.keep_snapshots)

    def finish(self, reason="completed"):
        """Idempotent cooperative finalization; a hard kill still retains older exports."""
        if reason not in ("completed", "budget_exhausted"):
            raise ValueError("Unknown cooperative completion reason")
        if self.closed:
            return
        if self.training_started is None or self.steps < 1:
            raise RuntimeError("No trained candidate to finalize")
        if not self.smoke_done:
            self._smoke()
        if self.best is None or self.remaining() > self.reserve_seconds() / self.config.finalization_safety_factor:
            self._validate()
        self._export()
        self.closed = True
        atomic_json(self.directory / "worker_finished.json",
                    dict(reason=reason, optimizer_steps=self.steps, finished_at=time.time(),
                         checkpoint_id=self.published_checkpoint))
        self.store.event(self.node_id, "worker_finished", reason=reason, elapsed_seconds=self.elapsed())
        print(f"Final Validation Score: {self.best['metric']}", flush=True)
