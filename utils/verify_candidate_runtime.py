"""CPU regressions for candidate validation, durable exports and timeout recovery.

Run: python utils/verify_candidate_runtime.py
No cluster, model downloads, private labels or LLM calls are used.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from engine.candidate_runtime.config import CandidateRuntimeConfig
from engine.candidate_runtime.integration import begin_execution, end_execution, export_results, settings
from engine.candidate_runtime.io import atomic_json, read_json
from engine.candidate_runtime.jigsaw import IDENTITIES, TASK_ID, prepare_contract, score
from engine.candidate_runtime.session import CandidateSession
from engine.candidate_runtime.store import ResultStore
from engine.executor import ExecutionResult, Interpreter


def pairwise_auc(labels, values):
    positive, negative = values[labels], values[~labels]
    return np.mean((positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative))


class Clock:
    """Advance callback wall time without sleeping or mocking result publication."""

    def __init__(self):
        self.value = time.time()

    def time(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.ws = self.run / "workspace"
        public = self.ws / "input"
        public.mkdir(parents=True)
        rng = np.random.default_rng(18)
        rows = 700
        x = rng.normal(size=(rows, 2))
        y = (x[:, 0] + rng.normal(size=rows) * 0.6 > 0).astype(float)
        frame = pd.DataFrame(dict(id=np.arange(rows), target=y, x=x[:, 0], z=x[:, 1]))
        for name in IDENTITIES:
            frame[name] = rng.binomial(1, 0.45, rows)
        frame.to_csv(public / "train.csv", index=False)
        frame.iloc[:80].assign(id=np.arange(10000, 10080)).drop(columns=["target"]).to_csv(public / "test.csv", index=False)
        self.runtime = CandidateRuntimeConfig(enabled=True, validation_fraction=0.3,
            smoke_steps=1, smoke_rows=16, first_validation_seconds=0.000001,
            validation_interval_seconds=0.000001, export_interval_seconds=0.000001,
            finalization_reserve_seconds=0.01)
        self.cfg = SimpleNamespace(workspace_dir=self.ws, log_dir=self.run / "logs", exp_id=TASK_ID,
            candidate_runtime=self.runtime, agent=SimpleNamespace(seed=51, time_limit=60, check_data_leakage=False),
            cpu_number=2, start_cpu_id=0)
        prepare_contract(self.ws, public, 51, self.runtime.validation_fraction)
        self.store = ResultStore(self.ws)

    def register(self, name="n1", code="print('candidate')", stage="draft"):
        node = SimpleNamespace(id=name, code=code, stage=stage, branch_id=1, parent=None)
        self.store.register(node, asdict(self.runtime), check_leakage=False)
        return node

    def session(self, name="n1"):
        self.register(name)
        now = time.time()
        session = CandidateSession(dict(workspace=str(self.ws), log_dir=str(self.run / "logs"),
            node_id=name, started_at=now, deadline=now + 60, config=asdict(self.runtime)))
        train = pd.read_csv(self.ws / "input/train.csv")
        test = pd.read_csv(self.ws / "input/test.csv")
        fit, val, test = session.split(train, test)
        model = {"weight": np.array([0.0, 1.0])}

        def predict(frame, indices):
            features = frame.iloc[indices][["x", "z"]].to_numpy()
            return 1 / (1 + np.exp(-features @ model["weight"]))

        def save(directory):
            np.save(directory / "weights.npy", model["weight"])

        def load(directory):
            model["weight"][:] = np.load(directory / "weights.npy", allow_pickle=False)

        session.bind(predict_validation=lambda indices: predict(val, indices),
                     predict_test=lambda indices: predict(test, indices), save_checkpoint=save, load_checkpoint=load)
        session.start_training(fit.id)
        return session, model

    def publish(self, name="n1"):
        session, model = self.session(name)
        session.step()
        session.finish()
        self.store.write_execution(name, status="completed", elapsed_seconds=session.elapsed())
        return session, model

    def timed_session(self, prediction_cost, budget=5400, save_cost=20, load_cost=15):
        self.runtime.smoke_steps = 5
        self.runtime.first_validation_seconds = 900
        self.runtime.validation_interval_seconds = 1800
        self.runtime.export_interval_seconds = 3600
        self.runtime.finalization_reserve_seconds = 900
        session, model = self.session()
        clock = Clock()
        timer = patch("engine.candidate_runtime.session.time", clock)
        timer.start()
        self.addCleanup(timer.stop)
        session.training_started = session.spec["started_at"] = clock.time()
        session.spec["deadline"] = clock.time() + budget
        calls = {"validation": [], "test": [], "save": [], "load": []}

        def prediction(kind, callback):
            def run(indices):
                calls[kind].append(indices.copy())
                clock.advance(prediction_cost(kind, len(indices), len(calls[kind])))
                return callback(indices)
            return run

        def checkpoint(kind, callback, cost):
            def run(path):
                calls[kind].append(path)
                clock.advance(cost)
                return callback(path)
            return run

        session.bind(predict_validation=prediction("validation", session.predict_validation),
                     predict_test=prediction("test", session.predict_test),
                     save_checkpoint=checkpoint("save", session.save_checkpoint, save_cost),
                     load_checkpoint=checkpoint("load", session.load_checkpoint, load_cost))
        return session, model, clock, calls

    def test_cold_start_and_fixed_inference_cost_do_not_stop_after_five_steps(self):
        session, model, clock, calls = self.timed_session(
            lambda kind, rows, call: (90 if call == 1 else 0) + 30 + rows * 0.05)
        deadline = session.spec["deadline"]
        for _ in range(6):
            model["weight"][0] += 0.01
            clock.advance(2)
            self.assertFalse(session.step())
        self.assertAlmostEqual(session.validation_seconds, 30 + len(session.answers) * 0.05, places=4)
        self.assertAlmostEqual(session.test_seconds, 34, places=4)
        self.assertEqual(session.reserve_seconds(), 900)
        self.assertAlmostEqual(session.training_seconds(), 12)
        self.assertGreater(session.elapsed(), 300)
        self.assertIsNone(session.best_validation_score)  # smoke never becomes a ranked score
        self.assertEqual(session.spec["deadline"], deadline)
        for kind, count in (("validation", len(session.answers)), ("test", len(session.test_ids))):
            self.assertEqual([len(i) for i in calls[kind]], [16, 16, 64])
            self.assertEqual(calls[kind][0][0], 0)
            self.assertEqual(calls[kind][0][-1], count - 1)

    def test_prediction_timing_flat_and_noisy_costs_stay_finite(self):
        session, _, clock, _ = self.timed_session(lambda *args: 0)
        for durations, expected in [([90, 30, 30], 30), ([90, 31, 30], 30 * 210 / 64),
                                    ([90, 1, 8], 8 * 210 / 64)]:
            with self.subTest(durations=durations):
                times = iter(durations)
                def callback(indices):
                    clock.advance(next(times))
                    return np.full(len(indices), 0.5)
                estimate, timing = session._estimate_prediction(callback, 210)
                self.assertAlmostEqual(estimate, expected)
                self.assertTrue(timing["calibrated"])

    def test_small_partition_and_deadline_skip_optional_timing_calls(self):
        session, _, clock, _ = self.timed_session(lambda *args: 0)
        calls = []
        def callback(indices):
            calls.append(indices.copy())
            clock.advance(5)
            return np.full(len(indices), 0.5)
        estimate, _ = session._estimate_prediction(callback, 10)
        self.assertEqual(estimate, 5)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0], np.arange(10))
        # Cross the minimum reserve during warmup, then during the warmed small call.
        for available, expected_calls in [(902, 1), (907, 2)]:
            calls.clear()
            session.spec["deadline"] = clock.time() + available
            deadline = session.spec["deadline"]
            estimate, timing = session._estimate_prediction(callback, 210)
            self.assertEqual(len(calls), expected_calls)
            self.assertFalse(timing["calibrated"])
            self.assertGreater(estimate, 0)
            self.assertEqual(session.spec["deadline"], deadline)

    def test_provisional_overestimate_recalibrates_and_resumes_current_model(self):
        # Small calls have costly per-row setup that full-partition vectorization avoids.
        session, model, clock, calls = self.timed_session(
            lambda kind, rows, call: rows * 8 if rows <= 64 else 100)
        model["weight"][:] = [0.3, 0.7]
        for _ in range(5):
            clock.advance(2)
            self.assertFalse(session.step())
        self.assertFalse(session.closed)
        self.assertEqual(session.last_validation_steps, 5)
        self.assertEqual(session.validation_seconds, 100)
        self.assertEqual(session.export_seconds, 215)  # reload + full validation + test
        self.assertEqual(session.reserve_seconds(), 900)
        self.assertEqual(len(calls["save"]), 1)
        self.assertEqual(len(calls["load"]), 1)
        self.assertEqual(len(self.store.valid_snapshots("n1")[0]), 1)
        np.testing.assert_array_equal(model["weight"], [0.3, 0.7])
        events = [json.loads(line) for line in
                  (session.store.metadata_dir("n1") / "events.jsonl").read_text().splitlines()]
        recalibrated = next(e for e in events if e["event"] == "budget_recalibrated")
        self.assertTrue(recalibrated["continue_training"])
        self.assertGreater(recalibrated["previous_reserve_seconds"], 5400)
        model["weight"][0] += 0.1
        clock.advance(2)
        self.assertFalse(session.step())
        self.assertEqual(session.steps, 6)
        self.assertAlmostEqual(model["weight"][0], 0.4)
        self.assertEqual(len(calls["load"]), 1)

    def test_real_expensive_cycle_stops_once_with_complete_result(self):
        session, _, clock, calls = self.timed_session(
            lambda kind, rows, call: rows * 8 if rows <= 64 else 1000)
        for _ in range(4):
            clock.advance(2)
            self.assertFalse(session.step())
        clock.advance(2)
        self.assertTrue(session.step())
        result = session.finish()
        self.assertEqual(result["reason"], "budget_exhausted")
        self.assertEqual(result["optimizer_steps"], 5)
        self.assertEqual(result["selected_optimizer_steps"], 5)
        self.assertEqual([len(i) for i in calls["validation"]], [16, 16, 64, 210, 210])
        self.assertEqual([len(i) for i in calls["test"]], [16, 16, 64, 80])
        self.assertEqual(len(calls["save"]), 1)
        self.assertTrue(Path(result["submission_path"]).is_file())
        self.assertGreater(session.remaining(), 0)
        counts = {k: len(v) for k, v in calls.items()}
        self.assertEqual(session.finish(), result)
        self.assertEqual({k: len(v) for k, v in calls.items()}, counts)

    def test_validation_and_export_do_not_advance_training_interval(self):
        session, model, clock, calls = self.timed_session(
            lambda kind, rows, call: 300 if kind == "validation" else 1800, budget=30000)
        for _ in range(5):
            clock.advance(2)
            self.assertFalse(session.step())
        # Smoke alone took 105 minutes, but first formal validation still needs 15 min training.
        self.assertEqual(session.training_seconds(), 10)
        self.assertIsNone(session.best)
        clock.advance(890)
        self.assertFalse(session.step())
        self.assertEqual(session.last_validation_steps, 6)
        self.assertEqual(session.training_seconds(), 900)
        before = len(calls["validation"])
        # Export takes >30 minutes. One subsequent training step must NOT trigger validation.
        clock.advance(1)
        self.assertFalse(session.step())
        self.assertEqual(len(calls["validation"]), before)
        clock.advance(1798)
        self.assertFalse(session.step())
        self.assertEqual(len(calls["validation"]), before)
        model["weight"][:] = [1, 0]
        clock.advance(1)
        self.assertFalse(session.step())
        self.assertEqual(session.last_validation_steps, 9)
        self.assertEqual(len(calls["validation"]), before + 1)
        # The older published model was not reloaded over the newer training state.
        np.testing.assert_array_equal(model["weight"], [1, 0])

    def test_actual_cycle_can_end_budget_at_same_optimizer_update(self):
        session, _, clock, calls = self.timed_session(
            lambda kind, rows, call: 1 if rows <= 64 else 1100)
        for _ in range(4):
            clock.advance(2)
            self.assertFalse(session.step())
        clock.advance(900)
        self.assertTrue(session.step())
        result = session.finish()
        self.assertEqual(result["reason"], "budget_exhausted")
        self.assertEqual(result["optimizer_steps"], 5)
        self.assertEqual(len(calls["validation"]), 5)
        self.assertEqual(len(calls["test"]), 4)
        self.assertEqual(len(self.store.valid_snapshots("n1")[0]), 1)

    def test_finish_returns_verified_immutable_result_and_is_idempotent(self):
        session, _ = self.session()
        self.assertIsNone(session.best_validation_score)
        self.assertIsNone(session.best_score)
        session.step()
        with patch.object(session, "predict_validation", side_effect=AssertionError("duplicate validation")), \
             patch.object(session, "predict_test", side_effect=AssertionError("duplicate inference")):
            result = session.finish()
            again = session.finish()
        self.assertEqual(result, again)
        self.assertIsNot(result, again)
        manifest = self.store.verify("n1", Path(result["submission_path"]).parent)
        self.assertEqual(result["checkpoint_id"], manifest["checkpoint_id"])
        self.assertEqual(result["snapshot_id"], manifest["snapshot_id"])
        self.assertAlmostEqual(result["best_validation_score"], manifest["metric"])
        self.assertEqual(session.best_score, result["best_validation_score"])
        self.assertEqual(session.best_validation_score, session.best_score)
        with self.assertRaises(AttributeError):
            session.best_validation_score = 1.0
        self.assertTrue(Path(result["validation_path"]).is_file())
        self.assertEqual(read_json(session.directory / "worker_finished.json"), result)
        result["best_validation_score"] = -1
        self.assertEqual(session.finish(), again)

    def test_finish_reports_selected_checkpoint_steps_instead_of_latest_steps(self):
        session, model, clock, calls = self.timed_session(lambda *args: 1, budget=20000)
        model["weight"][:] = [1, 0]
        clock.advance(901)
        session.step()
        first = session.best["metric"]
        model["weight"][:] = [0, 1]
        clock.advance(1801)
        session.step()
        self.assertEqual(session.last_validation_steps, 2)
        count = len(calls["validation"])
        result = session.finish()
        self.assertEqual(len(calls["validation"]), count)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["selected_optimizer_steps"], 1)
        self.assertEqual(result["best_validation_score"], first)
        self.assertEqual(self.store.verify("n1", Path(result["submission_path"]).parent)["metric"], first)

    def test_shared_prompt_documents_result_api(self):
        from engine.candidate_runtime.prompt import instructions
        prompt = " ".join(instructions())
        for api in ["result = session.finish()", "result['best_validation_score']", "result['submission_path']",
                    "session.best_validation_score", "session.best_score"]:
            self.assertIn(api, prompt)
        self.assertIn("without extra inference", prompt)

    def test_official_metric_matches_independent_pairwise_oracle(self):
        answers = pd.read_csv(self.ws / "candidate_results/contract/validation.csv")
        rng = np.random.default_rng(7)
        values = rng.uniform(size=len(answers))
        label = answers.target.to_numpy() >= 0.5
        parts = [[], [], []]
        for identity in IDENTITIES:
            group = answers[identity].to_numpy() >= 0.5
            for out, mask in zip(parts, [group, (group & ~label) | (~group & label), (group & label) | (~group & ~label)]):
                out.append(pairwise_auc(label[mask], values[mask]))
        expected = 0.25 * pairwise_auc(label, values) + 0.25 * sum(np.mean(np.asarray(p) ** -5) ** (-0.2) for p in parts)
        self.assertAlmostEqual(score(answers, values), expected, places=12)
        self.assertEqual(score(answers, np.full(len(answers), 0.5)), 0.5)
        with self.assertRaises(ValueError):
            score(answers, np.full(len(answers), np.nan))

    def test_fixed_split_rejects_reordering_and_leakage(self):
        session, _ = self.session()
        with self.assertRaises(ValueError):
            session.split(pd.read_csv(self.ws / "input/train.csv").iloc[::-1], pd.read_csv(self.ws / "input/test.csv"))
        with self.assertRaises(RuntimeError):
            session.start_training(session.train_ids)
        with self.assertRaises(ValueError):
            prepare_contract(self.ws, self.ws / "input", 52, 0.3)
        self.assertFalse(set(session.train_indices) & set(session.validation_indices))

    def test_complete_bundle_recomputes_and_recovers_without_journal(self):
        session, _ = self.publish()
        self.assertFalse((self.run / "logs/journal.json").exists())
        self.assertEqual(len(self.store.valid_snapshots("n1")[0]), 1)
        top = export_results(self.ws)
        self.assertTrue((top / "top1/submission.csv").exists())
        self.assertTrue((self.ws / "best_submission/submission.csv").exists())
        self.assertEqual(read_json(top / "top1/manifest.json")["checkpoint_id"], session.published_checkpoint)

    def test_partial_new_export_keeps_old_complete_snapshot(self):
        session, model = self.session()
        session.step()
        original = self.store.valid_snapshots("n1")[0][0]
        model["weight"][:] = [1.0, 0.0]
        session.last_validation_training_seconds = -1e6
        session.last_export = 0
        def fail_predict(indices):
            raise RuntimeError("test inference failed after checkpoint was saved")
        session.predict_test = fail_predict
        with self.assertRaises(RuntimeError):
            session.step()
        good, _ = self.store.valid_snapshots("n1")
        self.assertEqual([m["snapshot_id"] for m in good], [original["snapshot_id"]])
        self.assertGreater(session.best["metric"], original["metric"])
        export_results(self.ws)
        self.assertEqual(read_json(self.ws / "best_solution/manifest.json")["metric"], original["metric"])

    def test_corrupt_newest_checkpoint_falls_back(self):
        session, model = self.session()
        session.step()
        old = session.published_checkpoint
        model["weight"][:] = [1.0, 0.0]
        session.last_validation_training_seconds = -1e6
        session.last_export = 0
        session.step()
        new = session.published_checkpoint
        self.assertNotEqual(old, new)
        (session.directory / "checkpoints" / new / "model/weights.npy").write_bytes(b"corrupt")
        good, errors = self.store.valid_snapshots("n1")
        self.assertEqual(good[0]["checkpoint_id"], old)
        self.assertEqual(len(errors), 1)

    def test_load_mismatch_is_not_published(self):
        session, model = self.session()
        session.load_checkpoint = lambda _: model["weight"].fill(0)
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            session.step()
        self.assertEqual(self.store.valid_snapshots("n1")[0], [])

    def test_soft_deadline_exports_and_closes_without_new_node(self):
        session, _ = self.session()
        session.spec["deadline"] = time.time() + 0.001
        self.assertTrue(session.step())
        session.finish()
        self.assertEqual(read_json(session.directory / "worker_finished.json")["reason"], "budget_exhausted")
        self.assertEqual(len(list((self.store.root / "candidates").iterdir())), 1)
        with self.assertRaises(RuntimeError):
            session.step()

    def test_untrained_or_invalid_predictions_never_publish(self):
        session, _ = self.session()
        with self.assertRaises(RuntimeError):
            session.finish()
        session.predict_validation = lambda indices: np.full(len(indices), np.nan)
        with self.assertRaises(ValueError):
            session.step()
        self.assertEqual(self.store.valid_snapshots("n1")[0], [])

    def test_invalidation_withdraws_previous_best(self):
        self.publish()
        export_results(self.ws)
        self.store.invalidate("n1", "validation rows included in training")
        self.assertIsNone(export_results(self.ws))
        self.assertFalse((self.ws / "best_submission/submission.csv").exists())
        self.assertEqual(read_json(self.run / "logs/candidate_results/selection.json")["selected"], [])

    def test_one_candidate_one_ensemble_member_and_full_time_charged(self):
        session, model = self.session()
        session.step()
        model["weight"][:] = [1.0, 0.0]
        session.last_validation_training_seconds = -1e6
        session.last_export = 0
        session.step()
        self.store.write_execution("n1", status="timeout", elapsed_seconds=123.5)
        export_results(self.ws)
        self.assertFalse((self.ws / "top_solution/top2").exists())
        metric = (self.ws / "top_solution/top1/metric.txt").read_text()
        self.assertIn("123.500000", metric)
        self.assertIn("Execution Status: timeout", metric)

    def test_concurrent_export_and_failed_staging_preserve_best(self):
        self.publish("n1")
        session, model = self.session("n2")
        model["weight"][:] = [1.0, 0.0]
        session.step()
        session.finish()
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: export_results(self.ws), range(3)))
        before = (self.ws / "best_submission/submission.csv").read_bytes()
        self.assertEqual((self.ws / "best_solution/node_id.txt").read_text(), "n2")
        with patch("engine.candidate_runtime.integration.shutil.copy2", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                export_results(self.ws)
        self.assertEqual(before, (self.ws / "best_submission/submission.csv").read_bytes())

    def test_config_schema_and_opt_in(self):
        from config import Config
        from omegaconf import OmegaConf
        base = OmegaConf.load(ROOT / "config/config.yaml")
        merged = OmegaConf.merge(OmegaConf.structured(CandidateRuntimeConfig), base.candidate_runtime)
        self.assertIn("candidate_runtime", Config.__dataclass_fields__)
        self.assertFalse(merged.enabled)
        self.assertIsNone(merged.draft_budget_seconds)
        CandidateRuntimeConfig(**OmegaConf.to_container(merged)).validate()
        with self.assertRaises(ValueError):
            CandidateRuntimeConfig(smoke_steps=0).validate()

    def test_stage_budget_and_queue_wait_use_absolute_deadline(self):
        self.runtime.draft_budget_seconds = 15
        self.runtime.candidate_budget_seconds = 25
        self.register("draft", CHILD_CODE)
        self.register("improve", CHILD_CODE, stage="improve")
        started = time.time()
        _, draft_deadline = begin_execution(self.cfg, "draft", CHILD_CODE, started, 60, started + 40)
        _, improve_deadline = begin_execution(self.cfg, "improve", CHILD_CODE, started, 60, started + 40)
        self.assertEqual(draft_deadline, started + 15)
        self.assertEqual(improve_deadline, started + 25)
        with self.assertRaises(TimeoutError):
            begin_execution(self.cfg, "draft", CHILD_CODE, started, 60, started - 1)

    def test_session_budget_uses_candidate_admission_not_old_run_start(self):
        self.runtime.candidate_budget_seconds = 7200
        self.runtime.finalization_reserve_seconds = 900
        self.register("late_improve", CHILD_CODE, stage="improve")
        started = time.time()
        # Match the S59 failure: over 100 minutes passed before this candidate ran.
        atomic_json(self.store.root / "run.json", {"started_at": started - 6162})
        for whole_run_remaining, expected in [(10000, 7200), (1800, 1800)]:
            with self.subTest(whole_run_remaining=whole_run_remaining):
                with patch("engine.candidate_runtime.integration.time.time", return_value=started):
                    path, deadline = begin_execution(self.cfg, "late_improve", CHILD_CODE,
                                                     started, 21600, started + whole_run_remaining)
                session = CandidateSession(read_json(path))
                with patch("engine.candidate_runtime.session.time.time", return_value=started + 140):
                    self.assertEqual(deadline, started + expected)
                    self.assertAlmostEqual(session.elapsed(), 140)
                    self.assertAlmostEqual(session.remaining(), expected - 140)
                    self.assertGreater(session.remaining(), 900)

    def run_child(self, suffix="", timeout=20):
        code = CHILD_CODE + suffix
        self.register("child", code)
        with patch("engine.executor.visible_gpu_devices", return_value=[]):
            interpreter = Interpreter(self.ws, timeout=timeout, cfg=self.cfg)
        self.addCleanup(interpreter.terminate_all_subprocesses)
        return interpreter.run(code, "child")

    def test_real_training_subprocess_normal_completion(self):
        result = self.run_child(RESULT_CONSUMER)
        self.assertIsNone(result.exc_type, "".join(result.term_out))
        self.assertEqual(result.execution_status, "completed")
        self.assertTrue(self.store.valid_snapshots("child")[0])

    def test_real_training_subprocess_budget_stop_consumes_same_result(self):
        self.runtime.finalization_reserve_seconds = 30
        result = self.run_child(RESULT_CONSUMER)
        self.assertIsNone(result.exc_type, "".join(result.term_out))
        self.assertEqual(result.execution_status, "budget_exhausted")
        saved = read_json(self.store.candidate_dir("child") / "worker_finished.json")
        self.assertEqual(saved["optimizer_steps"], 1)
        self.assertEqual(saved["selected_optimizer_steps"], 1)
        self.assertTrue(self.store.valid_snapshots("child")[0])

    def test_real_subprocess_crash_after_publication(self):
        result = self.run_child("\nraise RuntimeError('simulated CUDA OOM after publication')\n")
        self.assertEqual(result.execution_status, "failed")
        self.assertTrue(self.store.valid_snapshots("child")[0])
        export_results(self.ws)
        self.assertTrue((self.ws / "best_submission/submission.csv").exists())

    def test_real_subprocess_hard_timeout_after_publication(self):
        result = self.run_child("\ntime.sleep(60)\n", timeout=6)
        self.assertEqual(result.exc_type, "TimeoutError", "".join(result.term_out))
        self.assertTrue(self.store.valid_snapshots("child")[0])
        top = export_results(self.ws)
        self.assertTrue((top / "top1/submission.csv").exists())

    def test_missing_protocol_fails_before_training(self):
        self.register("child", "raise AssertionError('must not run')")
        with patch("engine.executor.visible_gpu_devices", return_value=[]):
            interpreter = Interpreter(self.ws, cfg=self.cfg)
        result = interpreter.run("raise AssertionError('must not run')", "child")
        self.assertIn("missing runtime calls", "".join(result.term_out))
        self.assertEqual(result.execution_status, "failed")

    def parser_and_agent(self):
        import importlib
        unavailable = Mock(side_effect=AssertionError("No external model call expected"))
        fake_llm = patch.dict(sys.modules, {"llm": SimpleNamespace(FunctionSpec=lambda **kw: None,
                                                                 query=unavailable, generate=unavailable)})
        fake_llm.start()
        self.addCleanup(fake_llm.stop)
        parser = importlib.import_module("agents.result_parse_agent")
        agent = SimpleNamespace(cfg=self.cfg, acfg=self.cfg.agent, metric_maximize=True,
                                global_memory=None, branch_successful_nodes={})
        return parser, agent

    def test_parser_keeps_failed_artifact_but_routes_node_to_debug(self):
        self.publish()
        from engine.search_node import SearchNode
        parser, agent = self.parser_and_agent()
        for status, exception, healthy in [("completed", None, True), ("budget_exhausted", None, True),
                                           ("timeout", "TimeoutError", False), ("failed", "RuntimeError", False)]:
            node = SearchNode(id="n1", code="print('candidate')", stage="draft", branch_id=1)
            result = ExecutionResult(["training output"], 100.0, exception, execution_status=status)
            node = parser.run(agent, node, result)
            self.assertEqual(node.is_buggy, not healthy)
            self.assertEqual(node.is_valid, healthy)
            self.assertEqual(node.artifact_status, "scoreable")
            self.assertEqual(node.metric.is_worst, not healthy)
            self.assertIsNotNone(node.artifact_metric)

    def test_perfect_score_is_pending_until_leakage_review(self):
        session, _ = self.session()
        candidate_path = session.directory / "candidate.json"
        candidate = read_json(candidate_path)
        candidate["check_leakage"] = True
        atomic_json(candidate_path, candidate)
        session.predict_validation = lambda indices: session.answers.target.to_numpy()[indices]
        session.step()
        session.finish()
        self.assertEqual(self.store.valid_snapshots("n1")[0], [])
        self.cfg.agent.check_data_leakage = True
        parser, agent = self.parser_and_agent()
        from engine.search_node import SearchNode
        node = SearchNode(id="n1", code="print('candidate')", stage="draft")
        with patch.object(parser.data_leakage_agent, "run", return_value={
                "has_leakage": True, "confidence": "high", "reason": "validation labels used as predictions"}):
            parser.run(agent, node, ExecutionResult(["output"], 1, None, execution_status="completed"))
        self.assertTrue(node.is_buggy)
        self.assertTrue((session.directory / "invalidated.json").exists())
        self.assertIsNone(export_results(self.ws))

    def test_direction_comes_from_contract_without_llm(self):
        parser, agent = self.parser_and_agent()
        agent.metric_maximize = False
        parser.determine_metric_direction(agent)
        self.assertTrue(agent.metric_maximize)

    def test_real_timeout_during_new_csv_write_preserves_old_export(self):
        suffix = '''
def interrupted_csv(path):
    path.write_text("id,prediction\\n10000,")
    time.sleep(60)
s.store.publish(s.node_id, s.best["checkpoint_id"], interrupted_csv, s.contract, s.steps, s.elapsed)
'''
        result = self.run_child(suffix, timeout=6)
        self.assertEqual(result.exc_type, "TimeoutError")
        good, _ = self.store.valid_snapshots("child")
        self.assertTrue(good)
        export_results(self.ws)
        recovered = pd.read_csv(self.ws / "best_submission/submission.csv")
        self.assertEqual(len(recovered), 80)

    def test_real_timeout_before_first_export_has_no_result(self):
        code = "import time\ntime.sleep(60)\n" + CHILD_CODE
        self.register("child", code)
        with patch("engine.executor.visible_gpu_devices", return_value=[]):
            interpreter = Interpreter(self.ws, timeout=1, cfg=self.cfg)
        self.addCleanup(interpreter.terminate_all_subprocesses)
        result = interpreter.run(code, "child")
        self.assertEqual(result.exc_type, "TimeoutError")
        self.assertEqual(self.store.valid_snapshots("child")[0], [])

    def test_fixed_split_matches_across_two_arm_workspaces(self):
        other = self.run / "arm-f"
        contract = prepare_contract(other, self.ws / "input", 51, self.runtime.validation_fraction)
        self.assertEqual(read_json(contract / "manifest.json")["contract_id"],
                         read_json(self.store.root / "contract/manifest.json")["contract_id"])

    def test_leakage_review_failure_does_not_release_perfect_score(self):
        session, _ = self.session()
        candidate = read_json(session.directory / "candidate.json")
        candidate["check_leakage"] = True
        atomic_json(session.directory / "candidate.json", candidate)
        session.predict_validation = lambda indices: session.answers.target.to_numpy()[indices]
        session.step()
        self.cfg.agent.check_data_leakage = True
        parser, agent = self.parser_and_agent()
        from engine.search_node import SearchNode
        node = SearchNode(id="n1", code="print('candidate')", stage="draft")
        with patch.object(parser.data_leakage_agent, "run", return_value={
                "has_leakage": False, "confidence": "low", "reason": "API unavailable", "check_succeeded": False}):
            parser.run(agent, node, ExecutionResult(["output"], 1, None, execution_status="completed"))
        self.assertTrue(node.metric.is_worst)
        self.assertIn("review unavailable", node.analysis)
        self.assertFalse((session.directory / "review.json").exists())
        self.assertIsNone(export_results(self.ws))

    def test_cli_recovery_and_ensemble_produce_gradeable_csv(self):
        self.publish()
        runs = self.run / "cli-runs"
        run_name = "20260910_000000_jubias-runtime-test"
        destination = runs / run_name
        shutil.copytree(self.ws, destination / "workspace", symlinks=True)
        shutil.copytree(self.run / "logs", destination / "logs")
        result = subprocess.run([sys.executable, str(ROOT / "utils/recover_candidate_results.py"),
                                 "--run-dir", str(destination)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, str(ROOT / "utils/submission_fusion_utils.py"),
                                 "--runs_root", str(runs), "--exp_name", run_name, "--task_id", TASK_ID],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        outputs = list((destination / "workspace/ensembles_csv").glob("top1ens-*.csv"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(pd.read_csv(outputs[0])), 80)

    def test_archive_dereferences_outputs_without_model_weights(self):
        self.publish()
        export_results(self.ws)
        archive = self.run / "fetch.tgz"
        subprocess.run(["tar", "chzf", str(archive), "workspace/best_submission", "workspace/best_solution",
                        "logs/candidate_results"], cwd=self.run, check=True, capture_output=True)
        import tarfile
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            self.assertIn("workspace/best_submission/submission.csv", names)
            self.assertFalse(any("weights.npy" in name for name in names))
            self.assertTrue(tar.getmember("workspace/best_submission/submission.csv").isfile())

    def test_programmatic_entry_returns_artifact_without_healthy_journal_node(self):
        self.publish()
        self.store.write_execution("n1", status="timeout", elapsed_seconds=123)
        tree = ast.parse((ROOT / "__init__.py").read_text())
        experiment = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Experiment")
        method = next(n for n in experiment.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        namespace = {"Solution": SimpleNamespace, "time": time, "save_run": Mock()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "experiment-test", "exec"), namespace)
        experiment = SimpleNamespace(cfg=self.cfg, agent=SimpleNamespace(),
            interpreter=SimpleNamespace(cleanup_session=Mock()),
            journal=SimpleNamespace(get_best_node=lambda: None))
        solution = namespace["run"](experiment, 0)
        self.assertEqual(solution.code, "print('candidate')")
        self.assertGreater(solution.valid_metric, 0)


CHILD_CODE = '''
import numpy as np
import pandas as pd
import time
from engine.candidate_runtime import CandidateSession
s = CandidateSession.from_env()
train, val, test = s.split(pd.read_csv("input/train.csv"), pd.read_csv("input/test.csv"))
w = np.array([0.0, 1.0])
def predict(frame, indices):
    return 1 / (1 + np.exp(-frame.iloc[indices][["x", "z"]].to_numpy() @ w))
def save(directory):
    np.save(directory / "weights.npy", w)
def load(directory):
    w[:] = np.load(directory / "weights.npy", allow_pickle=False)
s.bind(predict_validation=lambda indices: predict(val, indices), predict_test=lambda indices: predict(test, indices),
       save_checkpoint=save, load_checkpoint=load)
s.start_training(train.id)
for i in range(5):
    x = train[["x", "z"]].to_numpy()
    prob = 1 / (1 + np.exp(-x @ w))
    w -= 0.1 * x.T @ (prob - train.target.to_numpy()) / len(train)
    if s.step():
        break
    time.sleep(0.02)
# Protocol hook exists, but this branch is deliberately false in crash/kill tests.
if False:
    s.finish()
'''


RESULT_CONSUMER = '''
from pathlib import Path
result = s.finish()
assert result == s.finish()
assert result["best_validation_score"] == s.best_validation_score == s.best_score
assert 0 <= result["best_validation_score"] <= 1
assert Path(result["submission_path"]).is_file()
assert Path(result["validation_path"]).is_file()
'''


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    unittest.main(verbosity=2)
