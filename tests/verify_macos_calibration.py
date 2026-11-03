"""Synthetic calibration lifecycle: real contracts/snapshots, no candidate execution."""

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.candidate_runtime.integration import end_execution, settings
from engine.candidate_runtime.diagnostics import validation_diagnostics
from engine.candidate_runtime.jigsaw import IDENTITIES, prepare_contract
from engine.candidate_runtime.session import CandidateSession
from engine.candidate_runtime.store import ResultStore
from engine.executor import ExecutionResult
from tests import verify_replay_improve as legacy
from utils.replay_macos import preparation, runtime as mac
import replay_improve as replay


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        legacy.ReplayTests.setUp(self)
        self.addCleanup(legacy.ReplayTests.tearDown, self)
        self.inputs = self.source / "workspace/input"
        rows = np.arange(128)
        train = pd.DataFrame({"id": rows, "target": rows % 2})
        for name in IDENTITIES:
            train[name] = (rows // 2) % 2
        train.to_csv(self.inputs / "train.csv", index=False)
        pd.DataFrame({"id": np.arange(1000, 1016)}).to_csv(self.inputs / "test.csv", index=False)
        self.cfg_path = self.source / "logs/config.yaml"
        cfg = replay.load_config(self.cfg_path)
        cfg.exp_id = mac.TASK
        cfg.agent.seed, cfg.agent.time_limit, cfg.exec.timeout = 51, 1200, 3600
        cfg.candidate_runtime.enabled = True
        cfg.candidate_runtime.validation_fraction = .5
        cfg.candidate_runtime.smoke_steps = 1
        cfg.candidate_runtime.smoke_rows = 8
        cfg.candidate_runtime.first_validation_seconds = .000001
        cfg.candidate_runtime.validation_interval_seconds = .000001
        cfg.candidate_runtime.export_interval_seconds = .000001
        cfg.candidate_runtime.finalization_reserve_seconds = .01
        OmegaConf.save(cfg, self.cfg_path)
        prepare_contract(self.source / "workspace", self.inputs, 51, .5)
        self.review_path = self.source / "workspace/candidate_results/candidates/parent/review.json"
        replay.write_json(self.review_path, {"status": "passed", "fixture": "synthetic source review"})
        self.original_files = {str(p.relative_to(self.source)): p.read_bytes()
                               for p in self.source.rglob("*") if p.is_file()}
        self.output = self.root / "mac_baseline"
        self.args = SimpleNamespace(runs_root=self.root, source_run="original", parent_node_id="parent",
                                    data_dir=None, dataset_dir=None, output_dir=self.output, execute=True)
        self.adapted = "print('synthetic adapted source; never executed')\n"
        self.adapter = SimpleNamespace(ADAPTER_VERSION="synthetic-mps-v1",
                                       adapt=Mock(return_value=self.adapted))
        self.environment = {"profile": "mac-mps", "training_device": "mps", "memory_device": "cpu"}
        self.interpreters = []

    def interpreter(self, workspace, *, cfg, timeout, **kwargs):
        instance = Mock(cfg=cfg, timeout=timeout)
        instance.run.side_effect = lambda code, node_id, immediate: self.publish(instance, code, node_id)
        self.interpreters.append(instance)
        return instance

    def publish(self, interpreter, code, node_id):
        """Stand in for execution with a constant predictor and real runtime artifacts."""
        self.assertEqual(code, self.adapted)
        self.assertEqual(interpreter.timeout, 1130)  # Original 1200 - parent completion at 70.
        started = time.time()
        self.assertAlmostEqual(interpreter.run_deadline - started, 1130, delta=1)
        cfg = interpreter.cfg
        session = CandidateSession(dict(workspace=str(cfg.workspace_dir), log_dir=str(cfg.log_dir),
                                        node_id=node_id, started_at=started,
                                        deadline=interpreter.run_deadline, config=settings(cfg)))
        fit, _, _ = session.split(pd.read_csv(cfg.workspace_dir / "input/train.csv"),
                                   pd.read_csv(cfg.workspace_dir / "input/test.csv"))

        def save(directory):
            (directory / "constant.txt").write_text("0.5")

        def load(directory):
            self.assertEqual((directory / "constant.txt").read_text(), "0.5")

        session.bind(predict_validation=lambda indices: np.full(len(indices), .5),
                     predict_test=lambda indices: np.full(len(indices), .5),
                     save_checkpoint=save, load_checkpoint=load)
        session.start_training(fit.id)
        session.step()
        session.finish()
        self.published = session
        if self.mode == "corrupt_checkpoint":
            checkpoints = list((session.directory / "checkpoints").glob("*/model/constant.txt"))
            self.assertTrue(checkpoints)
            for checkpoint in checkpoints:
                checkpoint.write_text("changed")
        error = "RuntimeError" if self.mode == "execution_failed" else None
        result = ExecutionResult(term_out=["synthetic predictor completed"], exec_time=time.time() - started,
                                 exc_type=error,
                                 exc_info={"message": "MPS backend out of memory (synthetic)"} if error else None)
        return end_execution(cfg, node_id, result, started, interpreter.run_deadline)

    def run_calibration(self, mode="completed"):
        self.mode = mode
        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch("utils.replay_macos.adapters.get_adapter", return_value=self.adapter), \
                patch("engine.executor.Interpreter", side_effect=self.interpreter):
            return preparation.prepare(self.args)

    def load_completed(self, manifest):
        with patch("utils.replay_macos.adapters.get_adapter", return_value=self.adapter):
            return mac.load_baseline(self.output, manifest["source"], self.environment)

    def test_calibration_publishes_verified_baseline_and_preserves_frozen_source(self):
        result = self.run_calibration()
        self.assertEqual(result["status"], "completed")
        manifest = json.loads((self.output / "baseline.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["source"]["replay_budget_seconds"], 1130)
        self.assertEqual(manifest["source_review_sha256"], mac.digest(self.review_path))
        self.assertEqual(manifest["adapter_version"], self.adapter.ADAPTER_VERSION)
        self.assertIsNone(manifest["private_dataset_dir"])
        checked, node = self.load_completed(manifest)  # Real ResultStore.verify and score recomputation.
        self.assertEqual(checked, manifest)
        self.assertAlmostEqual(node["metric"]["value"], .5)
        self.assertEqual(node["id"], result["effective_parent_id"])
        self.assertTrue(node["is_valid"])
        self.assertFalse(node["is_buggy"])
        self.assertGreater(node["ctime"], 10)  # New time never recomputes the original frozen budget.
        store = ResultStore(self.output / "calibration/workspace", self.output / "calibration/logs")
        review = json.loads((store.candidate_dir(node["id"]) / "review.json").read_text())
        self.assertEqual(review["status"], "passed")
        self.assertEqual(review["kind"], "locked-platform-adaptation-and-runtime-verification")
        self.assertFalse(review["llm_review"])
        self.assertEqual(review["source_sha256"], manifest["source"]["original_source_sha256"])
        self.assertEqual(review["adapted_source_sha256"], mac.digest(self.output / "solution.py"))
        diagnostics = validation_diagnostics(self.interpreters[0].cfg, SimpleNamespace(**node))
        self.assertTrue(diagnostics["available"], diagnostics)
        self.assertAlmostEqual(diagnostics["metric"], .5)
        self.assertEqual(diagnostics["selection"]["source_sha256"], review["adapted_source_sha256"])
        self.assertEqual(self.original_files, {str(p.relative_to(self.source)): p.read_bytes()
                                              for p in self.source.rglob("*") if p.is_file()})
        self.interpreters[0].cleanup_session.assert_called_once_with(-1)
        with self.assertRaisesRegex(ValueError, "already executed or failed"):
            self.run_calibration()
        self.assertEqual(len(self.interpreters), 1)

    def test_corrupted_checkpoint_cannot_complete_calibration(self):
        with self.assertRaisesRegex(RuntimeError, "no independently verified complete snapshot"):
            self.run_calibration("corrupt_checkpoint")
        manifest = json.loads((self.output / "baseline.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("snapshot", manifest["error"])
        self.assertNotIn("effective_parent_id", manifest)
        self.assertFalse((self.output / "baseline_node.json").exists())
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            self.load_completed(manifest)
        self.interpreters[0].cleanup_session.assert_called_once_with(-1)

    def test_failed_execution_with_good_snapshot_is_not_a_healthy_baseline(self):
        with self.assertRaisesRegex(RuntimeError, "calibration execution failed: RuntimeError"):
            self.run_calibration("execution_failed")
        manifest = json.loads((self.output / "baseline.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn("effective_parent_id", manifest)
        store = self.published.store
        good, rejected = store.valid_snapshots(self.published.node_id)
        self.assertTrue(good)
        self.assertFalse(rejected)
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            self.load_completed(manifest)
        self.interpreters[0].cleanup_session.assert_called_once_with(-1)

    def test_failed_execution_reports_cause_and_saved_log(self):
        with self.assertRaises(RuntimeError) as raised:
            self.run_calibration("execution_failed")
        execution_path = (self.output / "calibration/logs/executions"
                          / f"{self.published.node_id}.json")
        self.assertIn("MPS backend out of memory (synthetic)", str(raised.exception))
        self.assertIn(str(execution_path), str(raised.exception))
        execution = json.loads(execution_path.read_text())
        self.assertEqual(execution["exc_info"]["message"], "MPS backend out of memory (synthetic)")
        manifest = json.loads((self.output / "baseline.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn(str(raised.exception), manifest["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
