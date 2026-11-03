"""Offline Mac replay isolation, preparation and artifact validation regressions."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from engine.analogy.agent import AnalogyResult
from engine.search_node import SearchNode
from utils.metric import MetricValue
from utils.replay_macos import runtime as mac, preparation
from utils.replay_state import snapshot_journal
from tests import verify_replay_improve as legacy
import replay_improve as replay


class MacReplayTests(unittest.TestCase):
    def setUp(self):
        legacy.ReplayTests.setUp(self)
        self.cfg_path = self.source / "logs/config.yaml"
        cfg = replay.load_config(self.cfg_path)
        cfg.exp_id = mac.TASK
        cfg.candidate_runtime.enabled = True
        OmegaConf.save(cfg, self.cfg_path)
        inputs = self.source / "workspace/input"
        (inputs / "test.csv").write_text("id\n1\n")
        self.contract = {f"public_{split}_sha256": mac.digest(inputs / f"{split}.csv")
                         for split in ("train", "test")}
        self.contract_path = self.source / "workspace/candidate_results/contract/manifest.json"
        replay.write_json(self.contract_path, self.contract)
        replay.write_json(self.source / "workspace/candidate_results/candidates/parent/review.json", {"status": "passed"})
        self.snapshot = snapshot_journal(json.loads(self.before), "parent")
        self.budget = replay.remaining_budget(cfg, self.snapshot)
        self.expected = mac.source_identity(self.source, self.snapshot, self.budget)
        self.environment = {"profile": "mac-mps", "training_device": "mps", "memory_device": "cpu"}
        self.baseline_dir = self.root / "baseline"
        self.baseline_dir.mkdir()
        self.node = copy.deepcopy(self.snapshot["nodes"][-1])
        self.node.update(id="macparent", code="print('mps')", stage="improve", ctime=1800000000,
                         created_time="2026-10-30T00:00:00", finish_time="2026-10-30T00:01:00",
                         metric={"value": .7, "maximize": True}, best_snapshot_id="snapshot1",
                         analysis="Mac measured baseline", step=2)
        self.adapter = SimpleNamespace(ADAPTER_VERSION="synthetic-mps-v1",
                                       adapt=Mock(return_value=self.node["code"]))
        self.adapter_lookup = self.enterContext(patch("utils.replay_macos.adapters.get_adapter",
                                                     return_value=self.adapter))
        self.manifest = dict(version=1, status="completed", profile="mac-mps", source=self.expected,
                             adapter_version=self.adapter.ADAPTER_VERSION, environment=self.environment,
                             private_dataset_dir=None, effective_parent_id="macparent", metric=.7)
        (self.baseline_dir / "original_solution.py").write_text(self.snapshot["nodes"][-1]["code"])
        (self.baseline_dir / "solution.py").write_text(self.node["code"])
        replay.write_json(self.baseline_dir / "baseline_node.json", self.node)
        calibration = self.baseline_dir / "calibration/workspace/candidate_results"
        replay.write_json(calibration / "contract/manifest.json", self.contract)
        replay.write_json(calibration / "candidates/macparent/execution.json", {"status": "completed"})
        replay.write_json(calibration / "candidates/macparent/snapshots/snapshot1/manifest.json", {"checkpoint_id": "checkpoint1"})
        (calibration / "candidates/macparent/snapshots/snapshot1/validation.csv").write_text("id,prediction\n0,0.5\n")
        replay.write_json(calibration / "candidates/macparent/checkpoints/checkpoint1/manifest.json", {})
        self.manifest["files"] = {str(p.relative_to(self.baseline_dir)): mac.digest(p)
                                  for p in self.baseline_dir.rglob("*") if p.is_file()}
        self.save_manifest()
        self.verified = dict(metric=.7, maximize=True, source_sha256=mac.digest(self.baseline_dir / "solution.py"))
        self.addCleanup(legacy.ReplayTests.tearDown, self)

    def save_manifest(self):
        replay.write_json(self.baseline_dir / "baseline.json", self.manifest)

    def args(self, arm="A"):
        args = legacy.ReplayTests.args(self, arm)
        args.profile, args.prepared_baseline = "mac-mps", self.baseline_dir
        args.output_root, args.dataset_dir = self.root / "mac_runs", None
        return args

    def test_legacy_does_not_probe_or_import_mac_runtime(self):
        args = legacy.ReplayTests.args(self)
        with patch.object(replay, "_run", return_value="legacy") as run, patch.object(mac, "require_mps") as probe:
            self.assertEqual(replay.run(args), "legacy")
            run.assert_called_once_with(args)
            probe.assert_not_called()

    def test_manual_launcher_preflight_import_is_preserved(self):
        from utils.replay_macos import require_mps

        self.assertIs(require_mps, mac.require_mps)

    def test_invalid_profile_options_fail_before_any_output(self):
        args = self.args()
        with patch.object(replay, "_run") as run:
            args.profile = None
            with self.assertRaisesRegex(ValueError, "require --profile"):
                replay.run(args)
            args.profile, args.prepared_baseline = "mac-mps", None
            with self.assertRaisesRegex(ValueError, "requires --prepared-baseline"):
                replay.run(args)
            run.assert_not_called()

    def test_preflight_rejects_non_mac_and_fallback(self):
        with patch("platform.system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                mac.require_mps()
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"), \
                patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "fallback"):
                mac.require_mps()

    def test_process_environment_restored_after_failure(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                with mac.mac_process_environment():
                    self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
                    raise RuntimeError("test")
            self.assertNotIn("PYTORCH_ENABLE_MPS_FALLBACK", os.environ)

    def test_preflight_rejects_torch_with_mps_memory_growth(self):
        torch = SimpleNamespace(__version__="2.9.1")
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"), \
                patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}), \
                patch.dict(sys.modules, {"torch": torch}):
            with self.assertRaisesRegex(RuntimeError, "torch>=2.14.1.*memory fixes"):
                mac.require_mps()

    def test_preflight_accepts_fixed_torch_and_records_environment(self):
        torch = SimpleNamespace(__version__="2.14.1", backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True)))
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"), \
                patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}), \
                patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(mac.require_mps()["training_device"], "mps")

    def test_profile_retains_explicit_private_grading_opt_in(self):
        cfg = replay.load_config(self.cfg_path)
        self.assertIsNone(mac.apply_profile(cfg))
        self.assertEqual(cfg.agent.memory_embedding_device, "cpu")
        self.assertFalse(cfg.use_grading_server)
        self.assertEqual(mac.apply_profile(cfg, self.root), str(self.root.resolve()))
        cfg.exp_id = "another-task"
        with self.assertRaisesRegex(ValueError, "only Jigsaw"):
            mac.apply_profile(cfg)

    def test_memory_stays_on_cpu_from_construction_and_cuda_path_is_unchanged(self):
        from agents.memory.embedding_models import EmbeddingModel

        constructor = Mock()
        module = SimpleNamespace(SentenceTransformer=constructor)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            EmbeddingModel(model_name="test", device="cpu")
            constructor.assert_called_once_with("test", device="cpu")
            constructor.return_value.to.assert_called_once_with("cpu")
            constructor.reset_mock()
            EmbeddingModel(model_name="test", device="cuda")
            constructor.assert_called_once_with("test")
            constructor.return_value.to.assert_called_once_with("cuda")

    def test_default_prepare_writes_only_sources_and_is_repeatable(self):
        args = self.args()
        args.output_dir, args.execute = self.root / "prepared", False
        with patch("engine.candidate_runtime.jigsaw.load_contract", return_value=self.contract), \
                patch.object(preparation, "calibrate") as train:
            result = preparation.prepare(args)
            self.adapter_lookup.assert_called_once_with(self.expected["original_source_sha256"])
            self.adapter.adapt.assert_called_once_with(self.snapshot["nodes"][-1]["code"])
            self.assertFalse(result["training_started"])
            self.assertFalse((args.output_dir / "calibration").exists())
            saved = json.loads((args.output_dir / "baseline.json").read_text())
            self.assertIsNone(saved["private_dataset_dir"])
            self.assertEqual(saved["source"]["replay_budget_seconds"], self.budget)
            self.assertEqual(preparation.prepare(args), result)
            train.assert_not_called()
            (args.output_dir / "solution.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source changed"):
                preparation.prepare(args)

    def test_completed_baseline_checks_source_environment_and_measurement(self):
        with patch("engine.candidate_runtime.store.ResultStore.verify", return_value=self.verified):
            _, node = mac.load_baseline(self.baseline_dir, self.expected, self.environment)
            self.adapter_lookup.assert_called_once_with(self.expected["original_source_sha256"])
            self.assertEqual(node["id"], "macparent")
            with self.assertRaisesRegex(ValueError, "different source"):
                mac.load_baseline(self.baseline_dir, {**self.expected, "replay_budget_seconds": 1}, self.environment)
            with self.assertRaisesRegex(ValueError, "environment differs"):
                mac.load_baseline(self.baseline_dir, self.expected, {"python": "different"})
            self.adapter.ADAPTER_VERSION = "another-version"
            with self.assertRaisesRegex(ValueError, "different adapter version"):
                mac.load_baseline(self.baseline_dir, self.expected, self.environment)
            self.adapter.ADAPTER_VERSION = self.manifest["adapter_version"]
            self.verified["metric"] = .8
            with self.assertRaisesRegex(ValueError, "measurements"):
                mac.load_baseline(self.baseline_dir, self.expected, self.environment)

    def test_artifact_changes_or_unfinished_baseline_fail_closed(self):
        self.manifest["status"] = "prepared"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            mac.load_baseline(self.baseline_dir, self.expected, self.environment)
        self.manifest["status"] = "completed"
        self.save_manifest()
        (self.baseline_dir / "baseline_node.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            mac.load_baseline(self.baseline_dir, self.expected, self.environment)

    def test_bad_data_or_baseline_preserve_existing_arm(self):
        marker = self.args().output_root / "replay_paired/A/keep.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("previous result")
        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", side_effect=ValueError("baseline mismatch")):
            with self.assertRaisesRegex(ValueError, "baseline mismatch"):
                replay.run(self.args())
        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", return_value=(self.manifest, self.node)), \
                patch("engine.candidate_runtime.jigsaw.load_contract", return_value=self.contract):
            (self.source / "workspace/input/train.csv").write_text("changed")
            with self.assertRaisesRegex(ValueError, "does not match"):
                replay.run(self.args())
        self.assertEqual(marker.read_text(), "previous result")

    def test_protected_and_cluster_output_directories_rejected(self):
        for destination in (self.source, self.source.parent, self.source / "nested"):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                mac.ensure_separate(destination, self.source)
        args = self.args()
        args.output_root = args.runs_root
        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", return_value=(self.manifest, self.node)), \
                patch("engine.candidate_runtime.jigsaw.load_contract", return_value=self.contract):
            with self.assertRaisesRegex(ValueError, "separate --output-root"):
                replay.run(args)

    def test_four_arms_share_measured_parent_and_original_frozen_budget(self):
        seen = []

        def execute(agent, parent, text):
            seen.append((parent.id, parent.metric.value, agent.replay_budget_seconds, agent.replay_profile))
            self.assertEqual([n.id for n in agent.top_candidates], ["macparent"])
            self.assertEqual(agent.journal.nodes[1].metric.value, .9)
            self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
            return SearchNode(code="", id="child" + str(len(seen)), stage="improve", parent=parent,
                              metric=MetricValue(.75, maximize=True), is_valid=True, is_buggy=False)

        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", return_value=(self.manifest, self.node)), \
                patch("engine.candidate_runtime.jigsaw.load_contract", return_value=self.contract), \
                patch("engine.candidate_runtime.integration.prepare"), \
                patch("engine.executor.Interpreter"), \
                patch("utils.replay_inputs.prepare_input", return_value=("test", None, {})), \
                patch("engine.analogy.agent.retrieve_for_node", return_value=AnalogyResult(report_md="report")), \
                patch.object(replay, "execute_one", side_effect=execute), \
                patch.object(replay, "private_score") as private:
            for arm in "ABCD":
                result = replay.run(self.args(arm))
                self.assertEqual(result["parent_node_id"], "macparent")
                self.assertEqual(result["original_parent_id"], "parent")
                self.assertEqual(result["original_parent_metric"], .9)
                self.assertEqual(result["parent_metric"], .7)
                self.assertEqual(result["private_evaluation"]["status"], "unavailable")
            private.assert_not_called()
        self.assertEqual(set(seen), {("macparent", .7, self.budget, "mac-mps")})
        self.assertEqual(len(seen), 4)
        self.assertEqual((self.source / "logs/journal.json").read_bytes(), self.before)
        self.assertFalse((self.root / "replay_paired").exists())
        saved = json.loads((self.args().output_root / "replay_paired/experiment.json").read_text())
        self.assertEqual(saved["effective_parent_id"], "macparent")
        self.assertEqual(saved["replay_budget_seconds"], self.budget)


if __name__ == "__main__":
    unittest.main(verbosity=2)
