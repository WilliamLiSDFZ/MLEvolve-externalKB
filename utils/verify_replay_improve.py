"""Offline replay lifecycle checks, including real execution of a tiny generated program."""
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from engine.analogy.agent import AnalogyResult
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue
from utils.serialize import dump_json
import replay_improve as replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "original"
        (self.source / "logs").mkdir(parents=True)
        data = self.source / "workspace/input"
        data.mkdir(parents=True)
        (data / "train.csv").write_text("id,target\n0,1\n")
        cfg = OmegaConf.load(replay.ROOT / "config/config.yaml")
        cfg.data_dir, cfg.log_dir, cfg.workspace_dir = str(data), str(self.source / "logs"), str(self.source / "workspace")
        cfg.exp_name, cfg.exp_id = "original", "toy"
        cfg.goal, cfg.preprocess_data = "Toy task", False
        cfg.agent.use_global_memory, cfg.use_grading_server = False, False
        cfg.coldstart.use_coldstart = False
        cfg.agent.time_limit, cfg.exec.timeout = 7200, 30
        cfg.cpu_number = 1
        OmegaConf.save(cfg, self.source / "logs/config.yaml")
        root = SearchNode(code="", stage="root", id="root", ctime=0, metric=MetricValue(None))
        parent = SearchNode(code="print('original')", stage="draft", id="parent", parent=root,
                            branch_id=1, ctime=10, metric=MetricValue(.9, maximize=True),
                            is_buggy=False, is_valid=True,
                            created_time="2026-01-01T00:00:10", finish_time="2026-01-01T00:01:10",
                            prompt_input="# Task description\nClassify examples.\n# Instructions\nUse input.\n"
                                         "examine the dataset:\nOne training CSV.")
        journal = Journal()
        journal.append(root)
        journal.append(parent)
        dump_json(journal, self.source / "logs/journal.json")
        self.before = (self.source / "logs/journal.json").read_bytes()
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        (self.corpus / "manifest.json").write_text('{"records_sha1":"test-only"}')
        self.env = patch.dict(os.environ, {"LLM_API_KEY": "", "MLEVOLVE_REQUIRED_MODEL": "",
                                         "MLEVOLVE_REQUIRE_GPT6": "0"})
        self.env.start()
        self.old_tempdir = tempfile.tempdir

    def tearDown(self):
        # isolate_caches changes process env only; a normal job ends after one arm.
        self.env.stop()
        tempfile.tempdir = self.old_tempdir
        self.temp.cleanup()

    def args(self, arm="A"):
        return SimpleNamespace(runs_root=self.root, source_run="original", parent_node_id="parent",
                               experiment_id="paired", arm=arm, corpus=self.corpus, data_dir=None, seed=42)

    def test_each_arm_executes_one_child_without_touching_source_or_other_arms(self):
        seen = []
        code = "from pathlib import Path\nPath('submission/submission.csv').write_text('id,prediction\\n0,0.5\\n')\nprint('trained')\n"

        def improve(agent, parent, *, injection_text):
            seen.append((agent.task_desc, parent.id, injection_text, agent.replay_budget_seconds))
            return SearchNode(code=code, plan="test", stage="improve", parent=parent,
                              branch_id=parent.branch_id, prompt_input=injection_text)

        def parse(agent, child, result):
            self.assertIsNone(result.exc_type, result.term_out)
            child.absorb_exec_result(result)
            child.metric = MetricValue(.8, maximize=True)
            child.is_buggy, child.is_valid = False, True
            return child

        def prepare(arm, agent, parent, output_dir, seed):
            text = "" if arm == "A" else arm * replay.REPORT_CHARS
            (output_dir / "injection.txt").write_text(text)
            return text, None, {}

        with patch("engine.agent_search.clean_task_desc", side_effect=AssertionError("Must not rewrite task")), \
             patch("agents.result_parse_agent.determine_metric_direction", side_effect=AssertionError("Must restore metric")), \
             patch("agents.improve_agent.run", side_effect=improve), \
             patch("agents.code_review_agent.run", side_effect=lambda agent, node: node.code), \
             patch("agents.result_parse_agent.run", side_effect=parse), \
             patch("utils.replay_inputs.prepare_input", side_effect=prepare), \
             patch("engine.analogy.agent.retrieve_for_node", return_value=AnalogyResult(report_md="D report")) as analogy:
            for arm in "ABCD":
                result = replay.run(self.args(arm))
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["metric"], .8)
                self.assertTrue(Path(result["submission"]).is_file())
                self.assertEqual(result["private_evaluation"]["status"], "unavailable")
                folder = self.root / "replay_paired" / arm
                journal = json.loads((folder / "logs/journal.json").read_text())
                self.assertEqual(len(journal["nodes"]), 3)
                self.assertEqual(journal["node2parent"][result["child_node_id"]], "parent")
                self.assertTrue((folder / "logs/child_solution.py").is_file())
                self.assertFalse((folder / "logs/best_solution.py").exists())
                self.assertEqual((folder / "source/logs/journal.json").read_bytes(), self.before)
            analogy.assert_called_once()
            b_result = (self.root / "replay_paired/B/result.json").read_bytes()
            (self.root / "replay_paired/A/stale.txt").write_text("stale")
            replay.run(self.args("A"))
            self.assertFalse((self.root / "replay_paired/A/stale.txt").exists())
            self.assertEqual((self.root / "replay_paired/B/result.json").read_bytes(), b_result)
        self.assertEqual(len({(task, parent, budget) for task, parent, _, budget in seen}), 1)
        self.assertEqual([len(item[2]) for item in seen[:4]], [0, 12000, 12000, 8])
        self.assertEqual((self.source / "logs/journal.json").read_bytes(), self.before)
        self.assertEqual((self.source / "workspace/input/train.csv").read_text(), "id,target\n0,1\n")

    def test_destination_validation_and_pairing_precede_deletion(self):
        identity = {"parent": "parent", "seed": 1}
        arm = replay.prepare_arm(self.root, self.source, "pair", "A", identity)
        marker = arm / "preserve.txt"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "different inputs"):
            replay.prepare_arm(self.root, self.source, "pair", "A", {"parent": "different"})
        self.assertTrue(marker.exists())
        with self.assertRaises(ValueError):
            replay.prepare_arm(self.root, self.source, "../original", "A", identity)
        (self.root / "replay_pair/B").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            replay.prepare_arm(self.root, self.source, "pair", "B", identity)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            replay.prepare_arm(self.root, self.root / "replay_pair/A", "pair", "C", identity)
        self.assertEqual((self.source / "logs/journal.json").read_bytes(), self.before)

    def test_failure_is_recorded_and_empty_d_never_becomes_a(self):
        with patch("engine.analogy.agent.retrieve_for_node", return_value=AnalogyResult(reason="no report")), \
             patch("agents.improve_agent.run") as improve:
            with self.assertRaisesRegex(RuntimeError, "did not deliver"):
                replay.run(self.args("D"))
            improve.assert_not_called()
        result = json.loads((self.root / "replay_paired/D/result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIn("no report", result["error"])

    def test_failed_training_keeps_missing_metric_instead_of_zero(self):
        def improve(agent, parent, *, injection_text):
            return SearchNode(code="raise RuntimeError('training failed')", stage="improve",
                              parent=parent, branch_id=1)

        def parse(agent, child, result):
            self.assertIsNotNone(result.exc_type)
            child.absorb_exec_result(result)
            child.metric = MetricValue(None)
            child.is_buggy, child.is_valid = True, False
            return child

        with patch("agents.improve_agent.run", side_effect=improve), \
             patch("agents.code_review_agent.run", side_effect=lambda agent, node: node.code), \
             patch("agents.result_parse_agent.run", side_effect=parse):
            result = replay.run(self.args("A"))
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["metric"])
        self.assertIsNone(result["submission"])
        self.assertTrue(result["is_buggy"])
        self.assertEqual(result["private_evaluation"]["status"], "unavailable")

    def test_private_scoring_checks_grader_and_never_substitutes_failure_score(self):
        csv = self.root / "child.csv"
        csv.write_text("id,prediction\n0,0.5\n")
        cfg = SimpleNamespace(dataset_dir=self.root, exp_id="toy")
        registry = Mock()
        competition = registry.set_data_dir.return_value.get_competition.return_value
        competition.grader.return_value = .75
        modules = {"mlebench": SimpleNamespace(), "mlebench.registry": SimpleNamespace(registry=registry),
                   "mlebench.utils": SimpleNamespace(load_answers=Mock(), read_csv=Mock())}
        with patch.dict(sys.modules, modules), \
             patch("utils.mlebench_patch.grading_metadata", return_value={"metric_version": "test"}) as provenance:
            self.assertEqual(replay.private_score(cfg, csv)["score"], .75)
            provenance.side_effect = RuntimeError("Unpatched grader")
            registry.reset_mock()
            result = replay.private_score(cfg, csv)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("score", result)
            registry.set_data_dir.assert_not_called()

    def test_copied_input_is_not_a_shared_symlink(self):
        cfg = replay.load_config(self.source / "logs/config.yaml")
        snapshot = json.loads(self.before)
        original = self.source / "workspace/input/train.csv"
        shared = self.root / "shared.csv"
        original.rename(shared)
        original.symlink_to(shared)
        target = self.root / "isolated"
        target.mkdir()
        replay.copy_inputs(self.source, target, cfg, snapshot, self.source / "workspace/input")
        copied = target / "workspace/input/train.csv"
        self.assertFalse(copied.is_symlink())
        copied.write_text("changed")
        self.assertEqual(shared.read_text(), "id,target\n0,1\n")

    def test_parallel_arms_publish_one_complete_pairing_manifest(self):
        identity = {"parent": "parent", "settings": list(range(1000))}
        with ThreadPoolExecutor(max_workers=4) as pool:
            targets = list(pool.map(lambda arm: replay.prepare_arm(
                self.root, self.source, "concurrent", arm, identity), "ABCD"))
        self.assertEqual({p.name for p in targets}, set("ABCD"))
        self.assertTrue(all(p.is_dir() for p in targets))
        self.assertEqual(json.loads((self.root / "replay_concurrent/experiment.json").read_text()), identity)

    def test_future_global_memory_is_not_copied(self):
        cfg = replay.load_config(self.source / "logs/config.yaml")
        cfg.agent.use_global_memory = True
        records = [{"record_id": "node_parent", "description": "known"},
                   {"record_id": "node_future", "description": "secret future result"}]
        replay.write_json(self.source / "workspace/global_memory/records.json", records)
        target = self.root / "memory-copy"
        target.mkdir()
        replay.copy_inputs(self.source, target, cfg, json.loads(self.before), self.source / "workspace/input")
        actual = json.loads((target / "workspace/global_memory/records.json").read_text())
        self.assertEqual(actual, records[:1])

    def test_runtime_history_keeps_code_and_verified_diagnostics_without_weights(self):
        from utils.verify_runtime_diagnostics import DiagnosticTests
        from engine.candidate_runtime.diagnostics import build_runtime_context
        from engine.analogy.code_tools import CodeReadingSession

        fixture = DiagnosticTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        candidate = fixture.root / "candidates/current"
        (candidate / "solution.py").write_text("print('example')\n")
        (candidate / "checkpoints/checkpoint1/weights.bin").write_bytes(b"do not copy weights")
        fixture.node.execution_status = "completed"
        fixture.node.exec_time = 900
        target = self.root / "diagnostics-copy"
        shutil.copytree(fixture.root / "contract", target / "workspace/candidate_results/contract")
        replay.copy_runtime_history(fixture.run, target, {"nodes": [
            {"id": "current", "stage": "draft", "best_snapshot_id": "snapshot1"}]})
        cfg = SimpleNamespace(workspace_dir=target / "workspace", log_dir=target / "logs",
                              exp_id=fixture.cfg.exp_id, candidate_runtime=fixture.cfg.candidate_runtime)
        context = build_runtime_context(SimpleNamespace(cfg=cfg), fixture.node)
        self.assertTrue(context["public_validation"]["available"], context)
        session = CodeReadingSession([fixture.node], "current", workspace=cfg.workspace_dir, runtime_enabled=True)
        self.assertTrue(session.sources["current"]["available"])
        history = cfg.workspace_dir / "candidate_results/candidates"
        self.assertFalse((history / "current/checkpoints/checkpoint1/weights.bin").exists())
        (target / "source").mkdir()
        history.rename(target / "source/candidates")
        self.assertFalse(history.exists())
        self.assertTrue((target / "source/candidates/current/solution.py").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
