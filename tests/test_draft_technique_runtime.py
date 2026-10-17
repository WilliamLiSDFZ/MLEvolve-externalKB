import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from engine.coldstart import draft_techniques as dt


ROOT = Path(__file__).resolve().parents[1]


def make_agent(root):
    code = SimpleNamespace(model="gpt-5.6-terra", temp=1)
    acfg = SimpleNamespace(time_limit=1000, steps=50, code=code, search=SimpleNamespace(
        parallel_search_num=3, top_candidates_size=5, stagnation_window=5),
        use_global_memory=False, max_fusion_drafts=2)
    cfg = SimpleNamespace(exp_id="demo", log_dir=root / "logs", cpu_number=8,
        workspace_dir=root / "workspace", pretrain_model_dir="/models", agent=acfg,
        exec=SimpleNamespace(timeout=900), coldstart=SimpleNamespace(use_coldstart=False, description=""),
        draft_techniques=SimpleNamespace(enabled=True, cache_dir=str(root / "corpus"), context_window=258400))
    return SimpleNamespace(cfg=cfg, acfg=acfg, start_time=100, current_step=2,
                           task_desc="Predict labels", data_preview="Current training columns")


class DraftTechniqueRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agent = make_agent(self.root)

    def test_runtime_uses_outer_deadline_and_container_limits(self):
        limits = {"/sys/fs/cgroup/memory.max": 48_000,
                  "/sys/fs/cgroup/memory.current": 10_000}
        fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=90_000))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(name="GPU", total_memory=24_000)))
        with patch.dict(sys.modules, {"psutil": fake_psutil, "torch": fake_torch}), \
                patch.dict(os.environ, {"MLEVOLVE_RUN_DEADLINE": "500"}), \
                patch.object(dt.time, "time", return_value=200), \
                patch.object(dt, "_read_limit", side_effect=lambda path: limits.get(path)):
            runtime = dt.runtime_context(self.agent)
        self.assertEqual(runtime["remaining_seconds"], 300)
        self.assertEqual(runtime["single_execution_limit_seconds"], 300)
        self.assertEqual(runtime["observed_available_memory_bytes"], 38_000)
        self.assertEqual(runtime["parallel_experiments"], 3)
        self.assertEqual(runtime["configured_cpu_slots_per_experiment"], 2)
        self.assertEqual(runtime["visible_gpus"][0]["name"], "GPU")
        self.assertIn("not reserved", runtime["resource_sharing"])

    def _corpus(self, status="complete", task="demo"):
        directory = self.root / "corpus/demo"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"task_name": task, "status": status}))
        (directory / "techniques.json").write_text('[{"technique_id":"original"}]')
        return directory

    def test_run_snapshot_preserves_shared_corpus_and_records_actual_context(self):
        corpus = self._corpus(status="partial")
        before = {p.name: p.read_bytes() for p in corpus.iterdir()}
        audit = {"input_count": 1}
        selected = [{"technique_id": "original", "evidence": "original quote"}]
        with patch.object(dt, "runtime_context", return_value={"remaining_seconds": 300}) as runtime, \
                patch.object(dt, "rerank_techniques", return_value=(selected, audit)) as rank, \
                patch.object(dt, "render_techniques", return_value="selected text"), \
                self.assertLogs("MLEvolve", level="WARNING"):
            self.assertEqual(dt.prepare_draft_techniques(self.agent), "selected text")
        self.assertEqual(rank.call_args.args[1:4],
                         (self.agent.task_desc, self.agent.data_preview, runtime.return_value))
        self.assertEqual(before, {p.name: p.read_bytes() for p in corpus.iterdir()})
        logs = self.agent.cfg.log_dir
        self.assertEqual(json.loads((logs / "draft_techniques.json").read_text()), selected)
        self.assertEqual((logs / "draft_techniques.md").read_text(), "selected text")
        saved = json.loads((logs / "draft_technique_rerank.json").read_text())
        self.assertEqual(saved["corpus_status"], "partial")
        self.assertEqual(len(saved["corpus_sha256"]), 64)

    def test_invalid_or_missing_corpus_fails_before_ranking(self):
        with patch.object(dt, "rerank_techniques") as rank:
            with self.assertRaises(FileNotFoundError):
                dt.prepare_draft_techniques(self.agent)
            self._corpus(task="another-task")
            with self.assertRaises(ValueError):
                dt.prepare_draft_techniques(self.agent)
            rank.assert_not_called()

    def test_ranking_failure_is_not_replaced_by_file_order(self):
        self._corpus()
        with patch.object(dt, "runtime_context", return_value={"remaining_seconds": 300}), \
                patch.object(dt, "rerank_techniques", side_effect=ValueError("missing IDs")):
            with self.assertRaisesRegex(ValueError, "missing IDs"):
                dt.prepare_draft_techniques(self.agent)
        self.assertFalse(self.agent.cfg.log_dir.exists())

    def test_coordinator_initializes_once_before_any_steps_and_does_not_reset_timer(self):
        fake_agents = ModuleType("agents")
        for name in ("draft_agent", "improve_agent", "debug_agent", "evolution_agent", "fusion_agent",
                     "aggregation_agent", "code_review_agent", "result_parse_agent"):
            setattr(fake_agents, name, SimpleNamespace(determine_metric_direction=Mock()))
        preview = ModuleType("utils.data_preview")
        preview.generate = Mock(return_value="fresh preview")
        preview.clean_task_desc = lambda desc, cfg: desc
        modules = {"agents": fake_agents, "utils.data_preview": preview,
                   "config": SimpleNamespace(Config=object),
                   "engine.executor": SimpleNamespace(ExecutionResult=object),
                   "engine.search_node": SimpleNamespace(SearchNode=lambda **kw: SimpleNamespace(**kw), Journal=object),
                   "utils.metric": SimpleNamespace(WorstMetricValue=lambda: None)}
        for name in ("node_selection", "evaluation", "execution", "solution_manager"):
            modules[f"engine.{name}"] = ModuleType(f"engine.{name}")
        spec = importlib.util.spec_from_file_location("coordinator_tested", ROOT / "engine/agent_search.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), \
                patch.object(dt, "prepare_draft_techniques", return_value="ranked25") as prepare:
            spec.loader.exec_module(module)
            journal = MagicMock(nodes=[object()])
            agent = module.AgentSearch("task", self.agent.cfg, journal)
            self.assertEqual(agent.draft_techniques, "ranked25")
            self.assertIn("fresh preview", agent.data_preview)
            start = agent.search_start_time
            self.assertIsNotNone(start)
            agent._run_single_step = Mock(return_value=(False, None))
            for _ in range(3):
                agent.step(SimpleNamespace(stage="improve"), lambda: None)
            self.assertEqual(agent.search_start_time, start)
            prepare.assert_called_once_with(agent)
            preview.generate.assert_called_once()
            self.agent.cfg.draft_techniques.enabled = False
            baseline = module.AgentSearch("task", self.agent.cfg, journal)
            self.assertEqual(baseline.draft_techniques, "")
            self.assertIsNone(baseline.data_preview)
            self.assertEqual(prepare.call_count, 1)
            self.agent.cfg.draft_techniques.enabled = True
            prepare.side_effect = ValueError("bad ranking")
            with self.assertRaisesRegex(ValueError, "bad ranking"):
                module.AgentSearch("task", self.agent.cfg, journal)

    def test_shell_preserves_startup_failure_and_propagates_budget(self):
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        log = self.root / "calls.jsonl"
        python = bin_dir / "python"
        python.write_text(f"#!{sys.executable}\n" +
            "import json,os,sys\n"
            "with open(os.environ['CALL_LOG'],'a') as f:\n"
            " f.write(json.dumps({'args':sys.argv[1:], 'deadline':os.getenv('MLEVOLVE_RUN_DEADLINE')})+'\\n')\n"
            "sys.exit(int(os.environ.get('RUN_EXIT','7')) if sys.argv[1]=='run.py' else 0)\n")
        python.chmod(0o755)
        timeout = bin_dir / "timeout"
        timeout.write_text('#!/bin/sh\nshift 4\nexec "$@"\n')
        timeout.chmod(0o755)
        env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
               "CALL_LOG": str(log), "SKIP_GRADING_SERVER": "1", "TIME_LIMIT_SECS": "123"}
        result = subprocess.run(["bash", str(ROOT / "run_single_task.sh"), "demo", str(self.root)],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 1, "failed startup must not be hidden by successful fusion")
        self.assertIn("agent.time_limit=123", calls[0]["args"])
        self.assertGreater(float(calls[0]["deadline"]), 0)
        log.unlink()
        result = subprocess.run(["bash", str(ROOT / "run_single_task.sh"), "demo", str(self.root)],
                                env={**env, "RUN_EXIT": "124"}, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["args"][0], "utils/submission_fusion_utils.py")


if __name__ == "__main__":
    unittest.main()
