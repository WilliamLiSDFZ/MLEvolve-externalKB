"""CPU-only behavioral regressions: python utils/verify_execution_pipeline.py.

Uses real candidate subprocesses and synchronization barriers; mocks GPU discovery and
LLM generation only. No cluster, model downloads, API requests or training required.
"""

import ast
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.executor import ExecutionResult, Interpreter
from engine.gpu_devices import visible_gpu_devices
from engine.pipeline import run_search_pipeline
from engine.search_node import SearchNode, Journal


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.01)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def interpreter(self, devices, **kwargs):
        cfg = SimpleNamespace(cpu_number=8, start_cpu_id=0,
                              agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=7)))
        with patch("engine.executor.visible_gpu_devices", return_value=devices):
            interpreter = Interpreter(self.root, cfg=cfg, **kwargs)
        self.addCleanup(interpreter.terminate_all_subprocesses)
        return interpreter

    def test_device_masks_and_cpu(self):
        for mask, count, expected in [(None, 2, ["0", "1"]), ("7,2", 2, ["7", "2"]),
                                      ("GPU-abc,GPU-def", 2, ["GPU-abc", "GPU-def"]),
                                      ("MIG-GPU-abc/1/0", 1, ["MIG-GPU-abc/1/0"]),
                                      ("", 0, []), ("-1", 0, [])]:
            with self.subTest(mask=mask), patch.dict(os.environ, {}, clear=True), \
                    patch("torch.cuda.device_count", return_value=count):
                if mask is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = mask
                self.assertEqual(visible_gpu_devices(), expected)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,2"}), \
                patch("torch.cuda.device_count", return_value=2):
            with self.assertRaises(ValueError):
                visible_gpu_devices()

    def test_capacity_is_independent_of_search_workers(self):
        self.assertEqual(self.interpreter(["0"], max_parallel_run=3).max_parallel_run, 1)
        self.assertEqual(self.interpreter(["0", "1"]).max_parallel_run, 2)
        self.assertEqual(self.interpreter(["0", "1"], max_parallel_run=1).max_parallel_run, 1)
        self.assertEqual(self.interpreter([]).max_parallel_run, 1)
        with self.assertRaises(ValueError):
            self.interpreter(["0"], max_parallel_run=0)

    def test_real_subprocesses_do_not_overlap_on_same_device(self):
        for devices in [["7"], ["GPU-a", "GPU-b"]]:
            with self.subTest(devices=devices):
                interpreter = self.interpreter(devices)
                code = ("import os, time, json\n"
                        "start = time.monotonic()\n"
                        "time.sleep(0.25)\n"
                        "print(json.dumps([os.environ['CUDA_VISIBLE_DEVICES'], start, time.monotonic()]))\n")
                original_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda i: interpreter.run(code, str(i)), range(4)))
                self.assertTrue(all(r.exc_type is None for r in results))
                intervals = [json.loads(r.term_out[0]) for r in results]
                self.assertEqual(set(item[0] for item in intervals), set(devices))
                for device in devices:
                    ordered = sorted(item[1:] for item in intervals if item[0] == device)
                    self.assertTrue(all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])))
                if len(devices) == 2:
                    self.assertTrue(any(a[0] != b[0] and max(a[1], b[1]) < min(a[2], b[2])
                                        for a in intervals for b in intervals))
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original_mask)
                self.assertEqual(interpreter.current_parallel_run, 0)
                self.assertEqual(interpreter.status_map, [0] * len(devices))

    def test_failure_and_timeout_release_slot(self):
        interpreter = self.interpreter(["0"], timeout=0.15)
        self.assertEqual(interpreter.run("raise ValueError('bad')", "bad").exc_type, "ValueError")
        self.assertEqual(interpreter.run("import time; time.sleep(3)", "slow").exc_type, "TimeoutError")
        self.assertIsNone(interpreter.run("print('recovered')", "ok").exc_type)
        self.assertEqual(interpreter.current_parallel_run, 0)
        self.assertFalse(list(self.root.glob("runfile_*.py")))

    def test_launch_failure_releases_slot(self):
        interpreter = self.interpreter(["0"])
        with patch("engine.executor.subprocess.Popen", side_effect=OSError("launch failed")):
            self.assertEqual(interpreter.run("pass", "bad").exc_type, "RuntimeError")
        self.assertIsNone(interpreter.run("pass", "ok").exc_type)

    def test_waiting_time_does_not_consume_execution_timeout(self):
        interpreter = self.interpreter(["0"], timeout=0.15)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(interpreter.run, "import time; time.sleep(10)", "first")
            wait_until(lambda: interpreter.current_parallel_run == 1)
            second = pool.submit(interpreter.run, "print('second')", "second")
            self.assertEqual(first.result(timeout=5).exc_type, "TimeoutError")
            self.assertIsNone(second.result(timeout=5).exc_type)

    def test_termination_cancels_waiters_and_active_process(self):
        interpreter = self.interpreter(["0"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(interpreter.run, "import time; time.sleep(30)", "active")
            wait_until(lambda: bool(interpreter._active_procs))
            queued = pool.submit(interpreter.run, "print('must not start')", "queued")
            wait_until(lambda: bool(interpreter._slot_waiters))
            interpreter.terminate_all_subprocesses()
            active.result(timeout=5)
            with self.assertRaises(RuntimeError):
                queued.result(timeout=5)
        self.assertEqual(interpreter.current_parallel_run, 0)
        self.assertFalse(interpreter.check_current_status())

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_background_descendants_are_cleaned_up(self):
        interpreter = self.interpreter(["0"])
        child_code = "import time; from pathlib import Path; time.sleep(0.6); Path('orphan').touch()"
        code = ("import subprocess, sys\n"
                f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n")
        self.assertIsNone(interpreter.run(code, "parent").exc_type)
        time.sleep(0.8)
        self.assertFalse((self.root / "orphan").exists())

    def test_submission_and_model_filenames_remain_isolated(self):
        interpreter = self.interpreter(["0"])
        (self.root / "submission").mkdir()
        code = "from pathlib import Path\nPath('submission/submission.csv').write_text('x')\nPath('model.pt').write_text('m')\n"
        for node_id in ["one", "two"]:
            self.assertIsNone(interpreter.run(code, node_id).exc_type)
            self.assertTrue((self.root / "submission" / f"submission_{node_id}.csv").exists())
            self.assertTrue((self.root / f"model_{node_id}.pt").exists())


def load_draft_injector():
    # Execute the actual injection function without loading unrelated ML/LLM backends.
    source = ast.parse((ROOT / "agents/draft_agent.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_inject_analogy_draft")
    namespace = {"Any": Any, "logger": logging.getLogger("test"), "ANALOGY_SECTION_DRAFT": "analogy"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "draft_agent.py", "exec"), namespace)
    return namespace["_inject_analogy_draft"]


class PipelineTests(unittest.TestCase):
    def run_case(self, steps=3, drafts=3, workers=3, fail_generation=False, interrupt=False, analogy=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = SimpleNamespace(agent=SimpleNamespace(steps=steps, initial_drafts=drafts,
                              search=SimpleNamespace(parallel_search_num=workers)), log_dir=root,
                              cpu_number=4, start_cpu_id=0)
        first_executed = threading.Event()
        generation_done = threading.Event()
        generated = []
        executions = []
        memories = []
        prompts = []
        lock = threading.Lock()
        injector = load_draft_injector()
        outer = self

        class FakeAgent:
            def __init__(self):
                self.journal = Journal()
                self.virtual_root = SearchNode(code="", stage="root")
                self.journal.append(self.virtual_root)
                self.cfg = SimpleNamespace(analogy=SimpleNamespace(enabled=analogy, draft=analogy))
                self.calls = 0

            def step(self, exec_callback, node, execute_immediately=True):
                if not execute_immediately:
                    self.calls += 1
                    if self.calls == 2:
                        outer.assertTrue(first_executed.wait(5), "draft 1 must execute during draft 2 generation")
                        if interrupt:
                            raise KeyboardInterrupt()
                        if fail_generation:
                            raise RuntimeError("scripted generation failure")
                    outer.assertEqual(len(self.journal), 1, "execution results leaked into initial generation")
                    memories.append(self.virtual_root.fetch_child_memory())
                    prompt = {"Instructions": {}}
                    report = injector(self, prompt)
                    prompts.append(prompt)
                    self.virtual_root.add_expected_child_count()
                    result = SearchNode(code="print('draft')", plan=f"design {self.calls}", stage="draft",
                                        parent=self.virtual_root, analogy_report=report)
                    result.pending_execution = True
                    generated.append(result)
                    if self.calls == min(steps, drafts):
                        generation_done.set()
                    return result
                result = SearchNode(code="print('next')", stage="improve", parent=node or self.virtual_root)
                exec_callback(result.code, result.id, True)
                with lock:
                    self.journal.append(result)
                return result

            def execute_deferred_node(self, node, callback):
                outer.assertTrue(generation_done.is_set(), "result parsing began before generation barrier")
                node.absorb_exec_result(callback(node.code, node.id, True))
                node.is_buggy = False
                node.pending_execution = False
                with lock:
                    self.journal.append(node)
                return node

        def execute(code, node_id, reset):
            executions.append(node_id)
            result = interpreter.run(code, node_id, reset)
            first_executed.set()
            return result

        agent = FakeAgent()
        with patch("engine.executor.visible_gpu_devices", return_value=["GPU-test"]):
            interpreter = Interpreter(root, cfg=cfg)
        stop = interpreter.terminate_all_subprocesses
        self.addCleanup(stop)

        def record_stop():
            executions.append("stopped")
            stop()

        interpreter.terminate_all_subprocesses = record_stop
        fake_analogy = SimpleNamespace(retrieve_for_draft=lambda agent: "FULLTEXT_REPORT")
        with patch.dict(sys.modules, {"engine.analogy.agent": fake_analogy}):
            if interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    run_search_pipeline(agent, interpreter, cfg, execute, lambda: None)
                self.assertIn("stopped", executions)
                return
            run_search_pipeline(agent, interpreter, cfg, execute, lambda: None)
        self.assertEqual(len(agent.journal) - 1, steps)
        self.assertEqual(len(executions), len(set(executions)), "candidate executed twice")
        self.assertEqual(len(executions), steps)
        for node in generated:
            self.assertTrue((root / "executions" / f"{node.id}.json").exists())
        for memory in memories:
            self.assertNotIn("Validation Metric", memory)
            self.assertNotIn("Final Validation Score", memory)
        if len(memories) > 1:
            self.assertIn("design 1", memories[1])
        self.assertEqual(sum(bool(n.analogy_report) for n in generated), int(analogy and bool(generated)))

    def test_overlap_barrier_budget_and_exactly_once(self):
        for steps, drafts, workers in [(3, 3, 3), (1, 3, 3), (0, 3, 3), (6, 3, 3), (5, 0, 3), (4, 3, 1)]:
            with self.subTest(steps=steps, drafts=drafts, workers=workers):
                self.run_case(steps, drafts, workers)

    def test_f_injection_stays_on_first_draft(self):
        self.run_case(analogy=True)

    def test_generation_failure_does_not_lose_other_candidates(self):
        self.run_case(fail_generation=True)

    def test_interrupt_during_initial_generation(self):
        self.run_case(interrupt=True)

    def _assert_terminal_transport_stops_pipeline(self, *, during_initial):
        from llm.responses import ResponsesError

        class UnexpectedRetry(BaseException):
            """Stop a regressed pipeline immediately instead of hanging the test."""

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = SimpleNamespace(agent=SimpleNamespace(steps=20, initial_drafts=4 if during_initial else 0,
                              search=SimpleNamespace(parallel_search_num=2)), log_dir=root,
                              cpu_number=4, start_cpu_id=0)
        with patch("engine.executor.visible_gpu_devices", return_value=["GPU-test"]):
            interpreter = Interpreter(root, cfg=cfg)
        self.addCleanup(interpreter.terminate_all_subprocesses)
        failure = ResponsesError("synthetic terminal API failure", category="request_error", status_code=400)
        active_code = "import time\nfrom pathlib import Path\nPath('active-started').touch()\ntime.sleep(30)\n"
        queued_code = "from pathlib import Path\nPath('queued-started').touch()\n"
        expected_calls = 3 if during_initial else 2
        lock = threading.Lock()
        pools = []
        outer = self

        class RecordingPool(ThreadPoolExecutor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.shutdown_calls = []
                self.submitted = []
                pools.append(self)
                outer.addCleanup(lambda: ThreadPoolExecutor.shutdown(self, wait=True, cancel_futures=True))

            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                self.submitted.append(future)
                return future

            def shutdown(self, wait=True, *, cancel_futures=False):
                self.shutdown_calls.append((wait, cancel_futures))
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        class FakeAgent:
            calls = 0
            journal = [object()]  # A root only; failure must not trigger more search attempts.

            def step(self, exec_callback, node, execute_immediately=True):
                with lock:
                    self.calls += 1
                    call_number = self.calls
                if call_number > expected_calls:
                    raise UnexpectedRetry("terminal transport failure was retried/rescheduled")
                if call_number == expected_calls:
                    wait_until(lambda: (root / "active-started").exists())
                    raise failure
                if not execute_immediately:
                    return SimpleNamespace(id=f"initial-{call_number}", pending_execution=True,
                                           code=active_code if call_number == 1 else queued_code)
                exec_callback(active_code, "search-active", True)
                return None

            def execute_deferred_node(self, node, callback):
                raise UnexpectedRetry("failed initial generation crossed the result barrier")

        agent = FakeAgent()
        saved = []
        start = time.monotonic()
        with patch("engine.pipeline.ThreadPoolExecutor", RecordingPool):
            with self.assertRaises(ResponsesError) as caught:
                run_search_pipeline(agent, interpreter, cfg, interpreter.run, lambda: saved.append(True))
        self.assertIs(caught.exception, failure)
        self.assertEqual(agent.calls, expected_calls)
        self.assertTrue(interpreter._stopping)
        self.assertFalse((root / "queued-started").exists())
        self.assertEqual(saved, [])
        self.assertTrue(all(pool.shutdown_calls == [(False, True)] for pool in pools))
        wait_until(lambda: all(future.done() for pool in pools for future in pool.submitted))
        self.assertEqual(interpreter.current_parallel_run, 0)
        self.assertFalse(interpreter._active_procs)
        self.assertLess(time.monotonic() - start, 8, "terminal failure should not wait for the 30-second candidate")

    def test_terminal_transport_during_initial_drafts_aborts_and_cancels(self):
        self._assert_terminal_transport_stops_pipeline(during_initial=True)

    def test_terminal_transport_from_search_future_aborts_without_rescheduling(self):
        self._assert_terminal_transport_stops_pipeline(during_initial=False)

    def test_config_schema(self):
        from config import Config, ExecConfig
        from omegaconf import OmegaConf
        yaml_cfg = OmegaConf.load(ROOT / "config/config.yaml")
        yaml_cfg.data_dir = "/tmp/data"
        yaml_cfg.exp_name = "pipeline-test"
        self.assertEqual(set(yaml_cfg.exec), {field.name for field in fields(ExecConfig)})
        cfg = OmegaConf.merge(OmegaConf.structured(Config), yaml_cfg)
        self.assertIsNone(cfg.exec.max_parallel_run)
        self.assertEqual(cfg.agent.search.parallel_search_num, 3)
        self.assertEqual(OmegaConf.merge(cfg, OmegaConf.from_dotlist(["exec.max_parallel_run=1"])).exec.max_parallel_run, 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    unittest.main(verbosity=2)
