import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from engine.pipeline import run_search_pipeline


@dataclass
class RawResult:
    term_out: list[str]
    exec_time: float = 1.0
    exc_type: str | None = None
    exc_info: dict | None = None
    exc_stack: list | None = None


class ExecutionPipelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = SimpleNamespace(log_dir=self.root, agent=SimpleNamespace(
            steps=3, initial_drafts=3, time_limit=3600,
            search=SimpleNamespace(parallel_search_num=3)))
        self.interpreter = SimpleNamespace(max_parallel_run=3,
                                           terminate_all_subprocesses=Mock())
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("MLEVOLVE_RUN_DEADLINE", None)

    def run_case(self, steps=3, drafts=3, workers=3, fail_generation=False,
                 interrupt=None, overlap_executions=False, fail_execution=False,
                 overlap_search=False, fail_parse=False):
        self.cfg.agent.steps = steps
        self.cfg.agent.initial_drafts = drafts
        self.cfg.agent.search.parallel_search_num = workers
        self.interpreter.max_parallel_run = workers
        self.interpreter.terminate_all_subprocesses.reset_mock()
        first_execution = threading.Event()
        generation_done = threading.Event()
        if min(steps, drafts) == 0:
            generation_done.set()
        all_executions_started = threading.Event()
        lock = threading.Lock()
        generated, executions, parsed, pools, violations = [], [], [], [], []
        active = peak = 0
        saved = Mock()

        def expect(condition, message):
            # Production intentionally catches task exceptions; record assertions outside it.
            if not condition:
                violations.append(message)

        class RecordingPool(ThreadPoolExecutor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.shutdown_calls = []
                pools.append(self)

            def shutdown(self, wait=True, *, cancel_futures=False):
                self.shutdown_calls.append((wait, cancel_futures))
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        class FakeAgent:
            def __init__(self):
                self.journal = [object()]
                self.start_time = time.time()
                self.initial_calls = 0
                self.search_calls = 0

            def step(self, exec_callback, node=None, execute_immediately=True):
                if not execute_immediately:
                    self.initial_calls += 1
                    index = self.initial_calls
                    if index == 2:
                        expect(first_execution.wait(5),
                               "draft 1 execution waited for draft 2 generation")
                        if interrupt:
                            raise interrupt()
                        if fail_generation:
                            raise RuntimeError("scripted generation failure")
                    expect(parsed == [], "scores/memory leaked into initial generation")
                    expect(len(self.journal) == 1, "journal updated before initial barrier")
                    result = SimpleNamespace(id=f"draft-{index}", code=f"code-{index}",
                                             pending_execution=True, is_buggy=False, metric=None)
                    generated.append(result)
                    if index == min(steps, drafts):
                        generation_done.set()
                    return result
                with lock:
                    self.search_calls += 1
                    index = self.search_calls
                result = SimpleNamespace(id=f"search-{index}", code="search-code",
                                         is_buggy=False, metric=None)
                exec_callback(result.code, result.id, True)
                with lock:
                    self.journal.append(result)
                return result

            def execute_deferred_node(self, node, callback):
                expect(generation_done.is_set(), "parsed before generation barrier")
                result = callback(node.code, node.id, True)
                expect(result.term_out == [node.id], "raw result routed to the wrong node")
                if fail_parse and node.id == "draft-1":
                    raise RuntimeError("scripted parser failure")
                with lock:
                    parsed.append(node.id)
                    node.pending_execution = False
                    self.journal.append(node)
                return node

        def execute(code, node_id, reset):
            nonlocal active, peak
            with lock:
                executions.append(node_id)
                active += 1
                peak = max(peak, active)
                if active == workers:
                    all_executions_started.set()
            first_execution.set()
            try:
                if overlap_executions and (overlap_search or node_id.startswith("draft-")):
                    expect(all_executions_started.wait(5),
                           "initial executions did not overlap")
                if fail_execution and node_id == "draft-1":
                    raise RuntimeError("scripted executor failure")
                return RawResult([node_id])
            finally:
                with lock:
                    active -= 1

        agent = FakeAgent()
        try:
            with patch("engine.pipeline.ThreadPoolExecutor", RecordingPool):
                if interrupt:
                    with self.assertRaises(interrupt):
                        run_search_pipeline(agent, self.interpreter, self.cfg, execute, saved)
                else:
                    run_search_pipeline(agent, self.interpreter, self.cfg, execute, saved)
        finally:
            # Join fake callbacks even after the production path's nonblocking cancellation.
            for pool in pools:
                ThreadPoolExecutor.shutdown(pool, wait=True, cancel_futures=True)

        self.assertEqual(violations, [])
        self.assertEqual(len(executions), len(set(executions)), "candidate executed twice")
        for node in generated:
            if fail_execution and node.id == "draft-1":
                self.assertFalse((self.root / "executions" / f"{node.id}.json").exists())
                continue
            if node.id in executions:
                raw = json.loads((self.root / "executions" / f"{node.id}.json").read_text())
                self.assertEqual(raw["term_out"], [node.id])
                self.assertEqual(raw["exec_time"], 1.0)
        if interrupt:
            self.interpreter.terminate_all_subprocesses.assert_called_once()
            self.assertEqual(parsed, [])
            self.assertEqual(len(executions), 1)
            self.assertTrue(all(pool.shutdown_calls == [(False, True)] for pool in pools))
        else:
            self.assertEqual(len(agent.journal) - 1, steps)
            failed = int(fail_execution or fail_parse)
            self.assertEqual(len(executions), steps + failed)
            self.assertCountEqual(parsed, [node.id for node in generated
                                          if not (failed and node.id == "draft-1")])
            self.assertLessEqual(peak, workers)
            self.assertEqual(saved.call_count, steps + failed)
            if overlap_executions:
                self.assertEqual(peak, workers)

    def test_initial_execution_overlaps_generation_and_results_wait_for_barrier(self):
        self.run_case()

    def test_three_execution_slots_and_followup_search_respect_step_budget(self):
        self.run_case(steps=9, overlap_executions=True)

    def test_initial_execution_and_search_share_the_three_slot_limit(self):
        self.run_case(steps=7, drafts=1, overlap_executions=True, overlap_search=True)

    def test_zero_steps_fewer_steps_than_drafts_and_no_initial_drafts(self):
        for steps, drafts, workers in [(0, 3, 3), (1, 3, 3), (5, 0, 3),
                                       (4, 3, 1), (7, 5, 3)]:
            with self.subTest(steps=steps, drafts=drafts, workers=workers):
                self.run_case(steps, drafts, workers)

    def test_failed_generation_does_not_lose_other_candidates(self):
        with self.assertLogs("MLEvolve", level="ERROR") as captured:
            self.run_case(fail_generation=True)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("Initial draft 2 generation failed", captured.output[0])

    def test_executor_exception_does_not_deadlock_and_search_fills_remaining_budget(self):
        with self.assertLogs("MLEvolve", level="ERROR") as captured:
            self.run_case(fail_execution=True)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("Search task failed", captured.output[0])

    def test_parser_exception_keeps_raw_output_and_search_continues(self):
        with self.assertLogs("MLEvolve", level="ERROR") as captured:
            self.run_case(fail_parse=True)
        self.assertEqual(len(captured.records), 1)
        self.assertIn("Search task failed", captured.output[0])

    def test_interrupt_during_initial_generation_cancels_both_pools(self):
        for error in (KeyboardInterrupt, SystemExit):
            with self.subTest(error=error):
                self.run_case(interrupt=error)

    def test_expired_agent_or_outer_deadline_does_not_generate_or_execute(self):
        for outer_deadline in (False, True):
            with self.subTest(outer_deadline=outer_deadline):
                now = time.time()
                agent = SimpleNamespace(journal=[object()], step=Mock(),
                                        start_time=now if outer_deadline else now - 3601)
                execute = Mock()
                if outer_deadline:
                    os.environ["MLEVOLVE_RUN_DEADLINE"] = str(now - 1)
                run_search_pipeline(agent, self.interpreter, self.cfg, execute, Mock())
                agent.step.assert_not_called()
                execute.assert_not_called()

    def test_deadline_expiring_during_generation_does_not_launch_candidate(self):
        now = [1000.0]
        agent = SimpleNamespace(journal=[object()], start_time=1000.0)
        self.cfg.agent.time_limit = 10

        def generate(**kwargs):
            now[0] = 1011.0
            return SimpleNamespace(id="late", code="pass", pending_execution=True)

        agent.step = Mock(side_effect=generate)
        execute = Mock()
        with patch("engine.pipeline.time.time", side_effect=lambda: now[0]):
            run_search_pipeline(agent, self.interpreter, self.cfg, execute, Mock())
        self.assertEqual(agent.step.call_count, 1)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
