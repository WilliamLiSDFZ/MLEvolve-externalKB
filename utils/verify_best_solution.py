"""Best-result persistence regressions: python utils/verify_best_solution.py.

Uses real threads and temporary files, with no ML/LLM dependencies. Top-K disk
output is disabled in the focused race/failure checks to isolate the best result.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import solution_manager


@dataclass
class Metric:
    value: float
    maximize: bool

    def __lt__(self, other):
        return self.value < other.value if self.maximize else self.value > other.value


class DelayedLock:
    """Pause one writer before acquiring the real file lock, without sleeps."""

    def __init__(self):
        self.lock = threading.Lock()
        self.delayed_thread = None
        self.waiting = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        if threading.get_ident() == self.delayed_thread:
            self.waiting.set()
            if not self.release.wait(5):
                raise TimeoutError("Delayed writer was not released")
        self.lock.acquire()
        return self

    def __exit__(self, *args):
        self.lock.release()


class BestSolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workspace = Path(self.tmp.name)
        (self.workspace / "submission").mkdir()

    def agent(self, maximize=True):
        return SimpleNamespace(
            cfg=SimpleNamespace(workspace_dir=self.workspace),
            metric_maximize=maximize,
            best_node=None,
            save_node_lock=threading.Lock(),
            top_candidates=[],
            top_k=5,
        )

    def candidate(self, name, value, maximize=True, valid=True):
        node = SimpleNamespace(
            id=name, metric=Metric(value, maximize), is_valid=valid, is_buggy=False,
            code=f"print({name!r})\n", exec_time=1.0, branch_id=1, stage="draft",
        )
        self.submission(node).write_text(f"id,prediction\nexample,{value}\n", encoding="utf-8")
        return node

    def submission(self, node):
        return self.workspace / "submission" / f"submission_{node.id}.csv"

    def snapshot(self):
        paths = ["best_submission/submission.csv", "best_solution/solution.py",
                 "best_solution/node_id.txt", "best_solution/metric.txt"]
        return {name: (self.workspace / name).read_bytes() for name in paths}

    def assert_best(self, agent, expected):
        self.assertIs(agent.best_node, expected)
        files = self.snapshot()
        self.assertEqual(files["best_submission/submission.csv"], self.submission(expected).read_bytes())
        self.assertEqual(files["best_solution/solution.py"].decode("utf-8"), expected.code)
        self.assertEqual(files["best_solution/node_id.txt"].decode("utf-8"), expected.id)
        self.assertIn(f"Metric: {expected.metric.value}\n", files["best_solution/metric.txt"].decode())
        self.assertIn(f"Maximize: {agent.metric_maximize}\n", files["best_solution/metric.txt"].decode())

    def test_delayed_worse_writer_cannot_overwrite_better_submission(self):
        for maximize, first_score, best_score in [(True, 0.8, 0.9), (False, 0.2, 0.1)]:
            with self.subTest(maximize=maximize):
                agent = self.agent(maximize)
                first = self.candidate("first", first_score, maximize)
                best = self.candidate("best", best_score, maximize)
                lock = DelayedLock()
                agent.save_node_lock = lock

                def delayed_update():
                    lock.delayed_thread = threading.get_ident()
                    solution_manager.update_best_solution(agent, first)

                with patch.object(solution_manager, "save_top_candidates"), ThreadPoolExecutor(max_workers=2) as pool:
                    delayed = pool.submit(delayed_update)
                    try:
                        self.assertTrue(lock.waiting.wait(5), "First writer never reached the lock")
                        pool.submit(solution_manager.update_best_solution, agent, best).result(timeout=5)
                        self.assert_best(agent, best)
                    finally:
                        lock.release.set()
                    delayed.result(timeout=5)
                self.assert_best(agent, best)

    def test_selection_preserves_direction_ties_and_validity(self):
        for maximize, first_score, best_score in [(True, 0.8, 0.9), (False, 0.2, 0.1)]:
            with self.subTest(maximize=maximize):
                agent = self.agent(maximize)
                first = self.candidate("first", first_score, maximize)
                best = self.candidate("best", best_score, maximize)
                solution_manager.update_best_solution(agent, first)
                self.assert_best(agent, first)
                solution_manager.update_best_solution(agent, best)
                self.assert_best(agent, best)
                solution_manager.update_best_solution(agent, first)
                solution_manager.update_best_solution(agent, self.candidate("tie", best_score, maximize))
                invalid_score = 1.0 if maximize else 0.0
                solution_manager.update_best_solution(agent, self.candidate("invalid", invalid_score, maximize, valid=False))
                self.assert_best(agent, best)

    def test_preparation_failure_keeps_previous_best_and_files(self):
        agent = self.agent()
        first = self.candidate("first", 0.8)
        best = self.candidate("best", 0.9)
        solution_manager.update_best_solution(agent, first)
        before = self.snapshot()

        def failed_copy(source, target):
            Path(target).write_text("partial submission", encoding="utf-8")
            raise OSError("simulated copy failure")

        def failed_metric_write(path, *args):
            Path(path).write_text("partial metadata", encoding="utf-8")
            raise OSError("simulated metadata failure")

        for target, attribute, failure in [
            (solution_manager.shutil, "copy", failed_copy),
            (solution_manager, "write_metric_file", failed_metric_write),
        ]:
            with self.subTest(failure=attribute), patch.object(solution_manager, "save_top_candidates"):
                with patch.object(target, attribute, side_effect=failure):
                    with self.assertRaises(OSError):
                        solution_manager.update_best_solution(agent, best)
                self.assert_best(agent, first)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(list(self.workspace.glob(".best-solution-*")), [])

        # The same candidate can be committed after the I/O failure is resolved.
        solution_manager.update_best_solution(agent, best)
        self.assert_best(agent, best)

    def test_missing_submission_does_not_advance_best(self):
        agent = self.agent()
        first = self.candidate("first", 0.8)
        best = self.candidate("best", 0.9)
        solution_manager.update_best_solution(agent, first)
        self.submission(best).unlink()
        with patch.object(solution_manager, "save_top_candidates"):
            with self.assertRaises(FileNotFoundError):
                solution_manager.update_best_solution(agent, best)
        self.assert_best(agent, first)


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    unittest.main(verbosity=2)
