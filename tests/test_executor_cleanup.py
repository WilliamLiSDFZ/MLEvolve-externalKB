"""Exercise candidate launch/stop races without GPUs or subprocess side effects."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


ROOT = Path(__file__).resolve().parents[1]


def load_executor():
    spec = importlib.util.spec_from_file_location("_test_cleanup_executor", ROOT / "engine/executor.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        spec.name: module,
        "humanize": SimpleNamespace(naturaldelta=str),
        "dataclasses_json": SimpleNamespace(DataClassJsonMixin=object),
    }):
        spec.loader.exec_module(module)
    return module


class ExecutorCleanupTests(unittest.TestCase):
    def setUp(self):
        self.module = load_executor()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.interpreter = self.module.Interpreter(self.tmp.name, max_parallel_run=1)
        affinity = patch.object(self.module.os, "sched_getaffinity", return_value={0}, create=True)
        affinity.start()
        self.addCleanup(affinity.stop)
        self.proc = Mock(pid=12345, returncode=0)
        self.proc.poll.return_value = 0
        self.proc.communicate.return_value = ("completed\n", "")

    def assert_released(self):
        self.assertEqual(self.interpreter.current_parallel_run, 0)
        self.assertEqual(self.interpreter.status_map, [0])
        self.assertEqual(self.interpreter._active_procs, {})
        self.assertFalse(list(Path(self.tmp.name).glob("runfile_*.py")))

    def test_stop_rejects_later_runs_without_reserving_a_slot(self):
        self.interpreter.terminate_all_subprocesses()
        with patch.object(self.module.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "stopping"):
                self.interpreter.run("pass", "later")
        popen.assert_not_called()
        self.assertFalse(self.interpreter.check_current_status())
        self.assert_released()

    def test_stop_between_slot_reservation_and_launch_prevents_popen(self):
        def stop_before_launch(code, _id):
            self.interpreter.terminate_all_subprocesses()
            return code

        with patch.object(self.interpreter, "isolate_submission_path", side_effect=stop_before_launch), \
                patch.object(self.module.subprocess, "Popen") as popen, \
                self.assertLogs("MLEvolve", level="ERROR"):
            result = self.interpreter.run("pass", "candidate")
        self.assertEqual(result.exc_type, "RuntimeError")
        popen.assert_not_called()
        self.assert_released()

    def test_stop_waits_for_popen_to_be_registered(self):
        launching = threading.Event()
        finish_launch = threading.Event()
        stopping = threading.Event()
        stopped = threading.Event()
        killed = threading.Event()
        errors = []

        def popen(*args, **kwargs):
            launching.set()
            if not finish_launch.wait(5):
                raise AssertionError("Test did not release launch")
            return self.proc

        def communicate(**kwargs):
            if not killed.wait(5):
                raise AssertionError("Registered process was not stopped")
            return "", ""

        def stop():
            stopping.set()
            self.interpreter.terminate_all_subprocesses()
            stopped.set()

        def run():
            try:
                self.interpreter.run("pass", "candidate")
            except BaseException as exc:
                errors.append(exc)

        def signal_tree(proc, sig):
            if sig == signal.SIGKILL:
                killed.set()

        self.proc.communicate.side_effect = communicate
        with patch.object(self.module.subprocess, "Popen", side_effect=popen), \
                patch.object(self.interpreter, "_signal_process_tree", side_effect=signal_tree) as send:
            worker = threading.Thread(target=run, daemon=True)
            stopper = threading.Thread(target=stop, daemon=True)
            worker.start()
            try:
                self.assertTrue(launching.wait(5))
                stopper.start()
                self.assertTrue(stopping.wait(5))
                self.assertFalse(stopped.wait(0.05))
            finally:
                finish_launch.set()
                worker.join(5)
                if stopper.ident is not None:
                    stopper.join(5)
            self.assertFalse(worker.is_alive())
            self.assertFalse(stopper.is_alive())
            self.assertEqual(errors, [])
            send.assert_any_call(self.proc, signal.SIGTERM)
            send.assert_any_call(self.proc, signal.SIGKILL)
        self.assert_released()

    def test_success_cleans_descendants_before_releasing_slot(self):
        def signal_tree(proc, sig):
            self.assertEqual(self.interpreter.current_parallel_run, 1)
            self.assertEqual(self.interpreter.status_map, [1])
            self.assertIs(self.interpreter._active_procs[0], proc)

        with patch.object(self.module.subprocess, "Popen", return_value=self.proc) as popen, \
                patch.object(self.interpreter, "_signal_process_tree", side_effect=signal_tree) as send:
            result = self.interpreter.run("pass", "candidate")
        self.assertIsNone(result.exc_type)
        self.assertEqual(popen.call_args.kwargs["start_new_session"], os.name == "posix")
        send.assert_called_once_with(self.proc, signal.SIGKILL)
        self.assert_released()

    def test_timeout_interrupts_then_kills_the_entire_process_group(self):
        self.proc.communicate.side_effect = [
            subprocess.TimeoutExpired("candidate", 1),
            subprocess.TimeoutExpired("candidate", 2),
            ("partial output\n", ""),
        ]
        with patch.object(self.module.subprocess, "Popen", return_value=self.proc), \
                patch.object(self.interpreter, "_signal_process_tree") as send, \
                self.assertLogs("MLEvolve", level="WARNING"):
            result = self.interpreter.run("pass", "candidate")
        self.assertEqual(result.exc_type, "TimeoutError")
        self.assertIn("partial output\n", result.term_out)
        self.assertEqual(send.call_args_list, [
            call(self.proc, signal.SIGINT), call(self.proc, signal.SIGKILL),
            call(self.proc, signal.SIGKILL),
        ])
        self.assert_released()

    def test_stop_kills_group_when_parent_ignores_termination(self):
        self.interpreter._active_procs[0] = self.proc
        self.proc.wait.side_effect = [subprocess.TimeoutExpired("candidate", 2), 0]
        with patch.object(self.interpreter, "_signal_process_tree") as send:
            self.interpreter.terminate_all_subprocesses()
        self.assertEqual(send.call_args_list, [
            call(self.proc, signal.SIGTERM), call(self.proc, signal.SIGKILL),
        ])
        self.assertFalse(self.interpreter.check_current_status())

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_process_group_is_signalled_even_after_parent_exits(self):
        with patch.object(self.module.os, "killpg") as killpg:
            self.interpreter._signal_process_tree(self.proc, signal.SIGKILL)
        killpg.assert_called_once_with(self.proc.pid, signal.SIGKILL)
        with patch.object(self.module.os, "killpg", side_effect=ProcessLookupError):
            self.interpreter._signal_process_tree(self.proc, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
