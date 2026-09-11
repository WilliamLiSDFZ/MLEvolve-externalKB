"""
Python interpreter for executing code snippets via subprocess.
- Executes code in a separate Python process (avoids CUDA/fork issues).
- Captures stdout/stderr, exceptions and stack traces, execution time limit.
- Supports multiple parallel slots (max_parallel_run) with CPU pinning.
"""

import logging
import os
import signal
import sys
import threading
import time
import traceback
import subprocess
from dataclasses import dataclass
from pathlib import Path
from collections import deque

from engine.gpu_devices import visible_gpu_devices

import humanize
from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("MLEvolve")

@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    execution_status: str | None = None



class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int | None = None,
        cfg=None,
        **kwargs,
    ):
        """
        Executes Python code in subprocess(es). No fork/multiprocessing.Process to avoid CUDA issues.

        Args:
            working_dir: working directory of the agent
            timeout: timeout per code execution (seconds)
            agent_file_name: base name for runfile (actual names are runfile_0.py, ...)
            max_parallel_run: optional upper bound; at most one candidate per visible GPU
            cfg: config (start_cpu_id, cpu_number); search workers do not set execution capacity
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.cfg = cfg
        self.run_deadline = float("inf")
        if cfg is not None and getattr(getattr(cfg, "candidate_runtime", None), "enabled", False):
            self.run_deadline = min(time.time() + cfg.agent.time_limit,
                                    float(os.environ.get("MLEVOLVE_RUN_DEADLINE", "inf")))
        self.gpu_devices = visible_gpu_devices()
        if max_parallel_run is not None and max_parallel_run < 1:
            raise ValueError("max_parallel_run must be positive or None")
        self.start_cpu_id = int(cfg.start_cpu_id) if cfg else 0
        self.cpu_number = int(cfg.cpu_number) if cfg else (os.cpu_count() or 1)
        if self.cpu_number < 1:
            raise ValueError("cpu_number must be positive")
        available_cpus = (sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                          else list(range(os.cpu_count() or 1)))
        self.available_cpus = available_cpus[:self.cpu_number]
        requested_slots = max_parallel_run or len(self.gpu_devices) or 1
        capacity = len(self.gpu_devices) if self.gpu_devices else requested_slots
        self.max_parallel_run = min(requested_slots, capacity, len(self.available_cpus))
        if self.max_parallel_run < requested_slots:
            logger.info("Execution concurrency capped from %s to %s by visible GPUs/CPUs",
                        requested_slots, self.max_parallel_run)
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.current_parallel_run = 0
        self.status_map = [0] * self.max_parallel_run
        self.lock = threading.Condition()
        self._slot_waiters = deque()
        self._stopping = False
        self._procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        logger.info("Execution slots=%s, visible GPU IDs=%s (one candidate per GPU)",
                    self.max_parallel_run, self.gpu_devices or "CPU only")

    @staticmethod
    def _signal_process_tree(proc, sig):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            elif proc.poll() is None:
                proc.send_signal(sig)
        except ProcessLookupError:
            pass

    def terminate_all_subprocesses(self) -> None:
        """Terminate all active subprocesses (for graceful Ctrl+C exit)."""
        with self.lock:
            self._stopping = True
            self.lock.notify_all()
        with self._procs_lock:
            procs = list(self._active_procs.items())
        for slot_id, proc in procs:
            try:
                self._signal_process_tree(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                self._signal_process_tree(proc, signal.SIGKILL)
                proc.wait()
            except Exception as e:
                logger.warning(f"Error terminating subprocess slot {slot_id}: {e}")

    def check_current_status(self):
        """Check current parallel run number."""
        with self.lock:
            return not self._stopping and self.current_parallel_run < self.max_parallel_run

    def isolate_submission_path(self, code: str, _id) -> str:
        """Per-process submission filename to avoid write conflicts."""
        target = f"submission_{_id}.csv"

        code = code.replace("submission/submission.csv", f"submission/{target}")
        code = code.replace("/submission.csv", f"/{target}")

        for quote in ("'", '"'):
            code = code.replace(
                f"to_csv({quote}submission.csv",
                f"to_csv({quote}submission/{target}",
            )

        for quote in ("'", '"'):
            code = code.replace(f"{quote}submission.csv{quote}", f"{quote}{target}{quote}")

        return code
    
    def isolate_model_path(self, code, _id):
        """Replace generic model filenames in code to avoid multi-process conflicts."""
        if '.pth' not in code and '.bin' not in code and '.pt' not in code:
            return code

        modified_code = code

        generic_model_names = [
            "best_model.pth",
            "best_model.bin",
            "best_model.pt",
            "model_best.pth",
            "model_best.bin",
            "model_best.pt",
            "model.pth",
            "model.pt",
            "model.bin",
            "checkpoint.pth",
            "checkpoint.pt",
            "checkpoint.bin",
        ]

        generic_model_names.sort(key=len, reverse=True)

        for filename in generic_model_names:
            if filename not in modified_code:
                continue

            name, ext = filename.rsplit('.', 1)
            new_filename = f"{name}_{_id}.{ext}"

            modified_code = modified_code.replace(f"/{filename}", f"/{new_filename}")
            modified_code = modified_code.replace(f'"{filename}"', f'"{new_filename}"')
            modified_code = modified_code.replace(f"'{filename}'", f"'{new_filename}'")

        return modified_code
    
    def cleanup_session(self, process_id: int = -1) -> None:
        """Clean up resources for the given process slot."""
        pass

    def run(self, code: str, id, reset_session=True, working_dir: str | None = None):
        """
        Execute the provided Python command in a subprocess and return its output.

        Parameters:
            code: Python code to execute.
            reset_session: Reserved for future use.
            working_dir: Optional per-run working directory.

        Returns:
            ExecutionResult: output, exec_time, exc_type, exc_info, exc_stack.
        """
        return self._run_subprocess(code=code, id=id, working_dir=working_dir)

    def _run_subprocess(self, code: str, id, working_dir: str | None = None):
        """
        Execute code via subprocess (avoids CUDA fork issues).
        Aligned with multiprocessing mode for consistency.
        """
        logger.info("REPL is executing code via subprocess")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None

        with self.lock:
            ticket = object()
            self._slot_waiters.append(ticket)
            try:
                self.lock.wait_for(lambda: self._stopping or (
                    self._slot_waiters[0] is ticket and 0 in self.status_map))
                if self._stopping:
                    raise RuntimeError("Interpreter is stopping; candidate execution cancelled")
                process_id = self.status_map.index(0)
                self.status_map[process_id] = 1
                self.current_parallel_run += 1
            finally:
                self._slot_waiters.remove(ticket)
                self.lock.notify_all()
            logger.info("Assigned execution slot %s to candidate %s", process_id, id)

        start_time = time.time()
        runfile_path = None
        proc = None
        runtime_active = bool(self.cfg is not None and getattr(getattr(self.cfg, "candidate_runtime", None), "enabled", False))
        deadline = start_time + self.timeout
        execution_limit = self.timeout
        result = None
        
        try:
            # Divide the actual CPU allowance, including any remainder, among execution slots.
            start = process_id * len(self.available_cpus) // self.max_parallel_run
            end = (process_id + 1) * len(self.available_cpus) // self.max_parallel_run
            cpu_set = set(self.available_cpus[start:end])
            logger.info(f"has set process_id:{process_id} to use cpu: {cpu_set}")
            pre_code = ("import os\nos.sched_setaffinity(0, {cpu_set})\n".format(cpu_set=cpu_set)
                        if hasattr(os, "sched_setaffinity") else "")

            spec_path = None
            if runtime_active:
                from engine.candidate_runtime.integration import begin_execution
                spec_path, deadline = begin_execution(self.cfg, id, code, start_time, self.timeout, self.run_deadline)
                execution_limit = deadline - start_time

            code = self.isolate_submission_path(code=code, _id=id)
            code = self.isolate_model_path(code=code, _id=id)
            code = pre_code + code

            # decide runfile location and cwd
            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            runfile_path = run_wd / self.agent_file_name[process_id]
            run_wd.mkdir(parents=True, exist_ok=True)
            with open(runfile_path, "w") as f:
                f.write(code)

            cmd = [sys.executable, str(runfile_path)]
            child_env = {**os.environ, "PYTHONUNBUFFERED": "1",
                         "CUDA_VISIBLE_DEVICES": (self.gpu_devices[process_id]
                                                  if self.gpu_devices else "")}
            if spec_path is not None:
                child_env["MLEVOLVE_CANDIDATE_SPEC"] = str(spec_path)
                child_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent.parent),
                                                                     child_env.get("PYTHONPATH", "")]))
            with self.lock:
                if self._stopping:
                    raise RuntimeError("Interpreter stopped before candidate launch")
                proc = subprocess.Popen(
                    cmd, cwd=str(run_wd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, bufsize=1, env=child_env, start_new_session=(os.name == "posix"),
                )
                with self._procs_lock:
                    self._active_procs[process_id] = proc

            child_in_overtime = False
            exc_type = None
            exc_info = {}
            exc_stack = []
            
            try:
                stdout, stderr = proc.communicate(timeout=max(0.001, deadline - time.time()))
                exec_time = time.time() - start_time
                
                if proc.returncode != 0:
                    exc_type = "RuntimeError"
                    exc_info = {}
                    exc_stack = []
                    
                    if stderr:
                        stderr_text = stderr
                        exc_patterns = [
                            ("KeyboardInterrupt", "KeyboardInterrupt"),
                            ("TimeoutError", "TimeoutError"),
                            ("CUDA", "RuntimeError"),
                            ("cuda", "RuntimeError"),
                            ("ValueError", "ValueError"),
                            ("TypeError", "TypeError"),
                            ("AttributeError", "AttributeError"),
                            ("KeyError", "KeyError"),
                            ("IndexError", "IndexError"),
                            ("FileNotFoundError", "FileNotFoundError"),
                            ("ImportError", "ImportError"),
                            ("AssertionError", "AssertionError"),
                            ("NameError", "NameError"),
                            ("RuntimeError", "RuntimeError"),
                        ]
                        
                        for pattern, exc_name in exc_patterns:
                            if pattern in stderr_text:
                                exc_type = exc_name
                                break
                        
                        stderr_lines = stderr_text.splitlines()
                        for line in stderr_lines:
                            if 'File "' in line and 'line' in line:
                                try:
                                    file_start = line.find('File "') + 6
                                    file_end = line.find('"', file_start)
                                    if file_end > file_start:
                                        filename = line[file_start:file_end]
                                        line_start = line.find('line ') + 5
                                        line_end = line.find(',', line_start)
                                        if line_end == -1:
                                            line_end = line.find('\n', line_start)
                                        if line_end == -1:
                                            line_end = len(line)
                                        line_num_str = line[line_start:line_end].strip()
                                        if line_num_str.isdigit():
                                            line_num = int(line_num_str)
                                            func_name = ""
                                            if 'in ' in line:
                                                func_start = line.find('in ') + 3
                                                func_end = line.find('\n', func_start)
                                                if func_end == -1:
                                                    func_end = len(line)
                                                func_name = line[func_start:func_end].strip()
                                            
                                            filename_short = filename.replace(str(self.working_dir / self.agent_file_name[process_id]), self.agent_file_name[process_id])
                                            filename_short = os.path.basename(filename_short)
                                            exc_stack.append((filename_short, line_num, func_name, ""))
                                except Exception:
                                    pass
                        
                        for line in reversed(stderr_lines):
                            line = line.strip()
                            if line and not line.startswith("File") and not line.startswith("Traceback"):
                                if ":" in line:
                                    parts = line.split(":", 1)
                                    if len(parts) == 2:
                                        exc_info["message"] = parts[1].strip()
                                    break
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess timeout, sending SIGINT...")
                try:
                    self._signal_process_tree(proc, signal.SIGINT)
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("Subprocess failed to terminate after SIGINT, killing...")
                    self._signal_process_tree(proc, signal.SIGKILL)
                    stdout, stderr = proc.communicate()
                
                exec_time = time.time() - start_time
                exc_type = "TimeoutError"
                exc_info = {}
                exc_stack = []
            
            output: list[str] = []
            if stdout:
                output.extend(stdout.splitlines(keepends=True))
            if stderr:
                output.extend(stderr.splitlines(keepends=True))
            if not output:
                output = [""]
            if output and output[-1] and not output[-1].endswith("\n"):
                output.append("\n")

            if exc_type == "TimeoutError":
                output.append(
                    f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(execution_limit)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(execution_limit)})."
                )
            
            result = ExecutionResult(output, exec_time, exc_type, exc_info, exc_stack)
            return result
            
        except Exception as e:
            logger.error(f"Error in _run_subprocess: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            exec_time = time.time() - start_time if start_time else 0
            result = ExecutionResult(
                term_out=[f"Subprocess execution error: {str(e)}", error_trace],
                exec_time=exec_time,
                exc_type="TimeoutError" if isinstance(e, TimeoutError) else "RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
            return result
        finally:
            if process_id is not None:
                with self._procs_lock:
                    self._active_procs.pop(process_id, None)
            if proc is not None:
                try:
                    if proc.poll() is None:
                        logger.warning(f"Subprocess {process_id} still running, terminating...")
                        self._signal_process_tree(proc, signal.SIGTERM)
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Subprocess {process_id} failed to terminate, killing...")
                            self._signal_process_tree(proc, signal.SIGKILL)
                            proc.wait()
                    # A candidate's DataLoader/background children must not outlive its GPU slot.
                    self._signal_process_tree(proc, signal.SIGKILL)
                except Exception as e:
                    logger.warning(f"Error cleaning up subprocess {process_id}: {e}")
            
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove runfile after subprocess execution: {e}")
            
            if runtime_active and result is not None:
                try:
                    from engine.candidate_runtime.integration import end_execution
                    end_execution(self.cfg, id, result, start_time, deadline)
                except Exception:
                    logger.exception("Failed to save candidate exit state; snapshots remain recoverable")

            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1
                    self.lock.notify_all()
