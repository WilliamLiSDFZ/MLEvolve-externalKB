from dataclasses import dataclass

from llm import compile_prompt_to_md

from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from engine.search_node import SearchNode
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, _load_cfg, prep_cfg
from pathlib import Path
import time

@dataclass
class Solution:
    code: str
    valid_metric: float

class Experiment:

    def __init__(self, data_dir: str, goal: str, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)
        runtime_started = time.time()
        from utils.llm_preflight import record_configuration
        record_configuration(self.cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)
        from engine.candidate_runtime.integration import enabled, prepare
        if enabled(self.cfg):
            prepare(self.cfg, started_at=runtime_started)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec), cfg=self.cfg  # type: ignore
        )
        self.agent.executor = self.interpreter
        self.agent.runtime_deadline = min(self.interpreter.run_deadline, runtime_started + self.cfg.agent.time_limit)
        if enabled(self.cfg):
            self.interpreter.run_deadline = min(self.interpreter.run_deadline, runtime_started + self.cfg.agent.time_limit)
            self.agent.runtime_deadline = self.interpreter.run_deadline

    def run(self, steps: int) -> Solution:
        from engine.candidate_runtime.integration import enabled, export_results
        try:
            for _i in range(steps):
                if enabled(self.cfg) and time.time() >= self.interpreter.run_deadline:
                    break
                self.agent.step(node=None, exec_callback=self.interpreter.run)
                save_run(self.cfg, self.journal)
        finally:
            self.interpreter.cleanup_session()
            if enabled(self.cfg):
                export_results(self.cfg.workspace_dir, self.cfg.log_dir)

        best_node = self.journal.get_best_node()
        if enabled(self.cfg):
            from engine.candidate_runtime.io import read_json
            selection_path = self.cfg.log_dir / "candidate_results/selection.json"
            if selection_path.exists():
                selected = read_json(selection_path)["selected"]
                if selected:
                    return Solution(
                        code=(self.cfg.workspace_dir / "candidate_results/current/best_solution/solution.py").read_text(),
                        valid_metric=selected[0]["snapshot"]["metric"],
                    )
            raise RuntimeError("No verified complete candidate result is available")
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)

