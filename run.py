import atexit
import shutil
import signal
import os
import time
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.pipeline import run_search_pipeline
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from utils.logging_config import setup_logging
import torch



def run():
    cfg = load_cfg()
    run_started_at = time.time()
    from engine.candidate_runtime.integration import enabled, prepare, export_results
    run_deadline = min(run_started_at + cfg.agent.time_limit,
                       float(os.environ.get("MLEVOLVE_RUN_DEADLINE", "inf")))
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')
    from utils.llm_preflight import record_configuration
    record_configuration(cfg)

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg, task_desc=task_desc)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)
    if enabled(cfg):
        prepare(cfg, started_at=run_started_at)
        # Preserve configuration even if the process dies before a search node finishes.
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )
    agent.executor = interpreter
    agent.runtime_deadline = run_deadline  # observed search deadline, including runtime-off tasks
    if enabled(cfg):
        interpreter.run_deadline = run_deadline

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def handle_termination(signum, frame):
        # run_single_task.sh sends SIGTERM at the existing wall-clock deadline.
        # Raise through the pipeline so active process groups and queued work are stopped.
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
    try:
        run_search_pipeline(
            agent, interpreter, cfg, exec_callback,
            save_callback=lambda: save_run(cfg, journal),
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        interpreter.cleanup_session(-1)
        if enabled(cfg):
            try:
                export_results(cfg.workspace_dir, cfg.log_dir, agent.top_k)
            except Exception:
                logger.exception("Final export failed; recover from candidate_results on the PVC")


if __name__ == "__main__":
    run()
