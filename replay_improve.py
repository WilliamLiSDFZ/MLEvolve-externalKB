"""Run one isolated A/B/C/D improve experiment from a saved run and parent node.

See docs/replay_improve.md. This entry point never starts a new search tree.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REPORT_CHARS = 12000


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def component(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
        raise ValueError("Run and experiment IDs must be single directory names")
    return value


def prepare_arm(runs_root, source, experiment_id, arm, identity):
    """Publish the common pairing contract atomically; replace only our own arm."""
    if arm not in "ABCD" or len(arm) != 1:
        raise ValueError("arm must be A, B, C or D")
    experiment = Path(runs_root).resolve() / ("replay_" + component(experiment_id))
    target = experiment / arm
    if experiment.is_symlink() or target.is_symlink():
        raise ValueError("Experiment and arm directories must not be symlinks")
    source = Path(source).resolve()
    if source == experiment or source.is_relative_to(experiment) or experiment.is_relative_to(source):
        raise ValueError("Replay destination overlaps the original run")
    experiment.mkdir(parents=True, exist_ok=True)
    # link() publishes a complete file, so other arms cannot read a half-written manifest.
    with tempfile.NamedTemporaryFile(mode="w", dir=experiment, delete=False) as stream:
        json.dump(identity, stream, sort_keys=True)
        pending = Path(stream.name)
    try:
        try:
            os.link(pending, experiment / "experiment.json")
        except FileExistsError:
            if json.loads((experiment / "experiment.json").read_text()) != identity:
                raise ValueError("Experiment ID already belongs to different inputs; use a new ID")
    finally:
        pending.unlink()
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    return target


def load_config(path):
    import yaml
    from omegaconf import OmegaConf
    from config import Config

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
                           lambda loader, node: str(Path(*loader.construct_sequence(node))))
    saved = yaml.load(Path(path).read_text(), Loader=Loader)
    cfg = OmegaConf.merge(OmegaConf.structured(Config),
                          OmegaConf.merge(OmegaConf.load(ROOT / "config/config.yaml"), saved))
    for role in ("code", "feedback"):
        stage = getattr(cfg.agent, role)
        # Credentials come from this job, never from a historical plaintext config.
        stage.api_key = '${oc.env:LLM_API_KEY,""}'
        for field, env in (("model", "LLM_MODEL"), ("base_url", "LLM_BASE_URL"),
                           ("reasoning_effort", "LLM_REASONING_EFFORT")):
            if env in os.environ:
                setattr(stage, field, os.environ[env])
    cfg.analogy.report_char_budget = REPORT_CHARS
    cfg.analogy.enabled = False  # D is invoked explicitly, once, before improve.
    cfg.analogy.improve = True
    cfg.analogy.draft = False
    cfg.analogy.fulltext.enabled = True
    cfg.exec.max_parallel_run = 1
    return cfg


def remaining_budget(cfg, snapshot):
    parent = snapshot["nodes"][-1]
    if not parent.get("created_time") or not parent.get("finish_time"):
        raise ValueError("Parent creation/completion timestamps are required to restore its remaining budget")
    # Difference of source-local timestamps avoids assuming the source machine's timezone.
    duration = (datetime.fromisoformat(parent["finish_time"]) -
                datetime.fromisoformat(parent["created_time"])).total_seconds()
    completed_at = float(parent["ctime"]) + duration
    remaining = int(float(cfg.agent.time_limit) - (completed_at - snapshot["nodes"][0]["ctime"]))
    if remaining <= 0:
        raise ValueError("Selected parent completed after the original run budget expired")
    return min(int(cfg.agent.time_limit), remaining)


def rewrite_paths(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
    elif isinstance(value, list):
        value = [rewrite_paths(item, replacements) for item in value]
    elif isinstance(value, dict):
        value = {key: rewrite_paths(item, replacements) for key, item in value.items()}
    return value


def copy_inputs(source, arm_dir, cfg, snapshot, input_dir):
    workspace = arm_dir / "workspace"
    shutil.copytree(source / "logs", arm_dir / "source/logs")
    for name in ("working", "submission"):
        (workspace / name).mkdir(parents=True)
    shutil.copytree(input_dir, workspace / "input", symlinks=False)
    if input_dir != source / "workspace/input" and cfg.preprocess_data:
        from utils import preproc_data
        preproc_data(workspace / "input")
    replacements = sorted({
        str(cfg.workspace_dir): str(workspace),
        str(source / "workspace"): str(workspace),
        str(cfg.log_dir): str(arm_dir / "logs"),
        str(source / "logs"): str(arm_dir / "logs"),
        str(cfg.data_dir): str(workspace / "input"),
        str(input_dir): str(workspace / "input"),
    }.items(), key=lambda item: -len(item[0]))
    snapshot = rewrite_paths(snapshot, replacements)
    cfg.coldstart.description = rewrite_paths(cfg.coldstart.description, replacements)
    memory = source / "workspace/global_memory/records.json"
    if cfg.agent.use_global_memory:
        if not memory.is_file():
            raise FileNotFoundError("Source global_memory/records.json is required to restore enabled memory")
        allowed = {"node_" + node["id"] for node in snapshot["nodes"]}
        records = [record for record in json.loads(memory.read_text()) if record["record_id"] in allowed]
        write_json(workspace / "global_memory/records.json", rewrite_paths(records, replacements))
    if cfg.candidate_runtime.enabled:
        contract = source / "workspace/candidate_results/contract"
        if not contract.is_dir():
            raise FileNotFoundError("Source candidate_results/contract is required to preserve the exact validation split")
        shutil.copytree(contract, workspace / "candidate_results/contract")
        manifest = json.loads((contract / "manifest.json").read_text())
        for split in ("train", "test"):
            if file_hash(workspace / "input" / f"{split}.csv") != manifest[f"public_{split}_sha256"]:
                raise ValueError(f"Copied {split}.csv does not match the source validation contract")
        copy_runtime_history(source, arm_dir, snapshot)
    cfg.workspace_dir, cfg.log_dir = workspace, arm_dir / "logs"
    cfg.data_dir = workspace / "input"
    cfg.copy_data = True
    cfg.analogy.fulltext.cache_dir = str(arm_dir / "injection/paper_cache")
    cfg.torch_hub_dir = str(arm_dir / "cache/torch")
    cfg.log_dir.mkdir(exist_ok=True)
    return snapshot


def copy_runtime_history(source, arm_dir, snapshot):
    """Restore the source/diagnostics D can read, without copying model weights."""
    for node in snapshot["nodes"]:
        node_id = component(node["id"])
        if node["stage"] == "root":
            continue
        original = source / "workspace/candidate_results/candidates" / node_id
        target = arm_dir / "workspace/candidate_results/candidates" / node_id
        target.mkdir(parents=True)
        for name in ("solution.py", "candidate.json", "execution.json", "review.json",
                     "worker_finished.json", "invalidated.json"):
            if (original / name).is_file():
                shutil.copy2(original / name, target / name)
        metadata = source / "logs/candidate_results" / node_id
        if metadata.is_dir():
            shutil.copytree(metadata, arm_dir / "logs/candidate_results" / node_id)
        snapshot_id = node.get("best_snapshot_id")
        if snapshot_id:
            selected = original / "snapshots" / component(snapshot_id)
            manifest = json.loads((selected / "manifest.json").read_text())
            for relative in (f"snapshots/{snapshot_id}/manifest.json",
                             f"snapshots/{snapshot_id}/validation.csv",
                             f"checkpoints/{component(manifest['checkpoint_id'])}/manifest.json"):
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original / relative, destination)


def isolate_caches(arm_dir):
    for key, name in (("HF_HOME", "huggingface"), ("HF_HUB_CACHE", "huggingface/hub"),
                      ("HF_DATASETS_CACHE", "huggingface/datasets"),
                      ("TRANSFORMERS_CACHE", "huggingface/hub"), ("TORCH_HOME", "torch"),
                      ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("TRITON_CACHE_DIR", "triton"),
                      ("XDG_CACHE_HOME", "xdg"), ("TMPDIR", "tmp"),
                      ("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
        path = arm_dir / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]


@contextmanager
def grading_server(cfg):
    """Legacy result parsing needs its own local format validator; runtime mode does not."""
    if cfg.candidate_runtime.enabled or not cfg.use_grading_server:
        yield
        return
    if str(cfg.dataset_dir) in {"", "."}:
        raise ValueError("Legacy format validation requires the source config's dataset_dir")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ["GRADING_SERVER_PORT"] = str(port)
    command = [sys.executable, "-m", "engine.validation.format_server",
               f"dataset_dir={cfg.dataset_dir}", f"data_dir={cfg.data_dir}",
               "goal=Replay submission validation", f"log_dir={cfg.log_dir / 'validator'}",
               f"workspace_dir={cfg.workspace_dir / 'validator'}"]
    with (cfg.log_dir / "grading_server.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError("Replay grading server did not start; see logs/grading_server.log")
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def private_score(cfg, submission):
    if not submission.is_file():
        return {"status": "unavailable", "reason": "No scoreable child submission"}
    if str(cfg.dataset_dir) in {"", "."}:
        return {"status": "unavailable", "reason": "dataset_dir is not configured"}
    try:
        from utils.mlebench_patch import grading_metadata
        from mlebench.registry import registry
        from mlebench.utils import load_answers, read_csv
        provenance = grading_metadata(cfg.exp_id)
        competition = registry.set_data_dir(Path(cfg.dataset_dir)).get_competition(cfg.exp_id)
        score = competition.grader(read_csv(submission), load_answers(competition.answers))
        return {"status": "scored", "score": float(score), "provenance": provenance}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def execute_one(agent, parent, injection_text):
    from agents import improve_agent, code_review_agent, result_parse_agent
    from omegaconf import OmegaConf
    from utils.serialize import dump_json
    from engine import execution
    from engine.candidate_runtime import integration

    cfg = agent.cfg
    child = improve_agent.run(agent, parent, injection_text=injection_text)
    write_json(cfg.log_dir / "generated_node.json", {
        "id": child.id, "parent_id": parent.id, "plan": child.plan,
        "code": child.code, "prompt_input": child.prompt_input,
    })
    child.code = code_review_agent.run(agent, child)
    (cfg.log_dir / "child_solution.py").write_text(child.code)
    integration.register_candidate(cfg, child)
    interpreter = agent.executor
    interpreter.run_deadline = time.time() + agent.replay_budget_seconds
    with grading_server(cfg):
        result = interpreter.run(child.code, child.id, True)
        write_json(cfg.log_dir / "executions" / f"{child.id}.json", asdict(result))
        child = result_parse_agent.run(agent, child, result)
        execution.validate_executed_node(agent, child)
    child.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    agent.journal.append(child)
    dump_json(agent.journal, cfg.log_dir / "journal.json")
    OmegaConf.save(cfg, cfg.log_dir / "config.yaml")
    if integration.enabled(cfg):
        integration.export_results(cfg.workspace_dir, cfg.log_dir, agent.top_k)
    return child


def run(args):
    from omegaconf import OmegaConf
    from utils.replay_state import snapshot_journal, load_snapshot, restore_agent

    runs_root = args.runs_root.expanduser().resolve()
    source = runs_root / component(args.source_run)
    raw = json.loads((source / "logs/journal.json").read_text())
    snapshot = snapshot_journal(raw, args.parent_node_id)
    cfg = load_config(source / "logs/config.yaml")
    if args.corpus:
        cfg.analogy.corpus_path = str(args.corpus.expanduser().resolve())
    corpus_manifest = (Path(cfg.analogy.corpus_path) / "manifest.json") if cfg.analogy.corpus_path else None
    if args.arm in "CD" and corpus_manifest is None:
        raise ValueError("C/D require --corpus or a saved analogy.corpus_path")
    input_dir = args.data_dir or (source / "workspace/input")
    if not input_dir.is_dir() and not args.data_dir:
        input_dir = Path(cfg.data_dir)
    input_dir = input_dir.expanduser().absolute()
    if not input_dir.is_dir():
        raise FileNotFoundError("Public input data is missing; pass --data-dir with the prepared public split")
    seed = args.seed if args.seed is not None else int(cfg.agent.seed)
    budget = remaining_budget(cfg, snapshot)
    cfg.exec.timeout = min(int(cfg.exec.timeout), budget)
    common_cfg = json.loads(json.dumps(OmegaConf.to_container(cfg, resolve=True), default=str))
    for role in ("code", "feedback"):
        common_cfg["agent"][role]["api_key"] = "[from job environment]"
    identity = dict(version=1, experiment_id=args.experiment_id, source_run=args.source_run,
                    source_path=str(source.resolve()), parent_node_id=args.parent_node_id,
                    journal_sha256=file_hash(source / "logs/journal.json"),
                    source_config_sha256=file_hash(source / "logs/config.yaml"),
                    corpus_manifest_sha256=file_hash(corpus_manifest) if corpus_manifest else None,
                    input_dir=str(input_dir), seed=seed, injection_char_limit=REPORT_CHARS,
                    replay_budget_seconds=budget, config=common_cfg)
    arm_dir = prepare_arm(runs_root, source, args.experiment_id, args.arm, identity)
    result = dict(experiment_id=args.experiment_id, arm=args.arm, source_run=args.source_run,
                  parent_node_id=args.parent_node_id, seed=seed, status="preparing")
    write_json(arm_dir / "result.json", result)
    interpreter = None
    try:
        isolate_caches(arm_dir)
        snapshot = copy_inputs(source, arm_dir, cfg, snapshot, input_dir)
        write_json(cfg.log_dir / "restored_journal.json", snapshot)
        OmegaConf.save(cfg, cfg.log_dir / "config.yaml")
        from utils.logging_config import setup_logging
        from utils.llm_preflight import record_configuration
        from utils.seed import set_global_seed
        from utils.replay_analogy import _recover, _recover_draft
        from engine.agent_search import AgentSearch
        from engine.search_node import Journal
        from engine.executor import Interpreter
        from engine.candidate_runtime import integration
        from config import load_task_desc

        setup_logging(cfg)
        record_configuration(cfg)
        desc, preview = "", ""
        for node in reversed(snapshot["nodes"]):
            recovered_desc, recovered_preview = (_recover_draft(node) if node["stage"] == "draft" else _recover(node))
            desc, preview = desc or recovered_desc, preview or recovered_preview
        if not desc:
            cfg.desc_file = cfg.data_dir / "description.md" if (cfg.data_dir / "description.md").exists() else None
            desc = load_task_desc(cfg)
        if integration.enabled(cfg):
            integration.prepare(cfg)
        journal = load_snapshot(snapshot)
        parent = journal.nodes[-1]
        set_global_seed(seed)
        agent = AgentSearch(desc, cfg, Journal(), metric_maximize=parent.metric.maximize,
                            task_desc_is_clean=True)
        restore_agent(agent, journal, args.parent_node_id)
        if cfg.agent.use_global_memory and agent.global_memory is None:
            raise RuntimeError("Configured global memory could not be restored")
        if cfg.agent.use_global_memory:
            expected = json.loads((cfg.workspace_dir / "global_memory/records.json").read_text())
            if {r["record_id"] for r in expected} != {r.record_id for r in agent.global_memory.records}:
                raise RuntimeError("Global memory did not load all restored records")
        agent.replay_budget_seconds = budget
        agent.start_time = time.time() - (cfg.agent.time_limit - budget)
        agent.runtime_deadline = time.time() + budget
        agent.data_preview = preview or None
        if agent.data_preview is None:
            agent.update_data_preview()
        interpreter = Interpreter(cfg.workspace_dir, timeout=cfg.exec.timeout,
                                  agent_file_name=cfg.exec.agent_file_name, max_parallel_run=1, cfg=cfg)
        agent.executor = interpreter
        interpreter.run_deadline = agent.runtime_deadline
        (cfg.log_dir / "task_description.txt").write_text(str(agent.task_desc))
        (cfg.log_dir / "data_preview.txt").write_text(agent.data_preview)
        injection_dir = arm_dir / "injection"
        injection_dir.mkdir(exist_ok=True)
        if args.arm == "D":
            from engine.analogy.agent import retrieve_for_node
            cfg.analogy.enabled = True
            treatment = retrieve_for_node(agent, parent, with_result=True)
            cfg.analogy.enabled = False
            text = treatment.report_md
            write_json(injection_dir / "analogy_result.json", asdict(treatment))
            if not text.strip() or len(text) > REPORT_CHARS:
                raise RuntimeError(f"D did not deliver a report within the shared budget: {treatment.reason}")
            (injection_dir / "injection.txt").write_text(text)
        else:
            from utils.replay_inputs import prepare_input
            text, _, _ = prepare_input(args.arm, agent, parent, injection_dir, seed)
        # Input generation must not consume a different improve/training budget in C/D.
        set_global_seed(seed)
        agent.start_time = time.time() - (cfg.agent.time_limit - budget)
        agent.runtime_deadline = time.time() + budget
        result.update(status="running", injection_chars=len(text),
                      injection_sha256=hashlib.sha256(text.encode()).hexdigest())
        write_json(arm_dir / "result.json", result)
        history = cfg.workspace_dir / "candidate_results/candidates"
        if history.exists():
            history.rename(arm_dir / "source/candidates")
        child = execute_one(agent, parent, text)
        submission = cfg.workspace_dir / "submission" / f"submission_{child.id}.csv"
        if child.best_snapshot_id:
            submission = cfg.workspace_dir / "candidate_results/candidates" / child.id / "snapshots" / child.best_snapshot_id / "submission.csv"
        result.update(status="completed", child_node_id=child.id, parent_metric=parent.metric.value,
                      metric=child.metric.value, maximize=parent.metric.maximize,
                      is_buggy=child.is_buggy, is_valid=child.is_valid,
                      execution_status=child.execution_status, artifact_status=child.artifact_status,
                      submission=str(submission) if submission.is_file() else None,
                      private_evaluation=(private_score(cfg, submission)
                          if child.is_valid or child.artifact_status == "scoreable"
                          else {"status": "unavailable", "reason": "Child is not scoreable"}))
        return result
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if interpreter is not None:
            interpreter.cleanup_session(-1)
        write_json(arm_dir / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--parent-node-id", required=True, help="Exact ID of a successful completed parent")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--arm", required=True, choices=list("ABCD"))
    parser.add_argument("--corpus", type=Path, help="Paper corpus directory, shared by all four jobs")
    parser.add_argument("--data-dir", type=Path, help="Fallback/override prepared public data directory")
    parser.add_argument("--seed", type=int, help="Treatment/generation seed; defaults to source seed")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
