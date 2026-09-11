"""Controller integration. No search mutation occurs in worker-side snapshot writes."""

import ast
from dataclasses import asdict, fields
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .config import CandidateRuntimeConfig
from .io import atomic_json, read_json, seal_directory, sync_directory
from .jigsaw import prepare_contract
from .store import ResultStore

logger = logging.getLogger("MLEvolve")


def enabled(cfg):
    return bool(getattr(getattr(cfg, "candidate_runtime", None), "enabled", False))


def settings(cfg):
    value = CandidateRuntimeConfig(**{f.name: getattr(cfg.candidate_runtime, f.name, f.default)
                                      for f in fields(CandidateRuntimeConfig)})
    value.validate()
    return asdict(value)


def prepare(cfg, started_at=None):
    values = settings(cfg)
    contract = prepare_contract(cfg.workspace_dir, Path(cfg.workspace_dir) / "input",
                                cfg.agent.seed, values["validation_fraction"], cfg.exp_id)
    atomic_json(Path(cfg.workspace_dir) / "candidate_results/run.json",
                {"started_at": started_at if started_at is not None else time.time(), "runtime_config": values})
    return contract


def register_candidate(cfg, node):
    if enabled(cfg):
        store = ResultStore(cfg.workspace_dir, cfg.log_dir)
        store.register(node, settings(cfg), check_leakage=bool(cfg.agent.check_data_leakage))


def check_protocol(code):
    """Fail fast on missing hooks; actual callbacks/artifacts are checked at runtime too."""
    tree = ast.parse(code)
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    required = {"from_env", "split", "bind", "start_training", "step", "finish"}
    if not required <= calls:
        raise ValueError("CandidateRuntimeProtocolError: missing runtime calls: " + ", ".join(sorted(required - calls)))


def begin_execution(cfg, node_id, code, started_at, timeout, run_deadline):
    store = ResultStore(cfg.workspace_dir, cfg.log_dir)
    directory = store.candidate_dir(node_id)
    candidate = read_json(directory / "candidate.json")
    check_protocol(code)
    values = candidate["runtime_config"]
    budget_key = "draft_budget_seconds" if candidate["stage"] in ("draft", "fusion_draft") else "candidate_budget_seconds"
    budget = values[budget_key] or timeout
    deadline = min(started_at + timeout, started_at + budget, run_deadline)
    if deadline <= time.time():
        raise TimeoutError("Run budget expired while candidate was queued")
    spec = dict(workspace=str(store.workspace), log_dir=str(store.log_dir.resolve()), node_id=node_id,
                config=values, started_at=started_at, deadline=deadline)
    path = directory / "execution_spec.json"
    atomic_json(path, spec)
    store.write_execution(node_id, status="running", started_at=started_at, deadline=deadline)
    return path, deadline


def end_execution(cfg, node_id, result, started_at, deadline):
    store = ResultStore(cfg.workspace_dir, cfg.log_dir)
    finished = store.candidate_dir(node_id) / "worker_finished.json"
    if result.exc_type:
        status = "timeout" if result.exc_type == "TimeoutError" else "failed"
    elif finished.exists():
        status = read_json(finished)["reason"]
    else:
        status = "protocol_error"
        result.exc_type = "CandidateRuntimeProtocolError"
        result.term_out.append("\nRuntime session did not finish; use session.finish() after training.\n")
    store.write_execution(node_id, status=status, started_at=started_at, deadline=deadline,
                          finished_at=time.time(), elapsed_seconds=result.exec_time,
                          exc_type=result.exc_type)
    result.execution_status = status
    return result


def export_results(workspace, log_dir=None, top_k=6):
    """Recover final artifacts independently of journal, errors and LLM availability.

    A generation contains aligned code/metric/CSV sets. The `current` symlink is
    the single commit point; compatibility paths all resolve through that pointer.
    The process lock covers selection AND publication, including CLI recovery.
    """
    import fcntl
    store = ResultStore(workspace, log_dir)
    if not (store.root / "contract/manifest.json").exists():
        return None
    with (store.root / "export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        selected, summary = store.collect()
        selected = selected[:top_k]
        if not selected:
            # Do not leave a previously published but subsequently invalidated result current.
            current = store.root / "current"
            if current.is_symlink():
                current.unlink()
                sync_directory(store.root)
            selection_log = store.log_dir / "candidate_results/selection.json"
            if selection_log.exists():
                (store.log_dir / "best_solution.py").unlink(missing_ok=True)
            atomic_json(selection_log, {"version": 1, "selected": []})
            return None
        exports = store.root / "exports"
        exports.mkdir(exist_ok=True)
        generation = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=exports) as tmp:
            staging = Path(tmp) / "generation"
            staging.mkdir()
            for rank, item in enumerate(selected, 1):
                candidate, snapshot = item["candidate"], item["snapshot"]
                source = store.candidate_dir(candidate["node_id"]) / "snapshots" / snapshot["snapshot_id"]
                target = staging / "top_solution" / f"top{rank}"
                target.mkdir(parents=True)
                for name in ("solution.py", "submission.csv", "manifest.json"):
                    shutil.copy2(source / name, target / name)
                (target / "node_id.txt").write_text(candidate["node_id"])
                (target / "metric.txt").write_text(
                    f"Metric: {snapshot['metric']}\nMaximize: {snapshot['maximize']}\n"
                    f"Branch ID: {candidate['branch_id']}\nStage: {candidate['stage']}\n"
                    f"Execution Time(s): {item['charged_seconds']:.6f}\n"
                    f"Snapshot ID: {snapshot['snapshot_id']}\n"
                    f"Execution Status: {item['execution']['status']}\n")
            top = staging / "top_solution/top1"
            shutil.copytree(top, staging / "best_solution")
            (staging / "best_submission").mkdir()
            shutil.copy2(top / "submission.csv", staging / "best_submission/submission.csv")
            atomic_json(staging / "selection.json", {"version": 1, "selected": selected})
            seal_directory(staging, exports / generation)
        pointer = store.root / f".current-{generation}"
        pointer.symlink_to(Path("exports") / generation, target_is_directory=True)
        os.replace(pointer, store.root / "current")
        sync_directory(store.root)
        for name in ("top_solution", "best_solution", "best_submission"):
            compatibility = store.workspace / name
            if not compatibility.exists() and not compatibility.is_symlink():
                compatibility.symlink_to(Path("candidate_results/current") / name, target_is_directory=True)
            elif not compatibility.is_symlink():
                logger.warning("Preserving existing %s; recovered outputs are in candidate_results/current", compatibility)
        store.log_dir.mkdir(parents=True, exist_ok=True)
        # save_run skips its legacy journal-only best writer when the protocol is on.
        shutil.copy2(store.root / "current/best_solution/solution.py", store.log_dir / "best_solution.py")
        atomic_json(store.log_dir / "candidate_results/selection.json", {"version": 1, "selected": selected})
        logger.info("[artifact-best] node=%s snapshot=%s metric=%s status=%s",
                    selected[0]["candidate"]["node_id"], selected[0]["snapshot"]["snapshot_id"],
                    selected[0]["snapshot"]["metric"], selected[0]["execution"]["status"])
        return store.root / "current/top_solution"


def update_search_and_outputs(agent, node):
    from engine.solution_manager import update_top_candidates
    with agent.save_node_lock:
        if not node.is_buggy and node.is_valid and node.metric and node.metric.value is not None:
            update_top_candidates(agent, node)
            if agent.best_node is None or agent.best_node.metric < node.metric:
                agent.best_node = node
        try:
            export_results(agent.cfg.workspace_dir, agent.cfg.log_dir, agent.top_k)
        except Exception:
            # Artifact generation remains on disk; a later finalizer/recovery retries publication.
            logger.exception("Result export failed; immutable candidate snapshots are retained")


def parse_result(agent, node, result):
    from agents.result_parse_agent import _check_data_leakage, _save_to_global_memory
    from utils.metric import MetricValue, WorstMetricValue
    store = ResultStore(agent.cfg.workspace_dir, agent.cfg.log_dir)
    node.absorb_exec_result(result)
    node.execution_status = result.execution_status
    snapshots, rejected = store.valid_snapshots(node.id, allow_pending_review=True)
    node.metric = WorstMetricValue()
    node.is_valid = False
    node.is_buggy = True
    node.artifact_status = "unavailable"
    node.analysis = f"Execution status: {node.execution_status}.\n" + node.term_out[-5000:]
    if not snapshots:
        node.analysis += f"\nNo verified complete snapshot. Artifact errors: {rejected}"
        return node
    best = snapshots[0]
    node.metric = MetricValue(best["metric"], maximize=best["maximize"])
    try:
        review = _check_data_leakage(agent, node, {"metric": best["metric"]})
        if review is not None and review.get("check_succeeded") is False:
            raise RuntimeError("Data-leakage review did not complete")
    except Exception:
        if best["metric"] == 1.0 and agent.acfg.check_data_leakage:
            node.metric = WorstMetricValue()
            node.analysis += "\nPerfect score requires leakage review; review unavailable, artifact remains pending."
            return node
        raise
    if node.metric.is_worst:
        store.invalidate(node.id, node.analysis)
        return node
    atomic_json(store.candidate_dir(node.id) / "review.json", {"status": "passed"})
    node.artifact_status = "scoreable"
    node.best_snapshot_id = best["snapshot_id"]
    node.artifact_metric = best["metric"]
    node.analysis += (f"\nVerified snapshot {best['snapshot_id']}: local metric={best['metric']}, "
                      f"training updates={best['optimizer_steps']}, exported at {best['elapsed_seconds']:.1f}s. "
                      "The search metric is from this exact checkpoint and full prediction set.")
    healthy = result.exc_type is None and node.execution_status in ("completed", "budget_exhausted")
    node.is_buggy = not healthy
    node.is_valid = healthy
    if healthy:
        target = Path(agent.cfg.workspace_dir) / "submission" / f"submission_{node.id}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(store.candidate_dir(node.id) / "snapshots" / best["snapshot_id"] / "submission.csv", target)
        node.code_summary = f"{node.stage}; {node.execution_status}; local metric {best['metric']}"
        _save_to_global_memory(agent, node)
    else:
        node.metric = WorstMetricValue()
        node.analysis += "\nExecution failed after a usable snapshot; debug the code. The snapshot remains eligible for final submission."
    return node
