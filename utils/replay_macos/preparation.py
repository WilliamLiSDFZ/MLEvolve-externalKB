"""Prepare the locked Jigsaw Mac parent; only --execute starts calibration training."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    import replay_improve as replay
    from utils.replay_macos.adapters import get_adapter
    from utils.replay_macos.runtime import PROFILE, apply_profile, digest, ensure_separate, source_identity, verify_public_inputs
    from utils.replay_state import snapshot_journal

    source = args.runs_root.expanduser().resolve() / replay.component(args.source_run)
    snapshot = snapshot_journal(json.loads((source / "logs/journal.json").read_text()), args.parent_node_id)
    cfg = replay.load_config(source / "logs/config.yaml")
    budget = replay.remaining_budget(cfg, snapshot)
    original = snapshot["nodes"][-1]["code"]
    adapter = get_adapter(hashlib.sha256(original.encode()).hexdigest())
    code = adapter.adapt(original)
    private_dataset_dir = apply_profile(cfg, args.dataset_dir)
    inputs = args.data_dir or source / "workspace/input"
    inputs = inputs.expanduser().resolve()
    if not inputs.is_dir():
        raise FileNotFoundError("Prepared public input data is missing; pass --data-dir")
    contract_dir = source / "workspace/candidate_results/contract"
    verify_public_inputs(inputs, contract_dir)
    if cfg.agent.use_global_memory and not (source / "workspace/global_memory/records.json").is_file():
        raise FileNotFoundError("Original global memory records are required")
    review = source / "workspace/candidate_results/candidates" / args.parent_node_id / "review.json"
    if json.loads(review.read_text()).get("status") != "passed":
        raise ValueError("The original parent must have a passed source review")
    output = args.output_dir.expanduser().absolute()
    if output.is_symlink():
        raise ValueError("Mac baseline directory must not be a symlink")
    output = output.resolve()
    ensure_separate(output, source, inputs)
    identity = source_identity(source, snapshot, budget)
    manifest = dict(version=1, profile=PROFILE, status="prepared", source=identity,
                    adapter_version=adapter.ADAPTER_VERSION, input_dir=str(inputs),
                    private_dataset_dir=private_dataset_dir, source_review_sha256=digest(review))
    if output.exists():
        saved = json.loads((output / "baseline.json").read_text())
        if any(saved.get(key) != manifest[key] for key in ("version", "profile", "source", "adapter_version",
                                                         "input_dir", "private_dataset_dir", "source_review_sha256")):
            raise ValueError("Baseline directory belongs to different inputs; choose a new --output-dir")
        if saved.get("status") != "prepared":
            raise ValueError("Baseline already executed or failed; use a new directory for a new calibration")
        if (output / "original_solution.py").read_text() != original or (output / "solution.py").read_text() != code:
            raise ValueError("Prepared baseline source changed")
        manifest = saved
    else:
        output.mkdir(parents=True)
        (output / "original_solution.py").write_text(original)
        (output / "solution.py").write_text(code)
        manifest["files"] = {name: digest(output / name) for name in ("original_solution.py", "solution.py")}
        replay.write_json(output / "baseline.json", manifest)
    if args.execute:
        return calibrate(output, source, snapshot, cfg, inputs, manifest)
    return dict(status="prepared", directory=str(output), training_started=False,
                next_step="Review solution.py, then manually repeat this command with --execute")


def calibrate(output, source, snapshot, cfg, inputs, manifest):
    """Execute a deterministic adaptation, with no LLM generation or review calls."""
    import fcntl
    import replay_improve as replay
    from utils.replay_macos.runtime import digest, mac_process_environment, require_mps

    environment = require_mps()
    # Keep one operator from launching the same calibration twice.
    with (output / ".execute.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This Mac baseline calibration is already running") from None
        if json.loads((output / "baseline.json").read_text())["status"] != "prepared":
            raise ValueError("Baseline is no longer in prepared state")
        manifest.update(status="running", environment=environment)
        replay.write_json(output / "baseline.json", manifest)
        interpreter = None
        try:
            with mac_process_environment():
                from omegaconf import OmegaConf
                from engine.candidate_runtime import integration
                from engine.candidate_runtime.store import ResultStore
                from engine.executor import Interpreter
                from engine.search_node import SearchNode
                from utils.metric import MetricValue
                from utils.replay_state import load_snapshot
                from utils.serialize import dump_json, dumps_json

                calibration = output / "calibration"
                calibration.mkdir()
                replay.isolate_caches(calibration)
                snapshot = replay.copy_inputs(source, calibration, cfg, snapshot, inputs)
                integration.prepare(cfg)
                journal = load_snapshot(snapshot)
                parent = journal.nodes[-1]
                node = SearchNode(code=(output / "solution.py").read_text(), parent=parent,
                                  stage="improve", branch_id=parent.branch_id,
                                  plan="Deterministic Mac platform adaptation; no algorithmic improvement or LLM call",
                                  created_time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                budget = manifest["source"]["replay_budget_seconds"]
                cfg.exec.timeout = min(int(cfg.exec.timeout), budget)
                OmegaConf.save(cfg, cfg.log_dir / "config.yaml")
                integration.register_candidate(cfg, node)
                interpreter = Interpreter(cfg.workspace_dir, timeout=cfg.exec.timeout,
                                          agent_file_name=cfg.exec.agent_file_name, max_parallel_run=1, cfg=cfg)
                interpreter.run_deadline = time.time() + budget
                result = interpreter.run(node.code, node.id, True)
                execution_path = cfg.log_dir / "executions" / f"{node.id}.json"
                replay.write_json(execution_path, asdict(result))
                node.absorb_exec_result(result)
                node.execution_status = result.execution_status
                if result.exc_type or result.execution_status not in ("completed", "budget_exhausted"):
                    error = f"Mac calibration execution failed: {result.exc_type or result.execution_status}"
                    if result.exc_info and result.exc_info.get("message"):
                        error += f": {result.exc_info['message']}"
                    raise RuntimeError(f"{error}\nExecution log: {execution_path}")
                store = ResultStore(cfg.workspace_dir, cfg.log_dir)
                snapshots, rejected = store.valid_snapshots(node.id)
                if not snapshots:
                    raise RuntimeError(f"Mac calibration has no independently verified complete snapshot: {rejected}")
                best = snapshots[0]
                # This is a fixed platform-only patch of reviewed code, not an invented LLM review.
                review = dict(status="passed", kind="locked-platform-adaptation-and-runtime-verification",
                              source_review_sha256=manifest["source_review_sha256"],
                              source_sha256=manifest["source"]["original_source_sha256"],
                              adapted_source_sha256=digest(output / "solution.py"),
                              adapter_version=manifest["adapter_version"], llm_review=False)
                replay.write_json(store.candidate_dir(node.id) / "review.json", review)
                node.metric = MetricValue(best["metric"], maximize=best["maximize"])
                node.is_buggy, node.is_valid = False, True
                node.artifact_status = "scoreable"
                node.best_snapshot_id, node.artifact_metric = best["snapshot_id"], best["metric"]
                node.analysis = (f"Mac MPS calibration of original parent {parent.id}; local metric={best['metric']}. "
                                 "Platform migration only. Earlier source-run measurements were on the original GPU environment.")
                node.code_summary = "Fixed Mac MPS adaptation of the source model, measured on the unchanged validation contract"
                node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                node.local_best_node = node
                journal.append(node)
                dump_json(journal, cfg.log_dir / "journal.json")
                saved_node = json.loads(dumps_json(journal))["nodes"][-1]
                replay.write_json(output / "baseline_node.json", saved_node)
                manifest.update(status="completed", effective_parent_id=node.id,
                                metric=best["metric"], completed_at=time.time())
                candidate = store.candidate_dir(node.id)
                paths = [output / "original_solution.py", output / "solution.py", output / "baseline_node.json"]
                paths += [p for p in candidate.rglob("*") if p.is_file()]
                paths += [p for p in store.metadata_dir(node.id).rglob("*") if p.is_file()]
                paths += [p for p in (cfg.workspace_dir / "candidate_results/contract").iterdir() if p.is_file()]
                manifest["files"] = {str(p.relative_to(output)): digest(p) for p in sorted(paths)}
        except BaseException as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if interpreter is not None:
                interpreter.cleanup_session(-1)
            replay.write_json(output / "baseline.json", manifest)
    return dict(status="completed", directory=str(output), metric=manifest["metric"],
                effective_parent_id=manifest["effective_parent_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--parent-node-id", required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path, help="Optional local MLE-bench root for later private grading")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly start real calibration training; otherwise only prepare files")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
