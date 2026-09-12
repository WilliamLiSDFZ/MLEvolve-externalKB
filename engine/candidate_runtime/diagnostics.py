"""Read-only runtime facts and cached diagnostics from public validation only.

The search controller already verified the selected snapshot. This module checks
that selection and its small provenance files, but deliberately does not invoke
ResultStore.collect/verify or open model weights, test predictions or private data.
Missing facts stay unknown; diagnostics must never interrupt candidate generation.
"""

from collections import deque
import hashlib
import io
import json
import math
from pathlib import Path
import re

from .io import atomic_json

DIAGNOSTIC_VERSION = "jigsaw-public-components-v1"
CONTEXT_VERSION = 1
MIN_CLASS_SUPPORT = 20  # Diagnostic caution only; never affects the score.
_JSON_BYTES = 2 * 1024 * 1024
_VALIDATION_BYTES = 256 * 1024 * 1024
_EVENT_BYTES = 16 * 1024 * 1024
_EVENT_COUNT = 4096
_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid candidate or artifact ID")
    return value


def _bytes(path, *, limit=_JSON_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or non-regular fact file: {path.name}")
    if path.stat().st_size > limit:
        raise ValueError(f"Fact file exceeds read limit: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Fact file grew beyond read limit: {path.name}")
    return data


def _json(path):
    value = json.loads(_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {Path(path).name}")
    return value


def _optional_json(path, missing):
    try:
        return _json(path)
    except (OSError, ValueError) as exc:
        missing.append(f"{Path(path).name}: {exc}")
        return {}


def _pick(value, names):
    return {name: value.get(name) for name in names}


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _events(path):
    """Bound reads and tolerate a worker killed during its last event write."""
    if not path.is_file():
        return [], {"available": False, "reason": "events_missing"}
    if path.is_symlink():
        return [], {"available": False, "reason": "events_symlink"}
    retained = deque(maxlen=_EVENT_COUNT)
    malformed = count = total_bytes = 0
    truncated = False
    with path.open("rb") as stream:
        for _ in range(_EVENT_COUNT * 4):
            line = stream.readline(min(_JSON_BYTES, _EVENT_BYTES - total_bytes) + 1)
            if not line:
                break
            total_bytes += len(line)
            if total_bytes > _EVENT_BYTES or len(line) > _JSON_BYTES:
                truncated = True
                break
            if not line.endswith(b"\n"):
                malformed += 1
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise ValueError("Malformed event")
                retained.append(event)
                count += 1
            except (ValueError, UnicodeError):
                malformed += 1
        else:
            truncated = True
    return list(retained), dict(available=True, source="events.jsonl", events_read=count,
                               events_retained=len(retained), malformed_or_torn_lines=malformed,
                               truncated=truncated or count > len(retained))


def _verified_selection(cfg, node):
    """Check the journal-selected, reviewed snapshot without re-verifying weights."""
    node_id = _identifier(node.id)
    snapshot_id = _identifier(getattr(node, "best_snapshot_id", None))
    if getattr(node, "artifact_status", None) != "scoreable":
        raise ValueError("Node has no controller-verified scoreable snapshot")
    root = Path(cfg.workspace_dir) / "candidate_results"
    directory = root / "candidates" / node_id
    metadata = Path(cfg.log_dir) / "candidate_results" / node_id
    if (directory / "invalidated.json").exists() or (metadata / "invalidated.json").exists():
        raise ValueError("Candidate artifacts were invalidated by review")
    if _json(directory / "review.json").get("status") != "passed":
        raise ValueError("Candidate review has not passed")
    candidate = _json(directory / "candidate.json")
    snapshot_dir = directory / "snapshots" / snapshot_id
    snapshot = _json(snapshot_dir / "manifest.json")
    if candidate.get("node_id") != node_id or snapshot.get("node_id") != node_id or snapshot.get("snapshot_id") != snapshot_id:
        raise ValueError("Candidate/snapshot identity mismatch")
    source_hash = candidate.get("source_sha256", "")
    if not _HASH.fullmatch(source_hash) or snapshot.get("source_sha256") != source_hash:
        raise ValueError("Snapshot source hash does not match the candidate")
    contract_dir = root / "contract"
    contract = _json(contract_dir / "manifest.json")
    body = {k: v for k, v in contract.items() if k != "contract_id"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != contract.get("contract_id"):
        raise ValueError("Changed public validation contract")
    for key in ("contract_id", "metric_version", "maximize"):
        if snapshot.get(key) != contract.get(key):
            raise ValueError(f"Snapshot {key} does not match the public validation contract")
    if not _number(snapshot.get("optimizer_steps")) or snapshot["optimizer_steps"] < 1:
        raise ValueError("Selected snapshot has invalid optimizer steps")
    if not _number(snapshot.get("metric")):
        raise ValueError("Selected snapshot has a nonfinite metric")
    checkpoint_id = _identifier(snapshot.get("checkpoint_id"))
    checkpoint_bytes = _bytes(directory / "checkpoints" / checkpoint_id / "manifest.json")
    if hashlib.sha256(checkpoint_bytes).hexdigest() != snapshot.get("checkpoint_manifest_sha256"):
        raise ValueError("Checkpoint manifest hash mismatch")
    checkpoint = json.loads(checkpoint_bytes)
    for key in ("checkpoint_id", "contract_id", "optimizer_steps", "metric"):
        if checkpoint.get(key) != snapshot.get(key):
            raise ValueError(f"Checkpoint/snapshot {key} mismatch")
    expected_prediction_hash = snapshot.get("files", {}).get("validation.csv")
    if not isinstance(expected_prediction_hash, str) or not _HASH.fullmatch(expected_prediction_hash):
        raise ValueError("Selected snapshot has no validation prediction hash")
    if checkpoint.get("files", {}).get("validation.csv") != expected_prediction_hash:
        raise ValueError("Checkpoint and snapshot validation hashes differ")
    return dict(root=root, directory=directory, snapshot_dir=snapshot_dir, contract_dir=contract_dir,
                candidate=candidate, snapshot=snapshot, contract=contract)


def _hashed_validation(path, expected):
    if not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise ValueError(f"Missing validation hash for {Path(path).name}")
    data = _bytes(path, limit=_VALIDATION_BYTES)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Changed validation file: {Path(path).name}")
    return data


def _terms(report):
    yield "overall", report["overall"]
    for identity, components in report["identities"].items():
        for kind, component in components.items():
            yield f"{identity}/{kind}", component


def validation_diagnostics(cfg, node):
    """Full diagnostic report, or an explicit unavailable result.

    The cache contains derived public-validation numbers only, never candidate
    source, weights or examples. Both source CSV hashes are checked on cache hits.
    """
    if not getattr(getattr(cfg, "candidate_runtime", None), "enabled", False):
        return dict(available=False, supported=False, reason="candidate_runtime_disabled")
    from .jigsaw import IDENTITIES, METRIC_VERSION, TASK_ID, predictions, score_components
    if getattr(cfg, "exp_id", None) != TASK_ID:
        return dict(available=False, supported=False, reason="No diagnostics provider for this task")
    try:
        selection = _verified_selection(cfg, node)
        contract, snapshot = selection["contract"], selection["snapshot"]
        if contract.get("task_id") != TASK_ID or contract.get("metric_version") != METRIC_VERSION:
            return dict(available=False, supported=False, reason="No diagnostics provider for this task/metric")
        prediction_hash = snapshot["files"]["validation.csv"]
        labels_hash = contract.get("files", {}).get("validation.csv")
        prediction_bytes = _hashed_validation(selection["snapshot_dir"] / "validation.csv", prediction_hash)
        label_bytes = _hashed_validation(selection["contract_dir"] / "validation.csv", labels_hash)
        provenance = dict(diagnostic_version=DIAGNOSTIC_VERSION, contract_id=contract["contract_id"],
                          metric_version=contract["metric_version"], prediction_sha256=prediction_hash,
                          validation_labels_sha256=labels_hash)
        cache_key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache_path = Path(cfg.log_dir) / "candidate_results/diagnostics" / f"{cache_key}.json"
        report = None
        if cache_path.is_file():
            try:
                cached = _json(cache_path)
                if (cached.get("provenance") == provenance and cached.get("available") is True
                        and _number(cached.get("metric"))
                        and abs(cached["metric"] - snapshot["metric"]) <= 1e-10
                        and set(cached.get("identities", {})) == set(IDENTITIES)
                        and all({"subgroup", "bpsn", "bnsp"} == set(v) for v in cached["identities"].values())):
                    report = cached
            except (OSError, ValueError, TypeError):
                pass
        cache_hit = report is not None
        if report is None:
            import numpy as np
            import pandas as pd
            answers = pd.read_csv(io.BytesIO(label_bytes), dtype={"id": str})
            frame = pd.read_csv(io.BytesIO(prediction_bytes), dtype={"id": str})
            if list(frame.columns) != ["id", "prediction"]:
                raise ValueError("Validation prediction CSV must contain only id,prediction")
            if not {"id", "target", *IDENTITIES}.issubset(answers.columns):
                raise ValueError("Public validation labels lack required columns")
            for table in (answers, frame):
                if table.id.isna().any() or not table.id.is_unique:
                    raise ValueError("Validation IDs must be present and unique")
            if len(answers) != contract.get("validation_rows") or set(answers.id) != set(frame.id):
                raise ValueError("Validation prediction IDs do not match the public contract")
            if not np.isfinite(answers.target).all():
                raise ValueError("Public validation target labels must be finite")
            # Never assume predictions already follow label order; join by exact ID.
            values = predictions(frame.set_index("id").loc[answers.id, "prediction"].to_numpy(), len(answers))
            components = score_components(answers, values, allow_undefined=True)
            if components["metric"] is None or abs(components["metric"] - snapshot["metric"]) > 1e-10:
                raise ValueError("Public component metric differs from the verified snapshot")
            for _, term in _terms(components):
                term["support_sufficient"] = min(term["positive_count"], term["negative_count"]) >= MIN_CLASS_SUPPORT
            report = dict(available=True, supported=True, provenance=provenance, **components,
                          minimum_class_support=MIN_CLASS_SUPPORT,
                          interpretation="Observed public-validation components, not causal evidence or a reweighting recommendation")
            try:
                atomic_json(cache_path, report)
            except OSError:
                # Read-only replay directories should still yield useful diagnostics.
                report["cache_write"] = "unavailable"
        # Review can invalidate a candidate while this CPU computation is running.
        directory = selection["directory"]
        metadata = Path(cfg.log_dir) / "candidate_results" / str(node.id)
        if (directory / "invalidated.json").exists() or (metadata / "invalidated.json").exists():
            raise ValueError("Candidate invalidated during diagnostic generation")
        return dict(report, cache_hit=cache_hit, cache_key=cache_key,
                    selection=dict(node_id=node.id, snapshot_id=snapshot["snapshot_id"],
                                   checkpoint_id=snapshot["checkpoint_id"], source_sha256=snapshot["source_sha256"],
                                   optimizer_steps=snapshot["optimizer_steps"], origin="controller_verified_node.best_snapshot_id"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return dict(available=False, supported=True, reason=str(exc))


def compare_diagnostics(current, parent):
    """Only compare identical public contracts; small support is flagged explicitly."""
    if not current.get("available") or not parent.get("available"):
        return dict(comparable=False, reason="Current or parent verified public diagnostic unavailable")
    for key in ("contract_id", "metric_version", "validation_labels_sha256", "diagnostic_version"):
        if current["provenance"].get(key) != parent["provenance"].get(key):
            return dict(comparable=False, reason=f"Parent/current {key} differs")
    previous = dict(_terms(parent))
    deltas = []
    for name, term in _terms(current):
        base = previous[name]
        sufficient = bool(term["support_sufficient"] and base["support_sufficient"])
        same_support = all(term[k] == base[k] for k in ("rows", "positive_count", "negative_count"))
        comparable = sufficient and same_support and term["defined"] and base["defined"]
        deltas.append(dict(component=name, auc=term["auc"], parent_auc=base["auc"],
                           delta=term["auc"] - base["auc"] if comparable else None,
                           comparable=bool(comparable), positive_count=term["positive_count"],
                           negative_count=term["negative_count"],
                           reason=None if comparable else "Insufficient class support or changed component membership"))
    return dict(comparable=True, metric_delta=current["metric"] - parent["metric"],
                parent_selection=parent.get("selection"), components=deltas)


def _compact_diagnostics(report, comparison=None):
    if not report.get("available"):
        return report
    weak = [dict(component=name, **term) for name, term in _terms(report) if name != "overall"]
    weak.sort(key=lambda term: (term["auc"] is None, term["auc"] or 0.0, term["component"]))
    result = {k: report[k] for k in ("available", "supported", "provenance", "selection", "metric", "overall",
                                    "power_means", "minimum_class_support", "interpretation", "cache_hit", "cache_key")}
    result["weakest_components"] = weak[:6]
    result["components_total"] = 27
    result["components_shown"] = len(result["weakest_components"])
    result["all_components_in_diagnostic_cache"] = report.get("cache_write") != "unavailable"
    if comparison is not None:
        result["parent_comparison"] = {k: v for k, v in comparison.items() if k != "components"}
        if comparison.get("comparable"):
            deltas = comparison["components"]
            result["parent_comparison"]["largest_declines"] = sorted(
                (row for row in deltas if row["comparable"] and row["delta"] < 0),
                key=lambda row: row["delta"])[:5]
            result["parent_comparison"]["noncomparable_components"] = [
                row["component"] for row in deltas if not row["comparable"]]
    return result


def build_runtime_context(agent, node, *, include_diagnostics=True):
    """Build a bounded, JSON-ready packet using controller facts already on disk."""
    cfg = agent.cfg
    if not getattr(getattr(cfg, "candidate_runtime", None), "enabled", False):
        return dict(version=CONTEXT_VERSION, available=False, reason="candidate_runtime_disabled",
                    public_validation=dict(available=False, supported=False, reason="runtime_disabled"))
    missing = []
    try:
        node_id = _identifier(node.id)
        directory = Path(cfg.workspace_dir) / "candidate_results/candidates" / node_id
        metadata = Path(cfg.log_dir) / "candidate_results" / node_id
        candidate = _optional_json(directory / "candidate.json", missing)
        candidate_origin = "candidate_results/candidates/candidate.json"
        if not candidate:
            candidate = _optional_json(metadata / "candidate.json", missing)
            candidate_origin = "logs/candidate_results/candidate.json (metadata copy)"
        if candidate and candidate.get("node_id") != node_id:
            raise ValueError("Candidate metadata belongs to a different node")
        execution = _optional_json(directory / "execution.json", missing)
        if not execution:
            execution = _optional_json(metadata / "execution.json", missing)
        if execution and execution.get("node_id") != node_id:
            raise ValueError("Execution metadata belongs to a different node")
        finished = _optional_json(directory / "worker_finished.json", missing)
        if finished and finished.get("node_id") != node_id:
            raise ValueError("Worker completion metadata belongs to a different node")
        events, events_info = _events(metadata / "events.jsonl")
        validations = [e for e in events if e["event"] == "validation"]
        starts = [e for e in events if e["event"] == "training_started"]
        smoke = [e for e in events if e["event"] == "smoke_passed"]
        recalibrations = [e for e in events if e["event"] == "budget_recalibrated"]
        trajectory = [_pick(e, ("optimizer_steps", "metric", "duration_seconds", "elapsed_seconds")) for e in validations]
        trajectory_limit = 20
        shown = trajectory if len(trajectory) <= trajectory_limit else trajectory[:3] + trajectory[-17:]
        selected = dict(available=False, reason="No controller-verified selected snapshot")
        contract_info = {}
        try:
            selection = _verified_selection(cfg, node)
            snapshot = selection["snapshot"]
            selected = dict(available=True, origin="controller_verified_node.best_snapshot_id", **_pick(snapshot,
                ("snapshot_id", "checkpoint_id", "source_sha256", "contract_id", "metric_version", "metric",
                 "optimizer_steps", "export_step", "elapsed_seconds", "published_at")))
            contract_info = _pick(selection["contract"], ("contract_id", "task_id", "metric_version", "validation_rows", "seed"))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            selected["reason"] = str(exc)
        current = validation_diagnostics(cfg, node) if include_diagnostics else dict(available=False, reason="diagnostics_disabled")
        comparison = None
        parent = getattr(node, "parent", None)
        if include_diagnostics and current.get("available") and parent is not None:
            comparison = compare_diagnostics(current, validation_diagnostics(cfg, parent))
        observed_steps = [e["optimizer_steps"] for e in events if _number(e.get("optimizer_steps"))]
        wait = (execution["started_at"] - candidate["created_at"]
                if _number(execution.get("started_at")) and _number(candidate.get("created_at")) else None)
        result = dict(version=CONTEXT_VERSION, available=bool(candidate),
            provenance=dict(node_id=node_id, candidate_source=candidate_origin, events=events_info,
                            no_weight_or_test_reads=True, unknown_fields_are_null=True),
            candidate=_pick(candidate, ("node_id", "parent_id", "stage", "branch_id", "source_sha256")),
            selected_snapshot=selected, contract=contract_info,
            execution=dict(**_pick(execution, ("status", "started_at", "deadline", "finished_at", "elapsed_seconds", "exc_type")),
                           artifact_status=getattr(node, "artifact_status", None), worker_finish_reason=finished.get("reason"),
                           exception_summary=str(getattr(node, "exc_info", None) or "")[:2000] or None),
            training=dict(selected_checkpoint_optimizer_steps=selected.get("optimizer_steps"),
                          final_worker_optimizer_steps=finished.get("optimizer_steps"),
                          last_observed_optimizer_steps=max(observed_steps) if observed_steps else None,
                          note="Worker total is unknown without worker_finished.json; event steps are only a lower bound"),
            costs=dict(queue_wait_seconds=max(0.0, wait) if wait is not None else None,
                       before_training_wall_seconds=starts[0].get("elapsed_seconds") if starts else None,
                       validation_seconds_observed=sum(e.get("duration_seconds", 0) for e in validations if _number(e.get("duration_seconds"))),
                       export_seconds_observed=sum(e.get("duration_seconds", 0) for e in events if e["event"] == "export" and _number(e.get("duration_seconds"))),
                       smoke=smoke[-1] if smoke else None, budget_recalibrations=recalibrations[-3:],
                       net_training_seconds=None, checkpoint_save_seconds=None,
                       note="Validation/export totals cover retained events; preprocessing wall time includes setup and model loading"),
            validation_trajectory=dict(source="events.jsonl validation events; not a training-loss curve", values=shown,
                                       observed_count=len(trajectory), shown_count=len(shown), truncated=len(shown) < len(trajectory)),
            public_validation=_compact_diagnostics(current, comparison), missing=missing[:10])
        # Reject malformed persisted NaNs rather than propagating them into API JSON.
        json.dumps(result, allow_nan=False)
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return dict(version=CONTEXT_VERSION, available=False, reason=str(exc), missing=missing[:10])
