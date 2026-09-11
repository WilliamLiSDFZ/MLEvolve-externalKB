"""Durable candidate registry and independently verifiable result snapshots."""

import json
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd

from .io import atomic_json, check_hashes, digest, file_hashes, read_json, seal_directory
from .jigsaw import check_csv, load_contract, score


class ResultStore:
    def __init__(self, workspace, log_dir=None):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / "candidate_results"
        self.log_dir = Path(log_dir) if log_dir else self.workspace.parent / "logs"

    def candidate_dir(self, node_id):
        if not node_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in node_id):
            raise ValueError("Invalid candidate ID")
        return self.root / "candidates" / node_id

    def metadata_dir(self, node_id):
        return self.log_dir / "candidate_results" / node_id

    def register(self, node, config, check_leakage=True):
        directory = self.candidate_dir(node.id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "candidate.json"
        if path.exists():
            raise ValueError(f"Candidate {node.id} has already been registered")
        with (directory / "solution.py").open("w", encoding="utf-8") as stream:
            stream.write(node.code)
            stream.flush()
            os.fsync(stream.fileno())
        value = dict(node_id=node.id, parent_id=node.parent.id if node.parent else None,
                     stage=node.stage, branch_id=node.branch_id, created_at=time.time(),
                     source_sha256=digest(directory / "solution.py"), runtime_config=config,
                     status="queued", check_leakage=check_leakage,
                     plan=getattr(node, "plan", None), analogy_report=getattr(node, "analogy_report", None))
        atomic_json(path, value)
        atomic_json(self.metadata_dir(node.id) / "candidate.json", value)
        return value

    def write_execution(self, node_id, **value):
        value = dict(node_id=node_id, **value)
        atomic_json(self.candidate_dir(node_id) / "execution.json", value)
        atomic_json(self.metadata_dir(node_id) / "execution.json", value)

    def event(self, node_id, kind, **details):
        path = self.metadata_dir(node_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # One worker writes events for a candidate. A torn final line is ignored by readers.
        with path.open("a") as stream:
            stream.write(json.dumps(dict(event=kind, at=time.time(), **details), allow_nan=False) + "\n")
            stream.flush()

    def publish(self, node_id, checkpoint_id, write_submission, contract, step, elapsed):
        directory = self.candidate_dir(node_id)
        checkpoint = directory / "checkpoints" / checkpoint_id
        checkpoint_meta = read_json(checkpoint / "manifest.json")
        check_hashes(checkpoint, checkpoint_meta["files"])
        candidate = read_json(directory / "candidate.json")
        snapshots = directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        snapshot_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=snapshots) as tmp:
            staging = Path(tmp) / "bundle"
            staging.mkdir()
            shutil.copy2(directory / "solution.py", staging / "solution.py")
            shutil.copy2(checkpoint / "validation.csv", staging / "validation.csv")
            write_submission(staging / "submission.csv")
            test_ids = pd.read_csv(self.root / "contract/test_ids.csv", dtype=str).id.tolist()
            check_csv(staging / "submission.csv", test_ids)
            manifest = dict(version=1, node_id=node_id, snapshot_id=snapshot_id,
                            checkpoint_id=checkpoint_id, checkpoint_manifest_sha256=digest(checkpoint / "manifest.json"),
                            source_sha256=candidate["source_sha256"], contract_id=contract["contract_id"],
                            metric_version=contract["metric_version"], maximize=contract["maximize"],
                            metric=checkpoint_meta["metric"], optimizer_steps=checkpoint_meta["optimizer_steps"],
                            published_at=time.time(), elapsed_seconds=elapsed(),
                            export_step=step, files=file_hashes(staging))
            atomic_json(staging / "manifest.json", manifest)
            seal_directory(staging, snapshots / snapshot_id)
        # An advisory index: recovery scans committed directories if a crash loses this write.
        atomic_json(directory / "latest.json", {"snapshot_id": snapshot_id})
        atomic_json(self.metadata_dir(node_id) / "snapshots" / f"{snapshot_id}.json", manifest)
        self.event(node_id, "snapshot_published", snapshot_id=snapshot_id,
                   metric=manifest["metric"], elapsed_seconds=manifest["elapsed_seconds"])
        return manifest

    def verify(self, node_id, snapshot_dir, contract=None, allow_pending_review=False):
        directory = self.candidate_dir(node_id)
        contract = contract or load_contract(self.root / "contract")
        candidate = read_json(directory / "candidate.json")
        if (directory / "invalidated.json").exists():
            raise ValueError("Candidate artifacts were invalidated by review")
        snapshot_dir = Path(snapshot_dir)
        manifest = read_json(snapshot_dir / "manifest.json")
        if manifest["node_id"] != node_id or manifest["snapshot_id"] != snapshot_dir.name:
            raise ValueError("Snapshot identity mismatch")
        for key in ("contract_id", "metric_version", "maximize"):
            if manifest[key] != contract[key]:
                raise ValueError(f"Snapshot {key} does not match the holdout contract")
        if manifest["optimizer_steps"] < 1 or not math.isfinite(manifest["elapsed_seconds"]) or manifest["elapsed_seconds"] < 0:
            raise ValueError("Invalid training provenance")
        if manifest["source_sha256"] != candidate["source_sha256"] or digest(directory / "solution.py") != candidate["source_sha256"]:
            raise ValueError("Candidate source changed")
        check_hashes(snapshot_dir, manifest["files"])
        if digest(snapshot_dir / "solution.py") != candidate["source_sha256"]:
            raise ValueError("Snapshot source mismatch")
        checkpoint_id = manifest["checkpoint_id"]
        if not checkpoint_id.isalnum():
            raise ValueError("Invalid checkpoint ID")
        checkpoint = directory / "checkpoints" / checkpoint_id
        if digest(checkpoint / "manifest.json") != manifest["checkpoint_manifest_sha256"]:
            raise ValueError("Checkpoint manifest changed")
        checkpoint_meta = read_json(checkpoint / "manifest.json")
        check_hashes(checkpoint, checkpoint_meta["files"])
        if (checkpoint_meta["contract_id"] != contract["contract_id"]
                or checkpoint_meta["optimizer_steps"] != manifest["optimizer_steps"]
                or digest(checkpoint / "validation.csv") != digest(snapshot_dir / "validation.csv")):
            raise ValueError("Checkpoint and validation predictions are inconsistent")
        answers = pd.read_csv(self.root / "contract/validation.csv", dtype={"id": str})
        values = check_csv(snapshot_dir / "validation.csv", answers.id.tolist())
        metric = score(answers, values)
        if not math.isfinite(manifest["metric"]) or abs(metric - manifest["metric"]) > 1e-10:
            raise ValueError("Snapshot metric failed independent recomputation")
        if abs(metric - checkpoint_meta["metric"]) > 1e-10:
            raise ValueError("Checkpoint metric mismatch")
        if metric == 1.0 and candidate.get("check_leakage", True) and not allow_pending_review:
            review = directory / "review.json"
            if not review.exists() or read_json(review).get("status") != "passed":
                raise ValueError("Perfect local score awaits data-leakage review")
        ids = pd.read_csv(self.root / "contract/test_ids.csv", dtype=str).id.tolist()
        check_csv(snapshot_dir / "submission.csv", ids)
        return manifest

    def valid_snapshots(self, node_id, contract=None, allow_pending_review=False):
        good, errors = [], []
        for directory in sorted((self.candidate_dir(node_id) / "snapshots").glob("*")):
            if directory.name.startswith(".") or not directory.is_dir():
                continue
            try:
                good.append(self.verify(node_id, directory, contract, allow_pending_review))
            except Exception as exc:
                errors.append(dict(snapshot_id=directory.name, reason=str(exc)))
        good.sort(key=lambda s: (-s["metric"] if s["maximize"] else s["metric"], s["published_at"]))
        return good, errors

    def invalidate(self, node_id, reason):
        value = dict(reason=reason, at=time.time())
        atomic_json(self.candidate_dir(node_id) / "invalidated.json", value)
        atomic_json(self.metadata_dir(node_id) / "invalidated.json", value)

    def prune(self, node_id, best_checkpoint_id, keep):
        directory = self.candidate_dir(node_id)
        manifests = [read_json(p) for p in (directory / "snapshots").glob("*/manifest.json")
                     if not p.parent.name.startswith(".")]
        manifests.sort(key=lambda m: (-m["metric"] if m["maximize"] else m["metric"], m["published_at"]))
        retained = manifests[:keep]
        protected = {m["checkpoint_id"] for m in retained} | {best_checkpoint_id}
        for m in manifests[keep:]:
            shutil.rmtree(directory / "snapshots" / m["snapshot_id"])
        for p in (directory / "checkpoints").glob("*"):
            if p.is_dir() and not p.name.startswith(".") and p.name not in protected:
                shutil.rmtree(p)

    def collect(self):
        contract = load_contract(self.root / "contract")
        selected, rows = [], []
        for path in sorted((self.root / "candidates").glob("*/candidate.json")):
            candidate = read_json(path)
            node_id = candidate["node_id"]
            execution_path = path.parent / "execution.json"
            execution = read_json(execution_path) if execution_path.exists() else {"status": "queued"}
            good, errors = self.valid_snapshots(node_id, contract)
            best = good[0] if good else None
            # Charge the entire candidate execution, including work AFTER the selected snapshot.
            elapsed = execution.get("elapsed_seconds")
            if execution.get("status") == "running":
                elapsed = max(0, min(time.time(), execution["deadline"]) - execution["started_at"])
            if best:
                elapsed = max(float(elapsed or 0), best["elapsed_seconds"])
                selected.append(dict(candidate=candidate, snapshot=best, execution=execution,
                                     charged_seconds=elapsed))
            events_path = self.metadata_dir(node_id) / "events.jsonl"
            events = []
            if events_path.exists():
                for line in events_path.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            published = [e for e in events if e["event"] == "snapshot_published"]
            rows.append(dict(node_id=node_id, stage=candidate["stage"], execution=execution,
                             artifact_status="scoreable" if best else "unavailable",
                             selected_snapshot=best, rejected_snapshots=errors,
                             first_scoreable_at=min((s["published_at"] for s in good), default=None),
                             first_published_at=min((e["at"] for e in published), default=None),
                             validation_seconds=sum(e.get("duration_seconds", 0) for e in events if e["event"] == "validation"),
                             export_seconds=sum(e.get("duration_seconds", 0) for e in events if e["event"] == "export")))
        selected.sort(key=lambda item: (-item["snapshot"]["metric"] if contract["maximize"] else item["snapshot"]["metric"],
                                       item["snapshot"]["published_at"]))
        run_path = self.root / "run.json"
        summary = dict(version=1, contract=contract, candidates=rows,
                       run=read_json(run_path) if run_path.exists() else {})
        atomic_json(self.log_dir / "candidate_results/summary.json", summary)
        return selected, summary
