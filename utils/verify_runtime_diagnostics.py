"""CPU checks for public diagnostics, immutable provenance and unchanged scoring.

Run: python utils/verify_runtime_diagnostics.py
No model/cluster requests, training, test predictions or private labels are used.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.candidate_runtime import diagnostics
from engine.candidate_runtime.config import CandidateRuntimeConfig
from engine.candidate_runtime.io import atomic_json, digest, read_json
from engine.candidate_runtime.jigsaw import IDENTITIES, METRIC_VERSION, TASK_ID, score, score_components


def independent_auc(labels, values):
    positive, negative = values[labels], values[~labels]
    return float(np.mean((positive[:, None] > negative).astype(float)
                         + 0.5 * (positive[:, None] == negative)))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.ws, self.logs = self.run / "workspace", self.run / "logs"
        self.root = self.ws / "candidate_results"
        self.cfg = SimpleNamespace(workspace_dir=self.ws, log_dir=self.logs, exp_id=TASK_ID,
                                   candidate_runtime=CandidateRuntimeConfig(enabled=True))
        self.agent = SimpleNamespace(cfg=self.cfg)
        rng = np.random.default_rng(61)
        n = 600
        self.labels = np.tile([False, True], n // 2)
        self.answers = pd.DataFrame(dict(id=[f"row-{i}" for i in range(n)], target=self.labels.astype(float)))
        for identity in IDENTITIES:
            self.answers[identity] = rng.binomial(1, 0.5, n)
        self.values = np.clip(0.35 + 0.2 * self.labels + rng.normal(0, 0.2, n), 0, 1)
        directory = self.root / "contract"
        directory.mkdir(parents=True)
        self.answers.to_csv(directory / "validation.csv", index=False)
        contract = dict(version=1, task_id=TASK_ID, metric_version=METRIC_VERSION,
                        maximize=True, seed=61, validation_rows=n,
                        files={"validation.csv": digest(directory / "validation.csv")})
        contract["contract_id"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        atomic_json(directory / "manifest.json", contract)
        self.contract = contract
        self.node = self.candidate("current")

    def candidate(self, node_id, values=None, *, parent=None, reorder=False):
        values = self.values if values is None else values
        directory = self.root / "candidates" / node_id
        snapshot_id, checkpoint_id = "snapshot1", "checkpoint1"
        snapshot_dir = directory / "snapshots" / snapshot_id
        checkpoint_dir = directory / "checkpoints" / checkpoint_id
        snapshot_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)
        source_hash = hashlib.sha256(b"print('example')\n").hexdigest()
        atomic_json(directory / "candidate.json", dict(node_id=node_id, parent_id=parent.id if parent else None,
                    stage="improve" if parent else "draft", branch_id=1, created_at=10,
                    source_sha256=source_hash, runtime_config=asdict(self.cfg.candidate_runtime)))
        frame = pd.DataFrame(dict(id=self.answers.id, prediction=values))
        if reorder:
            frame = frame.sample(frac=1, random_state=3)
        frame.to_csv(snapshot_dir / "validation.csv", index=False)
        prediction_hash = digest(snapshot_dir / "validation.csv")
        checkpoint = dict(checkpoint_id=checkpoint_id, contract_id=self.contract["contract_id"],
                          optimizer_steps=100, metric=score(self.answers, values),
                          files={"validation.csv": prediction_hash, "model/weights.bin": "not-read"})
        atomic_json(checkpoint_dir / "manifest.json", checkpoint)
        snapshot = dict(node_id=node_id, snapshot_id=snapshot_id, checkpoint_id=checkpoint_id,
                        checkpoint_manifest_sha256=digest(checkpoint_dir / "manifest.json"),
                        source_sha256=source_hash, contract_id=self.contract["contract_id"],
                        metric_version=METRIC_VERSION, maximize=True, optimizer_steps=100,
                        metric=checkpoint["metric"], export_step=105, elapsed_seconds=900,
                        published_at=950, files={"validation.csv": prediction_hash})
        atomic_json(snapshot_dir / "manifest.json", snapshot)
        atomic_json(directory / "review.json", dict(status="passed"))
        atomic_json(directory / "execution.json", dict(node_id=node_id, status="completed", started_at=110,
                    finished_at=1010, elapsed_seconds=900, deadline=1100, exc_type=None))
        atomic_json(directory / "worker_finished.json", dict(node_id=node_id, reason="completed", optimizer_steps=150))
        meta = self.logs / "candidate_results" / node_id
        meta.mkdir(parents=True)
        events = [dict(event="training_started", at=170, elapsed_seconds=60),
                  dict(event="validation", at=710, optimizer_steps=100, metric=checkpoint["metric"],
                       elapsed_seconds=600, duration_seconds=25),
                  dict(event="export", at=910, duration_seconds=45),
                  dict(event="validation", at=1010, optimizer_steps=150, metric=checkpoint["metric"] - 0.01,
                       elapsed_seconds=900, duration_seconds=30)]
        (meta / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + '{"event":')
        return SimpleNamespace(id=node_id, parent=parent, stage="improve" if parent else "draft", branch_id=1,
                               artifact_status="scoreable", best_snapshot_id=snapshot_id, exc_info=None)

    def manifest(self, node=None):
        node = node or self.node
        return self.root / "candidates" / node.id / "snapshots" / node.best_snapshot_id / "manifest.json"

    def report(self, node=None):
        return diagnostics.validation_diagnostics(self.cfg, node or self.node)

    def test_score_parity_with_independent_pairwise_auc(self):
        parts = {"subgroup": [], "bpsn": [], "bnsp": []}
        for identity in IDENTITIES:
            group = self.answers[identity].to_numpy() >= 0.5
            masks = (group, (group & ~self.labels) | (~group & self.labels),
                     (group & self.labels) | (~group & ~self.labels))
            for kind, mask in zip(parts, masks):
                parts[kind].append(independent_auc(self.labels[mask], self.values[mask]))
        expected = 0.25 * independent_auc(self.labels, self.values)
        expected += 0.25 * sum(float(np.mean(np.power(v, -5.0)) ** -0.2) for v in parts.values())
        self.assertAlmostEqual(score(self.answers, self.values), expected, places=14)
        report = score_components(self.answers, self.values)
        self.assertEqual(report["metric"], score(self.answers, self.values))
        self.assertEqual(len(report["identities"]), 9)
        self.assertEqual(report["overall"]["positive_count"], 300)

    def test_score_undefined_and_zero_auc_behavior(self):
        broken = self.answers.copy()
        broken["male"] = 0
        with self.assertRaisesRegex(ValueError, "Undefined AUC for male/subgroup"):
            score(broken, self.values)
        diagnostic = score_components(broken, self.values, allow_undefined=True)
        self.assertIsNone(diagnostic["metric"])
        self.assertFalse(diagnostic["identities"]["male"]["subgroup"]["defined"])
        self.assertEqual(score(self.answers, 1 - self.labels.astype(float)), 0.0)

    def test_alignment_uses_exact_ids(self):
        shuffled = self.candidate("shuffled", reorder=True)
        regular, reordered = self.report(), self.report(shuffled)
        self.assertTrue(reordered["available"], reordered)
        self.assertEqual(regular["metric"], reordered["metric"])
        self.assertEqual(regular["identities"], reordered["identities"])

    def test_wrong_or_duplicate_ids_cannot_be_diagnosed(self):
        for duplicate in (False, True):
            node = self.candidate("duplicate" if duplicate else "wrong-id")
            path = self.manifest(node).parent / "validation.csv"
            frame = pd.read_csv(path, dtype={"id": str})
            frame.loc[0, "id"] = frame.loc[1, "id"] if duplicate else "different-id"
            frame.to_csv(path, index=False)
            snapshot = read_json(self.manifest(node))
            snapshot["files"]["validation.csv"] = digest(path)
            checkpoint_path = self.root / "candidates" / node.id / "checkpoints/checkpoint1/manifest.json"
            checkpoint = read_json(checkpoint_path)
            checkpoint["files"]["validation.csv"] = digest(path)
            atomic_json(checkpoint_path, checkpoint)
            snapshot["checkpoint_manifest_sha256"] = digest(checkpoint_path)
            atomic_json(self.manifest(node), snapshot)
            report = self.report(node)
            self.assertFalse(report["available"])
            self.assertIn("IDs", report["reason"])

    def test_cache_hit_rechecks_prediction_and_label_hashes(self):
        first, second = self.report(), self.report()
        self.assertTrue(first["available"], first)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        path = self.manifest().parent / "validation.csv"
        path.write_text(path.read_text() + "\n")
        self.assertIn("Changed validation file", self.report()["reason"])
        path.write_text(path.read_text()[:-1])
        labels = self.root / "contract/validation.csv"
        labels.write_text(labels.read_text() + "\n")
        self.assertIn("Changed validation file", self.report()["reason"])

    def test_selection_requires_reviewed_not_invalidated_snapshot(self):
        directory = self.root / "candidates/current"
        review = directory / "review.json"
        review.unlink()
        self.assertFalse(self.report()["available"])
        atomic_json(review, dict(status="pending"))
        self.assertFalse(self.report()["available"])
        atomic_json(review, dict(status="passed"))
        atomic_json(directory / "invalidated.json", dict(reason="leakage"))
        self.assertIn("invalidated", self.report()["reason"])
        (directory / "invalidated.json").unlink()
        self.node.best_snapshot_id = None
        self.assertFalse(self.report()["available"])

    def test_no_store_recovery_weights_or_test_prediction_reads(self):
        from engine.candidate_runtime.store import ResultStore
        original = Path.open
        opened = []

        def guarded(path, *args, **kwargs):
            opened.append(str(path))
            if "model" in path.parts or path.name in ("submission.csv", "test_ids.csv"):
                raise AssertionError(f"Unexpected large/private/test file: {path}")
            return original(path, *args, **kwargs)

        with patch.object(ResultStore, "collect", side_effect=AssertionError("collect forbidden")), \
             patch.object(ResultStore, "verify", side_effect=AssertionError("verify forbidden")), \
             patch.object(Path, "open", guarded):
            context = diagnostics.build_runtime_context(self.agent, self.node)
        self.assertTrue(context["public_validation"]["available"], context)
        self.assertTrue(opened)
        self.assertEqual(context["training"]["selected_checkpoint_optimizer_steps"], 100)
        self.assertEqual(context["training"]["final_worker_optimizer_steps"], 150)
        self.assertEqual(context["costs"]["before_training_wall_seconds"], 60)
        self.assertEqual(context["costs"]["validation_seconds_observed"], 55)
        self.assertEqual(context["costs"]["export_seconds_observed"], 45)
        self.assertEqual(context["costs"]["queue_wait_seconds"], 100)
        self.assertEqual(context["provenance"]["events"]["malformed_or_torn_lines"], 1)

    def test_parent_deltas_require_same_contract_and_sufficient_support(self):
        parent = self.candidate("parent", values=np.full(len(self.answers), 0.5))
        self.node.parent = parent
        current, previous = self.report(), self.report(parent)
        compared = diagnostics.compare_diagnostics(current, previous)
        self.assertTrue(compared["comparable"])
        self.assertAlmostEqual(compared["metric_delta"], current["metric"] - 0.5)
        previous["provenance"]["contract_id"] = "another-split"
        self.assertFalse(diagnostics.compare_diagnostics(current, previous)["comparable"])
        previous["provenance"]["contract_id"] = current["provenance"]["contract_id"]
        previous["identities"]["male"]["subgroup"]["support_sufficient"] = False
        comparison = diagnostics.compare_diagnostics(current, previous)
        term = next(t for t in comparison["components"] if t["component"] == "male/subgroup")
        self.assertFalse(term["comparable"])
        self.assertIsNone(term["delta"])

    def test_snapshot_provenance_mismatch_fails_closed(self):
        path = self.manifest()
        original = read_json(path)
        for key, wrong in (("contract_id", "wrong"), ("source_sha256", "0" * 64),
                           ("checkpoint_manifest_sha256", "0" * 64), ("node_id", "other")):
            atomic_json(path, dict(original, **{key: wrong}))
            self.assertFalse(self.report()["available"], key)
        atomic_json(path, original)

    def test_runtime_off_or_other_task_and_missing_completion(self):
        self.cfg.candidate_runtime.enabled = False
        self.assertEqual(diagnostics.build_runtime_context(self.agent, self.node)["reason"], "candidate_runtime_disabled")
        self.cfg.candidate_runtime.enabled = True
        self.cfg.exp_id = "another-task"
        self.assertFalse(self.report()["supported"])
        self.cfg.exp_id = TASK_ID
        (self.root / "candidates/current/worker_finished.json").unlink()
        context = diagnostics.build_runtime_context(self.agent, self.node)
        self.assertIsNone(context["training"]["final_worker_optimizer_steps"])
        self.assertEqual(context["training"]["last_observed_optimizer_steps"], 150)


if __name__ == "__main__":
    unittest.main(verbosity=2)
