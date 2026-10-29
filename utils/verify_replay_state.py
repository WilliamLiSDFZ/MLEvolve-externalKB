"""Offline journal replay regressions: python utils/verify_replay_state.py."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.replay_state import load_snapshot, restore_agent, snapshot_journal


def saved_node(node_id, step, value=None, branch=1, **changes):
    result = dict(id=node_id, step=step, stage="root" if step == 0 else "improve",
                  code="print('train')", branch_id=branch, parent=None, children=[],
                  metric={"value": value, "maximize": True if value is not None else None},
                  is_buggy=value is None, is_valid=value is not None, exec_time=1.0,
                  _term_out=["completed"], finish_time="2026-10-28T00:00:00")
    result.update(changes)
    return result


def journal_fixture():
    return {"nodes": [saved_node("root", 0, branch=None),
                      saved_node("first", 1, 0.8, stage="draft"),
                      saved_node("other", 2, 0.95, branch=2, stage="draft"),
                      saved_node("parent-full-id", 3, 0.7, expected_child_count=4, is_terminal=True),
                      saved_node("future", 4, 0.99)],
            "node2parent": {"first": "root", "other": "root", "parent-full-id": "first", "future": "parent-full-id"},
            "node2best_local_node": {"first": "future", "parent-full-id": "future"}, "__version": "2"}


class ReplayStateTests(unittest.TestCase):
    def test_prefix_excludes_future_and_rebuilds_ancestral_best_without_mutating_source(self):
        raw = journal_fixture()
        original = copy.deepcopy(raw)
        snapshot = snapshot_journal(raw, "parent-full-id")
        self.assertEqual(raw, original)
        self.assertEqual([node["id"] for node in snapshot["nodes"]], ["root", "first", "other", "parent-full-id"])
        self.assertNotIn("future", json.dumps(snapshot))
        self.assertEqual(snapshot["node2best_local_node"]["parent-full-id"], "first")
        self.assertFalse(snapshot["nodes"][-1]["is_terminal"])
        self.assertEqual(snapshot["nodes"][-1]["expected_child_count"], 0)
        self.assertEqual(snapshot["replay_snapshot"]["cutoff"], "journal-prefix")

    def test_minimizing_metric_reconstructs_the_correct_ancestor(self):
        raw = journal_fixture()
        for node in raw["nodes"][1:]:
            node["metric"]["maximize"] = False
        snapshot = snapshot_journal(raw, "parent-full-id")
        self.assertEqual(snapshot["node2best_local_node"]["parent-full-id"], "parent-full-id")

    def test_parent_requires_exact_id_and_successful_directed_metric(self):
        for parent_id in ("parent", "missing", "root"):
            with self.subTest(parent_id=parent_id), self.assertRaises(ValueError):
                snapshot_journal(journal_fixture(), parent_id)
        changes = [dict(is_buggy=True), dict(is_valid=False), dict(code=""), dict(code=None),
                   dict(metric={"value": 1, "maximize": None}),
                   dict(metric={"value": float("nan"), "maximize": True}),
                   dict(execution_status="running")]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                raw = journal_fixture()
                raw["nodes"][3].update(change)
                snapshot_journal(raw, "parent-full-id")

    def test_invalid_sequence_parent_links_and_duplicate_ids_are_rejected(self):
        for mutation in (lambda raw: raw["nodes"][2].update(step=3),
                         lambda raw: raw["node2parent"].update(first="future"),
                         lambda raw: raw["node2parent"].update(first="parent-full-id"),
                         lambda raw: raw["nodes"][4].update(id="first"),
                         lambda raw: raw["nodes"][2]["metric"].update(maximize=False)):
            raw = journal_fixture()
            mutation(raw)
            with self.assertRaises(ValueError):
                snapshot_journal(raw, "parent-full-id")

    def test_real_journal_roundtrip_restores_children_locks_and_agent_memory(self):
        from utils.serialize import dumps_json

        snapshot = snapshot_journal(journal_fixture(), "parent-full-id")
        journal = load_snapshot(snapshot)
        parent = journal.nodes[-1]
        self.assertEqual(parent.children, set())
        self.assertIs(parent.parent, journal.nodes[1])
        self.assertIs(parent.local_best_node, journal.nodes[1])
        parent.add_expected_child_count()
        self.assertEqual(parent.expected_child_count, 1)
        self.assertEqual(journal.nodes[1].expected_child_count, 0)
        restored = load_snapshot(json.loads(dumps_json(journal)))
        self.assertIs(restored.nodes[-1].local_best_node, restored.nodes[1])
        agent = SimpleNamespace(top_k=3, branch_all_nodes={99: ["stale"]}, best_metric=99)
        self.assertIs(restore_agent(agent, restored, "parent-full-id"), restored.nodes[-1])
        self.assertEqual(agent.current_step, 4)
        self.assertEqual([node.id for node in agent.branch_all_nodes[1]], ["first", "parent-full-id"])
        self.assertEqual(agent.branch_node_count, {1: 2, 2: 1})
        self.assertEqual(agent.next_branch_id, 3)
        self.assertEqual(agent.best_node.id, "other")
        self.assertEqual(agent.best_metric, 0.95)
        self.assertTrue(agent.metric_maximize)


if __name__ == "__main__":
    unittest.main()
