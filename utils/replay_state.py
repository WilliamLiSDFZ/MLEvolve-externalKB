"""Restore the completed journal prefix used for one controlled improve step."""

from __future__ import annotations

import copy
import json
import math
import threading


def _successful(node):
    metric = node.get("metric") or {}
    value = metric.get("value")
    return (node.get("is_buggy") is False and node.get("is_valid") is True
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and isinstance(metric.get("maximize"), bool))


def snapshot_journal(raw: dict, parent_id: str) -> dict:
    """Copy through the exact parent's journal position, excluding all later results.

    shortcut: this restores completed journal history, not in-flight search state;
    a faithful concurrent scheduler replay would require timestamped state snapshots.
    """
    source_nodes = raw.get("nodes", [])
    ids = [node.get("id") for node in source_nodes]
    if any(not isinstance(node_id, str) or not node_id for node_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Journal node IDs must be nonempty and unique")
    if parent_id not in ids:
        raise ValueError("parent_id must be the exact full ID of a saved node")
    parent_index = ids.index(parent_id)
    parent = source_nodes[parent_index]
    if (parent.get("stage") == "root" or not isinstance(parent.get("code"), str)
            or not parent["code"].strip()
            or not _successful(parent)
            or parent.get("execution_status") in {"queued", "running", "pending"}):
        raise ValueError("Replay parent must be a successful non-root node with code and a finite directed metric")

    nodes = copy.deepcopy(source_nodes[:parent_index + 1])
    if nodes[0].get("stage") != "root" or any(node.get("stage") == "root" for node in nodes[1:]):
        raise ValueError("Journal prefix must contain exactly one root at step zero")
    parents = raw.get("node2parent", {})
    node2parent = {}
    by_id = {}
    maximize = parent["metric"]["maximize"]
    for step, node in enumerate(nodes):
        node_id = node["id"]
        if type(node.get("step")) is not int or node["step"] != step:
            raise ValueError("Journal steps must match completion append order")
        parent_node_id = parents.get(node_id)
        if step == 0:
            if parent_node_id is not None:
                raise ValueError("Journal root cannot have a parent")
        elif parent_node_id not in by_id:
            raise ValueError(f"Node {node_id} has no earlier parent in the journal prefix")
        else:
            node2parent[node_id] = parent_node_id
        if _successful(node) and node["metric"]["maximize"] != maximize:
            raise ValueError("Successful journal nodes disagree on metric direction")
        # End-of-run tree counters can contain effects of excluded descendants.
        node.update(parent=None, children=[], local_best_node=None, child_count_lock=None,
                    visits=0, total_reward=0.0, _uct=0.0, is_terminal=False,
                    is_debug_success=False, continue_improve=False, improve_failure_depth=0,
                    lock=False, expected_child_count=0, alpha=1, beta=1, work_dir=None)
        by_id[node_id] = node

    best_local = {}
    for node in nodes:
        ancestors = []
        current = node
        while current is not None:
            if current.get("branch_id") == node.get("branch_id") and _successful(current):
                ancestors.append(current)
            current = by_id.get(node2parent.get(current["id"]))
        if ancestors:
            best = (max if maximize else min)(ancestors, key=lambda candidate: candidate["metric"]["value"])
            best_local[node["id"]] = best["id"]
    return {"nodes": nodes, "node2parent": node2parent, "node2best_local_node": best_local,
            "__version": "2", "replay_snapshot": {
                "cutoff": "journal-prefix", "parent_id": parent_id, "parent_step": parent_index,
                "tree_state": "counters reset; local best rebuilt from successful same-branch ancestors",
            }}


def load_snapshot(raw: dict):
    """Use the normal decoder, restoring local-best references and usable node locks."""
    from engine.search_node import Journal
    from utils.serialize import loads_json

    journal = loads_json(json.dumps(raw), Journal)
    nodes = {node.id: node for node in journal.nodes}
    for node in journal.nodes:
        node.child_count_lock = threading.Lock()
        best_id = raw.get("node2best_local_node", {}).get(node.id)
        node.local_best_node = nodes[best_id] if best_id is not None else None
    return journal


def restore_agent(agent, journal, parent_id: str):
    """Replace constructor state with completed history; caller owns paths and budgets."""
    from engine.solution_manager import update_top_candidates

    parent = next(node for node in journal.nodes if node.id == parent_id)
    agent.journal = journal
    agent.virtual_root = journal.nodes[0]
    agent.current_node = parent
    agent.current_node_list = [parent]
    agent.current_step = len(journal)
    agent.all_root = False
    agent.branch_all_nodes = {}
    agent.branch_successful_nodes = {}
    agent.branch_node_count = {}
    agent.top_candidates = []
    agent.metric_maximize = parent.metric.maximize
    agent.metric_maximize_reasoning = "Recovered from the selected parent's saved metric"
    for node in journal.nodes:
        if node.stage == "root" or node.branch_id is None:
            continue
        agent.branch_all_nodes.setdefault(node.branch_id, []).append(node)
        successful = agent.branch_successful_nodes.setdefault(node.branch_id, [])
        if node.is_buggy is False and node.is_valid is True and node.metric and node.metric.value is not None:
            successful.append(node)
            update_top_candidates(agent, node)
    agent.branch_node_count = {branch: len(nodes) for branch, nodes in agent.branch_all_nodes.items()}
    agent.next_branch_id = max(agent.branch_all_nodes, default=0) + 1
    agent.best_node = agent.top_candidates[0] if agent.top_candidates else None
    agent.best_metric = agent.best_node.metric.value if agent.best_node else None
    agent.best_metric_history = []
    return parent
