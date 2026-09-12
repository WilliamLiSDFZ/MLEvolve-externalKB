#!/usr/bin/env python3
"""CPU-only verification of analogy context and source tools; never calls an LLM.

Optional historical packet audit:
  python utils/verify_analogy_context.py --historical-root /path/to/fetched/results
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.analogy.code_tools import CodeReadingSession, CodeToolOptions
from engine.analogy.context import (ContextOptions, clean_execution_output, context_options,
                                    describe_plan, packet_from_search, resource_context)


def node(node_id="current", code="LR = 0.1\ndef loss(x):\n    return x * 2\n", **kwargs):
    values = dict(id=node_id, code=code, stage="improve", parent=None, children=[], plan="current plan",
                  metric=NS(value=0.9, maximize=True), execution_status="completed", exec_time=25,
                  artifact_status="scoreable", best_snapshot_id="best-checkpoint", branch_id=1,
                  ctime=1, _term_out=["start\n", "steps=250\n"], analysis="completed")
    values.update(kwargs)
    return NS(**values)


def agent(current, nodes=None, **options):
    cfg = NS(analogy=NS(context=ContextOptions(version=2, **options), code_tools=CodeToolOptions()),
             candidate_runtime=NS(enabled=False), workspace_dir="/tmp/context-test/workspace",
             log_dir="/tmp/context-test/logs", exec=NS(timeout=600), cpu_number=2)
    return NS(cfg=cfg, task_desc="# Task\nclassification\n## Evaluation\nAUC labels are continuous.",
              data_preview="train.csv", branch_all_nodes={1: nodes or [current]},
              branch_successful_nodes={1: [current]}, runtime_deadline=3000)


class ContextTests(unittest.TestCase):
    def test_plan_retains_changes_before_reason_and_removes_raw(self):
        plan = {"reason": "OLD_BCE_PROBLEM " * 2000, "module": "loss", "plan": "CURRENT_HALF_BCE_PLUS_RANKING",
                "raw_response": "DUPLICATE_SENTINEL" * 3000}
        current = node(plan=json.dumps(plan))
        packet = packet_from_search(agent(current), current, runtime_facts={"available": False})
        self.assertIn("CURRENT_HALF_BCE_PLUS_RANKING", packet.text)
        self.assertNotIn("DUPLICATE_SENTINEL", packet.text)
        self.assertLess(packet.text.index("CURRENT_HALF_BCE_PLUS_RANKING"), packet.text.index("OLD_BCE_PROBLEM"))
        self.assertTrue(packet.metadata["sections"]["plan"]["truncated"])

    def test_cleanup_before_truncation_preserves_exception(self):
        text = ("initial optimizer_steps=7\n" + "file.py:12: FutureWarning: old API\n  with torch.cuda.amp.autocast():\n" * 1000
                + "huggingface/tokenizers: The current process just got forked\n"
                + "To disable this warning, you can either:\n"
                + "\t- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)\n"
                + "Traceback (most recent call last):\n  File 'solution.py', line 20\n    raise RuntimeError('bad')\nRuntimeError: bad\n")
        cleaned, stats = clean_execution_output([text])
        self.assertIn("optimizer_steps=7", cleaned)
        self.assertIn("    raise RuntimeError", cleaned)
        self.assertEqual(stats["warning_lines_omitted"], 2003)
        current = node(_term_out=[text])
        packet = packet_from_search(agent(current), current, runtime_facts={"available": False})
        self.assertIn("optimizer_steps=7", packet.text)
        self.assertIn("RuntimeError: bad", packet.text)

    def test_packet_cap_keeps_snapshot_and_runtime_facts(self):
        current = node(plan="plan " * 10000, analysis="analysis " * 3000, _term_out=["logs " * 9000])
        cfg_agent = agent(current, max_packet_chars=18000)
        cfg_agent.task_desc = "task " * 9000
        facts = {"available": True, "selected_snapshot": {"snapshot_id": "verified-snapshot", "optimizer_steps": 4714},
                 "training": {"worker_final_steps": 4900}, "contract": {"contract_id": "same-contract"}}
        packet = packet_from_search(cfg_agent, current, runtime_facts=facts)
        self.assertLessEqual(len(packet.text), 18000)
        self.assertIn("verified-snapshot", packet.text)
        self.assertIn("4714", packet.text)
        self.assertIn("same-contract", packet.text)

    def test_only_executed_whitelisted_nodes_and_frozen_source(self):
        parent = node("parent", ctime=0)
        current = node(parent=parent)
        queued = node("queued", execution_status="queued", exec_time=None)
        other = node("other-branch", branch_id=2)
        cfg_agent = agent(current, [parent, current, queued])
        session = CodeReadingSession.from_search(cfg_agent, current)
        self.assertNotIn("queued", session.sources)
        self.assertEqual(session.dispatch("read_candidate_code", {"node_id": other.id})["status"], "error")
        current.code = "MUTATED = True"
        result = session.dispatch("read_candidate_code", {"node_id": current.id})
        self.assertIn("return x * 2", result["content"])
        self.assertNotIn("MUTATED", result["content"])
        self.assertEqual(result["origin"], "search_node_snapshot")

    def test_source_not_executed_and_ast_failure_readable(self):
        current = node(code="raise RuntimeError('must not execute')\ndef bad(:\n")
        session = CodeReadingSession([current], current.id)
        self.assertEqual(session.index_summary()["index"]["status"], "syntax_error")
        self.assertIn("must not execute", session.dispatch("read_candidate_code", {})["content"])

    def test_registered_source_hash_and_missing_fail_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = node()
            directory = Path(tmp) / "candidate_results/candidates/current"
            directory.mkdir(parents=True)
            (directory / "solution.py").write_text(current.code)
            meta = {"node_id": "current", "source_sha256": hashlib.sha256(current.code.encode()).hexdigest()}
            (directory / "candidate.json").write_text(json.dumps(meta))
            session = CodeReadingSession([current], current.id, workspace=tmp, runtime_enabled=True)
            self.assertEqual(session.dispatch("read_candidate_code", {})["origin"], "registered_solution")
            (directory / "solution.py").write_text("different")
            self.assertIn("return x * 2", session.dispatch("read_candidate_code", {})["content"])
            rejected = CodeReadingSession([current], current.id, workspace=tmp, runtime_enabled=True)
            self.assertIn("mismatch", rejected.dispatch("read_candidate_code", {})["reason"])
            (directory / "solution.py").unlink()
            self.assertFalse(CodeReadingSession([current], current.id, workspace=tmp, runtime_enabled=True).sources[current.id]["available"])

    def test_read_pagination_ranges_and_source_anchors(self):
        current = node(code="def loss(x):\n" + "    x += 1\n" * 8 + "    return x\n")
        session = CodeReadingSession([current], current.id, options=CodeToolOptions(default_read_lines=3))
        first = session.dispatch("read_candidate_code", {"symbol": "loss"})
        self.assertTrue(first["truncated"])
        self.assertEqual(first["continuation"]["start_line"], 4)
        second = session.dispatch("read_candidate_code", first["continuation"])
        self.assertTrue(second["content"].startswith("4: "))
        anchor = {"node_id": current.id, "source_sha256": first["source_sha256"], "start_line": 2, "end_line": 5}
        self.assertTrue(session.validate_anchor(anchor))
        self.assertFalse(session.validate_anchor({**anchor, "end_line": 10}))
        self.assertEqual(session.dispatch("read_candidate_code", {"start_line": -1})["status"], "error")
        self.assertEqual(session.dispatch("read_candidate_code", {"symbol": "loss", "start_line": 1})["status"], "error")
        self.assertEqual(session.dispatch("read_candidate_code", {"symbol": "missing"})["status"], "error")

    def test_long_lines_explicit_column_and_no_false_full_line_anchor(self):
        current = node(code="X = '" + "a" * 30000 + "'\n")
        session = CodeReadingSession([current], current.id)
        first = session.dispatch("read_candidate_code", {})
        self.assertEqual(first["continuation"]["start_line"], 1)
        self.assertGreater(first["continuation"]["start_column"], 0)
        self.assertFalse(session.validate_anchor({"node_id": current.id, "source_sha256": first["source_sha256"], "start_line": 1, "end_line": 1}))
        second = session.dispatch("read_candidate_code", first["continuation"])
        self.assertIn("column", second["content"])

    def test_diff_actual_change_and_budgets(self):
        parent = node("parent", code="def loss(x):\n    return x\n")
        current = node(parent=parent)
        session = CodeReadingSession([current, parent], current.id, options=CodeToolOptions(max_calls=2))
        result = session.dispatch("diff_candidate_code", {})
        self.assertIn("-    return x", result["content"])
        self.assertIn("+    return x * 2", result["content"])
        self.assertEqual(result["changed_symbols"][0]["symbol"], "loss")
        self.assertNotEqual(result["base"]["source_sha256"], result["target"]["source_sha256"])
        small = session.dispatch("read_candidate_code", {}, max_chars=100)
        self.assertLessEqual(len(json.dumps(small, ensure_ascii=False, separators=(",", ":"))), 100)
        self.assertEqual(session.dispatch("read_candidate_code", {})["status"], "budget_exhausted")
        self.assertEqual(session.ledger[0]["response"], result)

    def test_strict_schema_and_snapshot_provenance(self):
        current = node()
        session = CodeReadingSession([current], current.id)
        for tool in session.tools():
            schema = tool["function"]["parameters"]
            self.assertTrue(tool["function"]["strict"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["properties"]), set(schema["required"]))
        session.index_summary(register_anchors=False)
        self.assertEqual(session.anchors, [])
        packet_from_search(agent(current), current, code_session=session, runtime_facts={"available": False})
        self.assertTrue(session.anchors)
        self.assertFalse(session.validate_anchor({"node_id": "current", "source_sha256": "fake", "start_line": 1, "end_line": 1}))
        self.assertEqual(context_options(NS()).version, 1)

    def test_visible_runtime_paths_exclude_omitted_blocks(self):
        current = node()
        facts = {"available": True, "selected_snapshot": {"snapshot_id": "visible"},
                 "unreturned_detail": {"secret_evidence_marker": "z" * 10000},
                 "validation_trajectory": [{"optimizer_steps": i, "padding": "x" * 300} for i in range(30)]}
        packet = packet_from_search(agent(current, runtime_chars=2000), current, runtime_facts=facts)
        visible = packet.data["runtime_context"]
        self.assertNotIn("unreturned_detail", visible)
        self.assertIn("selected_snapshot", visible)
        self.assertIn("runtime_context.resources", packet.text)
        self.assertIn("resources", visible)
        self.assertGreater(visible["validation_trajectory"][0]["optimizer_steps"], 0)
        self.assertIn(current.id, packet.text)

    def test_long_diff_line_paginates_and_parent_alias_resolves(self):
        parent = node("ancestor", code="X = '" + "a" * 20000 + "'\n")
        current = node("descendant", parent=parent, code="X = '" + "b" * 20000 + "'\n")
        session = CodeReadingSession([current, parent], current.id)
        self.assertEqual(session.dispatch("read_candidate_code", {"node_id": "current"})["node_id"], current.id)
        parent_read = session.dispatch("read_candidate_code", {"node_id": "parent"})
        self.assertEqual(parent_read["continuation"]["node_id"], parent.id)
        self.assertEqual(session.dispatch("read_candidate_code", parent_read["continuation"])["node_id"], parent.id)
        self.assertEqual(session.dispatch("candidate_code_index", {"node_id": "parent"})["node_id"], parent.id)
        first = session.dispatch("diff_candidate_code", {})
        self.assertGreater(first["continuation"]["offset_column"], 0)
        second = session.dispatch("diff_candidate_code", first["continuation"])
        self.assertGreater(second["partial_diff_lines"][0]["start_column"], 0)
        self.assertLessEqual(session.chars_used, session.options.max_total_chars)

    def test_resources_deadline_and_no_promised_gpu(self):
        cfg_agent = agent(node())
        cfg_agent.executor = NS(run_deadline=float("inf"), current_parallel_run=1, _slot_waiters=[1, 2],
                                gpu_devices=["0"], cpu_number=4, max_parallel_run=1)
        cfg_agent.runtime_deadline = 1e12
        data = resource_context(cfg_agent, "draft")
        self.assertEqual(data["run_deadline_unix"], 1e12)
        self.assertEqual(data["queued_candidates"], 2)
        self.assertIn("not assigned", data["future_candidate_gpu"])


def historical_audit(root):
    """Rebuild the 20 S54–56 improve packets without future-node lookahead.

    Fetched results omit model weights and most candidate workspaces. Registered
    source is reconstructed into a temporary directory ONLY when journal source
    matches the original candidate.json hash. Runtime observations come from the
    fetched small JSON files and are labeled as replay metadata.
    """
    runs = [p for p in Path(root).glob("20260911_07*_jubias-anaf-s5[456]") if (p / "logs/journal.json").exists()]
    rows = []
    for run in sorted(runs):
        logs = run / "logs"
        journal = json.loads((logs / "journal.json").read_text())
        nodes = {v["id"]: NS(**v) for v in journal["nodes"]}
        relations = journal.get("node2parent", {})
        for n in nodes.values():
            parent_id = relations.get(n.id) or (n.parent if isinstance(n.parent, str) else None)
            n.parent = nodes.get(parent_id)
            n.children = [child for child in nodes.values() if relations.get(child.id) == n.id]
            n.metric = NS(**n.metric) if isinstance(n.metric, dict) else n.metric
        invocations = [json.loads(line) for line in (logs / "analogy/index.jsonl").read_text().splitlines() if line.strip()]
        with tempfile.TemporaryDirectory(prefix="analogy-context-replay-") as tmp:
            workspace = Path(tmp) / "workspace"
            for n in nodes.values():
                candidate_path = logs / "candidate_results" / n.id / "candidate.json"
                if not candidate_path.exists():
                    continue
                registered = json.loads(candidate_path.read_text())
                if hashlib.sha256(n.code.encode()).hexdigest() != registered["source_sha256"]:
                    raise AssertionError(f"historical source differs from registry: {run.name}/{n.id}")
                directory = workspace / "candidate_results/candidates" / n.id
                directory.mkdir(parents=True)
                (directory / "solution.py").write_text(n.code)
                (directory / "candidate.json").write_text(json.dumps(registered))
            for invocation in invocations:
                if invocation.get("stage") != "improve":
                    continue
                current = nodes[invocation["parent_id"]]
                prior = [n for n in nodes.values() if n.branch_id == current.branch_id and n.ctime <= current.ctime]
                cfg_agent = agent(current, prior)
                cfg_agent.cfg.workspace_dir = str(workspace)
                cfg_agent.cfg.log_dir = str(logs)
                cfg_agent.cfg.candidate_runtime.enabled = True
                # Description text is taken from the actual trace, not invented.
                trace = (logs / "analogy" / invocation["trace"]).read_text()
                cfg_agent.task_desc = trace.split("## Task (competition description; head and tail)", 1)[-1].split("## Available data", 1)[0]
                facts = {"available": True, "provenance": "historical fetched runtime metadata; no new scoring or inference"}
                metadata_dir = logs / "candidate_results" / current.id
                for source, key in (("candidate.json", "candidate"), ("execution.json", "execution")):
                    p = metadata_dir / source
                    if p.exists():
                        facts[key] = json.loads(p.read_text())
                snapshots = list((metadata_dir / "snapshots").glob("*.json"))
                selected = [json.loads(p.read_text()) for p in snapshots if p.stem == current.best_snapshot_id]
                if selected:
                    facts["selected_snapshot"] = selected[0]
                packet = packet_from_search(cfg_agent, current, runtime_facts=facts)
                source = packet.data["implementation_context"]
                assert source.get("origin") == "registered_solution", (run.name, current.id, source)
                assert len(packet.text) <= 80000
                assert "Planned changes" in packet.text or "Design intent" in packet.text
                if selected:
                    assert selected[0]["snapshot_id"] in packet.text
                    assert str(selected[0]["optimizer_steps"]) in packet.text
                rows.append({"run": run.name, "node_id": current.id, "packet_chars": len(packet.text),
                             "source_sha256": source["source_sha256"], "snapshot_visible": bool(selected),
                             "plan_chars": packet.metadata["sections"]["plan"]["returned_chars"]})
    if not rows:
        raise AssertionError("No historical S54–56 improve traces found")
    print(json.dumps({"historical_packets": len(rows), "snapshot_visible": sum(r["snapshot_visible"] for r in rows),
                      "max_packet_chars": max(r["packet_chars"] for r in rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContextTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.historical_root:
        historical_audit(args.historical_root)
