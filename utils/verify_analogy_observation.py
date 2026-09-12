"""Offline observation-loop tests: source evidence, tool replay and budgets.

Run: python utils/verify_analogy_observation.py
Uses real source/fulltext session code and mocked Responses calls. No network,
candidate execution, model inference or private validation labels are accessed.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.analogy import observed_loop, report_v2
from engine.analogy.agent import run_analogy_agent
from engine.analogy.code_tools import CodeReadingSession
from engine.analogy.context import ContextOptions, _runtime_text
from engine.analogy.fulltext import FullTextConfig, PaperReadingSession

SOURCE = 'LOSS = "bce"\n\ndef loss(logits, target):\n    return (logits - target) ** 2\n'
SHA = hashlib.sha256(SOURCE.encode()).hexdigest()
ANCHOR = {"node_id": "node1", "source_sha256": SHA, "start_line": 3, "end_line": 4}
ABSTRACT = "Pairwise comparisons model relative ordering rather than isolated pointwise classification."


class Corpus:
    digest = "synthetic-corpus"
    by_id = {"paper1": {"id": "paper1", "venue": "synthetic", "title": "Pairwise ranking",
                         "abstract": ABSTRACT, "tldr": "Compare positive and negative pairs."}}

    def __len__(self):
        return 1

    def __contains__(self, pid):
        return pid in self.by_id

    def search(self, query, k=10):
        return [{k: v for k, v in self.by_id["paper1"].items() if k != "abstract"}]

    def get(self, ids):
        return [copy.deepcopy(self.by_id[pid]) for pid in ids if pid in self.by_id]


def mechanism(**overrides):
    return {"title": "Pairwise objective", "paper_ids": ["paper1"], "bottleneck_idx": 0,
            "object_mappings": [{"source": "ordered items", "target": "toxic/clean examples", "rationale": "order metric"}],
            "shared_relations": "Relative order determines AUC.", "mechanism": "Compare positive and negative scores.",
            "intervention": "Test a small pairwise objective alongside the current objective.",
            "feasibility": "Reuse one batch and retain export callbacks.", "implementation_basis": "code",
            "code_refs": [copy.deepcopy(ANCHOR)], "runtime_evidence": [],
            "assumptions": "Labels and logits have aligned example order.",
            "target_fit": "The target metric depends on relative ordering.",
            "constraints": "Preserve the existing train/validation split and export callback.",
            "validation_plan": "Compare the fixed public metric at equal optimizer steps.",
            "rejection_criterion": "Reject if fixed-split public AUC decreases.",
            "limitations": "No private-score claim; limited pair support can add variance.",
            "evidence_refs": [{"paper_id": "paper1", "source": "abstract", "quote": ABSTRACT}], **overrides}


def report(**overrides):
    return {"bottlenecks": [{"statement": "Current loss uses individual residuals.", "objects": ["scores", "labels"],
                             "relations": ["order"], "evidence": "loss source"}],
            "observed_facts": [{"statement": "loss computes squared residuals.", "source": "code",
                                "evidence": "loss, lines 3-4", "code_refs": [copy.deepcopy(ANCHOR)]}],
            "hypotheses": ["Pairwise supervision may align with AUC."], "unknowns": ["Impact on fixed-split score."],
            "mechanisms": [mechanism()], **overrides}


def tool(name, args, call_id):
    if name == "read_candidate_code":
        args = {"node_id": None, "symbol": None, "start_line": None, "end_line": None,
                "start_column": None, "max_lines": None, **args}
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id,
            "name": name, "arguments": json.dumps(args), "status": "completed"}


def response(*calls, opaque=False):
    output = list(calls)
    if opaque:
        output.insert(0, {"type": "reasoning", "id": "rs_" + calls[0]["call_id"],
                          "encrypted_content": "synthetic-opaque-state", "summary": []})
    return {"id": "resp_synthetic", "model": "gpt-6-astra", "status": "completed", "output": output,
            "usage": {"input_tokens": 50, "output_tokens": 25}, "reasoning": {"effort": "high"}}


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.node = NS(id="node1", code=SOURCE, stage="improve", parent=None,
                       execution_status="completed", exec_time=1)
        self.session = CodeReadingSession([self.node], "node1", run_id="synthetic-run")
        self.corpus = Corpus()
        self.cfg = NS(model="gpt-6-astra", base_url="https://proxy.invalid/v1", api_key="synthetic-key",
                      reasoning_effort="high")
        self.fulltext = FullTextConfig(enabled=True, offline=True, cache_dir=str(self.root/"cache"))

    def execute(self, responses, *, session=True, options=None, **kwargs):
        self.requests = []
        client = Mock()
        def request(_client, **params):
            self.requests.append(copy.deepcopy(params))
            index = len(self.requests) - 1
            if index >= len(responses):
                raise AssertionError("Unexpected additional model request")
            return responses[index]() if callable(responses[index]) else copy.deepcopy(responses[index])
        with patch("llm.responses.make_response_client", return_value=client), \
             patch("llm.responses.request_response", side_effect=request):
            value = run_analogy_agent("Synthetic packet: available code and public runtime facts.", self.corpus, self.cfg,
                max_turns=kwargs.pop("max_turns", 6), context_options=options or ContextOptions(version=2),
                code_session=self.session if session else None, fulltext=kwargs.pop("fulltext", self.fulltext),
                report_char_budget=kwargs.pop("report_char_budget", 12000), **kwargs)
        self.client = client
        return value

    def evidence_turns(self):
        return [response(tool("candidate_code_index", {"node_id": None}, "index"),
                         tool("search_papers", {"query": "pairwise ranking"}, "search"), opaque=True),
                response(tool("read_candidate_code", {"node_id": None, "symbol": "loss"}, "read"),
                         tool("read_abstract", {"ids": ["paper1"]}, "abstract"), opaque=True)]

    def test_three_round_source_paper_submit_replays_all_calls(self):
        result = self.execute(self.evidence_turns() + [response(tool("submit_report", report(), "submit"))])
        self.assertEqual(result.turns, 3)
        self.assertTrue(result.report_md, result.reason)
        self.assertEqual(result.report["mechanisms"][0]["mechanism_id"], "m1")
        self.assertIn("lines", result.report["observed_facts"][0]["evidence"])
        self.assertEqual(result.report["mechanisms"][0]["evidence_level"], "abstract_only")
        history = self.requests[2]["input_items"]
        self.assertEqual(sum(x.get("type") == "reasoning" for x in history), 2)
        outputs = [x for x in history if x.get("type") == "function_call_output"]
        self.assertEqual([x["call_id"] for x in outputs], ["index", "search", "read", "abstract"])
        source_output = json.loads(next(x["output"] for x in outputs if x["call_id"] == "read"))
        self.assertEqual(source_output["source_sha256"], SHA)
        self.assertIn("return (logits - target) ** 2", source_output["content"])
        self.assertTrue(self.session.validate_anchor(ANCHOR))
        self.assertEqual(result.in_tokens, 150)  # billed total separate from per-request context cap
        self.client.close.assert_called_once()

    def test_unread_method_body_is_rejected_then_valid_report_resubmits(self):
        turns = [response(tool("candidate_code_index", {"node_id": None}, "index"),
                          tool("search_papers", {"query": "pairwise"}, "search"),
                          tool("read_abstract", {"ids": ["paper1"]}, "abstract")),
                 response(tool("submit_report", report(), "bad-submit")),
                 response(tool("read_candidate_code", {"symbol": "loss"}, "read")),
                 response(tool("submit_report", report(), "submit"))]
        result = self.execute(turns)
        self.assertEqual(result.turns, 4)
        self.assertTrue(result.report_md)
        rejected = [x for x in self.requests[2]["input_items"] if x.get("call_id") == "bad-submit"
                    and x.get("type") == "function_call_output"]
        self.assertEqual(json.loads(rejected[0]["output"])["status"], "rejected")

    def test_empty_report_arrays_are_valid(self):
        empty = {"bottlenecks": [], "observed_facts": [], "hypotheses": [], "unknowns": [], "mechanisms": []}
        result = self.execute([response(tool("submit_report", empty, "empty"))], max_turns=1)
        self.assertIsNotNone(result.report)
        self.assertFalse(result.report_md)
        self.assertIn("no structurally matching", result.reason)

    def test_report_budget_drops_entire_mechanisms_and_preserves_fields(self):
        self.session.dispatch("read_candidate_code", {"symbol": "loss"})
        long_condition = "retain condition " * 80
        raw = report(mechanisms=[mechanism(title="First", rejection_criterion=long_condition),
                                 mechanism(title="Second", rejection_criterion=long_condition)])
        reading = PaperReadingSession(self.corpus, self.fulltext)
        clean, problems = report_v2.validate(raw, {"paper1"}, self.corpus, 3, reading=reading,
            abstracts={"paper1": ABSTRACT}, code_session=self.session, runtime_context={}, mode="improve")
        self.assertFalse(problems)
        one = copy.deepcopy(clean)
        one["mechanisms"] = one["mechanisms"][:1]
        _, first_text = report_v2.render(one, self.corpus, 100000)
        retained, rendered = report_v2.render(clean, self.corpus, len(first_text) + 1)
        self.assertEqual(len(retained["mechanisms"]), 1)
        self.assertLessEqual(len(rendered), len(first_text) + 1)
        self.assertIn(long_condition, rendered)
        self.assertEqual(retained["mechanisms"][0]["rejection_criterion"], long_condition)
        self.assertNotIn("### Second", rendered)
        removed, text = report_v2.render(clean, self.corpus, 20)
        self.assertEqual(removed["mechanisms"], [])
        self.assertEqual(text, "")

    def test_malformed_submission_gets_tool_error_and_can_resubmit(self):
        invalid = report(mechanisms=[mechanism(bottleneck_idx="not an integer")])
        result = self.execute(self.evidence_turns() + [response(tool("submit_report", invalid, "bad-submit")),
                                                      response(tool("submit_report", report(), "submit"))])
        self.assertTrue(result.report_md, result.reason)
        self.assertEqual(result.turns, 4)
        self.assertIn("bad-submit", json.dumps(self.requests[3]["input_items"]))

    def test_source_freezes_and_runtime_unavailable_does_not_fall_back(self):
        self.node.code = "raise RuntimeError('must not execute or replace frozen source')"
        reply = self.session.dispatch("read_candidate_code", {"symbol": "loss"})
        self.assertEqual(reply["source_sha256"], SHA)
        self.assertIn("logits - target", reply["content"])
        unavailable = CodeReadingSession([self.node], "node1", workspace=self.root, runtime_enabled=True)
        reply = unavailable.dispatch("read_candidate_code", {"node_id": "node1"})
        self.assertEqual(reply["status"], "error")
        self.assertIn("source_unavailable", reply["reason"])
        self.assertNotIn("content", reply)
        self.assertFalse(unavailable.validate_anchor(ANCHOR))
        forbidden = self.session.dispatch("read_candidate_code", {"node_id": "../secrets"})
        self.assertIn("node_not_allowed", forbidden["reason"])

    def test_disabled_code_tools_are_not_exposed_or_executed(self):
        self.session.options.enabled = False
        empty = {"bottlenecks": [], "observed_facts": [], "hypotheses": [], "unknowns": [], "mechanisms": []}
        result = self.execute([response(tool("read_candidate_code", {"symbol": "loss"}, "denied")),
                               response(tool("submit_report", empty, "submit"))])
        names = {item["name"] for item in self.requests[0]["tools"]}
        self.assertFalse(names & {"candidate_code_index", "read_candidate_code", "diff_candidate_code"})
        self.assertEqual(self.session.calls_used, 0)
        self.assertEqual(self.session.ledger, [])
        denied = [item for item in self.requests[1]["input_items"]
                  if item.get("type") == "function_call_output" and item["call_id"] == "denied"]
        self.assertEqual(json.loads(denied[0]["output"])["status"], "unknown_tool")
        self.assertIsNotNone(result.report)

    def test_draft_wrapper_passes_visible_resource_evidence_into_real_loop(self):
        from engine.analogy import agent as retrieval
        acfg = NS(enabled=True, draft=True, corpus_path="unused", context=ContextOptions(version=2),
                  max_turns=1, top_k=10, max_mechanisms=3, report_char_budget=12000,
                  max_output_tokens=16384, fulltext=FullTextConfig(enabled=False))
        cfg = NS(analogy=acfg, agent=NS(code=self.cfg, time_limit=3600), cpu_number=2,
                 exec=NS(timeout=600), candidate_runtime=NS(enabled=False), log_dir=self.root)
        agent = NS(cfg=cfg, task_desc="Predict target; use a fixed validation split.",
                   data_preview="Two public columns.", use_coldstart=False)
        empty = {"bottlenecks": [], "hypotheses": [], "unknowns": [], "mechanisms": [],
                 "observed_facts": [{"statement": "CPU quota is 2.", "source": "runtime",
                                      "evidence": "resources.cpu_quota"}]}
        client = Mock()
        with patch.object(retrieval, "load_corpus", return_value=self.corpus), \
             patch("llm.responses.make_response_client", return_value=client), \
             patch("llm.responses.request_response", return_value=response(tool("submit_report", empty, "submit"))) as request, \
             patch.object(retrieval, "_write_artifacts") as write:
            result = retrieval.retrieve_for_draft(agent)
        self.assertEqual(result, "")  # Empty mechanisms are accepted, not rejected for missing runtime paths.
        request.assert_called_once()
        stored = write.call_args.args[3]
        self.assertIsNotNone(stored.report)
        self.assertEqual(stored.report["observed_facts"][0]["evidence"], "resources.cpu_quota")
        self.assertIn("no structurally matching", stored.reason)
        names = {item["name"] for item in request.call_args.kwargs["tools"]}
        self.assertNotIn("read_candidate_code", names)
        client.close.assert_called_once()

    def test_code_output_serialization_matches_budget_and_provenance(self):
        result = self.execute(self.evidence_turns() + [response(tool("submit_report", report(), "submit"))])
        outputs = {x["call_id"]: x["output"] for x in self.requests[2]["input_items"]
                   if x.get("type") == "function_call_output"}
        ledger = result.code_reads["ledger"]
        for entry, call_id in zip(ledger, ["index", "read"]):
            self.assertEqual(json.loads(outputs[call_id]), entry["response"])
            self.assertEqual(len(outputs[call_id]), self.session._size(entry["response"]))
        self.assertEqual(self.session.chars_used, sum(len(outputs[k]) for k in ["index", "read"]))

    def test_fulltext_budget_only_authorizes_returned_chunks(self):
        reading = PaperReadingSession(self.corpus, self.fulltext)
        chunks = [{"chunk_id": f"chunk{i}", "page": i + 1, "section": "Method", "text": str(i) * 1000,
                   "chars": 1000} for i in range(3)]
        reading.documents["paper1"] = {"chunks": chunks, "pdf_sha256": "pdf", "text_sha256": "text"}
        payload = observed_loop._paper_call(reading, "read_paper",
            {"paper_id": "paper1", "chunk_ids": [c["chunk_id"] for c in chunks]}, {"paper1"}, 3300)
        self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False)), 3300)
        kept = payload["chunks"]
        self.assertLess(len(kept), 3)
        self.assertTrue(kept)
        self.assertEqual(set(reading.delivered), {("paper1", c["chunk_id"]) for c in kept})
        self.assertEqual(reading.chars, sum(c["chars"] for c in kept))
        self.assertEqual(reading.events[-1]["result"], payload)

    def test_near_context_limit_forces_final_submit_and_ignores_new_reads(self):
        empty = {"bottlenecks": [], "observed_facts": [], "hypotheses": [], "unknowns": [], "mechanisms": []}
        # Estimate is held below limit but remaining read allowance is exhausted.
        with patch.object(observed_loop.ContextBudget, "estimate", return_value=10000), \
             patch.object(observed_loop.ContextBudget, "tool_chars", return_value=0):
            result = self.execute([response(tool("read_candidate_code", {"symbol": "loss"}, "blocked"),
                                            tool("submit_report", empty, "submit"))])
        self.assertEqual(self.requests[0]["tool_choice"], {"type": "function", "name": "submit_report"})
        self.assertEqual(self.session.calls_used, 0)
        self.assertTrue(result.context["turn_budgets"][0]["finish_only"])
        self.assertIn("context_budget_exhausted", "\n".join(result.trace))

    def test_context_limits_conversation_not_cumulative_billing(self):
        options = ContextOptions(version=2, max_input_tokens=1000, endpoint_context_tokens=10000,
                                 input_safety_tokens=100, final_report_reserve_tokens=100)
        budget = observed_loop.ContextBudget(options, 1000)
        budget.encoding = None
        messages, tools = [{"role": "user", "content": "test"}], []
        budget.observe(messages, tools, 100)
        first = budget.estimate(messages, tools)
        budget.observe(messages, tools, 100)
        self.assertEqual(budget.estimate(messages, tools), first)
        messages.append({"type": "function_call_output", "call_id": "c", "output": "x" * 200})
        self.assertGreater(budget.estimate(messages, tools), first)

    def test_observed_usage_calibrates_only_unchanged_prefix_with_conservative_suffix(self):
        budget = observed_loop.ContextBudget(ContextOptions(version=2), 16384)
        budget.encoding = None
        messages = [{"role": "user", "content": "paper and code " * 1000},
                    {"type": "reasoning", "encrypted_content": "opaque" * 35000, "summary": []}]
        tools = [{"type": "function", "name": "submit_report"}]
        before_observation = budget.estimate(messages, tools)
        self.assertGreater(before_observation, 200000)
        budget.observe(messages, tools, 50000)
        calibrated = budget.estimate(messages, tools)
        self.assertLess(calibrated, 62000)
        self.assertGreater(calibrated, 50000)  # API measurement still has a safety margin.
        suffix = {"type": "function_call_output", "call_id": "c", "output": "结果" * 4000}
        messages.append(suffix)
        suffix_bytes = len(json.dumps(suffix, ensure_ascii=False).encode("utf-8"))
        self.assertEqual(budget.estimate(messages, tools), calibrated + suffix_bytes + 2 + 64)
        # Missing usage must neither treat the unmeasured suffix as known nor discard
        # the older valid measurement. Changed prefixes/tools invalidate its use.
        estimate = budget.estimate(messages, tools)
        budget.observe(messages, tools, None)
        self.assertEqual(budget.estimate(messages, tools), estimate)
        original = messages[0]["content"]
        messages[0]["content"] = "changed prompt " + original
        self.assertGreater(budget.estimate(messages, tools), before_observation)
        messages[0]["content"] = original
        changed_tools = tools + [{"type": "function", "name": "read_candidate_code"}]
        self.assertGreater(budget.estimate(messages, changed_tools), before_observation)
        without_usage = observed_loop.ContextBudget(ContextOptions(version=2), 16384)
        without_usage.encoding = None
        without_usage.observe(messages, tools, 0)
        self.assertGreater(without_usage.estimate(messages, tools), 200000)

    def test_large_opaque_history_and_rejected_report_can_resubmit_with_actual_usage(self):
        oversized = report(unknowns=["Missing public validation details: " + "x" * 20000])
        turns = self.evidence_turns() + [response(tool("submit_report", oversized, "oversized"), opaque=True),
                                      response(tool("submit_report", report(), "compact"))]
        for index, turn in enumerate(turns):
            turn["usage"]["input_tokens"] = 50000
            for item in turn["output"]:
                if item["type"] == "reasoning":
                    item["encrypted_content"] = str(index) * 60000
        with patch.dict(sys.modules, {"tiktoken": None}):
            result = self.execute(turns)
        self.assertTrue(result.report_md, result.reason)
        self.assertEqual(result.turns, 4)
        self.assertLessEqual(len(result.report_md), 12000)
        self.assertEqual(result.context["input_limit_tokens"], 196608)
        last_request = self.requests[3]
        self.assertGreater(len(json.dumps(last_request["input_items"]).encode("utf-8")), 200000)
        self.assertLess(result.context["turn_budgets"][3]["estimated_input_tokens"], 196608)
        self.assertIn("12000 characters total", self.requests[0]["input_items"][0]["content"])
        preserved = [item for item in last_request["input_items"] if item.get("type") == "reasoning"]
        self.assertEqual([item["encrypted_content"] for item in preserved],
                         [str(index) * 60000 for index in range(3)])
        rejection = next(json.loads(item["output"]) for item in last_request["input_items"]
                         if item.get("type") == "function_call_output" and item["call_id"] == "oversized")
        self.assertEqual(rejection["status"], "rejected")
        self.assertEqual(rejection["rendered_char_budget"], 12000)
        self.assertGreater(rejection["rendered_chars_with_first_mechanism"], 12000)
        self.assertIn("shorten shared facts/hypotheses/unknowns", rejection["note"])

    def test_runtime_citations_use_only_visible_fields_and_visible_list_indices(self):
        raw_runtime = {"available": True, "validation_trajectory": [
            {"steps": i, "padding": "x" * 200} for i in range(8)], "large_details": "y" * 8000}
        text, visible, omitted = _runtime_text(raw_runtime, 1400)
        self.assertTrue(omitted)
        self.assertNotIn("large_details", visible)
        self.assertLess(len(visible["validation_trajectory"]), 8)
        self.assertGreater(visible["validation_trajectory"][0]["steps"], 0)
        self.assertIn("visible list indices start at 0", text)
        reading = PaperReadingSession(self.corpus, self.fulltext)
        path = "validation_trajectory.0.steps"
        runtime_report = report(observed_facts=[{"statement": "Public validation steps were recorded.",
            "source": "runtime", "evidence": path}], mechanisms=[mechanism(
                implementation_basis="runtime", code_refs=[], runtime_evidence=[path])])
        clean, problems = report_v2.validate(runtime_report, {"paper1"}, self.corpus, 3,
            reading=reading, abstracts={"paper1": ABSTRACT}, code_session=None,
            runtime_context=visible, mode="improve")
        self.assertFalse(problems)
        self.assertEqual(len(clean["mechanisms"]), 1)
        runtime_report["observed_facts"][0]["evidence"] = "large_details"
        clean, problems = report_v2.validate(runtime_report, {"paper1"}, self.corpus, 3,
            reading=reading, abstracts={"paper1": ABSTRACT}, code_session=None,
            runtime_context=visible, mode="improve")
        self.assertTrue(problems)
        self.assertFalse(clean["mechanisms"])

    def test_legacy_gpt5_v1_keeps_existing_loop(self):
        cfg = copy.copy(self.cfg)
        cfg.model = "gpt-5"
        fake = Mock()
        fake.chat.completions.create.side_effect = RuntimeError("synthetic stop")
        with patch("openai.OpenAI", return_value=fake), patch.object(observed_loop, "run") as new_loop:
            result = run_analogy_agent("old packet", self.corpus, cfg, max_turns=1,
                                      context_options=ContextOptions(version=1))
        new_loop.assert_not_called()
        fake.chat.completions.create.assert_called_once()
        self.assertIsNone(result.context)
        self.assertIn("synthetic stop", result.reason)


if __name__ == "__main__":
    unittest.main()
