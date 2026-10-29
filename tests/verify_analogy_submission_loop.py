"""Offline P0 submission-loop/artifact regressions.

Run: python tests/verify_analogy_submission_loop.py
Reuses observation fixtures and mocks model transport; no API, cluster, training,
or private-label access. Artifact writes are limited to temporary directories.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import verify_analogy_observation as fixtures
from engine.analogy import agent, observed_loop, report_v2


def abstention():
    return {"report_schema_revision": 2, "bottlenecks": [], "observed_facts": [],
            "hypotheses": [], "unknowns": [], "mechanisms": [],
            "abstention_reason": "No supported intervention fits the remaining budget."}


class SubmissionLoopTests(unittest.TestCase):
    # Reuse only fixture helpers, not ObservationTests' test_* methods.
    setUp = fixtures.ObservationTests.setUp
    execute = fixtures.ObservationTests.execute
    evidence_turns = fixtures.ObservationTests.evidence_turns

    def outputs(self, request_index):
        return {item["call_id"]: json.loads(item["output"])
                for item in self.requests[request_index]["input_items"]
                if item.get("type") == "function_call_output"}

    def test_initial_submit_at_12_then_two_repairs_stays_within_14(self):
        invalid = fixtures.report(observed_facts=[{
            "statement": "A public metric is present.", "source": "runtime",
            "evidence": "public_validation.missing"}])
        turns = self.evidence_turns()
        turns.extend(fixtures.response(fixtures.tool("read_abstract", {"ids": ["paper1"]}, f"read{turn}"))
                     for turn in range(3, 12))
        turns.extend(fixtures.response(fixtures.tool("submit_report", value, f"submit{turn}"))
                     for turn, value in [(12, invalid), (13, invalid), (14, fixtures.report())])
        result = self.execute(turns, max_turns=14)
        self.assertEqual(len(self.requests), 14)
        self.assertEqual(result.turns, 14)
        self.assertEqual(result.delivery_status, "accepted_complete", result.reason)
        self.assertEqual(result.context["first_submit_deadline_turn"], 12)
        self.assertEqual([a["turn"] for a in result.submission_attempts], [12, 13, 14])
        self.assertEqual([a["correcting"] for a in result.submission_attempts], [False, True, True])
        forced = {"type": "function", "name": "submit_report"}
        self.assertTrue(all(r["tool_choice"] == "auto" for r in self.requests[:11]))
        self.assertEqual(self.requests[11]["tool_choice"], forced)
        self.assertEqual(self.requests[12]["tool_choice"], "auto")
        self.assertEqual(self.requests[13]["tool_choice"], forced)
        self.assertEqual(self.outputs(12)["submit12"]["remaining_turns"], 2)
        self.assertEqual(self.outputs(13)["submit13"]["remaining_turns"], 1)
        self.assertNotIn("shorten", self.outputs(12)["submit12"]["note"].lower())

    def test_correction_allows_one_known_read_but_blocks_search_and_second_read(self):
        invalid = fixtures.report(mechanisms=[fixtures.mechanism(evidence_refs=[{
            "paper_id": "paper1", "source": "abstract", "quote": "A quote never returned by this tool."}])])
        turns = self.evidence_turns() + [
            fixtures.response(fixtures.tool("submit_report", invalid, "bad")),
            fixtures.response(
                fixtures.tool("search_papers", {"query": "unrelated expanded search"}, "search-denied"),
                fixtures.tool("open_paper", {"paper_id": "paper1"}, "open-denied"),
                fixtures.tool("read_abstract", {"ids": ["paper1"]}, "repair-read"),
                fixtures.tool("read_candidate_code", {"symbol": "loss"}, "second-denied")),
            fixtures.response(fixtures.tool("submit_report", fixtures.report(), "fixed"))]
        result = self.execute(turns, max_turns=14)
        self.assertEqual(result.delivery_status, "accepted_complete", result.reason)
        self.assertEqual(result.turns, 5)
        outputs = self.outputs(4)
        for call_id in ("search-denied", "open-denied", "second-denied"):
            self.assertEqual(outputs[call_id]["status"], "correction_only")
        self.assertEqual(outputs["repair-read"]["papers"][0]["id"], "paper1")
        self.assertEqual(result.queries, ["pairwise ranking"])
        self.assertEqual(len(self.session.ledger), 2)
        self.assertEqual(result.fulltext["documents"], {})
        self.assertEqual(self.requests[4]["tool_choice"], {"type": "function", "name": "submit_report"})

    def test_malformed_submit_json_is_preserved_then_repaired(self):
        malformed = fixtures.tool("submit_report", {}, "malformed")
        malformed["arguments"] = '{"mechanisms": ['
        result = self.execute(self.evidence_turns() + [fixtures.response(malformed),
            fixtures.response(fixtures.tool("submit_report", fixtures.report(), "fixed"))], max_turns=14)
        self.assertEqual(result.delivery_status, "accepted_complete", result.reason)
        first = result.submission_attempts[0]
        self.assertEqual(first["raw_arguments"], malformed["arguments"])
        self.assertIsNone(first["raw_report"])
        self.assertEqual(first["status"], "rejected")
        self.assertEqual(first["issues"][0]["code"], "invalid_arguments")
        self.assertEqual(first["turn"], 3)
        self.assertTrue(result.submission_attempts[1]["correcting"])
        self.assertEqual(self.outputs(3)["malformed"]["status"], "rejected")

    def test_targeted_read_and_fixed_submit_can_share_one_response(self):
        invalid = fixtures.report(mechanisms=[fixtures.mechanism(evidence_refs=[{
            "paper_id": "paper1", "source": "abstract", "quote": "A quote never returned by this tool."}])])
        result = self.execute(self.evidence_turns() + [
            fixtures.response(fixtures.tool("submit_report", invalid, "bad")),
            fixtures.response(fixtures.tool("read_abstract", {"ids": ["paper1"]}, "repair-read"),
                              fixtures.tool("submit_report", fixtures.report(), "fixed"))], max_turns=14)
        self.assertEqual(result.delivery_status, "accepted_complete", result.reason)
        self.assertEqual(result.turns, 4)
        self.assertEqual([a["turn"] for a in result.submission_attempts], [3, 4])
        self.assertTrue(result.submission_attempts[-1]["correcting"])

    def test_feedback_has_total_utf8_budget_without_mutating_full_issues(self):
        issues = [{"code": "runtime_path_unavailable", "location": f"observed_facts[{i}].runtime_evidence[0]",
                   "received": {"large_value": "😀" * 3000}, "expected": "可见路径" * 1000,
                   "repair_hint": "请修复路径，保留全部证据。" * 1000} for i in range(30)]
        attempt = {"status": "rejected", "issues": issues}
        original = json.dumps(attempt, ensure_ascii=False)
        encoded = observed_loop._feedback(attempt, 2)
        self.assertLessEqual(len(encoded.encode("utf-8")), 4096)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["status"], "rejected")
        self.assertEqual(decoded["remaining_turns"], 2)
        self.assertEqual(len(decoded["issues"]) + decoded["issues_not_shown"], 30)
        self.assertEqual(decoded["issues"][0]["location"], "observed_facts[0].runtime_evidence[0]")
        self.assertNotIn("shorten", decoded["note"].lower())
        self.assertEqual(json.dumps(attempt, ensure_ascii=False), original)

    def test_final_validation_failure_is_distinct_from_abstention(self):
        invalid = fixtures.report(observed_facts=[{
            "statement": "A public metric is present.", "source": "runtime", "evidence": "missing"}])
        failed = self.execute(self.evidence_turns() + [
            fixtures.response(fixtures.tool("submit_report", invalid, "invalid"))], max_turns=3)
        self.assertEqual(failed.delivery_status, "failed")
        self.assertEqual(failed.failure_kind, "validation")
        self.assertFalse(failed.report_md)
        self.assertIsNone(failed.report)
        declined = self.execute([fixtures.response(fixtures.tool("submit_report", abstention(), "decline"))],
                                max_turns=1)
        self.assertEqual(declined.delivery_status, "abstained", declined.reason)
        self.assertFalse(declined.failure_kind)
        self.assertFalse(declined.report_md)
        self.assertEqual(declined.reason, abstention()["abstention_reason"])
        self.assertEqual(declined.submission_attempts[0]["status"], "abstained")

    def test_transport_failure_before_any_submission_is_not_abstention(self):
        def broken_transport():
            raise RuntimeError("synthetic transport failure")
        result = self.execute([broken_transport], max_turns=14)
        self.assertEqual(result.delivery_status, "failed")
        self.assertEqual(result.failure_kind, "transport")
        self.assertEqual(result.submission_attempts, [])
        self.assertEqual(result.turns, 1)
        self.client.close.assert_called_once()

    def test_partial_acceptance_keeps_original_mapping_through_budget_trim(self):
        broken_anchor = {**fixtures.ANCHOR, "source_sha256": "not-the-visible-source"}
        submitted = fixtures.report(mechanisms=[
            fixtures.mechanism(title="First valid mechanism"),
            fixtures.mechanism(title="Unsupported middle mechanism", code_refs=[broken_anchor]),
            fixtures.mechanism(title="Last valid mechanism")])
        # Measure the complete first mechanism after real validation; choose a
        # budget that fits it but cannot fit the other independent valid block.
        self.session.dispatch("read_candidate_code", {"symbol": "loss"})
        clean, problems = report_v2.validate(fixtures.report(mechanisms=submitted["mechanisms"][:1]),
            {"paper1"}, self.corpus, 3, reading=fixtures.PaperReadingSession(self.corpus, self.fulltext),
            abstracts={"paper1": fixtures.ABSTRACT}, code_session=self.session,
            runtime_context={}, mode="improve")
        self.assertFalse(problems)
        _, first_text = report_v2.render(clean, self.corpus, 100000)
        result = self.execute(self.evidence_turns() + [
            fixtures.response(fixtures.tool("submit_report", submitted, "partial"))],
            max_turns=14, report_char_budget=len(first_text) + 1)
        self.assertEqual(result.delivery_status, "accepted_partial", result.reason)
        self.assertEqual([m["title"] for m in result.report["mechanisms"]], ["First valid mechanism"])
        self.assertEqual(result.report["mechanisms"][0]["mechanism_id"], "m1")
        self.assertLessEqual(len(result.report_md), len(first_text) + 1)
        attempt = result.submission_attempts[0]
        self.assertEqual(attempt["raw_report"], submitted)
        self.assertEqual([(m["original_index"], m["mechanism_id"], m["retained"])
                          for m in attempt["mechanism_mapping"]], [(0, "m1", True), (2, "m2", False)])
        self.assertEqual([m["original_index"] for m in attempt["dropped_mechanisms"]], [1])
        self.assertEqual([m["original_index"] for m in attempt["budget_dropped_mechanisms"]], [2])
        self.assertIn("rendered_budget_exceeded", [i["code"] for i in attempt["issues"]])
        self.assertNotIn("Last valid mechanism", result.report_md)
        self.assertNotIn("Unsupported middle mechanism", result.report_md)

    def test_artifacts_keep_attempts_normalization_and_delivery_metadata(self):
        malformed = fixtures.tool("submit_report", {}, "malformed")
        malformed["arguments"] = "not json"
        runtime = {"public_validation": {"metric": 0.9, "components": [0.8]}}
        repaired = fixtures.report(observed_facts=[{
            "statement": "The metric and a component were recorded.", "source": "runtime",
            "evidence": "public_validation.metric; public_validation.components[0]"}])
        result = self.execute(self.evidence_turns() + [fixtures.response(malformed),
            fixtures.response(fixtures.tool("submit_report", repaired, "fixed"))],
            max_turns=14, runtime_context=runtime)
        self.assertEqual(result.delivery_status, "accepted_complete", result.reason)
        agent._write_artifacts(self.root, "node1", "visible packet", result, self.corpus, {"stage": "improve"})
        index = json.loads((self.root / "analogy/index.jsonl").read_text())
        stored = json.loads((self.root / "analogy" / index["context_trace"]).read_text())
        self.assertEqual(stored["submission_attempts"], result.submission_attempts)
        self.assertEqual(stored["report"], result.report)
        self.assertEqual(stored["delivery_status"], "accepted_complete")
        self.assertEqual(stored["failure_kind"], "")
        self.assertEqual(stored["submission_attempts"][0]["raw_arguments"], "not json")
        final = stored["submission_attempts"][-1]
        self.assertEqual(final["raw_report"], repaired)
        self.assertTrue(final["normalizations"])
        self.assertEqual(final["normalized_report"]["observed_facts"][0]["runtime_evidence"],
                         ["public_validation.metric", "public_validation.components.0"])
        self.assertTrue(index["ok"])
        self.assertEqual(index["first_submit_turn"], 3)
        self.assertEqual(index["submission_count"], 2)
        self.assertTrue(index["correction_succeeded"])
        self.assertEqual(index["report_schema_revision"], 2)
        self.assertEqual(index["final_mechanism_count"], 1)
        self.assertEqual(index["stage"], "improve")


if __name__ == "__main__":
    unittest.main()
