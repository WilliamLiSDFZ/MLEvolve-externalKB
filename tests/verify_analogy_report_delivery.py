#!/usr/bin/env python3
"""CPU-only regression checks for the P0 report evidence/delivery contract.

No network, LLM, candidate execution, private data or persistent artifacts.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.analogy import report_v2
from engine.analogy.agent import TOOLS
from engine.analogy.code_tools import CodeReadingSession


ABSTRACT = "Pairwise comparisons model relative ordering rather than isolated pointwise classification."
FULLTEXT = "The adaptive weights are updated after each observed group loss and normalized over all groups."
RUNTIME = {"public_validation": {"metric": 0.91, "weakest_components": [{"auc": 0.8}]},
           "execution": {"status": "completed"}, "unknown_value": "unknown", "null_value": None}


class Corpus:
    by_id = {"paper1": {"id": "paper1", "title": "Ordering", "venue": "Synthetic"}}

    def __contains__(self, pid):
        return pid in self.by_id


def mechanism(**changes):
    return {"title": "Adaptive ordering", "paper_ids": ["paper1"], "bottleneck_idx": 0,
            "mechanism": "Update weights from observed losses.",
            "intervention": "Evaluate normalized adaptive weights.",
            "object_mappings": [{"source": "groups", "target": "metric cells"}],
            "shared_relations": "Relative ordering matters.", "feasibility": "Use current data.",
            "implementation_basis": "runtime", "code_refs": [],
            "runtime_evidence": ["public_validation.metric"],
            "assumptions": "Each cell has valid pairs.", "target_fit": "Target is based on ordering.",
            "constraints": "Keep the fixed public split.", "validation_plan": "Compare fixed-split scores.",
            "rejection_criterion": "Reject worse scores.", "limitations": "No claim on private scores.",
            "evidence_refs": [{"paper_id": "paper1", "source": "abstract", "quote": ABSTRACT}], **changes}


def report(**changes):
    return {"report_schema_revision": 2,
            "bottlenecks": [{"statement": "Metric depends on relative order.", "evidence": "public metric"}],
            "observed_facts": [{"statement": "A public metric is available.", "source": "runtime",
                "evidence": "Public validation reported its selected score.",
                "runtime_evidence": ["public_validation.metric"]}],
            "hypotheses": [], "unknowns": [], "mechanisms": [mechanism()], **changes}


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.reading = NS(delivered={("paper1", "chunk-1"): {
            "text": FULLTEXT, "page": 1, "section": "Method"}},
            documents={"paper1": {"pdf_sha256": "pdf", "text_sha256": "text"}})
        self.corpus = Corpus()

    def validate(self, raw, **kwargs):
        options = dict(reading=self.reading, abstracts={"paper1": ABSTRACT}, code_session=None,
                       runtime_context=copy.deepcopy(RUNTIME), mode="improve")
        options.update(kwargs)
        return report_v2.validate_detailed(raw, {"paper1"}, self.corpus, 3, **options)

    def test_new_runtime_array_and_revision_remain_independent_of_context(self):
        result = self.validate(report())
        self.assertFalse(result["issues"])
        self.assertEqual(result["report"]["context_version"], 2)
        self.assertEqual(result["report"]["report_schema_revision"], 2)
        self.assertEqual(result["mechanism_mapping"], [{"original_index": 0, "mechanism_id": "m1"}])
        schema = report_v2.extend_tools(TOOLS, mode="improve")[-1]["function"]["parameters"]
        self.assertNotIn("report_schema_revision", schema["required"])
        self.assertIn("runtime_evidence", schema["properties"]["observed_facts"]["items"]["properties"])

    def test_legacy_single_semicolon_and_numeric_brackets_are_audited(self):
        raw = report()
        raw.pop("report_schema_revision")
        fact = raw["observed_facts"][0]
        fact.pop("runtime_evidence")
        fact["evidence"] = " public_validation.metric ; public_validation.weakest_components[0].auc "
        original = copy.deepcopy(raw)
        result = self.validate(raw)
        self.assertFalse(result["issues"])
        self.assertEqual(raw, original)
        expected = ["public_validation.metric", "public_validation.weakest_components.0.auc"]
        self.assertEqual(result["report"]["observed_facts"][0]["runtime_evidence"], expected)
        self.assertEqual(result["normalizations"][0]["received"], fact["evidence"])
        self.assertEqual(result["normalizations"][0]["normalized"], expected)

    def test_legacy_mixed_good_bad_references_do_not_keep_only_first(self):
        raw = report()
        raw.pop("report_schema_revision")
        fact = raw["observed_facts"][0]
        fact.pop("runtime_evidence")
        fact["evidence"] = "public_validation.metric; public_validation.missing"
        result = self.validate(raw)
        self.assertFalse(result["shared_facts_valid"])
        self.assertFalse(result["report"]["mechanisms"])
        issue = result["issues"][0]
        self.assertEqual(issue["location"], "observed_facts[0].runtime_evidence[1]")
        self.assertEqual(issue["received"], "public_validation.missing")
        self.assertNotIn("shorten", issue["repair_hint"])

    def test_historical_s57_semicolon_fact_normalizes_every_visible_reference(self):
        # Minimal fact/path fixture from fetched S57, not a reinterpretation of its
        # whole report or a replay with later/private context.
        # 20260912_082449_jubias-anaf-gpt56sol-s57/logs/analogy/
        # d4f0a7b619fc467985adb10f1e0ce7e7_005.md
        # sha256: 4290c5e347b1ce428327ac85adc90dfa25119b665f1ac97a1e661ec5312e79f0
        paths = "selected_snapshot.optimizer_steps; execution.status; validation_trajectory.observed_count"
        visible = {"selected_snapshot": {"optimizer_steps": 1495},
                   "execution": {"status": "budget_exhausted"},
                   "validation_trajectory": {"observed_count": 1}}
        raw = report(observed_facts=[{"statement": "The recorded snapshot and execution state are available.",
            "source": "runtime", "evidence": paths}], mechanisms=[])
        raw.pop("report_schema_revision")
        result = self.validate(raw, runtime_context=visible)
        self.assertFalse(result["issues"])
        self.assertEqual(result["report"]["observed_facts"][0]["runtime_evidence"], paths.split("; "))
        hidden = copy.deepcopy(visible)
        hidden.pop("execution")
        rejected = self.validate(raw, runtime_context=hidden)
        self.assertFalse(rejected["shared_facts_valid"])
        self.assertEqual(rejected["issues"][0]["location"], "observed_facts[0].runtime_evidence[1]")

    def test_historical_s58_bracket_fact_resolves_the_original_visible_item(self):
        # 20260912_081346_jubias-anaf-gpt56sol-s58/logs/analogy/
        # c03c57a6fc0c44739acab52ac4ed5bf1_002.md
        # sha256: 1214a4af153f868491e75bc14ee037591288363a02660b601a382d31b5d7bae7
        component = {"component": "homosexual_gay_or_lesbian/bpsn", "auc": 0.7235202266676758,
                     "rows": 7497, "positives": 7067, "negatives": 430,
                     "defined": True, "support_sufficient": True}
        raw = report(observed_facts=[{"statement": "A weakest public component was displayed.",
            "source": "runtime", "evidence": "public_validation.weakest_components[0]"}], mechanisms=[])
        raw.pop("report_schema_revision")
        visible = {"public_validation": {"weakest_components": [component]}}
        result = self.validate(raw, runtime_context=visible)
        self.assertFalse(result["issues"])
        self.assertEqual(result["report"]["observed_facts"][0]["runtime_evidence"],
                         ["public_validation.weakest_components.0"])
        self.assertTrue(self.validate(raw, runtime_context={"public_validation": {}})["issues"])

    def test_revision_two_requires_array_instead_of_legacy_evidence_fallback(self):
        for paths in (None, "public_validation.metric"):
            with self.subTest(paths=paths):
                raw = report()
                fact = raw["observed_facts"][0]
                fact["evidence"] = "public_validation.metric"
                if paths is None:
                    fact.pop("runtime_evidence")
                else:
                    fact["runtime_evidence"] = paths
                result = self.validate(raw)
                self.assertTrue(result["issues"])
                self.assertFalse(result["shared_facts_valid"])

    def test_unknown_omitted_and_invalid_index_paths_fail(self):
        for path in ("unknown_value", "null_value", "private_hidden", "public_validation.weakest_components.1.auc",
                     "public_validation.weakest_components[-1].auc", "public_validation.metric;"):
            with self.subTest(path=path):
                raw = report()
                raw["observed_facts"][0]["runtime_evidence"] = [path]
                result = self.validate(raw)
                self.assertTrue(result["issues"])
                self.assertFalse(result["report"]["mechanisms"])

    def test_runtime_terminal_values_match_visible_catalog_rules(self):
        raw = report(observed_facts=[{"statement": "A value was displayed.", "source": "runtime",
            "evidence": "Visible scalar.", "runtime_evidence": ["value"]}], mechanisms=[],
            abstention_reason="No suitable intervention.")
        for value in (None, "unknown", " UNKNOWN ", "", "   ", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assertTrue(self.validate(raw, runtime_context={"value": value})["issues"])
        for value in (False, 0, 0.0, "0"):
            with self.subTest(value=value):
                self.assertFalse(self.validate(raw, runtime_context={"value": value})["issues"])

    def test_every_runtime_reference_checked_even_for_code_based_mechanism(self):
        session, anchor = self.source()
        raw = report(mechanisms=[mechanism(implementation_basis="code", code_refs=[anchor],
            runtime_evidence=["public_validation.metric", "unseen_field"])])
        result = self.validate(raw, code_session=session)
        self.assertTrue(result["shared_facts_valid"])
        self.assertFalse(result["report"]["mechanisms"])
        self.assertEqual(result["issues"][0]["location"], "mechanisms[0].runtime_evidence[1]")

    def source(self):
        node = NS(id="candidate1", code="def loss(x):\n    return x * 2\n", stage="improve",
                  execution_status="completed", exec_time=1, parent=None)
        session = CodeReadingSession([node], node.id)
        response = session.dispatch("read_candidate_code", {"symbol": "loss"})
        anchor = {"node_id": node.id, "source_sha256": response["source_sha256"],
                  "start_line": 1, "end_line": 2}
        return session, anchor

    def test_exact_source_hash_and_read_coverage_preserved(self):
        session, anchor = self.source()
        raw = report(mechanisms=[mechanism(implementation_basis="code", code_refs=[anchor])])
        self.assertFalse(self.validate(raw, code_session=session)["issues"])
        raw["mechanisms"][0]["code_refs"][0]["source_sha256"] = "wrong"
        result = self.validate(raw, code_session=session)
        self.assertEqual(result["issues"][0]["code"], "code_anchor_unread")
        self.assertIn("candidate1", result["issues"][0]["repair_hint"])

    def test_bad_fulltext_quote_does_not_silently_downgrade_to_valid_abstract(self):
        refs = mechanism()["evidence_refs"] + [{"paper_id": "paper1", "source": "full_text",
            "chunk_id": "chunk-1", "quote": "These exact words were never returned to the model."}]
        raw = report(mechanisms=[mechanism(title="Bad fulltext", evidence_refs=refs),
                                 mechanism(title="Independent valid mechanism")])
        result = self.validate(raw)
        self.assertEqual([m["title"] for m in result["report"]["mechanisms"]], ["Independent valid mechanism"])
        self.assertEqual(result["mechanism_mapping"], [{"original_index": 1, "mechanism_id": "m1"}])
        self.assertEqual(result["dropped_mechanisms"][0]["original_index"], 0)
        self.assertEqual(result["issues"][0]["location"], "mechanisms[0].evidence_refs[1].quote")
        self.assertEqual(result["normalized_report"]["mechanisms"][0]["evidence_refs"], refs)

    def test_valid_fulltext_preserves_evidence_metadata(self):
        refs = [{"paper_id": "paper1", "source": "full_text", "chunk_id": "chunk-1", "quote": FULLTEXT}]
        result = self.validate(report(mechanisms=[mechanism(evidence_refs=refs)]))
        self.assertFalse(result["issues"])
        item = result["report"]["mechanisms"][0]
        self.assertEqual(item["evidence_level"], "full_text")
        self.assertEqual(item["evidence_refs"][0]["pdf_sha256"], "pdf")
        self.assertEqual(item["evidence_refs"][0]["quote"], FULLTEXT)

    def test_invalid_shared_fact_blocks_otherwise_valid_complete_mechanisms(self):
        raw = report(mechanisms=[mechanism(), mechanism(title="Another")])
        raw["observed_facts"].append({"statement": "A state changed.", "source": "code",
            "evidence": "Unread lines", "code_refs": [], "runtime_evidence": []})
        result = self.validate(raw)
        self.assertFalse(result["report"]["mechanisms"])
        self.assertEqual([m["original_index"] for m in result["dropped_mechanisms"]], [0, 1])
        self.assertEqual(len(result["normalized_report"]["observed_facts"]), 2)

    def test_missing_field_points_to_original_mechanism_index(self):
        result = self.validate(report(mechanisms=[mechanism(validation_plan=""), mechanism(title="Good")]))
        self.assertEqual(result["issues"][0]["location"], "mechanisms[0].validation_plan")
        self.assertEqual(result["mechanism_mapping"], [{"original_index": 1, "mechanism_id": "m1"}])
        self.assertEqual(set(result["issues"][0]), {"code", "location", "received", "expected", "repair_hint"})

    def test_unseen_paper_invalidates_whole_mechanism_instead_of_removing_only_citation(self):
        result = self.validate(report(mechanisms=[mechanism(paper_ids=["paper1", "invented"])]))
        self.assertFalse(result["report"]["mechanisms"])
        self.assertIn("paper_not_seen", {i["code"] for i in result["issues"]})

    def test_legacy_and_revision_two_abstention(self):
        empty = {"bottlenecks": [], "mechanisms": [], "observed_facts": [], "hypotheses": [], "unknowns": []}
        self.assertFalse(self.validate(empty)["issues"])
        self.assertTrue(self.validate({**empty, "report_schema_revision": 2})["issues"])
        result = self.validate({**empty, "report_schema_revision": 2, "abstention_reason": "No grounded match."})
        self.assertFalse(result["issues"])
        self.assertEqual(result["report"]["abstention_reason"], "No grounded match.")

    def test_malformed_model_fields_return_issues_without_throwing(self):
        cases = [[], report(observed_facts=[{"statement": "x", "evidence": "y", "source": []}]),
                 report(mechanisms=[mechanism(implementation_basis=[])]),
                 report(mechanisms=[mechanism(bottleneck_idx="not-an-index")])]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertTrue(self.validate(raw)["issues"])

    def test_render_keeps_whole_mechanisms_and_explicit_runtime_references(self):
        clean = self.validate(report(mechanisms=[mechanism(), mechanism(title="Second")]))["report"]
        one = copy.deepcopy(clean)
        one["mechanisms"] = one["mechanisms"][:1]
        _, text = report_v2.render(one, self.corpus, 100000)
        retained, text = report_v2.render(clean, self.corpus, len(text) + 1)
        self.assertEqual(len(retained["mechanisms"]), 1)
        self.assertIn("runtime_evidence=", text)
        self.assertNotIn("### Second", text)
        self.assertIn("Reject worse scores.", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
