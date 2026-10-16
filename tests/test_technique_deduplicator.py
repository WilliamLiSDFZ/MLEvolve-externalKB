import copy
import json
import unittest
from unittest.mock import patch

from corpus_retriever.technique_deduplicator import (
    SYSTEM_PROMPT, deduplicate_techniques,
)


def technique(technique_id, name="Subtype auxiliary heads"):
    return {
        "technique_id": technique_id,
        "name": name,
        "paper_id": f"paper-{technique_id}",
        "paper_title": f"Paper {technique_id}",
        "paper_finding": "Auxiliary labels improve the shared representation.",
        "proposed_experiment": "Compare a single head with subtype auxiliary heads.",
        "task_adaptation": "Keep soft toxicity targets in the primary head.",
        "requirements_and_limits": "Requires subtype labels.",
        "evidence": [{"chunk_id": f"{technique_id}:0", "quote": "Original quote."}],
    }


def group(keep_id, *member_ids):
    return {"keep_id": keep_id, "member_ids": list(member_ids) or [keep_id],
            "reason": "Same experimental intervention."}


class TechniqueDeduplicatorTests(unittest.TestCase):
    def test_representatives_preserve_original_records_and_input_order(self):
        techniques = [technique("a"), technique("b"), technique("c", "GroupDRO")]
        before = copy.deepcopy(techniques)
        groups = [group("c"), group("b", "a", "b")]
        with patch("corpus_retriever.technique_deduplicator._query",
                   return_value={"groups": groups}):
            retained, audit = deduplicate_techniques(techniques, None)
        self.assertEqual(retained, [before[1], before[2]])
        self.assertEqual(techniques, before)
        self.assertEqual(audit, {"status": "complete", "input_count": 3,
                                 "output_count": 2, "groups": groups})

    def test_query_receives_only_the_requested_comparison_fields(self):
        techniques = [technique("a"), technique("b")]
        techniques[0]["fulltext"] = "Must not be sent."
        techniques[0]["available_data"] = "Must not be sent either."
        cfg = object()
        with patch("corpus_retriever.technique_deduplicator._query",
                   return_value={"groups": [group("a", "a", "b")]}) as query:
            deduplicate_techniques(techniques, cfg)
        prompt, passed_cfg = query.call_args.args
        self.assertIs(passed_cfg, cfg)
        self.assertEqual(json.loads(prompt), {"techniques": [
            {"technique_id": t["technique_id"], "name": t["name"],
             "paper": {"paper_id": t["paper_id"], "title": t["paper_title"]},
             "paper_finding": t["paper_finding"],
             "proposed_experiment": t["proposed_experiment"]}
            for t in techniques
        ]})

    def test_distinct_techniques_are_kept_regardless_of_perceived_value(self):
        techniques = [technique("a", "Generic baseline"),
                      technique("b", "Costly speculative experiment")]
        groups = [group("b"), group("a")]
        with patch("corpus_retriever.technique_deduplicator._query",
                   return_value={"groups": groups}):
            retained, audit = deduplicate_techniques(techniques, None)
        self.assertEqual(retained, techniques)
        self.assertEqual(audit["output_count"], 2)

    def test_empty_and_singleton_inputs_do_not_call_the_model(self):
        for techniques in [[], [technique("a")]]:
            with self.subTest(count=len(techniques)):
                with patch("corpus_retriever.technique_deduplicator._query") as query:
                    retained, audit = deduplicate_techniques(techniques, None)
                query.assert_not_called()
                self.assertEqual(retained, techniques)
                self.assertEqual(audit["status"], "complete")
                self.assertEqual(audit["input_count"], len(techniques))
                self.assertEqual(audit["output_count"], len(techniques))
                self.assertEqual([member for entry in audit["groups"]
                                  for member in entry["member_ids"]],
                                 [t["technique_id"] for t in techniques])

    def test_invalid_partitions_cannot_silently_drop_or_duplicate_records(self):
        techniques = [technique("a"), technique("b"), technique("c")]
        invalid = {
            "missing member": [group("a"), group("b")],
            "unknown member": [group("a", "a", "missing"), group("b"), group("c")],
            "unknown representative": [group("missing", "a"), group("b"), group("c")],
            "representative not in own group": [group("b", "a"), group("b", "b", "c")],
            "overlapping groups": [group("a", "a", "b"), group("c", "b", "c")],
            "repeated member": [group("a", "a", "a"), group("b"), group("c")],
            "empty groups": [],
        }
        for case, groups in invalid.items():
            with self.subTest(case=case):
                with patch("corpus_retriever.technique_deduplicator._query",
                           return_value={"groups": groups}):
                    with self.assertRaises(ValueError):
                        deduplicate_techniques(techniques, None)

    def test_malformed_model_responses_are_rejected(self):
        techniques = [technique("a"), technique("b")]
        responses = [
            None, [], {}, {"groups": "a,b"},
            {"groups": [group("a")], "rejected": ["b"]},
            {"groups": ["a", "b"]},
            {"groups": [{"keep_id": "a", "member_ids": ["a", "b"]}]},
            {"groups": [{"keep_id": "a", "member_ids": [], "reason": "Empty"}]},
            {"groups": [{"keep_id": "a", "member_ids": "ab", "reason": "Wrong type"}]},
            {"groups": [{"keep_id": "a", "member_ids": ["a", "b"], "reason": "  "}]},
            {"groups": [{"keep_id": "a", "member_ids": ["a", "b"], "reason": 1}]},
            {"groups": [{**group("a", "a", "b"), "replacement_name": "Rewritten"}]},
        ]
        for response in responses:
            with self.subTest(response=response):
                with patch("corpus_retriever.technique_deduplicator._query",
                           return_value=response):
                    with self.assertRaises(ValueError):
                        deduplicate_techniques(techniques, None)

    def test_context_limit_includes_system_prompt_without_truncating(self):
        techniques = [technique("a"), technique("b")]
        with patch("corpus_retriever.technique_deduplicator._query",
                   return_value={"groups": [group("a"), group("b")]}) as query:
            deduplicate_techniques(techniques, None)
        original_size = len(query.call_args.args[0])
        padding = 200001 - len(SYSTEM_PROMPT) - original_size
        self.assertGreater(padding, 0)
        techniques[0]["paper_finding"] += "x" * padding
        before = copy.deepcopy(techniques)
        with patch("corpus_retriever.technique_deduplicator._query") as query:
            with self.assertRaises(ValueError):
                deduplicate_techniques(techniques, None)
        query.assert_not_called()
        self.assertEqual(techniques, before)

    def test_service_failure_propagates_for_pipeline_fallback(self):
        failure = RuntimeError("Service unavailable")
        with patch("corpus_retriever.technique_deduplicator._query", side_effect=failure):
            with self.assertRaises(RuntimeError) as raised:
                deduplicate_techniques([technique("a"), technique("b")], None)
        self.assertIs(raised.exception, failure)

    def test_auth_and_dependency_errors_remain_fatal(self):
        denied = RuntimeError("Invalid model credentials")
        denied.status_code = 401
        forbidden = RuntimeError("Provider denied access")
        forbidden.code = 403
        for failure, expected in [(denied, PermissionError), (forbidden, PermissionError),
                                  (PermissionError("Denied"), PermissionError),
                                  (ImportError("Missing SDK"), ImportError)]:
            with self.subTest(failure=failure):
                with patch("corpus_retriever.technique_deduplicator._query",
                           side_effect=failure):
                    with self.assertRaises(expected):
                        deduplicate_techniques([technique("a"), technique("b")], None)


if __name__ == "__main__":
    unittest.main()
