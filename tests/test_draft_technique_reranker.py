import copy
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from engine.coldstart.technique_reranker import (
    CORE_FIELDS, OUTPUT_TOKENS, RANKING_SCHEMA, SYSTEM_PROMPT,
    check_context_budget, render_techniques, rerank_techniques,
)


def technique(technique_id):
    return {
        "technique_id": technique_id, "name": "Subtype auxiliary heads",
        "paper_id": f"paper-{technique_id}", "paper_title": "Auxiliary task study",
        "paper_finding": "Auxiliary labels improve the shared representation.",
        "task_relevance": "Subtype labels are available for toxicity prediction.",
        "task_adaptation": "Keep soft toxicity targets in the primary head.",
        "proposed_experiment": "Compare single-head and subtype auxiliary-head AUC.",
        "requirements_and_limits": "Requires subtype labels and a shared encoder.",
        "evidence": [{"chunk_id": f"{technique_id}:0", "quote": "Original quote."}],
        "source": "Long source information excluded from ranking.",
    }


def rank(*technique_ids):
    return {"ranking": [{"technique_id": value, "reason": "Feasible within the stated budget."}
                        for value in technique_ids]}


class DraftTechniqueRerankerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            agent=SimpleNamespace(code=SimpleNamespace(model="gpt-5.6-terra")),
            draft_techniques=SimpleNamespace(context_window=258400))
        self.runtime = {"time_remaining_seconds": 1800, "execution_timeout_seconds": 600,
                        "gpus": [{"name": "Test GPU", "memory_bytes": 8000000000}],
                        "parallel_experiments": 3, "ram_bytes": 16000000000}

    def rerank(self, techniques):
        return rerank_techniques(techniques, "Maximize toxicity AUC", "Train has subtype labels",
                                 self.runtime, self.cfg)

    def test_top_25_follow_complete_ranking_and_preserve_original_records(self):
        techniques = [technique(str(i)) for i in range(39)]
        before = copy.deepcopy(techniques)
        ids = [item["technique_id"] for item in reversed(techniques)]
        with patch("engine.coldstart.technique_reranker._query", return_value=rank(*ids)):
            selected, audit = self.rerank(techniques)
        self.assertEqual([item["technique_id"] for item in selected], ids[:25])
        self.assertIs(selected[0], techniques[-1])
        self.assertEqual(techniques, before)
        self.assertEqual(audit["selected_ids"], ids[:25])
        self.assertEqual(audit["input_count"], 39)
        self.assertEqual(audit["output_count"], 25)
        self.assertEqual(audit["model"], "gpt-5.6-terra")

    def test_all_candidates_and_task_data_runtime_reach_model_and_audit(self):
        techniques = [technique("a"), technique("b")]
        with patch("engine.coldstart.technique_reranker._query", return_value=rank("b", "a")) as query:
            selected, audit = self.rerank(techniques)
        prompt, passed_cfg = query.call_args.args
        payload = json.loads(prompt)
        self.assertIs(passed_cfg, self.cfg)
        self.assertEqual(payload, audit["input"])
        self.assertEqual(payload["runtime"], self.runtime)
        self.assertEqual(payload["task_description"], "Maximize toxicity AUC")
        self.assertEqual(payload["data_description"], "Train has subtype labels")
        self.assertEqual([item["technique_id"] for item in payload["techniques"]], ["a", "b"])
        for item in payload["techniques"]:
            self.assertEqual(set(item), set(CORE_FIELDS) | {"paper_id"})
        self.assertEqual(audit["system_prompt"], SYSTEM_PROMPT)
        self.assertEqual(audit["response_schema"], RANKING_SCHEMA)
        self.assertEqual(len(selected), 2)

    def test_empty_input_is_rejected_and_singleton_skips_model(self):
        with patch("engine.coldstart.technique_reranker._query") as query:
            with self.assertRaises(ValueError):
                self.rerank([])
            techniques = [technique("a")]
            selected, audit = self.rerank(techniques)
        query.assert_not_called()
        self.assertEqual(selected, techniques)
        self.assertEqual(audit["selected_ids"], ["a"])

    def test_incomplete_unknown_and_repeated_rankings_are_rejected(self):
        for response in [rank("a"), rank("a", "missing"), rank("a", "a", "b"), rank()]:
            with self.subTest(response=response):
                with patch("engine.coldstart.technique_reranker._query", return_value=response):
                    with self.assertRaises(ValueError):
                        self.rerank([technique("a"), technique("b")])

    def test_malformed_model_responses_are_rejected(self):
        responses = [None, [], {}, {"ranking": "ab"}, {**rank("a", "b"), "extra": True},
                     {"ranking": ["a", "b"]},
                     {"ranking": [{"technique_id": "a"}]},
                     {"ranking": [{"technique_id": "a", "reason": "  "}]},
                     {"ranking": [{"technique_id": "a", "reason": 1}]},
                     {"ranking": [{"technique_id": [], "reason": "Bad ID"}]},
                     {"ranking": [{"technique_id": "a", "reason": "Reason", "score": 1}]}]
        for response in responses:
            with self.subTest(response=response):
                with patch("engine.coldstart.technique_reranker._query", return_value=response):
                    with self.assertRaises(ValueError):
                        self.rerank([technique("a"), technique("b")])

    def test_malformed_corpus_is_rejected_before_calling_model(self):
        invalid = [None, {}, [None], [technique("a"), technique("a")],
                   [{**technique("a"), "paper_id": 7}]]
        for field in CORE_FIELDS:
            for value in [None, "", "  ", []]:
                invalid.append([{**technique("a"), field: value}])
        with patch("engine.coldstart.technique_reranker._query") as query:
            for techniques in invalid:
                with self.subTest(techniques=techniques):
                    with self.assertRaises(ValueError):
                        self.rerank(techniques)
        query.assert_not_called()

    def test_optional_paper_id_and_renderer_preserve_all_core_text(self):
        item = technique("a")
        del item["paper_id"]
        item["proposed_experiment"] += "完整内容" * 1000
        rendered = json.loads(render_techniques([item]))
        self.assertIsNone(rendered[0]["paper_id"])
        self.assertEqual(rendered[0]["proposed_experiment"], item["proposed_experiment"])
        self.assertNotIn("evidence", rendered[0])
        self.assertNotIn("source", rendered[0])

    def test_budget_uses_utf8_includes_overhead_and_output_and_accepts_boundary(self):
        text = "论文 abc"
        total = len(text.encode("utf-8")) + 2048 + OUTPUT_TOKENS
        budget = check_context_budget(text, total)
        self.assertEqual(budget["required_tokens_upper_bound"], total)
        self.assertEqual(budget["remaining_tokens_lower_bound"], 0)
        with self.assertRaisesRegex(ValueError, "nothing was truncated"):
            check_context_budget(text, total - 1)

    def test_rerank_budget_includes_system_payload_and_response_schema(self):
        techniques = [technique("a"), technique("b")]
        with patch("engine.coldstart.technique_reranker._query", return_value=rank("a", "b")) as query:
            _, audit = self.rerank(techniques)
        prompt = query.call_args.args[0]
        total = len((SYSTEM_PROMPT + prompt + json.dumps(RANKING_SCHEMA)).encode("utf-8"))
        self.assertEqual(audit["context_budget"]["input_token_upper_bound"], total)
        self.cfg.draft_techniques.context_window = total + 2048 + OUTPUT_TOKENS - 1
        before = copy.deepcopy(techniques)
        with patch("engine.coldstart.technique_reranker._query") as query:
            with self.assertRaisesRegex(ValueError, "nothing was truncated"):
                self.rerank(techniques)
        query.assert_not_called()
        self.assertEqual(techniques, before)

    def test_model_failure_propagates_without_using_file_order(self):
        failure = RuntimeError("Model service unavailable")
        with patch("engine.coldstart.technique_reranker._query", side_effect=failure):
            with self.assertRaises(RuntimeError) as raised:
                self.rerank([technique("a"), technique("b")])
        self.assertIs(raised.exception, failure)

    def test_query_uses_current_code_model_and_structured_output(self):
        query = Mock(return_value=rank("b", "a"))
        with patch.dict(sys.modules, {"llm": SimpleNamespace(query=query, FunctionSpec=SimpleNamespace)}):
            self.rerank([technique("a"), technique("b")])
        arguments = query.call_args.kwargs
        self.assertEqual(arguments["model"], self.cfg.agent.code.model)
        self.assertIs(arguments["cfg"], self.cfg)
        self.assertEqual(arguments["system_message"], SYSTEM_PROMPT)
        self.assertEqual(arguments["temperature"], 0)
        self.assertEqual(arguments["max_tokens"], OUTPUT_TOKENS)
        self.assertEqual(arguments["func_spec"].json_schema, RANKING_SCHEMA)

    def test_invalid_context_configuration_is_rejected(self):
        for window, output in [(0, 1), (-1, 1), (True, 1), ("258400", 1), (10000, -1)]:
            with self.subTest(window=window, output=output):
                with self.assertRaises(ValueError):
                    check_context_budget("Input", window, output)


if __name__ == "__main__":
    unittest.main()
