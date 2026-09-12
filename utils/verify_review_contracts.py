"""CPU regressions for real review schemas and agents through the Responses SDK.

Run: python utils/verify_review_contracts.py
Only HTTP responses and grading are mocked; no API calls, training or credentials.
"""

from contextlib import contextmanager
from itertools import product
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm
from llm import openai as backend
from llm.responses import ResponsesError
from agents import code_review_agent, result_parse_agent
from engine.search_node import SearchNode


class ReviewContractTests(unittest.TestCase):
    model = "gpt-6-astra"

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        stage = NS(model=self.model, reasoning_effort="high", temp=1,
                   max_output_tokens=16384, api_key="synthetic-secret", base_url="https://proxy.invalid/v1")
        acfg = NS(code=stage, feedback=NS(**vars(stage)), use_diff_mode=True,
                  use_global_memory=False, check_data_leakage=False)
        cfg = NS(agent=acfg, log_dir=self.root, workspace_dir=self.root,
                 candidate_runtime=NS(enabled=False))
        self.agent = NS(acfg=acfg, cfg=cfg, task_desc="Synthetic AUC task", global_memory=None,
                        metric_maximize=True, metric_maximize_reasoning="AUC is maximized")

    @contextmanager
    def replies(self, spec, payloads):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(request.url.path, "/v1/responses")
            self.assertEqual(body["model"], self.model)
            self.assertEqual(body["reasoning"], {"effort": "high"})
            self.assertEqual(body["tools"][0]["parameters"], spec.json_schema)
            self.assertFalse(body["tools"][0]["strict"])
            payload = payloads[min(len(requests) - 1, len(payloads) - 1)]
            return httpx.Response(200, json={
                "id": "resp_fixture", "model": self.model, "status": "completed",
                "reasoning": {"effort": "high"}, "usage": {"input_tokens": 10, "output_tokens": 10},
                "output": [{"type": "function_call", "name": spec.name, "call_id": "call_fixture",
                            "arguments": json.dumps(payload)}]})

        def client(stage):
            return openai.OpenAI(api_key=stage.api_key, base_url=stage.base_url, max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        with patch.object(backend, "make_response_client", side_effect=client):
            yield requests

    def test_approved_null_and_omitted_code_reach_agent_without_retry(self):
        for runtime, fields in product((False, True), ({"revised_code": None}, {})):
            with self.subTest(runtime=runtime, fields=fields):
                self.agent.cfg.candidate_runtime.enabled = runtime
                node = SearchNode(code="x = 1\n", stage="draft")
                payload = dict(needs_revision=False, reasoning="No required changes.", **fields)
                with self.replies(code_review_agent.CODE_REVIEW_SPEC, [payload]) as requests:
                    self.assertEqual(code_review_agent.run(self.agent, node), node.code)
                self.assertEqual(len(requests), 1)

    def test_revision_diff_is_applied(self):
        node = SearchNode(code="x = 1\n", stage="draft")
        payload = dict(needs_revision=True, reasoning="Fix the value.",
                       revised_code="<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE")
        with self.replies(code_review_agent.CODE_REVIEW_SPEC, [payload]) as requests:
            self.assertEqual(code_review_agent.run(self.agent, node), "x = 2")
        self.assertEqual(len(requests), 1)

    def test_required_revision_without_code_still_uses_bounded_review_retry(self):
        node = SearchNode(code="x = 1\n", stage="draft")
        missing = dict(needs_revision=True, reasoning="Fix needed.", revised_code=None)
        approved = dict(needs_revision=False, reasoning="No change needed.", revised_code=None)
        with patch.object(code_review_agent.time, "sleep"), \
                self.replies(code_review_agent.CODE_REVIEW_SPEC, [missing, approved]) as requests:
            self.assertEqual(code_review_agent.run(self.agent, node), node.code)
        self.assertEqual(len(requests), 2)
        with patch.object(code_review_agent.time, "sleep"), \
                self.replies(code_review_agent.CODE_REVIEW_SPEC, [missing]) as requests:
            self.assertEqual(code_review_agent.run(self.agent, node), node.code)
        self.assertEqual(len(requests), 3)  # Preserve the existing bounded fallback.

    def test_failed_execution_null_metric_returns_buggy_node(self):
        for use_memory in (False, True):
            with self.subTest(use_memory=use_memory):
                self.agent.acfg.use_global_memory = use_memory
                payload = dict(is_bug=True, summary="Training failed.", metric=None, lower_is_better=False)
                if use_memory:
                    payload["code_summary"] = "Synthetic failing candidate."
                node = SearchNode(code="raise ValueError('training failed')", stage="draft")
                execution = NS(term_out=["ValueError: training failed"], exec_time=1,
                               exc_type="ValueError", exc_info={}, exc_stack=[])
                spec = result_parse_agent.get_review_func_spec(use_memory)
                with self.replies(spec, [payload]) as requests:
                    self.assertIs(result_parse_agent.run(self.agent, node, execution), node)
                self.assertEqual(len(requests), 1)
                self.assertTrue(node.is_buggy)
                self.assertTrue(node.metric.is_worst)
                self.assertEqual(node.analysis, payload["summary"])
                self.assertEqual(node.code_summary, payload.get("code_summary"))

    def test_successful_scores_and_direction_check_remain_intact(self):
        for use_memory in (False, True):
            for score, lower in ((0.91, False), (0, False), (0.91, True)):
                with self.subTest(use_memory=use_memory, score=score, lower=lower):
                    self.agent.acfg.use_global_memory = use_memory
                    payload = dict(is_bug=False, summary="Training finished.", metric=score, lower_is_better=lower)
                    if use_memory:
                        payload["code_summary"] = "Synthetic successful candidate."
                    node = SearchNode(code="print('score')", stage="draft")
                    (self.root/"submission").mkdir(exist_ok=True)
                    (self.root/"submission"/f"submission_{node.id}.csv").write_text("id,prediction\n1,0.1\n")
                    execution = NS(term_out=["Training finished"], exec_time=1,
                                   exc_type=None, exc_info={}, exc_stack=[])
                    with self.replies(result_parse_agent.get_review_func_spec(use_memory), [payload]) as requests, \
                            patch.object(result_parse_agent, "_validate_format_with_retry") as grade:
                        self.assertIs(result_parse_agent.run(self.agent, node, execution), node)
                    grade.assert_called_once()
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(node.is_buggy, lower)  # Opposite direction is still rejected.
                    self.assertEqual(node.metric.value, None if lower else score)

    def test_wrong_types_and_missing_required_fields_still_fail_validation(self):
        review = dict(needs_revision=False, reasoning="Approved.", revised_code=None)
        result = dict(is_bug=True, summary="Failed.", metric=None, lower_is_better=False)
        cases = [(code_review_agent.CODE_REVIEW_SPEC, {**review, "revised_code": 123}),
                 (code_review_agent.CODE_REVIEW_SPEC, {**review, "needs_revision": "false"}),
                 (code_review_agent.CODE_REVIEW_SPEC, {k: v for k, v in review.items() if k != "reasoning"}),
                 (result_parse_agent.get_review_func_spec(False), {**result, "metric": "0.91"}),
                 (result_parse_agent.get_review_func_spec(False), {**result, "metric": True}),
                 (result_parse_agent.get_review_func_spec(False), {k: v for k, v in result.items() if k != "metric"}),
                 (result_parse_agent.get_review_func_spec(True), result)]
        for spec, payload in cases:
            with self.subTest(function=spec.name, payload=payload), self.replies(spec, [payload]) as requests:
                with self.assertRaises(ResponsesError) as raised:
                    llm.query("Synthetic review", None, model=self.model, cfg=self.agent.cfg,
                              role="code", func_spec=spec)
                self.assertEqual(raised.exception.category, "invalid_output")
                self.assertEqual(len(requests), 1)


class SolReviewContractTests(ReviewContractTests):
    model = "gpt-5.6-sol"


if __name__ == "__main__":
    logging.getLogger("MLEvolve").setLevel(logging.CRITICAL)
    print(f"Real review schemas/agents, OpenAI SDK {openai.__version__}; HTTP and grading mocked.")
    unittest.main()
