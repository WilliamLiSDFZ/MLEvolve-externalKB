"""Real material planner/reviewer wrappers with mocked Responses HTTP only."""

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import httpx
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import code_review_agent
from agents.planner.base_planner import run_planner
from agents.planner.planner_with_memory import refine_plan_to_json
from agents.replay_material import MaterialProtocol
from engine.search_node import SearchNode
from llm import openai as backend
from llm.responses import ResponsesError
from tests.verify_replay_material import material_text, plan_for


class MaterialTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.text = material_text("C")
        self.material = MaterialProtocol(self.root / "material", self.text, "C", "x = 1\n")
        stage = NS(model="gpt-5.6-sol", reasoning_effort="high", temp=1, max_output_tokens=16384,
                   api_key="synthetic-secret", base_url="https://proxy.invalid/v1")
        acfg = NS(code=stage, feedback=NS(**vars(stage)), use_diff_mode=True, use_global_memory=False)
        cfg = NS(agent=acfg, log_dir=self.root, workspace_dir=self.root,
                 candidate_runtime=NS(enabled=False))
        self.agent = NS(acfg=acfg, cfg=cfg, task_desc="Synthetic task", replay_profile="mac-mps",
                        replay_material=self.material, global_memory=None, executor=Mock())

    @contextmanager
    def response(self, raw_output):
        requests = []

        def handler(request):
            self.assertEqual(request.url.path, "/v1/responses")
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(body["model"], "gpt-5.6-sol")
            self.assertEqual(body["reasoning"], {"effort": "high"})
            if body.get("tools"):
                output = [{"type": "function_call", "name": body["tools"][0]["name"],
                           "call_id": "call_fixture", "arguments": raw_output}]
            else:
                output = [{"type": "message", "role": "assistant",
                           "content": [{"type": "output_text", "text": raw_output}]}]
            completed = {"id": "resp_fixture", "model": "gpt-5.6-sol", "status": "completed",
                         "usage": {"input_tokens": 10, "output_tokens": 10}, "output": output}
            if body.get("stream"):
                event = {"type": "response.completed", "response": completed}
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content="event: response.completed\ndata: " + json.dumps(event) + "\n\n")
            return httpx.Response(200, json=completed)

        def client(stage):
            return openai.OpenAI(api_key=stage.api_key, base_url=stage.base_url, max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        with patch.object(backend, "make_response_client", side_effect=client):
            yield requests

    def plan(self):
        return run_planner(
            agent_instance=self.agent,
            prompt_base={"Introduction": "Plan a change", "Task description": "Synthetic task",
                         "Instructions": {}, "Previous solution": {"Code": "x = 1"}},
            data_preview="Synthetic rows", context={}, your_task_section="Choose a training improvement",
            assistant_suffix="Return the structured plan.",
        )

    def review(self):
        return code_review_agent.run(self.agent, SearchNode(code="x = 2\n", stage="improve"))

    def refine(self):
        self.agent.global_memory = NS(retrieve_similar_records=Mock(return_value=[]))
        return refine_plan_to_json(
            self.agent, "Improve training sampling.",
            {"Introduction": "Plan a change", "Task description": "Synthetic task",
             "Instructions": {}, "Previous solution": {"Code": "x = 1"}},
            "Synthetic rows", {},
        )

    def assert_complete_material_on_wire(self, request):
        contents = "\n".join(item["content"] for item in request["input"])
        if self.text:
            self.assertEqual(contents.count(self.text), 1)
            arm = self.material.audit["arm"]
            self.assertIn(f"START_{arm}", contents)
            self.assertIn(f"END_{arm}", contents)
        else:
            self.assertIn("<replay_material>\n\n</replay_material>", contents)

    def test_successful_plan_and_review_keep_full_material_through_real_sdk(self):
        payload = plan_for("C", "adopt")
        del payload["parse_success"]
        with self.response(json.dumps(payload)) as requests:
            plan = self.plan()
        self.assertEqual(len(requests), 1)
        self.assert_complete_material_on_wire(requests[0])
        self.assertIn("material_decision", requests[0]["text"]["format"]["schema"]["required"])
        self.material.accept_plan(plan)
        payload = dict(needs_revision=False, reasoning="No correction needed.", revised_code=None,
                       material_review=dict(verdict="consistent", reason="The planned change is present.",
                                            evidence=["x = 2"]))
        with self.response(json.dumps(payload)) as requests:
            self.assertEqual(self.review(), "x = 2\n")
        self.assertEqual(len(requests), 1)
        self.assert_complete_material_on_wire(requests[0])
        self.assertIn("material_review", requests[0]["tools"][0]["parameters"]["required"])
        self.assertEqual(self.material.audit["material_review"]["verdict"], "consistent")
        self.assertEqual(len(self.material.audit["calls"]), 2)
        self.agent.executor.run.assert_not_called()

    def test_nested_decision_is_normalized_once_for_both_planners_and_all_arms(self):
        for operation, stage in ((self.plan, "planner_direct"), (self.refine, "planner_refine")):
            for arm, decision in (("A", "no_material"), ("B", "reject"),
                                  ("C", "adopt"), ("D", "adopt")):
                with self.subTest(stage=stage, arm=arm):
                    self.text = material_text(arm)
                    self.material = MaterialProtocol(self.root / f"{stage}_{arm}", self.text,
                                                     arm, "x = 1\n")
                    self.agent.replay_material = self.material
                    payload = plan_for(arm, decision)
                    del payload["parse_success"]
                    expected = json.loads(json.dumps(payload))
                    payload["plan"]["material_decision"] = payload.pop("material_decision")
                    raw = json.dumps(payload)
                    with self.response(raw) as requests:
                        plan = operation()
                    self.assertEqual(len(requests), 1)
                    self.assert_complete_material_on_wire(requests[0])
                    self.assertEqual(plan["material_decision"], expected["material_decision"])
                    self.assertEqual(plan["plan"], expected["plan"])
                    self.assertEqual(len(self.material.audit["calls"]), 1)
                    record = json.loads((self.material.directory / self.material.audit["calls"][0]).read_text())
                    self.assertEqual(record["stage"], stage)
                    self.assertEqual(record["status"], "completed")
                    self.assertEqual(record["response"], raw)
                    self.assertEqual(json.loads(record["normalized_response"]), expected)
                    self.assertEqual(record["normalization"], "moved plan.material_decision to material_decision")
                    self.assertEqual(record["error_category"], "invalid_output")
                    self.assertIn("ResponsesError", record["error"])
        self.agent.executor.run.assert_not_called()

    def test_ambiguous_or_invalid_nested_decision_is_still_terminal(self):
        mutations = {
            "duplicate": lambda p: p.update(material_decision=p["plan"]["material_decision"]),
            "invalid_choice": lambda p: p["plan"]["material_decision"].update(decision="maybe"),
            "bad_quote": lambda p: p["plan"]["material_decision"].update(source_quotes=["absent quote"]),
            "wrong_arm": lambda p: p["plan"]["material_decision"].update(decision="no_material"),
            "other_schema_error": lambda p: p.update(reason=123),
        }
        for operation, stage in ((self.plan, "planner_direct"), (self.refine, "planner_refine")):
            for name, mutate in mutations.items():
                with self.subTest(stage=stage, error=name):
                    payload = plan_for("C", "adopt")
                    del payload["parse_success"]
                    payload["plan"]["material_decision"] = payload.pop("material_decision")
                    mutate(payload)
                    self.assert_terminal_failure(operation, json.dumps(payload), stage)

    def test_legacy_planners_do_not_normalize_nested_material_decisions(self):
        self.agent.replay_material = None
        payload = plan_for("C", "adopt")
        del payload["parse_success"]
        payload["plan"]["material_decision"] = payload.pop("material_decision")
        raw = json.dumps(payload)
        for operation in (self.plan, self.refine):
            with self.subTest(planner=operation.__name__), self.response(raw) as requests:
                with self.assertRaises(ResponsesError) as raised:
                    operation()
            self.assertEqual(raised.exception.category, "invalid_output")
            self.assertEqual(raised.exception.raw_output, raw)
            self.assertEqual(len(requests), 1)
            self.assertNotIn("material_decision", requests[0]["text"]["format"]["schema"]["required"])
        self.assertEqual(self.material.audit["calls"], [])

    def test_invalid_generate_output_is_terminal_and_raw_output_is_audited(self):
        missing = plan_for("C", "adopt")
        del missing["parse_success"], missing["material_decision"]
        for raw in (json.dumps(missing), "{broken planner JSON"):
            with self.subTest(raw=raw):
                self.assert_terminal_failure(self.plan, raw, "planner_direct")

    def test_invalid_query_arguments_are_terminal_and_raw_arguments_are_audited(self):
        self.material.accept_plan(plan_for("C", "adopt"))
        missing = dict(needs_revision=False, reasoning="Approved.", revised_code=None)
        for raw in (json.dumps(missing), "{broken review JSON"):
            with self.subTest(raw=raw):
                self.assert_terminal_failure(self.review, raw, "review")

    def assert_terminal_failure(self, operation, raw, stage):
        count = len(self.material.audit["calls"])
        with self.response(raw) as requests:
            with self.assertRaises(ResponsesError) as raised:
                operation()
        self.assertEqual(raised.exception.category, "invalid_output")
        self.assertTrue(raised.exception.transport_retry_exhausted)
        self.assertEqual(raised.exception.raw_output, raw)
        self.assertEqual(len(requests), 1)
        self.assert_complete_material_on_wire(requests[0])
        self.assertEqual(len(self.material.audit["calls"]), count + 1)
        record = json.loads((self.material.directory / self.material.audit["calls"][-1]).read_text())
        self.assertEqual(record["stage"], stage)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_category"], "invalid_output")
        self.assertEqual(record["response"], raw)
        self.agent.executor.run.assert_not_called()


if __name__ == "__main__":
    logging.getLogger("MLEvolve").setLevel(logging.CRITICAL)
    unittest.main(verbosity=2)
