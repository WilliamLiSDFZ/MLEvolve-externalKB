"""Offline GPT-6 transport regressions using the real SDK and httpx MockTransport.

Run: python utils/verify_gpt6_transport.py
No endpoint, credentials, model requests or GPU are used. Supports the project's
pinned openai==1.66.3 through its generic JSON and SSE transport.
"""

from __future__ import annotations

import copy
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import httpx
import openai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm
from llm import openai as backend
from llm.responses import (ResponsesError, function_tool, request_response,
                           response_function_calls, response_text, response_usage,
                           should_retry_outer, is_gpt6_model, uses_responses, supports_reasoning_effort)
from llm.model_profiles import supports_sampling_params, get_profile


MODEL = "gpt-6-astra"
ERROR_REPRESENTATIONS = ("json", "sse_error", "sse_sdk_error", "sse_failed", "sdk_flat", "sdk_nested")
TRANSIENT_ERROR_CODES = ("request_timeout", "server_is_overloaded")
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"},
          "note": {"type": "string"}}, "required": ["ok"], "additionalProperties": False}


def completed(text="ok", *, output=None, **fields):
    return {"id": "resp_test", "model": MODEL, "status": "completed", "created_at": 1,
            "reasoning": {"effort": "high"}, "usage": {"input_tokens": 11, "output_tokens": 7},
            "output": output if output is not None else [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            **fields}


def call(name="submit", args='{"ok":true}', call_id="call_1"):
    return {"type": "function_call", "id": "fc_1", "call_id": call_id,
            "name": name, "arguments": args, "status": "completed"}


def sse(*events):
    body = "".join("event: " + item["type"] + "\ndata: " + json.dumps(item) + "\n\n" for item in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


class TransportTests(unittest.TestCase):
    def client(self, replies):
        self.requests = []
        def handler(request):
            self.requests.append({"path": request.url.path, "host": request.url.host,
                                  "body": json.loads(request.content)})
            index = len(self.requests) - 1
            reply = replies[min(index, len(replies) - 1)]
            if isinstance(reply, BaseException):
                raise reply
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200, json=reply)
        client = openai.OpenAI(api_key="synthetic-secret", base_url="https://proxy.invalid/v1",
                              max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(client.close)
        return client

    def cfg(self, log_dir=None):
        def stage(host, effort):
            return NS(model=MODEL, api_key="synthetic-secret", base_url=f"https://{host}/v1",
                      reasoning_effort=effort, max_output_tokens=16384)
        return NS(agent=NS(code=stage("code.invalid", "high"),
                           feedback=stage("feedback.invalid", "medium")), log_dir=log_dir)

    def request(self, client, **params):
        return request_response(client, model=MODEL, input_items=[{"role": "user", "content": "probe"}],
                                retry_delay=0, **params)

    def test_exact_request_shape_and_usage(self):
        result = self.request(self.client([completed()]))
        self.assertEqual(response_text(result), "ok")
        self.assertEqual(response_usage(result), (11, 7))
        request = self.requests[0]
        self.assertEqual(request["path"], "/v1/responses")
        body = request["body"]
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(body["max_output_tokens"], 16384)
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertIs(body["store"], False)
        for param in ["temperature", "top_p", "top_logprobs", "logprobs", "max_tokens", "reasoning_effort"]:
            self.assertNotIn(param, body)

    def test_two_rounds_preserve_all_output_and_opaque_reasoning(self):
        opaque = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-test",
                  "future_opaque_field": {"unknown": "must survive SDK"}}
        first = completed(output=[opaque, call("candidate_code_index", '{"node_id":"current"}')])
        second = completed(output=[call("read_candidate_code", '{"symbol":"train"}', "call_2")])
        client = self.client([first, second, completed("observed")])
        history = [{"role": "user", "content": "Read source"}]
        tools = [function_tool("candidate_code_index", "index", SCHEMA),
                 function_tool("read_candidate_code", "read", SCHEMA)]
        for expected, result in [("candidate_code_index", {"symbol": "train"}),
                                 ("read_candidate_code", {"code": "observed"})]:
            response = request_response(client, model=MODEL, input_items=history, tools=tools)
            tc = response_function_calls(response)[0]
            self.assertEqual(tc["name"], expected)
            history.extend(response["output"])
            history.append({"type": "function_call_output", "call_id": tc["call_id"], "output": json.dumps(result)})
        final = request_response(client, model=MODEL, input_items=history, tools=tools, tool_choice="none")
        self.assertEqual(response_text(final), "observed")
        self.assertIn(opaque, self.requests[1]["body"]["input"])
        self.assertIn(opaque, self.requests[2]["body"]["input"])
        outputs = [item for item in self.requests[2]["body"]["input"] if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["call_1", "call_2"])

    def test_completed_stream(self):
        final = completed("complete")
        client = self.client([sse({"type": "response.output_text.delta", "delta": "partial"},
                                 {"type": "response.completed", "response": final})])
        response = self.request(client, stream=True)
        self.assertEqual(response_text(response), "complete")
        self.assertTrue(self.requests[0]["body"]["stream"])

    def test_truncated_stream_never_returns_partial_and_has_bounded_retry(self):
        reply = lambda: sse({"type": "response.output_text.delta", "delta": "partial"})
        client = self.client([reply(), reply(), reply()])
        with self.assertRaises(ResponsesError) as ctx:
            self.request(client, stream=True, max_attempts=20)
        self.assertEqual(len(self.requests), 3)
        self.assertFalse(should_retry_outer(ctx.exception))

    def error_reply(self, representation, code):
        error = {"code": code, "message": "synthetic error"}
        if representation == "json":
            return completed(status="failed", error=error, text="discard this partial output")
        delta = {"type": "response.output_text.delta", "delta": "discard this partial output"}
        if representation == "sse_error":
            terminal = {"type": "error", **error}
        elif representation == "sse_sdk_error":
            # openai==1.66.3 raises APIError for this nested SSE envelope.
            terminal = {"type": "error", "error": error}
        elif representation == "sse_failed":
            terminal = {"type": "response.failed", "response": completed(status="failed", error=error)}
        elif representation in {"sdk_flat", "sdk_nested"}:
            return openai.APIError("synthetic error", request=httpx.Request("POST", "https://proxy.invalid/v1/responses"),
                                   body=error if representation == "sdk_flat" else {"error": error})
        else:
            raise AssertionError(representation)
        return sse(delta, terminal)

    def test_transient_codes_retry_all_error_representations_and_discard_partial(self):
        for representation, code in product(ERROR_REPRESENTATIONS, TRANSIENT_ERROR_CODES):
            with self.subTest(representation=representation, code=code):
                streaming = representation.startswith("sse_")
                success = sse({"type": "response.completed", "response": completed("fresh result")}) if streaming else completed("fresh result")
                reply = self.error_reply(representation, code)
                client = self.client([reply, success])
                # Inject SDK exceptions after HTTP so they cannot be converted to
                # APIConnectionError by the SDK's transport-exception handler.
                if representation.startswith("sdk_"):
                    with patch.object(client, "post", side_effect=[reply, success]) as post:
                        result = self.request(client)
                    self.assertEqual(post.call_count, 2)
                    self.assertEqual(post.call_args_list[0], post.call_args_list[1])
                else:
                    result = self.request(client, stream=streaming)
                    self.assertEqual(len(self.requests), 2)
                    self.assertEqual(self.requests[0], self.requests[1])
                self.assertEqual(response_text(result), "fresh result")

    def test_transient_codes_exhaust_at_three_attempts(self):
        for representation, code in product(ERROR_REPRESENTATIONS, TRANSIENT_ERROR_CODES):
            with self.subTest(representation=representation, code=code):
                replies = [self.error_reply(representation, code) for _ in range(3)]
                client = self.client(replies)
                if representation.startswith("sdk_"):
                    with patch.object(client, "post", side_effect=replies) as post:
                        with self.assertRaises(ResponsesError) as ctx:
                            self.request(client, max_attempts=20)
                    self.assertEqual(post.call_count, 3)
                else:
                    with self.assertRaises(ResponsesError) as ctx:
                        self.request(client, stream=representation.startswith("sse_"), max_attempts=20)
                    self.assertEqual(len(self.requests), 3)
                self.assertEqual(ctx.exception.category, "transient_exhausted")
                self.assertEqual(ctx.exception.attempts, 3)
                self.assertFalse(should_retry_outer(ctx.exception))

    def test_deterministic_error_codes_remain_terminal(self):
        for representation in ERROR_REPRESENTATIONS:
            for code in ["invalid_api_key", "invalid_parameter", "model_not_found", "insufficient_quota", "unknown_proxy_error"]:
                with self.subTest(representation=representation, code=code):
                    reply = self.error_reply(representation, code)
                    client = self.client([reply])
                    if representation.startswith("sdk_"):
                        with patch.object(client, "post", side_effect=reply) as post:
                            with self.assertRaises(ResponsesError) as ctx:
                                self.request(client)
                        self.assertEqual(post.call_count, 1)
                    else:
                        with self.assertRaises(ResponsesError) as ctx:
                            self.request(client, stream=representation.startswith("sse_"))
                        self.assertEqual(len(self.requests), 1)
                    self.assertEqual(ctx.exception.attempts, 1)
                    self.assertNotEqual(ctx.exception.category, "transient_exhausted")
                    self.assertFalse(should_retry_outer(ctx.exception))

    def test_incomplete_refusal_and_model_mismatch_fail_once(self):
        examples = [
            (completed(status="incomplete", incomplete_details={"reason": "max_output_tokens"}), "incomplete"),
            (completed(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "No"}]}]), "refusal"),
            (completed(model="gpt-5"), "model_mismatch"),
            (completed(reasoning={"effort": "none"}), "reasoning_mismatch"),
            (completed(output=[{"type": "function_call", "name": "submit", "arguments": "{}"}]), "protocol"),
        ]
        for response, category in examples:
            with self.subTest(category=category):
                with self.assertRaises(ResponsesError) as ctx:
                    self.request(self.client([response]))
                self.assertEqual(ctx.exception.category, category)
                self.assertEqual(len(self.requests), 1)

    def test_http_retries_only_transient(self):
        for status, code, count in [(400, "invalid_parameter", 1), (401, "invalid_api_key", 1),
                                     (429, "insufficient_quota", 1), (429, "rate_limit_exceeded", 3),
                                     (503, "server_error", 3)]:
            with self.subTest(status=status, code=code):
                replies = [httpx.Response(status, json={"error": {"code": code, "message": "synthetic"}})
                           for _ in range(count)]
                with self.assertRaises(ResponsesError):
                    self.request(self.client(replies), max_attempts=20)
                self.assertEqual(len(self.requests), count)
        client = self.client([httpx.Response(503, json={"error": {"code": "server_error"}}), completed()])
        self.assertEqual(response_text(self.request(client)), "ok")
        self.assertEqual(len(self.requests), 2)

    def test_legacy_optional_tool_schema_is_explicit_non_strict(self):
        original = copy.deepcopy(SCHEMA)
        client = self.client([completed(output=[call()])])
        self.request(client, tools=[{"type": "function", "function": {
            "name": "submit", "description": "submit", "parameters": SCHEMA}}],
            tool_choice={"type": "function", "function": {"name": "submit"}})
        body = self.requests[0]["body"]
        self.assertIs(body["tools"][0]["strict"], False)
        self.assertEqual(body["tools"][0]["parameters"], original)
        self.assertEqual(body["tool_choice"], {"type": "function", "name": "submit"})
        self.assertEqual(SCHEMA, original)

    def test_query_contract_feedback_role_and_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = self.cfg(folder)
            response = completed(output=[call()], reasoning={"effort": "medium"})
            client = self.client([response])
            spec = llm.FunctionSpec(name="submit", description="submit", json_schema=SCHEMA)
            stages = []
            def factory(stage):
                stages.append(stage)
                return client
            with patch.object(backend, "make_response_client", side_effect=factory):
                result = llm.query(system_message="system", user_message="probe", model=MODEL,
                                   func_spec=spec, cfg=cfg, role="feedback")
            self.assertEqual(result, {"ok": True})
            self.assertIs(stages[0], cfg.agent.feedback)
            self.assertEqual(self.requests[0]["body"]["reasoning"]["effort"], "medium")
            records = [json.loads(line) for line in (Path(folder)/"llm_calls.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["role"], "feedback")
            self.assertEqual(records[0]["endpoint_host"], "feedback.invalid")
            self.assertEqual(records[0]["usage"], response["usage"])
            self.assertNotIn("synthetic-secret", json.dumps(records))

    def test_low_level_query_tuple_and_invalid_schema(self):
        client = self.client([completed("hello")])
        with patch.object(backend, "make_response_client", return_value=client):
            value = backend.query("system", "probe", cfg=self.cfg(), model=MODEL, role="code")
        self.assertEqual(len(value), 5)
        self.assertEqual(value[0], "hello")
        self.assertEqual(value[2:4], (11, 7))
        client = self.client([completed(output=[call(args='{"ok":"wrong"}')])])
        spec = llm.FunctionSpec(name="submit", description="submit", json_schema=SCHEMA)
        with patch.object(backend, "make_response_client", return_value=client):
            with self.assertRaises(ResponsesError) as ctx:
                backend.query("system", "probe", cfg=self.cfg(), model=MODEL, func_spec=spec)
        self.assertEqual(ctx.exception.category, "invalid_output")

    def test_generate_structured_stream_contract_and_stop(self):
        final = completed('{"ok":true}')
        client = self.client([sse({"type": "response.completed", "response": final})])
        with patch.object(backend, "make_response_client", return_value=client):
            result = llm.generate(prompt="probe", cfg=self.cfg(), json_schema=SCHEMA)
        self.assertEqual(json.loads(result), {"ok": True})
        fmt = self.requests[0]["body"]["text"]["format"]
        self.assertEqual(fmt["schema"], SCHEMA)
        self.assertFalse(fmt["strict"])
        client = self.client([sse({"type": "response.completed", "response": completed("before STOP after")})])
        with patch.object(backend, "make_response_client", return_value=client):
            self.assertEqual(llm.generate("probe", cfg=self.cfg(), stop_tokens=["STOP"]), "before ")
        self.assertNotIn("stop", self.requests[0]["body"])

    def test_error_telemetry_without_credentials_or_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.client([completed(status="incomplete")])
            with patch.object(backend, "make_response_client", return_value=client):
                with self.assertRaises(ResponsesError):
                    backend.query("private-prompt", "probe", cfg=self.cfg(folder), model=MODEL)
            raw = (Path(folder)/"llm_calls.jsonl").read_text()
            info = json.loads(raw)
            self.assertEqual(info["error"]["category"], "incomplete")
            self.assertNotIn("private-prompt", raw)
            self.assertNotIn("synthetic-secret", raw)

    def test_legacy_chat_path_retained(self):
        cfg = self.cfg()
        cfg.agent.code.model = "gpt-5"
        choice = NS(message=NS(content="legacy", tool_calls=None), finish_reason="stop")
        completion = NS(choices=[choice], usage=NS(prompt_tokens=1, completion_tokens=2), model="gpt-5", created=1)
        client = NS(chat=NS(completions=NS(create=lambda **kwargs: completion)))
        with patch.object(backend, "OpenAI", return_value=client), patch.object(backend, "request_response") as responses:
            value = backend.query("system", "probe", cfg=cfg, model="gpt-5")
        self.assertEqual(value[0], "legacy")
        responses.assert_not_called()
        self.assertFalse(supports_sampling_params(MODEL))
        self.assertEqual(get_profile(MODEL), {})
        self.assertTrue(is_gpt6_model(MODEL))
        self.assertFalse(is_gpt6_model("gpt-60"))


class SolTransportTests(TransportTests):
    """Run the entire SDK contract/retry matrix against the explicit Sol route."""

    def setUp(self):
        self.model_patch = patch.dict(globals(), {"MODEL": "gpt-5.6-sol"})
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def test_legacy_chat_path_retained(self):
        # The compatibility route still exists, but Sol itself must use Responses.
        cfg = self.cfg()
        cfg.agent.code.model = "gpt-5.6-terra"
        completion = NS(choices=[NS(message=NS(content="legacy", tool_calls=None), finish_reason="stop")],
                        usage=NS(prompt_tokens=1, completion_tokens=2), model="gpt-5.6-terra", created=1)
        client = NS(chat=NS(completions=NS(create=lambda **kwargs: completion)))
        with patch.object(backend, "OpenAI", return_value=client), patch.object(backend, "request_response") as responses:
            result = backend.query("system", "probe", cfg=cfg, model="gpt-5.6-terra")
        self.assertEqual(result[0], "legacy")
        responses.assert_not_called()
        self.assertFalse(is_gpt6_model(MODEL))
        self.assertTrue(uses_responses(MODEL))
        self.assertTrue(uses_responses("provider/gpt-5.6-sol-2026-09-01"))
        self.assertFalse(uses_responses("gpt-5.6-solar"))
        self.assertFalse(uses_responses("gpt-5.6-terra"))
        self.assertFalse(supports_sampling_params(MODEL))
        self.assertEqual(get_profile(MODEL), {})
        self.assertTrue(supports_reasoning_effort(MODEL, "none"))
        self.assertFalse(supports_reasoning_effort("gpt-6-astra", "none"))


if __name__ == "__main__":
    print(f"Testing actual OpenAI SDK {openai.__version__}; all HTTP is mocked.")
    unittest.main()
