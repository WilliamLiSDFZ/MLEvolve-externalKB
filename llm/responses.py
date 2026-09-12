"""GPT-6 Responses transport, including stateless tool and reasoning replay.

Use the SDK's generic JSON endpoint rather than its generated Responses models:
openai==1.66.3 predates GPT-6 and newer opaque reasoning fields. Keeping raw
dictionaries preserves every output item for the next request without an SDK
upgrade. This adapter owns retries; callers must not retry ResponsesError.
"""

from __future__ import annotations

import copy
import logging
import re
import time
from typing import Any

import openai
import httpx
from openai import OpenAI, Stream

logger = logging.getLogger("MLEvolve")
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_TRANSIENT_CODES = {"server_error", "rate_limit_exceeded", "timeout", "overloaded"}


class ResponsesError(RuntimeError):
    """A terminal adapter failure; the transport has already handled retries."""

    retryable = False
    transport_retry_exhausted = True

    def __init__(self, message: str, *, category: str, attempts: int = 1,
                 status_code: int | None = None, response_info: dict | None = None):
        super().__init__(message)
        self.category = category
        self.attempts = attempts
        self.status_code = status_code
        self.response_info = response_info or {}


class _TransientResponseError(RuntimeError):
    pass


def is_gpt6_model(model: str | None) -> bool:
    # Mirror profiles' provider-prefix support without accepting gpt-60.
    name = (model or "").lower().split("/")[-1]
    return name == "gpt-6" or name.startswith("gpt-6-")


def should_retry_outer(exc: BaseException) -> bool:
    """Legacy caller loops may retry other errors, never transport-owned ones."""
    return not isinstance(exc, ResponsesError)


def make_response_client(stage, *, timeout: float = 1200.0) -> OpenAI:
    return OpenAI(api_key=stage.api_key, base_url=stage.base_url or None,
                  timeout=timeout, max_retries=0)


def function_tool(name: str, description: str, parameters: dict,
                  *, strict: bool = False) -> dict:
    """Legacy schemas must explicitly opt out of Responses strict normalization."""
    return {"type": "function", "name": name, "description": description,
            "parameters": copy.deepcopy(parameters), "strict": strict}


def responses_tools(tools: list[dict]) -> list[dict]:
    """Accept legacy Chat tools and already-flat Responses function tools."""
    result = []
    for tool in tools:
        if tool.get("type") != "function":
            raise ResponsesError("Only function tools are supported", category="configuration")
        fn = tool.get("function", tool)
        result.append(function_tool(fn["name"], fn.get("description", ""),
                                    fn.get("parameters", {}),
                                    strict=bool(fn.get("strict", tool.get("strict", False)))))
    return result


def response_text(response: dict) -> str:
    return "".join(content.get("text", "")
                   for item in response.get("output", []) if item.get("type") == "message"
                   for content in item.get("content", []) if content.get("type") == "output_text")


def response_function_calls(response: dict) -> list[dict]:
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]


def response_usage(response: dict) -> tuple[int, int]:
    usage = response.get("usage") or {}
    return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)


def response_info(response: dict, *, requested_model: str | None = None,
                  reasoning_effort: str | None = None) -> dict:
    """Safe metadata only: never log opaque encrypted state or model reasoning."""
    return {"model": response.get("model"), "requested_model": requested_model,
            "created": response.get("created_at"), "response_id": response.get("id"),
            "endpoint_type": "responses", "sdk_version": openai.__version__,
            "reasoning_effort": reasoning_effort,
            "returned_reasoning_effort": (response.get("reasoning") or {}).get("effort"),
            "status": response.get("status"), "usage": response.get("usage"),
            "incomplete_details": response.get("incomplete_details")}


def _validate_response(response: dict, model: str, effort: str) -> dict:
    if not isinstance(response, dict):
        raise ResponsesError("Responses body is not a JSON object", category="protocol")
    info = response_info(response, requested_model=model, reasoning_effort=effort)
    error = response.get("error") or {}
    if error:
        code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
        if code in _TRANSIENT_CODES:
            raise _TransientResponseError(f"Responses server failure ({code})")
        raise ResponsesError(f"Responses error ({code})", category="response_error", response_info=info)
    if response.get("status") != "completed":
        raise ResponsesError(f"Responses status={response.get('status')!r}; partial output discarded",
                             category="incomplete", response_info=info)
    returned = response.get("model") or ""
    if not re.fullmatch(re.escape(model) + r"(?:-\d{4}-\d{2}-\d{2})?", returned):
        raise ResponsesError(f"Requested model {model!r}, received {returned!r}",
                             category="model_mismatch", response_info=info)
    returned_effort = (response.get("reasoning") or {}).get("effort")
    if returned_effort not in {None, effort}:
        raise ResponsesError(f"Requested reasoning {effort!r}, received {returned_effort!r}",
                             category="reasoning_mismatch", response_info=info)
    output = response.get("output")
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise ResponsesError("Responses output is missing or malformed", category="protocol", response_info=info)
    for item in output:
        if item.get("type") == "function_call" and not (
                isinstance(item.get("call_id"), str) and item["call_id"]
                and isinstance(item.get("name"), str) and item["name"]
                and isinstance(item.get("arguments"), str)):
            raise ResponsesError("Malformed Responses function call", category="protocol", response_info=info)
        if item.get("type") == "message":
            if any(content.get("type") == "refusal" for content in item.get("content", [])):
                raise ResponsesError("Model refused the request", category="refusal", response_info=info)
    return response


def _read_stream(stream) -> dict:
    """Only a completed response is usable; EOF and partial deltas are failures."""
    try:
        for event in stream:
            if not isinstance(event, dict):
                raise ResponsesError("Malformed Responses SSE event", category="protocol")
            # SDK 1.66 wraps non-response event names in {event, data}.
            if "event" in event and isinstance(event.get("data"), dict):
                event = {"type": event["event"], **event["data"]}
            event_type = event.get("type")
            if event_type in {"response.completed", "response.incomplete", "response.failed"}:
                response = event.get("response")
                if not isinstance(response, dict):
                    raise ResponsesError("Terminal SSE event lacks response", category="protocol")
                return response
            if event_type == "error":
                code = event.get("code", "unknown")
                if code in _TRANSIENT_CODES:
                    raise _TransientResponseError(f"Responses SSE error ({code})")
                raise ResponsesError(f"Responses SSE error ({code})", category="response_error")
        raise _TransientResponseError("Responses stream ended without a terminal event")
    finally:
        stream.close()


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                        httpx.TransportError, _TransientResponseError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        body = getattr(exc, "body", None) or {}
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        if code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return False
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    if isinstance(exc, openai.APIError):
        body = getattr(exc, "body", None) or {}
        return isinstance(body, dict) and body.get("code") in _TRANSIENT_CODES
    return False


def request_response(client: OpenAI, *, model: str, input_items: list[dict] | str,
                     reasoning_effort: str = "high", max_output_tokens: int = 16384,
                     tools: list[dict] | None = None, tool_choice: Any = None,
                     text_format: dict | None = None, stream: bool = False,
                     max_attempts: int = 3, retry_delay: float = 3) -> dict:
    """Request raw Responses output. Append ALL output then function_call_output.

    We use store=false and explicitly request encrypted reasoning content so the
    caller can replay state without server-side persistence. Never reconstruct
    an assistant message from text alone in a tool loop.
    """
    if not is_gpt6_model(model) or reasoning_effort not in _EFFORTS:
        raise ResponsesError("GPT-6 requires an explicit supported model/effort", category="configuration")
    if max_output_tokens <= 0 or max_attempts <= 0:
        raise ResponsesError("Token budget and attempts must be positive", category="configuration")
    payload = {"model": model, "input": input_items, "reasoning": {"effort": reasoning_effort},
               "max_output_tokens": max_output_tokens, "store": False,
               "include": ["reasoning.encrypted_content"]}
    if tools is not None:
        payload["tools"] = responses_tools(tools)
        payload["parallel_tool_calls"] = False
    if tool_choice is not None:
        if isinstance(tool_choice, dict) and "function" in tool_choice:
            tool_choice = {"type": "function", "name": tool_choice["function"]["name"]}
        payload["tool_choice"] = tool_choice
    if text_format is not None:
        payload["text"] = {"format": text_format}
    if stream:
        payload["stream"] = True
    # Hard upper bound also protects callsites still passing legacy max_retries=20.
    attempts = min(int(max_attempts), 3)
    for attempt in range(1, attempts + 1):
        try:
            if stream:
                response = _read_stream(client.post("/responses", cast_to=dict[str, Any], body=payload,
                                                     stream=True, stream_cls=Stream[dict[str, Any]]))
            else:
                response = client.post("/responses", cast_to=dict[str, Any], body=payload)
            response = _validate_response(response, model, reasoning_effort)
            logger.info("Responses completed: %s", response_info(
                response, requested_model=model, reasoning_effort=reasoning_effort))
            return response
        except ResponsesError as exc:
            exc.attempts = attempt
            raise
        except Exception as exc:
            transient = _is_transient(exc)
            if not transient or attempt >= attempts:
                status = getattr(exc, "status_code", None)
                category = "transient_exhausted" if transient else "request_error"
                raise ResponsesError(
                    f"Responses request failed ({type(exc).__name__}, HTTP {status}, attempts={attempt})",
                    category=category, attempts=attempt, status_code=status) from exc
            logger.warning("Transient Responses failure (%s), retry %s/%s", type(exc).__name__, attempt, attempts)
            time.sleep(min(max(0, retry_delay) * (2 ** (attempt - 1)), 30))
    raise AssertionError("unreachable")
