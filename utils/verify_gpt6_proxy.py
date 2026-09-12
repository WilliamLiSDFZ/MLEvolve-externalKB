#!/usr/bin/env python3
"""Small, standalone GPT-6 Responses compatibility probe (Python 3.10+, stdlib only).

Uses LLM_BASE_URL and LLM_API_KEY, as the experiment Jobs do. Deliberately ignores
LLM_MODEL: a stale Secret must not turn this into a GPT-5 test. No model fallback,
automatic retries, package installs, training, or changes to the shared proxy.

Example (credentials supplied through the environment, never command arguments):
    python utils/verify_gpt6_proxy.py --output-dir /tmp/gpt6-proxy-test

Seven short generation requests on a successful run, plus GET /models. Defaults:
gpt-6-astra, reasoning high, 8192 output tokens/request (including reasoning),
120 seconds/request. Exit 0 = all required checks passed; 1 = at least one failed;
2 = configuration/report error. The model inventory is advisory, not a pass gate.
Each case checkpoints report.json and prints a result; final JSON is also in logs.

This tests the HTTP protocol proposed for migration, NOT the existing MLEvolve
SDK wrappers, throughput, or long-context quality. Returned model names/effort are
proxy-reported evidence, not independent proof of the upstream model or effort.
The tools use synthetic data only and never read or execute candidate code.

Protocol references:
https://developers.openai.com/api/docs/guides/latest-model
https://developers.openai.com/api/docs/guides/function-calling
https://developers.openai.com/api/docs/guides/streaming-responses
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ProbeError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def redact(value, api_key):
    text = str(value).replace(api_key, "[REDACTED]") if api_key else str(value)
    text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    return re.sub(
        r'(?i)("(?:api_key|access_token|refresh_token|authorization|encrypted_content)"\s*:\s*")[^"]*',
        r'\1[REDACTED]', text,
    )


@contextmanager
def request_deadline(seconds):
    # In the Job (Linux) and on macOS this also bounds slow/dripping SSE streams,
    # not just socket inactivity. The command runs on the main thread.
    def expired(signum, frame):
        raise TimeoutError(f"request exceeded {seconds:g}s wall-clock deadline")

    if not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def output_text(response):
    return "".join(
        part.get("text", "")
        for item in response.get("output", []) if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    )


def read_sse(stream, evidence, started):
    """Read actual Responses SSE; a 200, [DONE], or partial delta is insufficient."""
    pending, deltas, events = [], [], Counter()
    completed = None
    done = False
    total_bytes = 0

    def consume():
        nonlocal completed, done
        if not pending:
            return
        data = "\n".join(pending)
        pending.clear()
        if data == "[DONE]":
            done = True
            return
        event = json.loads(data)
        kind = event.get("type", "unknown")
        events[kind] += 1
        evidence["sse_events"] = dict(events)
        if kind in {"error", "response.failed", "response.incomplete"}:
            raise ProbeError(f"SSE {kind}: {json.dumps(event)[:1200]}")
        if kind == "response.output_text.delta":
            evidence.setdefault("first_text_delta_seconds", round(time.monotonic() - started, 3))
            deltas.append(event.get("delta", ""))
        if kind == "response.completed":
            require(completed is None, "duplicate response.completed event")
            completed = event.get("response")

    for raw in stream:
        total_bytes += len(raw)
        require(total_bytes <= 8 * 1024 * 1024, "SSE exceeded the probe's 8 MiB safety limit")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            consume()
            # A terminal event is enough; some proxies keep HTTP open afterward.
            if completed is not None or done:
                break
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip(" "))
    consume()
    require(isinstance(completed, dict), "SSE ended without response.completed")
    require(bool(deltas) and bool("".join(deltas)), "SSE contained no text deltas")
    require("".join(deltas) == output_text(completed), "SSE deltas differ from completed output")
    return completed


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProbeError(f"HTTP redirect {code}; configure the final API base URL explicitly")


class Probe:
    def __init__(self, args, api_key):
        self.args, self.api_key = args, api_key
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = Path(args.output_dir) / f"{stamp}-{secrets.token_hex(3)}" / "report.json"
        self.path.parent.mkdir(parents=True, exist_ok=False)
        self.opener = urllib.request.build_opener(NoRedirect())
        self.case = None
        self.report = {
            "started_at": stamp, "status": "running", "base_url": args.base_url,
            "requested_model": args.model, "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens, "timeout_seconds": args.timeout,
            "transport": "stdlib HTTP /v1/responses (not MLEvolve SDK integration)",
            "python_version": sys.version.split()[0],
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": [], "warnings": [],
        }
        self.save()

    def warn(self, message):
        if message not in self.report["warnings"]:
            self.report["warnings"].append(message)

    def save(self):
        encoded = redact(json.dumps(self.report, ensure_ascii=False, indent=2), self.api_key)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(encoded + "\n", encoding="utf-8")
        temp.replace(self.path)

    def run_case(self, name, operation, *, required=True):
        self.case = {"name": name, "required": required, "status": "running", "requests": []}
        self.report["cases"].append(self.case)
        self.save()
        print(f"[RUN ] {name}", flush=True)
        started = time.monotonic()
        try:
            details = operation()
            self.case.update(status="pass", details=details)
        except Exception as exc:
            self.case.update(status="fail" if required else "warn",
                             error=redact(f"{type(exc).__name__}: {exc}", self.api_key)[:2000])
        self.case["elapsed_seconds"] = round(time.monotonic() - started, 3)
        self.save()
        print(f"[{self.case['status'].upper():4}] {name} ({self.case['elapsed_seconds']}s) "
              f"{self.case.get('error', '')}", flush=True)

    def request(self, path, payload=None, *, stream=False):
        info = {"path": path, "stream": stream}
        self.case["requests"].append(info)
        started = time.monotonic()
        req = urllib.request.Request(
            self.args.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "Accept": "text/event-stream" if stream else "application/json"},
        )
        try:
            with request_deadline(self.args.timeout):
                with self.opener.open(req, timeout=self.args.timeout) as response:
                    info.update(http_status=response.status, request_id=response.headers.get("x-request-id"))
                    if stream:
                        require("text/event-stream" in response.headers.get("Content-Type", ""),
                                "stream=True returned non-SSE Content-Type")
                        result = read_sse(response, info, started)
                    else:
                        raw = response.read(8 * 1024 * 1024 + 1)
                        require(len(raw) <= 8 * 1024 * 1024, "response exceeded 8 MiB")
                        result = json.loads(raw)
            if path == "/responses":
                self.validate_response(result, info)
            return result
        except urllib.error.HTTPError as exc:
            info["http_status"] = exc.code
            # Keep the error read bounded too; even an upstream error can stall.
            with request_deadline(min(self.args.timeout, 10)):
                body = exc.read(4096).decode(errors="replace")
            hint = {401: "authentication", 403: "access denied", 404: "route/model unavailable",
                    429: "rate limit/quota"}.get(exc.code, "request or upstream error")
            raise ProbeError(f"HTTP {exc.code} ({hint}): {redact(body, self.api_key)[:1200]}") from None
        finally:
            info["elapsed_seconds"] = round(time.monotonic() - started, 3)

    def validate_response(self, response, info):
        require(isinstance(response, dict), "Responses body is not an object")
        info.update(response_id=response.get("id"), returned_model=response.get("model"),
                    status=response.get("status"), usage=response.get("usage"),
                    returned_reasoning=response.get("reasoning"),
                    incomplete_details=response.get("incomplete_details"))
        require(response.get("status") == "completed", f"response not completed: {info}")
        require(not response.get("error"), f"response.error: {response.get('error')}")
        returned_model = response.get("model") or ""
        # Accept only the exact requested name or its dated snapshot; never another family.
        require(re.fullmatch(re.escape(self.args.model) + r"(?:-\d{4}-\d{2}-\d{2})?", returned_model),
                f"model mismatch: requested {self.args.model!r}, returned {returned_model!r}")
        effort = (response.get("reasoning") or {}).get("effort")
        require(effort in {None, self.args.reasoning_effort}, f"reasoning effort mismatch: {effort!r}")
        if effort is None:
            self.warn("Proxy did not echo reasoning.effort; acceptance does not prove it was applied.")
        if not response.get("usage"):
            self.warn("Some responses omit usage; token accounting is not verified.")
        output = response.get("output")
        require(isinstance(output, list) and all(isinstance(item, dict) for item in output),
                "missing/malformed response.output")
        info["output_types"] = [item.get("type") for item in output]
        for item in output:
            for content in item.get("content", []) if item.get("type") == "message" else []:
                require(content.get("type") != "refusal", "model returned a refusal")

    def respond(self, input_items, **extra):
        payload = {"model": self.args.model, "input": input_items,
                   "instructions": "Follow this synthetic API compatibility test precisely. Keep replies short.",
                   "reasoning": {"effort": self.args.reasoning_effort},
                   "max_output_tokens": self.args.max_output_tokens, "store": False}
        payload.update(extra)
        return self.request("/responses", payload, stream=bool(payload.get("stream")))

    def inventory(self):
        models = self.request("/models")
        ids = [item.get("id", "") for item in models.get("data", [])]
        if self.args.model not in ids:
            self.warn("Requested model absent from /models; real Responses checks are authoritative.")
        return {"model_count": len(ids), "requested_model_listed": self.args.model in ids,
                "gpt6_models": [name for name in ids if "gpt-6" in name.lower()]}

    def text(self, *, stream=False):
        marker = "GPT6_OK_" + secrets.token_hex(6)
        response = self.respond(f"Reply with exactly {marker}, with no other text.", stream=stream)
        text = output_text(response).strip()
        require(text == marker, f"text mismatch: {text[:200]!r}")
        return {"exact_marker_matched": True}

    def structured(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"},
                  "value": {"type": "integer"}}, "required": ["ok", "value"],
                  "additionalProperties": False}
        # Conflict is intentional: if the proxy silently strips the schema and
        # returns free text, this should fail rather than pass by prompt obedience.
        response = self.respond("Reply with the plain word SCHEMA_IGNORED if permitted. "
                                "If a JSON schema constrains the output, use ok=true and value=42.",
                                text={"format": {"type": "json_schema", "name": "proxy_probe",
                                                 "strict": True, "schema": schema}})
        value = json.loads(output_text(response))
        require(isinstance(value, dict) and set(value) == {"ok", "value"}
                and value["ok"] is True and type(value["value"]) is int and value["value"] == 42,
                f"JSON schema/content mismatch: {value!r}")
        return {"strict_json_valid": True}

    @staticmethod
    def function(name, parameter, *, strict=True):
        parameters = {"type": "object", "properties": {parameter: {"type": "string"}},
                      "required": [parameter], "additionalProperties": False}
        if not strict:
            parameters["properties"]["optional_note"] = {"type": "string"}
        return {"type": "function", "name": name, "description": f"Synthetic test: {name}.",
                "parameters": parameters, "strict": strict}

    @staticmethod
    def function_call(response, name, expected):
        calls = [item for item in response["output"] if item.get("type") == "function_call"]
        require(len(calls) == 1, f"expected one function call; received {len(calls)}")
        call = calls[0]
        require(call.get("name") == name, f"wrong function: {call.get('name')!r}")
        require(isinstance(call.get("call_id"), str) and bool(call["call_id"]), "missing call_id")
        args = json.loads(call.get("arguments", ""))
        require(args == expected, f"function arguments mismatch: {args!r}")
        return call

    def legacy_tool(self):
        tool = self.function("submit_probe", "marker", strict=False)
        marker = secrets.token_hex(6)
        response = self.respond(f"Call submit_probe with marker={marker}. Omit optional_note.",
                                tools=[tool], tool_choice={"type": "function", "name": tool["name"]},
                                parallel_tool_calls=False)
        self.function_call(response, "submit_probe", {"marker": marker})
        return {"non_strict_optional_field_omitted": True}

    def tool_loop(self):
        tools = [self.function("candidate_code_index", "node_id"),
                 self.function("read_candidate_code", "symbol")]
        history = [{"role": "user", "content": "First call candidate_code_index with node_id=current; "
                    "then read_candidate_code with the exact symbol returned by the index. "
                    "Finally reply with only the marker returned by read_candidate_code."}]
        symbol, marker = "probe_" + secrets.token_hex(6), "TOOL_OK_" + secrets.token_hex(6)
        reasoning_items = encrypted_items = 0
        call_ids = []
        stages = [("candidate_code_index", {"node_id": "current"}, {"symbol": symbol}),
                  ("read_candidate_code", {"symbol": symbol}, {"marker": marker})]
        for name, expected, result in stages:
            response = self.respond(history, tools=tools, parallel_tool_calls=False,
                                    tool_choice={"type": "function", "name": name},
                                    include=["reasoning.encrypted_content"])
            call = self.function_call(response, name, expected)
            require(call["call_id"] not in call_ids, "call_id reused across tool turns")
            call_ids.append(call["call_id"])
            # Preserve ALL output items, including opaque reasoning state. Never
            # log encrypted state or replace these with assistant text only.
            reasoning_items += sum(item.get("type") == "reasoning" for item in response["output"])
            encrypted_items += sum(bool(item.get("encrypted_content")) for item in response["output"])
            history.extend(response["output"])
            history.append({"type": "function_call_output", "call_id": call["call_id"],
                            "output": json.dumps(result)})
        response = self.respond(history, tools=tools, tool_choice="none",
                                include=["reasoning.encrypted_content"])
        require(output_text(response).strip() == marker, "final response did not consume the tool result")
        if reasoning_items == 0 or encrypted_items == 0:
            self.warn("Tool continuation worked, but no encrypted reasoning was returned; opaque reasoning "
                      "state replay is not fully verified.")
        return {"model_turns": 3, "call_ids": call_ids, "tool_result_matched": True,
                "continuation": "store=false, full output replay with function_call_output",
                "reasoning_items_replayed": reasoning_items, "encrypted_items_replayed": encrypted_items}

    def run(self):
        print(f"Target: {self.args.base_url} model={self.args.model} reasoning={self.args.reasoning_effort}",
              flush=True)
        print(f"Report: {self.path}", flush=True)
        self.run_case("models_inventory", self.inventory, required=False)
        self.run_case("responses_text", self.text)
        self.run_case("responses_strict_json", self.structured)
        self.run_case("responses_sse", lambda: self.text(stream=True))
        self.run_case("responses_non_strict_function", self.legacy_tool)
        self.run_case("responses_two_tool_rounds", self.tool_loop)
        failed = [case["name"] for case in self.report["cases"] if case["status"] == "fail"]
        self.report.update(status="fail" if failed else "pass", failed_checks=failed,
                           finished_at=datetime.now(timezone.utc).isoformat())
        self.save()
        print("\n=== GPT-6 PROXY REPORT ===", flush=True)
        print(redact(json.dumps(self.report, ensure_ascii=False, indent=2), self.api_key), flush=True)
        print(f"\n{'FAIL' if failed else 'PASS'}: HTTP compatibility probe; "
              "MLEvolve SDK migration and sustained-load compatibility remain untested.", flush=True)
        return int(bool(failed))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("TEST_MODEL", "gpt-6-astra"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("TEST_REASONING_EFFORT", "high"),
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", default="/tmp/gpt6-proxy-test")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    url = urllib.parse.urlsplit(args.base_url)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        parser.error("set LLM_BASE_URL to the API root (e.g. http://cliproxy:8317/v1), without credentials/query")
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        parser.error("LLM_API_KEY is missing; use the mlevolve-llm-proxy Secret")
    if not args.model.startswith("gpt-6"):
        parser.error("this probe requires an explicit gpt-6 model; no automatic fallback")
    if args.timeout <= 0 or args.max_output_tokens <= 0:
        parser.error("timeout and max-output-tokens must be positive")
    try:
        return Probe(args, api_key).run()
    except Exception as exc:
        print(redact(f"FATAL: {type(exc).__name__}: {exc}", api_key), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
