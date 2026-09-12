#!/usr/bin/env python3
"""Verify GPT-5.6 Sol/high through the actual MLEvolve Responses wrappers.

Run on CPU: python utils/verify_sol_proxy.py --output-dir /tmp/sol-proxy-test
Credentials come from LLM_API_KEY/LLM_BASE_URL, or --config pointing to an existing
run's saved config. Never pass credentials in arguments. Six short requests on a
successful run test streaming text, JSON, feedback and two synthetic tool rounds.
No model fallback, real candidate tools, training or shared-config writes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm
from llm.responses import (function_tool, make_response_client, request_response,
                           response_function_calls, response_text, response_usage)
from utils.verify_gpt6_proxy import redact, request_deadline

MODEL = "gpt-5.6-sol"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/sol-proxy-test"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    saved = None
    if args.config:
        from omegaconf import OmegaConf
        saved = OmegaConf.load(args.config).agent.code
    key = os.environ.get("LLM_API_KEY") or (saved.api_key if saved else "")
    base = args.base_url or os.environ.get("LLM_BASE_URL") or (saved.base_url if saved else "")
    if not key or not base:
        parser.error("Supply LLM_API_KEY/LLM_BASE_URL or a saved --config with credentials")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = NS(model=MODEL, api_key=key, base_url=base, reasoning_effort="high", max_output_tokens=8192)
    cfg = NS(agent=NS(code=stage, feedback=NS(**vars(stage))), log_dir=args.output_dir)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "marker": {"type": "string"}},
              "required": ["ok", "marker"], "additionalProperties": False}
    marker = "sol-" + secrets.token_hex(4)
    report = {"model": MODEL, "reasoning_effort": "high", "cases": [], "status": "running"}
    logging.getLogger("MLEvolve").setLevel(logging.ERROR)

    def save():
        (args.output_dir/"report.json").write_text(redact(json.dumps(report, indent=2), key) + "\n")

    def run(name, operation):
        started = time.monotonic()
        try:
            with request_deadline(args.timeout):
                details = operation()
            result = dict(name=name, status="pass", details=details)
        except Exception as exc:
            result = dict(name=name, status="fail", error=redact(f"{type(exc).__name__}: {exc}", key)[:1000])
        result["seconds"] = round(time.monotonic() - started, 3)
        report["cases"].append(result)
        save()
        print(json.dumps(result), flush=True)

    def text_case():
        text = llm.generate(f"Reply with exactly {marker} and nothing else.", cfg, max_retries=3)
        check(text.strip() == marker, "Text stream did not match the marker")
        return {"wrapper": "llm.generate", "marker_matched": True}

    def json_case():
        value = json.loads(llm.generate(f'Return JSON with ok=true and marker="{marker}".', cfg, json_schema=schema))
        check(value == {"ok": True, "marker": marker}, "Structured result mismatch")
        return {"wrapper": "llm.generate/json_schema", "json_matched": True}

    def feedback_case():
        spec = llm.FunctionSpec(name="submit_check", description="Submit the compatibility result", json_schema=schema)
        value = llm.query("Use submit_check to return the requested result.", f'ok=true, marker="{marker}"',
                          model=MODEL, cfg=cfg, role="feedback", func_spec=spec)
        check(value == {"ok": True, "marker": marker}, "Feedback function result mismatch")
        return {"wrapper": "llm.query/feedback", "function_matched": True}

    def tool_case():
        history = [{"role": "user", "content": "Call read_fixture with key=blue, then verify_fixture with the returned "
                    "marker, then return exactly the verified marker. This is a synthetic test."}]
        tools = [function_tool("read_fixture", "Read synthetic data", {"type": "object", "properties": {
                 "key": {"type": "string"}}, "required": ["key"]}),
                 function_tool("verify_fixture", "Verify the returned marker", {"type": "object", "properties": {
                 "marker": {"type": "string"}}, "required": ["marker"]})]
        calls, tokens, opaque = [], [], 0
        with make_response_client(stage, timeout=args.timeout) as client:
            for name, expected in [("read_fixture", {"key": "blue"}), ("verify_fixture", {"marker": marker})]:
                response = request_response(client, model=MODEL, reasoning_effort="high", max_output_tokens=8192,
                    input_items=history, tools=tools, tool_choice={"type": "function", "name": name})
                tc = response_function_calls(response)
                check(len(tc) == 1 and tc[0]["name"] == name, "Expected one matching tool call")
                check(json.loads(tc[0]["arguments"]) == expected, "Tool arguments did not consume earlier context")
                check(tc[0]["call_id"] not in calls, "Tool call ID reused")
                calls.append(tc[0]["call_id"])
                tokens.append(response_usage(response))
                opaque += sum(bool(item.get("encrypted_content")) for item in response["output"])
                history.extend(response["output"])
                history.append({"type": "function_call_output", "call_id": tc[0]["call_id"],
                                "output": json.dumps({"marker": marker, "verified": name == "verify_fixture"})})
            final = request_response(client, model=MODEL, reasoning_effort="high", max_output_tokens=8192,
                                     input_items=history, tools=tools, tool_choice="none")
            check(response_text(final).strip() == marker, "Final answer did not consume the tool results")
            tokens.append(response_usage(final))
        return {"model_turns": 3, "tool_rounds": len(calls), "encrypted_items_replayed": opaque,
                "input_tokens": sum(x[0] for x in tokens), "output_tokens": sum(x[1] for x in tokens)}

    for name, operation in [("text_stream", text_case), ("structured_stream", json_case),
                             ("feedback_function", feedback_case), ("two_tool_rounds", tool_case)]:
        run(name, operation)
        if report["cases"][-1]["status"] == "fail":
            break  # Do not spend further calls when this endpoint/model already failed.
    report["status"] = "pass" if len(report["cases"]) == 4 and all(
        x["status"] == "pass" for x in report["cases"]) else "fail"
    save()
    print("SOL_PROBE=" + json.dumps(report), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
