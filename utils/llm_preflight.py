"""Record effective model slots before generation, without revealing credentials."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

from engine.analogy.context import context_options

logger = logging.getLogger("MLEvolve")


def record_configuration(cfg):
    from llm.responses import uses_responses, supports_reasoning_effort
    context = context_options(getattr(cfg, "analogy", None))
    try:
        sdk = version("openai")
    except PackageNotFoundError:
        sdk = "unavailable"
    slots = {}
    for role in ("code", "feedback"):
        stage = getattr(cfg.agent, role)
        model = str(stage.model)
        effort = str(getattr(stage, "reasoning_effort", "high"))
        tokens = int(getattr(stage, "max_output_tokens", 16384))
        if uses_responses(model) and not supports_reasoning_effort(model, effort):
            raise ValueError(f"Responses {role} slot has unsupported reasoning_effort={effort!r}")
        if tokens <= 0:
            raise ValueError(f"{role}.max_output_tokens must be positive")
        endpoint = urlsplit(stage.base_url or "https://api.openai.com/v1")
        slots[role] = {"model": model, "reasoning_effort": effort, "max_output_tokens": tokens,
                       "endpoint_type": "responses" if uses_responses(model) else "legacy",
                       "endpoint_host": endpoint.hostname}
    slots["analogy"] = {**slots["code"], "max_output_tokens": int(getattr(cfg.analogy, "max_output_tokens", 16384))}
    if slots["analogy"]["max_output_tokens"] <= 0:
        raise ValueError("analogy.max_output_tokens must be positive")
    require_gpt6 = os.environ.get("MLEVOLVE_REQUIRE_GPT6", "0").lower() in {"1", "true", "yes"}
    required_model = os.environ.get("MLEVOLVE_REQUIRED_MODEL") or None
    required_effort = os.environ.get("MLEVOLVE_REQUIRED_REASONING_EFFORT", "high")
    if require_gpt6:
        if required_model not in {None, "gpt-6-astra"} or required_effort != "high":
            raise ValueError("Conflicting legacy GPT-6 and explicit experiment model requirements")
        required_model = "gpt-6-astra"
    if required_model and any(slot["model"] != required_model or slot["reasoning_effort"] != required_effort
                              for slot in slots.values()):
        raise ValueError(f"This experiment requires {required_model}/{required_effort} in code, feedback and analogy; "
                         "check Secret/env/dotlist overrides")
    if required_model and context.version != 2:
        raise ValueError("This experiment requires analogy.context.version=2")
    effective_input_cap = min(context.max_input_tokens, context.endpoint_context_tokens -
                              slots["analogy"]["max_output_tokens"] - context.input_safety_tokens)
    if effective_input_cap <= context.final_report_reserve_tokens:
        raise ValueError("analogy context window cannot reserve a final report")
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "openai_sdk_version": sdk,
              "slots": slots, "context": asdict(context), "effective_analogy_input_cap": effective_input_cap,
              "require_gpt6": require_gpt6, "required_model": required_model,
              "required_reasoning_effort": required_effort if required_model else None,
              "live_compatibility_checked_here": False,
              "note": "Effective configuration only; actual returned model/usage appears in per-call telemetry."}
    path = Path(cfg.log_dir) / "llm_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("LLM configuration: %s (context v%s, input cap %s)", slots, context.version, effective_input_cap)
    return record
