"""Per-run LLM request metadata; excludes prompts, credentials and opaque state."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .responses import response_info

_LOCK = threading.Lock()
logger = logging.getLogger("MLEvolve")


def record_call(cfg, *, stage, role: str, requested_model: str, effort: str,
                max_output_tokens: int, elapsed: float, response: dict | None = None,
                error: BaseException | None = None, context_budget: dict | None = None) -> None:
    log_dir = getattr(cfg, "log_dir", None)
    if not log_dir:
        return
    info = response_info(response or {}, requested_model=requested_model, reasoning_effort=effort)
    endpoint = urlsplit(stage.base_url or "https://api.openai.com/v1")
    info.update(timestamp=datetime.now(timezone.utc).isoformat(), role=role,
                endpoint_host=endpoint.hostname, max_output_tokens=max_output_tokens,
                elapsed_seconds=round(elapsed, 3), context_budget=context_budget,
                outcome="error" if error else "success")
    if error is not None:
        info["error"] = {"type": type(error).__name__, "category": getattr(error, "category", None),
                         "http_status": getattr(error, "status_code", None),
                         "attempts": getattr(error, "attempts", None)}
        if getattr(error, "response_info", None):
            info["error"]["response_info"] = error.response_info
    try:
        with _LOCK:
            path = Path(log_dir) / "llm_calls.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(info, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # Logging must not discard a paid model response.
        logger.warning("Could not append LLM telemetry", exc_info=True)
