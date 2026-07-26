"""Model-specific parameter profiles for OpenAI-compatible backends.

Usage:
    profile = get_profile(model_name, use_thinking=True)
    # Returns a dict with any subset of:
    #   temperature, top_p, presence_penalty  — standard OpenAI Chat params
    #   top_k, enable_thinking                — go into extra_body (Qwen-specific)

    thinking_extra = get_thinking_extra_body(model_name)
    # Returns model-specific extra_body params for enabling thinking mode

To add a new model family, add an entry to _PROFILES below.
Each entry has two modes: "thinking" and "non_thinking".
Only include params that differ from provider defaults — missing keys are skipped.
Longer prefixes take precedence (e.g. "gpt-4o" wins over "gpt").
"""

from __future__ import annotations

_PROFILES: dict[str, dict] = {
    # ── Qwen series ──────────────────────────────────────────────────────
    "qwen": {
        "thinking": {
            # Precise coding tasks
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0.0, "enable_thinking": True,
        },
        "non_thinking": {
            # General tasks (used for planner / structured output)
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "presence_penalty": 1.5, "enable_thinking": False,
        },
    },

    # ── GPT series ───
    "gpt": {
        "thinking": {
            "temperature": 1.0,
        },
        "non_thinking": {
            "temperature": 0.7,
        },
    },

    # ── Kimi series (K2.5, K2.6) ───
    # Kimi reasoning models only allow temperature=1
    "kimi": {
        "thinking": {
            "temperature": 1.0, "top_p": 0.95,
        },
        "non_thinking": {
            "temperature": 1.0, "top_p": 0.95,
        },
    },

    # ── DeepSeek series (V4-pro, V4-flash) ───
    "deepseek": {
        "thinking": {
            "temperature": 1.0,
        },
        "non_thinking": {
            "temperature": 1.0,
        },
    },

    # ── Claude series (Opus 4.6/4.7, Sonnet 4.6) ───
    # Adaptive thinking is the recommended mode on Opus 4.6+/Sonnet 4.6+;
    # required on Opus 4.7. No budget_tokens needed.
    "claude": {
        "thinking": {
            "temperature": 1.0,
        },
        "non_thinking": {
            "temperature": 1.0,
        },
    },

    # ── Fallback for any unrecognised model ──────────────────────────────────
    "default": {
        "thinking":     {},
        "non_thinking": {},
    },
}

# Model-specific extra_body params for enabling thinking/reasoning mode.
# Synced from agentic-mle llm_client.py _MODEL_CONFIGS.
_THINKING_EXTRA_BODY: dict[str, dict] = {
    "qwen":     {"enable_thinking": True},
    "kimi":     {},                          # Kimi enables thinking by default
    "deepseek": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "gpt":      {},
    # Claude Opus 4.6/4.7 + Sonnet 4.6: adaptive thinking is the recommended
    # mode (required on Opus 4.7). Auto-enables interleaved thinking.
    "claude":   {"thinking": {"type": "adaptive"}},
}

# Models that only support {"type": "json_object"}, not json_schema + strict.
_NO_JSON_SCHEMA_PREFIXES = ("deepseek",)

# Models where thinking mode and json_schema are mutually exclusive.
# generate() will drop json_schema for these models to keep thinking enabled,
# relying on prompt instructions + post-processing for JSON extraction.
_THINKING_JSON_INCOMPATIBLE = ("qwen",)

# Models that don't support tool_choice="required" / specific function targeting.
# Claude with extended thinking only supports tool_choice="auto" or "none";
# specific tool name will return a 400 error.
_NO_TOOL_CHOICE_REQUIRED_PREFIXES = ("kimi", "deepseek", "claude")

# Models that REJECT sampling params (temperature / top_p / presence_penalty) outright —
# sending them returns a 400. Sampling params were removed on Claude Opus 4.7+ and Fable 5.
_NO_SAMPLING_PARAMS_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8", "claude-fable", "fable")


def normalize_model_name(model_name: str) -> str:
    """Lowercase and drop any provider prefix.

    Gateways namespace models (e.g. OpenRouter's 'anthropic/claude-opus-4-8'), which
    would otherwise defeat every startswith() check below.
    """
    return (model_name or "").lower().rsplit("/", 1)[-1]


def thinking_json_incompatible(model_name: str) -> bool:
    """Return True for models that cannot use thinking + json_schema simultaneously."""
    name = normalize_model_name(model_name)
    return any(name.startswith(p) for p in _THINKING_JSON_INCOMPATIBLE)


def supports_json_schema(model_name: str) -> bool:
    """Return False for models that require json_object instead of json_schema+strict."""
    name = normalize_model_name(model_name)
    return not any(name.startswith(p) for p in _NO_JSON_SCHEMA_PREFIXES)


def supports_tool_choice_required(model_name: str) -> bool:
    """Return False for models that don't support tool_choice=required."""
    name = normalize_model_name(model_name)
    return not any(name.startswith(p) for p in _NO_TOOL_CHOICE_REQUIRED_PREFIXES)


def supports_sampling_params(model_name: str) -> bool:
    """Return False for models that 400 when temperature/top_p/etc. are sent."""
    name = normalize_model_name(model_name)
    return not any(name.startswith(p) for p in _NO_SAMPLING_PARAMS_PREFIXES)


def get_thinking_extra_body(model_name: str) -> dict:
    """Return model-specific extra_body params for thinking mode (synced from agentic-mle)."""
    name = normalize_model_name(model_name)
    for key in sorted(_THINKING_EXTRA_BODY, key=len, reverse=True):
        if name.startswith(key):
            return dict(_THINKING_EXTRA_BODY[key])
    return {}


def get_profile(model_name: str, use_thinking: bool = True) -> dict:
    """Return parameter dict for model_name.

    Matches by longest prefix (case-insensitive). Falls back to 'default'.
    """
    name = normalize_model_name(model_name)
    for key in sorted(_PROFILES, key=len, reverse=True):
        if key == "default":
            continue
        if name.startswith(key):
            mode = "thinking" if use_thinking else "non_thinking"
            return dict(_PROFILES[key][mode])
    mode = "thinking" if use_thinking else "non_thinking"
    return dict(_PROFILES["default"][mode])
