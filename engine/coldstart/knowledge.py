"""Build guidance description for agent from task/model JSON."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("MLEvolve")

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"
METHODOLOGY_MAP_JSON = Path(__file__).resolve().parent / "methodology_map.json"

# Techniques are joined by this separator by every builder (methodology_agent._render,
# ondemand._assemble / _assemble_techniques). Truncation cuts on it so a budget never
# splits a technique mid-sentence.
TECHNIQUE_SEPARATOR = "\n\n---\n\n"

LOG_PREVIEW_HEAD_LINES = 8      # the first 3 are the fixed section header, so this shows ~5
LOG_PREVIEW_TAIL_LINES = 4
LOG_PREVIEW_LINE_CHARS = 160    # technique bodies are prose paragraphs on one line


def preview_text(text: str, head: int = LOG_PREVIEW_HEAD_LINES,
                 tail: int = LOG_PREVIEW_TAIL_LINES,
                 width: int = LOG_PREVIEW_LINE_CHARS) -> str:
    """Render text as its first and last few non-blank lines, with an elision marker.

    Logging the injected knowledge in full floods the run log with thousands of characters at
    every call; logging only its length — which is what this module did after the guidance
    split — makes it impossible to tell afterwards *which* techniques were selected, and that
    is the one thing worth knowing when a run's results are being interpreted. Head plus tail
    identifies the content and shows that it terminated cleanly, and the per-line cap keeps a
    single prose paragraph from undoing the point of eliding.
    """
    if not text or not text.strip():
        return "(empty)"

    def _clip(ln: str) -> str:
        return ln if len(ln) <= width else ln[:width] + " …"

    lines = [_clip(ln) for ln in text.splitlines() if ln.strip()]
    if len(lines) <= head + tail:
        return "\n".join(lines)
    omitted = len(lines) - head - tail
    return "\n".join(
        lines[:head]
        + [f"    ... [{omitted} more lines, {len(text)} chars total] ..."]
        + lines[-tail:]
    )


def text_digest(text: str) -> str:
    """Short stable id for a block of injected text.

    Lets two arms of a paired run be compared with `grep digest` — identical digests confirm
    both arms retrieved the same knowledge, which the experiment design assumes but has so far
    only been inferred from matching candidate counts.
    """
    import hashlib
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:8]


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_models_for_task(
    task_name: str, tasks: Dict, models: Dict
) -> List[Dict[str, str]]:
    """Match model list for task from knowledge by task name."""
    if task_name not in tasks:
        return []
    category = tasks[task_name]  # flat string: "General Image", "NLP", etc.
    if category not in models:
        return []
    matched = []
    for m_name, m_info in models[category].items():
        matched.append({
            "model_name": m_name,
            "description": m_info.get("Description", ""),
            "code_template": m_info.get("Code_template", ""),
        })
    return matched


def _build_guidance_text(task_name: str, tasks: Dict, models: Dict) -> str:
    """Build guidance text from task name and knowledge."""
    model_list = collect_models_for_task(task_name, tasks, models)
    if not model_list:
        return "None model"
    lines = []
    for i, m in enumerate(model_list):
        lines.append(f"\nModel{i+1}: {m['model_name']}\n")
        lines.append(f"Description:{m['description']}\n")
        lines.append("Code template (MUST copy exactly — do NOT change model variant names or file paths):\n```python\n" + m["code_template"] + "\n```")
    return "\n".join(lines)


def get_init_solution_paths(exp_id: str) -> List[str]:
    """Load init solution paths for exp_id from engine/coldstart/init_solution_paths.json."""
    if not INIT_SOLUTION_JSON.exists():
        return []
    try:
        data = _load_json(str(INIT_SOLUTION_JSON))
        paths = data.get(exp_id)
        if isinstance(paths, list):
            return [str(p) for p in paths if p]
        return []
    except Exception:
        return []


def _extract_positive_sections(text: str) -> List[str]:
    """Extract ## [POSITIVE] sections from a *_methodology.md file."""
    sections = []
    pattern = re.compile(r'^## \[POSITIVE\] (.+?)$\n(.*?)(?=^## \[|^# [^#]|\Z)', re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        title = match.group(1).strip()
        body = match.group(2).strip()
        sections.append(f"**[POSITIVE] {title}**\n{body}")
    return sections


def _build_methodology_text(task_name: str, methodology_kb_path: str) -> str:
    """Static path: extract only [POSITIVE] entries for task_name via methodology_map.json.

    No-ops (returns "") when methodology_map.json is absent, so static mode is opt-in.
    """
    if not METHODOLOGY_MAP_JSON.exists():
        return ""
    try:
        mapping = _load_json(str(METHODOLOGY_MAP_JSON))
    except Exception:
        return ""

    folders = mapping.get(task_name, [])
    if not folders:
        return ""

    kb_base = Path(methodology_kb_path)
    all_entries = []
    for folder in folders:
        parts = folder.split("/", 1)
        if len(parts) != 2:
            continue
        venue_year, category = parts
        cat_dir = kb_base / venue_year / category
        if not cat_dir.exists():
            continue
        for md_file in sorted(cat_dir.glob("*_methodology.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            all_entries.extend(_extract_positive_sections(text))

    if not all_entries:
        return ""

    return (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following actionable techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(all_entries)
    )


def trim_methodology_text(text: str, token_budget: int) -> str:
    """Trim assembled technique text to a token budget, cutting on technique boundaries.

    The header is preserved and whole techniques are dropped from the end; a partial
    technique is never emitted. Returns "" for empty input.
    """
    if not text or not text.strip():
        return ""
    budget_chars = max(1, int(token_budget)) * 4      # ~4 chars/token
    if len(text) <= budget_chars:
        return text

    parts = text.split(TECHNIQUE_SEPARATOR)
    kept, used = [], 0
    for part in parts:
        add = len(part) + (len(TECHNIQUE_SEPARATOR) if kept else 0)
        if kept and used + add > budget_chars:
            break
        kept.append(part)
        used += add
    if len(kept) <= 1 and len(parts) > 1:
        # The header block alone already exceeds the budget: keep header + first technique
        # rather than returning a bare header with no content.
        kept = parts[:2]
    return TECHNIQUE_SEPARATOR.join(kept)


def build_guidance_description(cfg: Any, task_desc: str = "") -> str:
    """Return the PRETRAINED-MODEL guidance, and stash literature techniques on the config.

    The two used to be concatenated into one string. That was a mislabelling bug: draft_agent
    renders this return value inside a block headed "Pretrained Model Strategy" whose
    Option A is annotated "SOTA models with proven performance" and followed by "you MUST copy
    the Code template EXACTLY". Appending paper-derived techniques there presented them to the
    model as pretrained-model recommendations under an instruction that does not apply to them.

    They are now kept apart: the return value is the model guidance only, and the retrieved
    techniques go to `cfg.coldstart.methodology_text` for callers to render under their own
    heading. Consumers: draft_agent (its own section) and, when
    `coldstart.inject_into_improve` is set, improve_agent.
    """
    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    text = _build_guidance_text(cfg.exp_id, tasks, models)
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))

    # Methodology / literature KB retrieval (opt-in: only when methodology_kb_path is set).
    # Mode: "vector" (semantic retrieval) | "llm" (category match) | "static" (methodology_map.json)
    #     | "lazy" (abstract index + on-demand extraction)
    methodology_text = ""
    methodology_kb_path = getattr(cfg, "methodology_kb_path", "") or ""
    if methodology_kb_path:
        mode = str(getattr(cfg, "methodology_retrieval", "vector")).lower()
        if mode == "static" or not task_desc:
            methodology_text = _build_methodology_text(cfg.exp_id, methodology_kb_path)
        else:
            from engine.coldstart.methodology_agent import build_methodology_guidance
            methodology_text = build_methodology_guidance(task_desc, methodology_kb_path, cfg)

    try:
        cfg.coldstart.methodology_text = methodology_text or ""
    except Exception:  # pragma: no cover - config objects are OmegaConf/dataclass, both settable
        logger.warning("could not store methodology_text on cfg.coldstart; "
                       "literature techniques will not be injected")

    if methodology_text:
        logger.info("Knowledge injected at draft: %d chars, digest %s\n%s",
                    len(methodology_text), text_digest(methodology_text),
                    preview_text(methodology_text))
        # Also dump the FULL text next to the journal. The log preview is elided to keep the
        # log readable, which means a finished run does not record what knowledge it actually
        # received — and without that, "did the agent use the retrieved techniques?" cannot be
        # answered afterwards from the run directory alone. It is a few KB, write-only, and it
        # makes every run self-documenting. Failure here must never affect the run.
        try:
            log_dir = Path(getattr(cfg, "log_dir", "") or ".")
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "injected_knowledge.md").write_text(methodology_text, encoding="utf-8")
        except Exception as e:  # pragma: no cover - diagnostics must not break a 12 h run
            logger.warning("could not write injected_knowledge.md: %s", e)
    elif methodology_kb_path:
        logger.info("Knowledge injected at draft: NOTHING (kb path set but retrieval "
                    "returned empty) — this arm is running as an expensive baseline")

    return text
