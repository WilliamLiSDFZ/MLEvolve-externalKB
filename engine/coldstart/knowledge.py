"""Build guidance description for agent from task/model JSON."""
import json
import re
from pathlib import Path
from typing import Dict, List, Any

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"
METHODOLOGY_MAP_JSON = Path(__file__).resolve().parent / "methodology_map.json"


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


def build_guidance_description(cfg: Any, task_desc: str = "") -> str:

    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    text = _build_guidance_text(cfg.exp_id, tasks, models)
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))

    # Methodology / literature KB retrieval (opt-in: only when methodology_kb_path is set).
    methodology_kb_path = getattr(cfg, "methodology_kb_path", "") or ""
    if methodology_kb_path:
        use_dynamic = getattr(cfg, "methodology_dynamic", False)
        if use_dynamic and task_desc:
            from engine.coldstart.methodology_agent import build_methodology_guidance
            methodology_text = build_methodology_guidance(task_desc, methodology_kb_path, cfg.agent.code)
        else:
            methodology_text = _build_methodology_text(cfg.exp_id, methodology_kb_path)
        if methodology_text:
            text += methodology_text

    return text
