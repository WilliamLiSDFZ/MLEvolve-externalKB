"""Build guidance description for agent from task/model JSON."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger("MLEvolve")

INIT_SOLUTION_JSON = Path(__file__).resolve().parent / "init_solution_paths.json"


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


def build_guidance_description(cfg: Any, task_desc: str = "") -> str:
    """Return the PRETRAINED-MODEL guidance for this task.

    This is the only cold-start knowledge: literature retrieval no longer happens here. It
    runs per improve node instead (engine/analogy), because the query it needs — the diagnosed
    local bottleneck of the current methodology — does not exist before the first experiment.
    `task_desc` is accepted for call-compatibility and unused.
    """
    tasks = _load_json(cfg.coldstart.task_json_path)
    models = _load_json(cfg.coldstart.model_json_path)
    text = _build_guidance_text(cfg.exp_id, tasks, models)
    torch_hub_dir = getattr(cfg, "torch_hub_dir", "") or ""
    if torch_hub_dir:
        text = text.replace("{TORCH_HUB_DIR}", torch_hub_dir.rstrip("/"))

    # Record WHAT THE CORPUS CONTAINED for this run (arm D), separately from what any node was
    # injected with. The IMPORT is inside the try, not just the call: a deploy that copies
    # knowledge.py without kb_snapshot.py would otherwise raise ImportError here — in a function
    # every arm calls, at cold start — and kill the job in its first minute. That happened
    # (essay s47, BackoffLimitExceeded before a single node). A diagnostic must not be able to
    # end a run, and "must not" has to include failing to load.
    try:
        from engine.coldstart.kb_snapshot import write_kb_snapshot
        write_kb_snapshot(cfg)
    except Exception as e:  # pragma: no cover
        logger.warning("KB snapshot unavailable (%s: %s) — continuing without it",
                       type(e).__name__, e)

    return text
