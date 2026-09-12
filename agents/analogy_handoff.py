"""Explicit mechanism selection and child-scoped code provenance for context v2.

Selection is a planner declaration, not proof that code implements a mechanism or
that a score change was caused by it. No agent/parent mutable state is used.
"""

from copy import deepcopy
import difflib
import hashlib
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger("MLEvolve")
CONTEXT_KEY = "_analogy_context_v2"
MAX_MECHANISM_CHARS = 32768


def report_from_prompt(prompt):
    report = prompt.get(CONTEXT_KEY) if isinstance(prompt, dict) else None
    if not isinstance(report, dict) or report.get("context_version") != 2:
        return None
    return report


def attach_report(prompt, report):
    if isinstance(report, dict) and report.get("context_version") == 2:
        prompt[CONTEXT_KEY] = deepcopy(report)


def _mechanisms(report):
    result = {}
    for mechanism in report.get("mechanisms", []) if isinstance(report, dict) else []:
        if isinstance(mechanism, dict) and isinstance(mechanism.get("mechanism_id"), str):
            key = mechanism["mechanism_id"]
            if key in result:
                return {}  # Ambiguous IDs cannot establish an explicit selection.
            result[key] = mechanism
    return result


def planning_schema(base_schema, prompt):
    report = report_from_prompt(prompt)
    if report is None:
        return base_schema
    schema = deepcopy(base_schema)
    schema["properties"].update({
        "selected_mechanism_id": {"type": ["string", "null"], "enum": [None, *_mechanisms(report)],
                                  "description": "Select at most one offered mechanism ID; null explicitly rejects all."},
        "analogy_decision_reason": {"type": "string", "maxLength": 4000,
                                    "description": "Why this mechanism is appropriate or why all offered mechanisms were rejected."},
        "analogy_adaptation": {"type": "string", "maxLength": 6000,
                               "description": "Concrete adaptation and preserved constraints; empty string when rejecting all."},
    })
    # Optional fields preserve legacy plan contracts. Missing fields mean unknown,
    # never implicit adoption and never automatic rejection.
    return schema


def planning_instruction(prompt, *, full_report=False):
    report = report_from_prompt(prompt)
    if report is None:
        return ""
    text = ("\n# Explicit analogy decision\n"
            "The mechanisms below are suggestions supported by recorded evidence, not instructions. "
            "Choose at most ONE mechanism or reject all. In your JSON plan include "
            "selected_mechanism_id (one offered ID or null), analogy_decision_reason, and analogy_adaptation. "
            "Check source evidence, assumptions, preserved constraints and validation/rejection criteria. "
            "A selection records intent; it does not establish implementation correctness or causality.\n"
            f"Offered mechanism IDs: {', '.join(_mechanisms(report)) or '[none]'}\n")
    if full_report:
        text += "\nOriginal structured analogy report:\n" + json.dumps(report, ensure_ascii=False) + "\n"
    return text


def _plan_object(plan):
    if isinstance(plan, dict):
        return plan
    if not isinstance(plan, str):
        return None
    try:
        value = json.loads(plan)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    # Full rewrite may include this explicitly requested JSON line before prose.
    # Do not search arbitrary natural-language braces or infer adoption by name.
    match = re.search(r"(?m)^\s*ANALOGY_ADOPTION:\s*", plan)
    if match:
        try:
            value, _ = json.JSONDecoder().raw_decode(plan[match.end():])
            return value if isinstance(value, dict) else None
        except ValueError:
            pass
    return None


def adoption_from_plan(prompt, plan):
    report = report_from_prompt(prompt)
    if report is None:
        return None
    result = dict(version=1, context_version=2, status="unknown", selected_mechanism_id=None,
                  decision_reason=None, adaptation=None, selected_mechanism=None,
                  interpretation="Planner declaration only; implementation and causal effects are not inferred")
    parsed = _plan_object(plan)
    if not parsed or parsed.get("parse_success") is False or "selected_mechanism_id" not in parsed:
        result["unknown_reason"] = "No parseable explicit analogy selection in the generated plan"
        return result
    key = parsed["selected_mechanism_id"]
    reason = parsed.get("analogy_decision_reason")
    adaptation = parsed.get("analogy_adaptation")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
        result["unknown_reason"] = "Missing or invalid explicit adoption/rejection reason"
        return result
    if key is None:
        result.update(status="rejected", decision_reason=reason, adaptation="")
        return result
    mechanisms = _mechanisms(report)
    if not isinstance(key, str) or key not in mechanisms:
        result["unknown_reason"] = "Selection is not a single offered mechanism ID"
        return result
    if not isinstance(adaptation, str) or not adaptation.strip() or len(adaptation) > 6000:
        result["unknown_reason"] = "Selected mechanism has no valid concrete adaptation"
        return result
    mechanism = deepcopy(mechanisms[key])
    if len(json.dumps(mechanism, ensure_ascii=False)) > MAX_MECHANISM_CHARS:
        result["unknown_reason"] = "Selected mechanism exceeds complete handoff budget; no partial constraints forwarded"
        return result
    result.update(status="selected", selected_mechanism_id=key, decision_reason=reason,
                  adaptation=adaptation, selected_mechanism=mechanism)
    return result


def selected_mechanism_brief(adoption):
    if not adoption or adoption.get("status") != "selected":
        return ""
    return ("\n# Selected analogy mechanism and implementation constraints\n"
            "Implement only the planner's selected adaptation. The original mechanism below retains its "
            "source references, assumptions, constraints, validation plan and rejection criterion. "
            "Check that the code preserves these details; do not assume the analogy is causally proven.\n"
            + json.dumps({"selected_mechanism_id": adoption["selected_mechanism_id"],
                          "decision_reason": adoption["decision_reason"], "adaptation": adoption["adaptation"],
                          "original_mechanism": adoption["selected_mechanism"]}, ensure_ascii=False) + "\n")


def _code_provenance(node, *, phase):
    parent_code = getattr(getattr(node, "parent", None), "code", "") or ""
    child_code = node.code or ""
    diff_lines = difflib.unified_diff(parent_code.splitlines(keepends=True), child_code.splitlines(keepends=True),
                                     fromfile="parent/solution.py", tofile="child/solution.py")
    patch = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in diff_lines)
    return dict(phase=phase, parent_source_sha256=hashlib.sha256(parent_code.encode()).hexdigest(),
                child_source_sha256=hashlib.sha256(child_code.encode()).hexdigest(),
                diff_sha256=hashlib.sha256(patch.encode()).hexdigest(), code_changed=parent_code != child_code), patch


def _write_handoff(node, patch):
    from engine.candidate_runtime.io import atomic_json
    adoption = node.analogy_adoption
    path = adoption.get("artifact_path")
    if not path:
        return
    try:
        atomic_json(path, dict(version=1, parent_node_id=getattr(node.parent, "id", None), child_node_id=node.id,
                               adoption=adoption, actual_diff=patch,
                               outcome_reference={"node_id": node.id, "journal": "logs/journal.json",
                                                  "runtime_metadata": f"logs/candidate_results/{node.id}/",
                                                  "note": "Read final metric, artifact and validity status from this child; no causal attribution"}))
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("[analogy] cannot persist child handoff %s: %s", node.id, exc)
        adoption["artifact_write_error"] = str(exc)


def record_child_handoff(agent, node, prompt, *, generation_mode):
    adoption = adoption_from_plan(prompt, node.plan)
    if adoption is None:
        return
    provenance, patch = _code_provenance(node, phase="generated_before_code_review")
    adoption.update(generation_mode=generation_mode, code_provenance=provenance,
                    execution_outcome=dict(status="pending", node_id=node.id),
                    artifact_path=str(Path(agent.cfg.log_dir) / "analogy/handoffs" / f"{node.id}.json"))
    node.analogy_adoption = adoption
    _write_handoff(node, patch)


def update_execution_handoff(node, execution_result):
    """Refresh source hashes after code review, while metric parsing is still pending."""
    if not isinstance(getattr(node, "analogy_adoption", None), dict):
        return
    provenance, patch = _code_provenance(node, phase="executed_source")
    node.analogy_adoption["code_provenance"] = provenance
    node.analogy_adoption["execution_outcome"] = dict(
        node_id=node.id, status=getattr(execution_result, "execution_status", None) or "executed",
        exc_type=execution_result.exc_type, elapsed_seconds=execution_result.exec_time,
        metric_status="See the child's final journal metric and artifact status after result parsing")
    _write_handoff(node, patch)
