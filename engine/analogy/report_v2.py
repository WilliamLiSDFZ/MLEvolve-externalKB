"""Evidence-aware report validation and atomic rendering for context version 2."""
from __future__ import annotations

import copy
import json


ANCHOR_SCHEMA = {"type": "object", "properties": {
    "node_id": {"type": "string"}, "source_sha256": {"type": "string"},
    "start_line": {"type": "integer", "minimum": 1},
    "end_line": {"type": "integer", "minimum": 1}},
    "required": ["node_id", "source_sha256", "start_line", "end_line"],
    "additionalProperties": False}


def extend_tools(tools, *, mode):
    tools = copy.deepcopy(tools)
    report = next(t["function"]["parameters"] for t in tools
                  if t["function"]["name"] == "submit_report")
    report["properties"].update({
        "observed_facts": {"type": "array", "maxItems": 10, "items": {
            "type": "object", "properties": {
                "statement": {"type": "string"}, "evidence": {"type": "string"},
                "source": {"type": "string", "enum": ["code", "runtime", "task"]},
                "code_refs": {"type": "array", "items": ANCHOR_SCHEMA}},
            "required": ["statement", "evidence", "source"]}},
        "hypotheses": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "unknowns": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    })
    report["required"] += ["observed_facts", "hypotheses", "unknowns"]
    mechanism = report["properties"]["mechanisms"]["items"]
    mechanism["properties"].update({
        "implementation_basis": {"type": "string", "enum": ["code", "runtime", "task"]},
        "code_refs": {"type": "array", "items": ANCHOR_SCHEMA, "maxItems": 6},
        "runtime_evidence": {"type": "array", "items": {"type": "string"},
                             "description": "Exact dotted paths relative to runtime_context in the packet"},
        **{name: {"type": "string", "maxLength": 1500} for name in
           ("assumptions", "target_fit", "constraints", "validation_plan", "rejection_criterion")},
    })
    mechanism["required"] = list(dict.fromkeys(mechanism["required"] + [
        "implementation_basis", "code_refs", "runtime_evidence", "assumptions", "target_fit",
        "constraints", "validation_plan", "rejection_criterion"]))
    return tools


def _strings(value):
    return isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)


def _anchors(refs, session):
    return (isinstance(refs, list) and len(refs) <= 6
            and all(isinstance(ref, dict) and session is not None and session.validate_anchor(ref)
                    for ref in refs))


def _runtime_path(path, data):
    if not isinstance(path, str) or not path:
        return False
    value = data
    for key in path.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return False
    return value is not None and value != "unknown"


def validate(report, seen_ids, corpus, max_mechanisms, *, reading, abstracts,
             code_session, runtime_context, mode):
    from .agent import validate_report
    problems = []
    empty = {"context_version": 2, "bottlenecks": [], "mechanisms": [],
             "observed_facts": [], "hypotheses": [], "unknowns": []}
    if not isinstance(report, dict):
        return empty, ["report must be an object"]
    facts = report.get("observed_facts")
    if (not isinstance(facts, list) or len(facts) > 10
            or any(not _strings(report.get(k)) for k in ("hypotheses", "unknowns"))):
        return empty, ["supply observed_facts, hypotheses and unknowns as arrays (empty is allowed)"]
    for index, fact in enumerate(facts):
        if (not isinstance(fact, dict) or not all(isinstance(fact.get(k), str) and fact[k].strip()
                                                for k in ("statement", "evidence", "source"))):
            return empty, ["each observed fact needs statement, evidence and source"]
        if fact["source"] not in {"code", "runtime", "task"}:
            return empty, ["unknown fact source"]
        if fact["source"] == "code" and (not fact.get("code_refs") or not _anchors(fact["code_refs"], code_session)):
            bad_refs = [j for j, ref in enumerate(fact.get("code_refs") or [])
                        if code_session is None or not code_session.validate_anchor(ref)]
            return empty, [f"observed_facts[{index}]: code fact cites lines not actually returned by a source "
                           f"tool/index (invalid code_refs indices: {bad_refs}); copy exact node IDs and hashes "
                           "from the packet/tool replies, read missing lines, or narrow/remove this claim"]
        if fact["source"] == "runtime" and not _runtime_path(fact["evidence"], runtime_context):
            return empty, ["runtime fact evidence must be an available dotted runtime_context path"]
    clean = {**empty, "observed_facts": copy.deepcopy(facts),
             "hypotheses": list(report["hypotheses"]), "unknowns": list(report["unknowns"])}
    raw_mechanisms = report.get("mechanisms")
    if not isinstance(raw_mechanisms, list):
        return empty, ["mechanisms must be an array"]
    # Validate one mechanism at a time so removed citations cannot shift its extra fields.
    for candidate in raw_mechanisms:
        if not isinstance(candidate, dict):
            problems.append("mechanism must be an object")
            continue
        basis = candidate.get("implementation_basis")
        refs = candidate.get("code_refs", [])
        runtime_refs = candidate.get("runtime_evidence", [])
        fields = ("assumptions", "target_fit", "constraints", "validation_plan", "rejection_criterion")
        if any(not isinstance(candidate.get(k), str) or not candidate[k].strip() for k in fields):
            problems.append("mechanism needs complete assumptions, fit, constraints, validation and rejection conditions")
            continue
        if not _anchors(refs, code_session) or not _strings(runtime_refs):
            problems.append("invalid code_refs/runtime_evidence")
            continue
        if mode == "improve":
            if basis == "code" and not refs:
                problems.append("method-specific intervention requires code lines actually read")
                continue
            if basis == "runtime" and (not runtime_refs or
                    not all(_runtime_path(path, runtime_context) for path in runtime_refs)):
                problems.append("resource intervention requires available runtime_context evidence")
                continue
            if basis not in {"code", "runtime"}:
                problems.append("improve intervention must be grounded in code or runtime evidence")
                continue
        elif basis != "task":
            problems.append("draft has no executed candidate; use implementation_basis=task")
            continue
        try:
            legacy, issues = validate_report({"bottlenecks": report.get("bottlenecks", []),
                                             "mechanisms": [candidate]}, seen_ids, corpus, 1,
                                            reading=reading, abstracts=abstracts)
        except (TypeError, ValueError, KeyError):
            problems.append("malformed mechanism/bottleneck fields; resubmit using the tool schema")
            continue
        problems.extend(issues)
        if not legacy["mechanisms"]:
            continue
        item = legacy["mechanisms"][0]
        # Keep entire fields; legacy rendering clips some fields at 1000 characters.
        for key in (*fields, "limitations"):
            if key in candidate:
                item[key] = candidate[key]
        item.update(implementation_basis=basis, code_refs=copy.deepcopy(refs), runtime_evidence=runtime_refs,
                    mechanism_id=f"m{len(clean['mechanisms']) + 1}")
        clean["bottlenecks"] = legacy["bottlenecks"]
        clean["mechanisms"].append(item)
        if len(clean["mechanisms"]) >= max_mechanisms:
            break
    if not raw_mechanisms:
        # No supported diagnosis is a valid outcome, including zero bottlenecks.
        if report.get("bottlenecks"):
            legacy, issues = validate_report(report, seen_ids, corpus, max_mechanisms)
            clean["bottlenecks"] = legacy["bottlenecks"]
            problems.extend(issues)
    return clean, problems


def render(report, corpus, budget_chars, mode="improve"):
    """Return the exact report and Markdown retained under the total budget."""
    from .agent import render_report
    clean = copy.deepcopy(report)

    def one_render():
        if not clean["mechanisms"]:
            return ""
        facts = "\n".join(f"- {f['statement']} [source={f['source']}; evidence={f['evidence']}; "
                          f"code_refs={json.dumps(f.get('code_refs', []), ensure_ascii=False)}]"
                          for f in clean["observed_facts"]) or "- None confirmed."
        prefix = ("## Implementation evidence\n\nObserved facts:\n" + facts + "\n\nHypotheses:\n" +
                  "\n".join(f"- {s}" for s in clean["hypotheses"]) + "\n\nUnknowns:\n" +
                  "\n".join(f"- {s}" for s in clean["unknowns"]) + "\n\n")
        # An effectively unlimited legacy render gives whole mechanisms; this layer
        # applies the one actual budget to the combined evidence and all mechanism fields.
        body = render_report(clean, corpus, 10**12, mode=mode)
        for item in clean["mechanisms"]:
            heading = "### " + item["title"]
            replacement = (heading + f" [{item['mechanism_id']}]\n"
                           f"**Implementation basis**: {item['implementation_basis']}\n"
                           f"**Code locations read**: {json.dumps(item['code_refs'], ensure_ascii=False)}\n"
                           f"**Runtime evidence**: {json.dumps(item['runtime_evidence'])}\n"
                           f"**Constraints to preserve**: {item['constraints']}\n"
                           f"**Explicit rejection condition**: {item['rejection_criterion']}")
            if "evidence_refs" not in item:
                replacement += (f"\n**Source assumptions**: {item['assumptions']}\n"
                                f"**Fit**: {item['target_fit']}\n**Validation**: {item['validation_plan']}")
            body = body.replace(heading + "\n", replacement + "\n", 1)
        return prefix + body

    while clean["mechanisms"]:
        text = one_render()
        if len(text) <= budget_chars:
            return clean, text
        clean["mechanisms"].pop()  # whole lowest-priority mechanisms only
    return clean, ""
