"""Analogy agent: from an improve node's search state to cross-domain mechanism suggestions.

Runs once per improve node (hooked in `agents/improve_agent.py`). Given the parent node's
design, validation behaviour and history, an LLM (a) diagnoses the local bottlenecks of the
CURRENT methodology, (b) rewrites each as short queries in the vocabulary other subfields use
for the same relational structure, (c) searches the paper corpus with BM25 and reads abstracts,
and (d) submits a report mapping the found mechanisms back to this pipeline as concrete
interventions. Structure-mapping (objects + relations, mapped by function) follows arXiv
2605.11258; the "query = diagnosed local problem, not the task" framing follows the design doc
(`Agentic_Knowledge_Base/docs/analogy_bm25_agent_design.md`).

Shape: an OpenAI tools loop (same pattern as the KB repo's plugin_a2_insighter.py) with three
tools — search_papers, read_abstract, submit_report — and a hard turn cap. The LLM does the
analogy; BM25 only does the lookup. Two hard rules keep it honest and harmless:

* a mechanism may only cite paper ids that appeared in this run's search results — anything
  else is dropped at validation, so the report cannot hallucinate citations;
* nothing here may end a run. Every failure path returns an empty report and the improve node
  proceeds without it (`retrieve_for_node` never raises).

Per-node artefacts: `logs/analogy/<parent_id>_<n>.md` (packet, every tool call and its hits,
the report) and one line in `logs/analogy/index.jsonl`; the rendered report is also stored on
the child node (`SearchNode.analogy_report`) so `journal.json` carries what each node saw.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.analogy.corpus import PaperCorpus, load_corpus

logger = logging.getLogger("MLEvolve")

# ------------------------------------------------------------------ packet (the agent's input)

# Head + tail of the competition description. Tail included on purpose: in classic Kaggle
# descriptions the ML content (metric, label semantics) sits at the END under "Evaluation" —
# the head-only rule scored 0/10 on spooky (semantic_retrieval_design.md §18).
_TASK_HEAD, _TASK_TAIL = 4500, 1500
_DATA_CHARS = 2000
# utils.data_preview.clean_task_desc appends submission-format and metric-alignment blocks
# (each headed by a line of "=") to every task description. They are run-mechanics the packet's
# "Available data" section already covers, and they would otherwise eat the whole tail budget.
_APPENDED_BLOCK = re.compile(r"\n=+\n\*\*(?:REQUIRED SUBMISSION FORMAT|TASK AND METRIC ALIGNMENT REQUIREMENT)\*\*")
_FIELD_CHARS = 1500
_ATTEMPTS_CHARS = 2500
_TRAJECTORY_NODES = 6


def _clip(text: Any, n: int, tail: bool = False) -> str:
    s = str(text or "").strip()
    if len(s) <= n:
        return s
    return ("…" + s[-n:]) if tail else (s[:n] + "…")


def _fmt_metric(value: Any, maximize: Any) -> str:
    if value is None:
        return "n/a"
    direction = "higher is better" if maximize else ("lower is better" if maximize is False else "direction unknown")
    return f"{value:.5f} ({direction})" if isinstance(value, (int, float)) else f"{value} ({direction})"


def build_packet(*, task_desc: str, data_preview: str, node_id: str, stage: str, design: str,
                 code_summary: str, analysis: str, metric_value: Any, metric_maximize: Any,
                 branch_best: Any, term_out: str, attempts: str,
                 trajectory: List[Dict[str, Any]]) -> str:
    """Render the search state as markdown. Plain-data signature so utils/replay_analogy.py can
    rebuild a packet from journal.json without live SearchNode objects."""
    desc = task_desc or ""
    m = _APPENDED_BLOCK.search(desc)
    if m:
        desc = desc[:m.start()]
    if len(desc) > _TASK_HEAD + _TASK_TAIL:
        desc = desc[:_TASK_HEAD] + "\n\n[... middle of the description omitted ...]\n\n" + desc[-_TASK_TAIL:]
    traj = "\n".join(
        f"- {t.get('stage', '?')}: metric {t.get('metric') if t.get('metric') is not None else 'n/a'}"
        f"{' (buggy)' if t.get('is_buggy') else ''} — {_clip(t.get('plan', ''), 200)}"
        for t in trajectory) or "(none)"
    return f"""# SEARCH STATE

## Task (competition description; head and tail)
{desc.strip()}

## Available data (workspace listing / preview)
{_clip(data_preview, _DATA_CHARS)}

## Current solution — node {node_id} (stage: {stage})
Design / plan:
{_clip(design, _FIELD_CHARS)}

Code summary:
{_clip(code_summary, _FIELD_CHARS) or '(none)'}

## Validation behaviour
Metric of this node: {_fmt_metric(metric_value, metric_maximize)}
Best metric on this branch so far: {_fmt_metric(branch_best, metric_maximize)}

Execution summary:
{_clip(analysis, _FIELD_CHARS) or '(none)'}

Tail of the run output:
```
{_clip(term_out, _FIELD_CHARS, tail=True) or '(empty)'}
```

## Improvement attempts already made from this node
{_clip(attempts, _ATTEMPTS_CHARS) or '(none yet)'}

## Branch trajectory (oldest -> newest)
{traj}
"""


def packet_from_search(agent: Any, parent_node: Any) -> str:
    """Build the packet from the live AgentSearch instance and the node being improved."""
    branch = list(getattr(agent, "branch_all_nodes", {}).get(parent_node.branch_id, []) or [])
    branch = sorted(branch, key=lambda n: getattr(n, "ctime", 0.0))[-_TRAJECTORY_NODES:]
    trajectory = [{"stage": n.stage, "is_buggy": n.is_buggy,
                   "metric": (n.metric.value if n.metric is not None else None),
                   "plan": n.plan} for n in branch]

    maximize = parent_node.metric.maximize if parent_node.metric is not None else getattr(agent, "metric_maximize",
                                                                                          None)
    succ = [n for n in getattr(agent, "branch_successful_nodes", {}).get(parent_node.branch_id, [])
            if n.metric is not None and n.metric.value is not None]
    branch_best = None
    if succ:
        vals = [n.metric.value for n in succ]
        branch_best = max(vals) if maximize is not False else min(vals)

    try:
        attempts = parent_node.fetch_child_memory(include_code=False)
    except Exception:
        attempts = ""
    return build_packet(
        task_desc=getattr(agent, "task_desc", "") or "",
        data_preview=getattr(agent, "data_preview", "") or "",
        node_id=parent_node.id, stage=parent_node.stage,
        design=parent_node.plan or "", code_summary=getattr(parent_node, "code_summary", "") or "",
        analysis=parent_node.analysis or "",
        metric_value=(parent_node.metric.value if parent_node.metric is not None else None),
        metric_maximize=maximize, branch_best=branch_best,
        term_out=parent_node.term_out if parent_node._term_out else "",
        attempts=attempts, trajectory=trajectory)


# ------------------------------------------------------------------ prompts & tools

SYSTEM_PROMPT = """You are a research-methodology analyst embedded in an automated machine-learning \
engineering search. A candidate solution to a Kaggle-style competition has just been trained and \
evaluated; its state is in the user message. Your job is NOT to propose the next tweak yourself. \
It is to find, in a corpus of {n_papers} recent ML papers, mechanisms that solved the SAME PROBLEM \
STRUCTURE in OTHER subfields, and to map them back onto this pipeline as concrete interventions.

Work in four steps.

STEP 1 - DIAGNOSE (write this out, before any tool call). From the search state, identify at most \
3 local bottlenecks of the CURRENT methodology. A bottleneck is a property of the pipeline, not of \
the competition's topic: an objective that does not match the metric, a symmetry or invariance the \
model violates, the scale at which information is fused, evidence the model ignores, a resource \
constraint forcing a bad trade-off, a label structure the loss ignores. "The score is low" is not a \
bottleneck. For each one write: objects (the pipeline entities involved, by FUNCTIONAL role), \
relations (how they constrain each other; what is violated or missing), evidence (which line of the \
search state shows it).

STEP 2 - ABSTRACT INTO QUERIES. For each bottleneck write 2-4 search queries of 3-6 technical terms \
each, in the vocabulary OTHER subfields use for the same relational structure. Never use the \
competition's own domain nouns (its dataset, entities or field-specific words). Map by function, not \
by surface similarity - "delivers payload" is a good mapping basis, "is liquid" is not. Two examples \
of the translation expected:
  - "swapping the two candidate answers should permute the predicted probabilities, but the model is \
not symmetric"  ->  `permutation equivariance symmetrization`, `pairwise comparison antisymmetry`, \
`group averaging test-time symmetrization`
  - "the target is 2-D but the signal lives on a short depth axis whose absolute offset is arbitrary" \
->  `nuisance variable invariance marginalization`, `shift invariant pooling projection`, \
`3D to 2D aggregation depth invariant`
The corpus is title + tldr + abstract matched lexically (BM25): short, specific mechanism terms \
work; sentences do not. If a query returns unrelated papers, change the vocabulary - do not add \
words. The same mechanism often has several names across subfields; try more than one.

STEP 3 - SEARCH AND READ. Call search_papers for each query (several calls per turn are fine). \
Judge structural match from the tldr; call read_abstract on the few that look isomorphic to confirm \
the mechanism. Papers from the competition's own subfield count only if the mechanism transfers; \
prefer other subfields. You have at most {max_turns} assistant turns in total, so search broadly early.

STEP 4 - MAP BACK. Call submit_report with at most {max_mechanisms} mechanisms. Each must name its \
bottleneck, give explicit object mappings (search-state entity <-> paper entity, one-line rationale \
each), the shared relational structure, what the paper did, and a concrete intervention for THIS \
pipeline - specific enough to become one improvement step: what changes, where in the pipeline, what \
you expect to observe. Judge feasibility against the "Available data" section: a mechanism needing a \
modality, annotation or compute this competition does not have is infeasible, say so. Cite only paper \
ids that appeared in your search results; anything else is discarded at validation.

If nothing structurally matching exists in the corpus, submit the bottlenecks with an empty \
mechanisms list - that is a valid answer. Do not pad the report with generic advice."""

_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "bottlenecks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "relations": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "string"},
                },
                "required": ["statement"],
            },
        },
        "mechanisms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bottleneck_idx": {"type": "integer", "description": "0-based index into bottlenecks"},
                    "title": {"type": "string", "description": "short name of the mechanism"},
                    "paper_ids": {"type": "array", "items": {"type": "string"},
                                  "description": "ids from search_papers results only"},
                    "object_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"source": {"type": "string"}, "target": {"type": "string"},
                                           "rationale": {"type": "string"}},
                            "required": ["source", "target"],
                        },
                    },
                    "shared_relations": {"type": "string"},
                    "mechanism": {"type": "string", "description": "what the paper did, 2-3 sentences"},
                    "intervention": {"type": "string",
                                     "description": "the concrete change to THIS pipeline, 2-4 sentences"},
                    "feasibility": {"type": "string"},
                },
                "required": ["title", "paper_ids", "mechanism", "intervention"],
            },
        },
    },
    "required": ["bottlenecks", "mechanisms"],
}

TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search_papers",
        "description": "BM25 search over the paper corpus (title + tldr + abstract). Use 3-6 "
                       "technical terms. Returns the top-k papers as id, venue, title, tldr, score "
                       "- no abstracts.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_abstract",
        "description": "Full abstracts for up to 8 papers by id. Only ids returned by an earlier "
                       "search_papers call are accepted.",
        "parameters": {"type": "object",
                       "properties": {"ids": {"type": "array", "items": {"type": "string"},
                                              "maxItems": 8}},
                       "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "submit_report",
        "description": "Finish: submit the diagnosed bottlenecks and the mechanisms mapped back "
                       "onto this pipeline.",
        "parameters": _REPORT_SCHEMA}},
]

_MAX_TOOL_RESULT_CHARS = 20000
_MAX_OUTPUT_TOKENS = 6000
_NUDGE = ("Continue with the tools: search_papers / read_abstract to keep looking, or "
          "submit_report to finish. Reply with a tool call.")


# ------------------------------------------------------------------ result & validation

@dataclass
class AnalogyResult:
    report_md: str = ""  # "" -> nothing to inject
    report: Optional[dict] = None
    reason: str = ""  # why report_md is empty
    turns: int = 0
    queries: List[str] = field(default_factory=list)
    paper_ids: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    in_tokens: int = 0
    out_tokens: int = 0
    seconds: float = 0.0


def validate_report(report: Any, seen_ids: set, corpus: PaperCorpus,
                    max_mechanisms: int) -> tuple[dict, List[str]]:
    """Coerce the submitted report to the schema and enforce the citation rule.

    Returns (clean_report, problems). A mechanism survives only if it keeps at least one paper id
    that (a) the agent actually saw in a search result this run and (b) exists in the corpus.
    """
    problems: List[str] = []
    if not isinstance(report, dict):
        return {"bottlenecks": [], "mechanisms": []}, ["report is not an object"]

    bottlenecks = []
    for b in (report.get("bottlenecks") or []):
        if isinstance(b, dict) and str(b.get("statement", "")).strip():
            bottlenecks.append({
                "statement": str(b["statement"]).strip(),
                "objects": [str(x) for x in (b.get("objects") or [])][:8],
                "relations": [str(x) for x in (b.get("relations") or [])][:8],
                "evidence": str(b.get("evidence", "")).strip(),
            })
    if not bottlenecks:
        problems.append("no bottleneck with a statement")

    mechanisms = []
    for m in (report.get("mechanisms") or []):
        if not isinstance(m, dict):
            continue
        title = str(m.get("title", "")).strip()
        ids = [str(x) for x in (m.get("paper_ids") or [])]
        kept = [i for i in ids if i in seen_ids and i in corpus]
        rejected = [i for i in ids if i not in kept]
        if rejected:
            problems.append(f"'{title or '?'}': dropped uncited/unknown ids {rejected}")
        if not title or not kept or not str(m.get("intervention", "")).strip():
            problems.append(f"'{title or '?'}': discarded (needs title, a cited paper id, and an intervention)")
            continue
        mechanisms.append({
            "bottleneck_idx": int(m.get("bottleneck_idx", 0) or 0),
            "title": title,
            "paper_ids": kept[:4],
            "object_mappings": [
                {"source": str(om.get("source", "")), "target": str(om.get("target", "")),
                 "rationale": str(om.get("rationale", ""))}
                for om in (m.get("object_mappings") or []) if isinstance(om, dict)][:6],
            "shared_relations": str(m.get("shared_relations", "")).strip(),
            "mechanism": str(m.get("mechanism", "")).strip(),
            "intervention": str(m.get("intervention", "")).strip(),
            "feasibility": str(m.get("feasibility", "")).strip(),
        })
    if len(mechanisms) > max_mechanisms:
        problems.append(f"kept the first {max_mechanisms} of {len(mechanisms)} mechanisms")
        mechanisms = mechanisms[:max_mechanisms]
    return {"bottlenecks": bottlenecks, "mechanisms": mechanisms}, problems


REPORT_HEADING = "## Cross-domain mechanism suggestions (analogy search on this node's bottleneck)"


def render_report(report: dict, corpus: PaperCorpus, budget_chars: int) -> str:
    """Markdown for the improve prompt. Each mechanism is a `### ` block so the adoption judge
    (KB repo measure_adoption.py, TECHNIQUE_HEADING) sees one technique per mechanism."""
    if not report.get("mechanisms"):
        return ""
    lines = [REPORT_HEADING, "", "Diagnosed bottlenecks of the current solution:"]
    for i, b in enumerate(report.get("bottlenecks") or []):
        ev = f" — evidence: {b['evidence']}" if b.get("evidence") else ""
        lines.append(f"{i}. {b['statement']}{ev}")
    blocks = []
    for m in report["mechanisms"]:
        cites = "; ".join(
            f"{corpus.by_id[i]['title']} ({corpus.by_id[i]['venue']}, `{i}`)" for i in m["paper_ids"])
        maps = "; ".join(
            f"{om['source']} ↔ {om['target']}" + (f" ({om['rationale']})" if om.get("rationale") else "")
            for om in m["object_mappings"]) or "(not given)"
        blocks.append("\n".join([
            f"### {m['title']}",
            f"*Addresses bottleneck {m['bottleneck_idx']}. Source: {cites}*",
            "",
            f"**Shared problem structure**: {m['shared_relations'] or '(not given)'}",
            f"**Object mappings (this pipeline ↔ source)**: {maps}",
            f"**Mechanism in the source**: {m['mechanism']}",
            f"**Proposed intervention here**: {m['intervention']}",
            f"**Feasibility with the available data**: {m['feasibility'] or '(not assessed)'}",
        ]))
    head = "\n".join(lines) + "\n"
    out, used = [], len(head)
    for b in blocks:  # whole mechanisms only; never cut one mid-block
        if out and used + len(b) + 2 > budget_chars:
            break
        out.append(b)
        used += len(b) + 2
    return head + "\n" + "\n\n".join(out) + "\n"


# ------------------------------------------------------------------ the loop

def _chat_params(model: str, base_url: str, messages: List[dict]) -> dict:
    """Mirror the per-model rules of llm/openai.py for a tools call."""
    from llm.model_profiles import is_openai_reasoning_model, uses_max_completion_tokens
    params: dict = {
        "model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
        ("max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"): _MAX_OUTPUT_TOKENS,
    }
    if is_openai_reasoning_model(model):
        # Function tools on /v1/chat/completions require reasoning_effort='none' (llm/openai.py).
        # Consequence worth knowing when reading traces: on gpt-5.x the diagnosis is written as
        # visible text (STEP 1 in the prompt), not reasoned privately.
        params["extra_body"] = {"reasoning_effort": "none"}
    # No thinking params for Claude here, unlike llm/openai.py: this is a MULTI-turn tool loop,
    # and replaying assistant tool-call turns without their thinking blocks through an
    # OpenAI-compatible proxy is exactly the case those endpoints reject. tool_choice stays
    # "auto", which every model family accepts (see _NO_TOOL_CHOICE_REQUIRED_PREFIXES).
    return params


def _tool_message(msg: Any) -> dict:
    """Assistant turn re-encoded as a plain dict (no SDK-specific fields) for the next request."""
    return {"role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments or "{}"}}
                           for tc in (msg.tool_calls or [])]}


def run_analogy_agent(packet_md: str, corpus: PaperCorpus, llm_cfg: Any, *, max_turns: int = 10,
                      top_k: int = 10, max_mechanisms: int = 3,
                      report_char_budget: int = 8000) -> AnalogyResult:
    """One agent episode. Raises only on programming errors; API/parse failures are caught by
    the caller (`retrieve_for_node`), which turns them into an empty report."""
    from openai import OpenAI

    model = str(getattr(llm_cfg, "model", "") or "")
    client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url or None, timeout=600.0)
    system = SYSTEM_PROMPT.format(n_papers=len(corpus), max_turns=max_turns,
                                  max_mechanisms=max_mechanisms)
    messages: List[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": packet_md}]
    res = AnalogyResult()
    seen_ids: set = set()
    nudged = False
    t0 = time.time()

    for turn in range(1, max_turns + 1):
        res.turns = turn
        resp = client.chat.completions.create(**_chat_params(model, llm_cfg.base_url or "", messages))
        usage = getattr(resp, "usage", None)
        res.in_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        res.out_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        msg = resp.choices[0].message
        if msg.content:
            res.trace.append(f"[turn {turn}] assistant:\n{msg.content.strip()[:4000]}")

        if not msg.tool_calls:
            if nudged:
                res.reason = "assistant stopped without submit_report"
                break
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": _NUDGE})
            nudged = True
            continue

        messages.append(_tool_message(msg))
        done = False
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "search_papers":
                q = str(args.get("query", "")).strip()
                try:
                    k = max(1, min(int(args.get("k") or top_k), 20))
                except (TypeError, ValueError):
                    k = top_k
                hits = corpus.search(q, k=k) if q else []
                seen_ids.update(h["id"] for h in hits)
                res.queries.append(q)
                content = json.dumps(hits, ensure_ascii=False)
                res.trace.append(f"[turn {turn}] search_papers({q!r}, k={k}) ->\n" + "\n".join(
                    f"    {h['score']:6.2f}  [{h['venue']}] {h['title'][:100]}  ({h['id']})" for h in hits))
            elif name == "read_abstract":
                ids = [str(i) for i in (args.get("ids") or [])][:8]
                allowed = [i for i in ids if i in seen_ids]
                rejected = [i for i in ids if i not in seen_ids]
                payload: Any = corpus.get(allowed)
                if rejected:
                    payload = {"papers": payload,
                               "rejected_ids": rejected,
                               "note": "only ids returned by search_papers can be read"}
                content = json.dumps(payload, ensure_ascii=False)
                res.trace.append(f"[turn {turn}] read_abstract({ids}) -> {len(allowed)} abstracts"
                                 + (f", rejected {rejected}" if rejected else ""))
            elif name == "submit_report":
                clean, problems = validate_report(args, seen_ids, corpus, max_mechanisms)
                res.trace.append(f"[turn {turn}] submit_report -> {len(clean['mechanisms'])} mechanism(s)"
                                 + (f"; problems: {problems}" if problems else ""))
                if clean["mechanisms"] or not (args.get("mechanisms") or []):
                    # Accepted: either something survived, or the agent honestly found nothing.
                    res.report = clean
                    res.paper_ids = sorted({i for m in clean["mechanisms"] for i in m["paper_ids"]})
                    res.report_md = render_report(clean, corpus, report_char_budget)
                    if not clean["mechanisms"]:
                        res.reason = "agent found no structurally matching mechanism"
                    content = "accepted"
                    done = True
                else:
                    content = ("rejected: " + "; ".join(problems) +
                               ". Cite only paper ids from search_papers results and give an "
                               "intervention for each mechanism, then call submit_report again.")
            else:
                content = f"unknown tool {name}"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": content[:_MAX_TOOL_RESULT_CHARS]})
            if done:
                break
        if done:
            break
    else:
        res.reason = f"no report within {max_turns} turns"

    res.seconds = time.time() - t0
    return res


# ------------------------------------------------------------------ entry point for improve_agent

_INDEX_LOCK = threading.Lock()
_INVOCATIONS = 0


def _write_artifacts(log_dir: Path, parent_id: str, packet_md: str, res: AnalogyResult,
                     corpus: PaperCorpus, extra: dict) -> None:
    """Trace file + one index line. Failure here is logged, never raised."""
    global _INVOCATIONS
    try:
        adir = log_dir / "analogy"
        adir.mkdir(parents=True, exist_ok=True)
        with _INDEX_LOCK:
            _INVOCATIONS += 1
            n = _INVOCATIONS
        trace_path = adir / f"{parent_id}_{n:03d}.md"
        body = [f"# analogy agent — parent {parent_id} (invocation {n})", "",
                f"corpus {corpus.digest} ({len(corpus)} papers) | turns {res.turns} | "
                f"tokens in/out {res.in_tokens}/{res.out_tokens} | {res.seconds:.0f}s | "
                f"{'report' if res.report_md else 'NO REPORT: ' + res.reason}", "",
                "## Packet", "", packet_md, "", "## Trace", ""]
        body += [t + "\n" for t in res.trace]
        body += ["## Report (as injected)", "", res.report_md or "(empty)", ""]
        if res.report is not None:
            body += ["## Report (raw JSON)", "", "```json",
                     json.dumps(res.report, ensure_ascii=False, indent=2), "```", ""]
        trace_path.write_text("\n".join(body), encoding="utf-8")
        line = {"parent_id": parent_id, "invocation": n, "trace": trace_path.name,
                "ok": bool(res.report_md), "reason": res.reason, "turns": res.turns,
                "n_queries": len(res.queries), "queries": res.queries,
                "paper_ids": res.paper_ids, "report_chars": len(res.report_md),
                "in_tokens": res.in_tokens, "out_tokens": res.out_tokens,
                "seconds": round(res.seconds, 1), "corpus": corpus.digest, **extra}
        with _INDEX_LOCK:
            with (adir / "index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover - diagnostics must not break a run
        logger.warning("[analogy] could not write trace for %s: %s: %s", parent_id, type(e).__name__, e)


def retrieve_for_node(agent: Any, parent_node: Any) -> str:
    """Run the agent for one improve node and return the report markdown ("" = inject nothing).

    Never raises. Reads cfg.analogy.* and cfg.agent.code (the model slot).
    """
    cfg = agent.cfg
    acfg = getattr(cfg, "analogy", None)
    if acfg is None or not getattr(acfg, "enabled", False):
        return ""
    try:
        corpus = load_corpus(str(getattr(acfg, "corpus_path", "") or ""))
    except Exception as e:
        logger.warning("[analogy] corpus unavailable (%s: %s) — improving without it",
                       type(e).__name__, e)
        return ""
    if corpus is None:
        return ""
    packet = ""
    try:
        packet = packet_from_search(agent, parent_node)
        res = run_analogy_agent(
            packet, corpus, cfg.agent.code,
            max_turns=int(getattr(acfg, "max_turns", 10)),
            top_k=int(getattr(acfg, "top_k", 10)),
            max_mechanisms=int(getattr(acfg, "max_mechanisms", 3)),
            report_char_budget=int(getattr(acfg, "report_char_budget", 8000)))
    except Exception as e:
        res = AnalogyResult(reason=f"{type(e).__name__}: {e}")
        res.trace.append(f"EXCEPTION: {type(e).__name__}: {e}")
        logger.warning("[analogy] node %s: agent failed (%s: %s) — improving without it",
                       parent_node.id, type(e).__name__, e)

    _write_artifacts(Path(getattr(cfg, "log_dir", "") or "."), parent_node.id, packet, res, corpus,
                     extra={"branch_id": parent_node.branch_id,
                            "parent_metric": (parent_node.metric.value if parent_node.metric is not None else None)})
    if res.report_md:
        logger.info("[analogy] node %s: %d mechanism(s) from %d quer%s in %d turns, %d chars, "
                    "papers %s", parent_node.id, len(res.report["mechanisms"]), len(res.queries),
                    "y" if len(res.queries) == 1 else "ies", res.turns, len(res.report_md),
                    res.paper_ids)
    else:
        logger.info("[analogy] node %s: no report (%s) after %d turns, %d queries",
                    parent_node.id, res.reason or "?", res.turns, len(res.queries))
    return res.report_md
