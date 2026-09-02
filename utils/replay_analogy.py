"""Replay the analogy agent on one node of an EXISTING run — no GPU, no data, no 12-hour run.

    python utils/replay_analogy.py --run ~/nautilus/results/<run_dir> --node <id-prefix or step> \
        --corpus /path/to/paper_corpus [--desc description.md] [--out trace.md] [--packet-only]

Rebuilds the search-state packet from `logs/journal.json` (plan, code summary, execution
summary, metric, output tail, sibling attempts, branch trajectory), runs the agent against the
corpus with the LLM configured by LLM_MODEL / LLM_BASE_URL / LLM_API_KEY, and prints the trace
and the report exactly as improve_agent would have injected it. This is the fast feedback loop
for prompt and tokenizer changes — the equivalent of the old probe_retrieval.py — and the way
to judge report quality on several tasks before spending cluster time (design doc §6.2).

The task description and data preview are not stored in journal.json. Pass --desc to supply
the description; without it, both are recovered from the node's own `prompt_input` (the improve
prompt embeds them), which works for every run written by this codebase so far.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.analogy.agent import build_packet, run_analogy_agent  # noqa: E402
from engine.analogy.corpus import load_corpus  # noqa: E402


def _prompt_text(node: dict) -> str:
    raw = node.get("prompt_input") or ""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return str(raw)
    parts: list[str] = []

    def walk(x):
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(obj)
    return "\n".join(parts)


def _recover(node: dict) -> tuple[str, str]:
    """(task description, data preview) from the improve prompt stored on the node."""
    text = _prompt_text(node)
    desc = ""
    m = re.search(r"# Task description\n(.*?)(?:\n# Memory\n|\n# Instructions\n)", text, re.S)
    if m:
        desc = m.group(1).strip()
    preview = ""
    m = re.search(r"review the dataset:\n(.*?)\nThe current solution uses", text, re.S)
    if m:
        preview = m.group(1).strip()
    return desc, preview


def _find_node(nodes: list[dict], key: str) -> dict:
    if key.isdigit():
        for n in nodes:
            if int(n.get("step", -1)) == int(key):
                return n
    hits = [n for n in nodes if str(n.get("id", "")).startswith(key)]
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(f"node {key!r}: {len(hits)} matches (give a longer id prefix or a step number)")


def _attempts(parent: dict, nodes: list[dict], node2parent: dict) -> str:
    kids = [n for n in nodes if node2parent.get(n["id"]) == parent["id"]]
    out = []
    for i, n in enumerate(kids, 1):
        m = n.get("metric") or {}
        block = [f"Attempt #{i}:", f"Design: {str(n.get('plan') or '')[:800]}"]
        if n.get("is_buggy"):
            block.append("Results: The implementation of this design has bugs.")
        else:
            if n.get("analysis"):
                block.append(f"Results: {n['analysis']}")
            if isinstance(m, dict) and m.get("value") is not None:
                block.append(f"Validation Metric: {m['value']}")
        out.append("\n".join(block))
    return "\n\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (contains logs/journal.json)")
    ap.add_argument("--node", required=True, help="node id prefix, or its step number")
    ap.add_argument("--corpus", required=True, help="dir with records.jsonl + manifest.json")
    ap.add_argument("--desc", help="description.md (default: recovered from the node's prompt)")
    ap.add_argument("--out", help="write the full trace here (markdown)")
    ap.add_argument("--packet-only", action="store_true", help="print the packet and stop")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-mechanisms", type=int, default=3)
    args = ap.parse_args()

    jr = Path(args.run) / "logs" / "journal.json"
    j = json.loads(jr.read_text(encoding="utf-8"))
    nodes = j.get("nodes", [])
    node2parent = j.get("node2parent", {})
    node = _find_node(nodes, args.node)

    desc, preview = _recover(node)
    if args.desc:
        desc = Path(args.desc).read_text(encoding="utf-8")
    if not desc:
        print("WARN: no task description recovered; pass --desc", file=sys.stderr)

    m = node.get("metric") or {}
    maximize = m.get("maximize") if isinstance(m, dict) else None
    branch = sorted([n for n in nodes if n.get("branch_id") == node.get("branch_id")
                     and n.get("stage") != "root"], key=lambda n: n.get("ctime", 0))
    succ = [n["metric"]["value"] for n in branch
            if not n.get("is_buggy") and isinstance(n.get("metric"), dict)
            and n["metric"].get("value") is not None]
    branch_best = (max(succ) if maximize is not False else min(succ)) if succ else None
    term = node.get("_term_out")
    term_out = "".join(term) if isinstance(term, list) else (term if isinstance(term, str) and term != "<OMITTED>" else "")

    packet = build_packet(
        task_desc=desc, data_preview=preview, node_id=node["id"], stage=node.get("stage", "?"),
        design=node.get("plan") or "", code_summary=node.get("code_summary") or "",
        analysis=node.get("analysis") or "",
        metric_value=(m.get("value") if isinstance(m, dict) else None), metric_maximize=maximize,
        branch_best=branch_best, term_out=term_out,
        attempts=_attempts(node, nodes, node2parent),
        trajectory=[{"stage": n.get("stage"), "is_buggy": n.get("is_buggy"),
                     "metric": (n.get("metric") or {}).get("value") if isinstance(n.get("metric"), dict) else None,
                     "plan": n.get("plan") or ""} for n in branch[-6:]])
    print(packet)
    if args.packet_only:
        return 0

    corpus = load_corpus(args.corpus)
    if corpus is None:
        return 1
    llm = SimpleNamespace(model=os.environ.get("LLM_MODEL", "gpt-5.6-terra"),
                          base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                          api_key=os.environ.get("LLM_API_KEY", ""))
    if not llm.api_key:
        print("FATAL: LLM_API_KEY not set", file=sys.stderr)
        return 1
    print(f"\n=== running analogy agent: {llm.model} @ {llm.base_url}, corpus {corpus.digest} "
          f"({len(corpus)} papers) ===\n")
    res = run_analogy_agent(packet, corpus, llm, max_turns=args.max_turns,
                            max_mechanisms=args.max_mechanisms)
    for t in res.trace:
        print(t, "\n")
    print("=== REPORT (as injected) ===\n")
    print(res.report_md or f"(empty: {res.reason})")
    print(f"\nturns {res.turns} | queries {len(res.queries)} | papers {res.paper_ids} | "
          f"tokens in/out {res.in_tokens}/{res.out_tokens} | {res.seconds:.0f}s")
    if args.out:
        Path(args.out).write_text(
            "# packet\n\n" + packet + "\n\n# trace\n\n" + "\n\n".join(res.trace) +
            "\n\n# report\n\n" + (res.report_md or f"(empty: {res.reason})") + "\n" +
            ("\n```json\n" + json.dumps(res.report, ensure_ascii=False, indent=2) + "\n```\n"
             if res.report else ""), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
