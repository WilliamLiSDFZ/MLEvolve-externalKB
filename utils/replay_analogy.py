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

Draft mode (arm E) replays the task-structure variant that runs before the first draft:

    python utils/replay_analogy.py --draft --desc description.md [--data preview.txt] --corpus ...
    python utils/replay_analogy.py --draft --run <run_dir> --corpus ...     # desc/preview from the run's first draft

It needs no node: the packet is the task, a data preview and a nominal resource budget
(--time-limit-h / --exec-timeout-h / --gpu). This is the offline check the design doc asks for
before launching arm E (analogy_draft_injection_design.md §5).
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

from engine.analogy.agent import build_packet, build_task_packet, run_analogy_agent  # noqa: E402
from engine.analogy.corpus import load_corpus  # noqa: E402
from engine.analogy.fulltext import FullTextConfig  # noqa: E402


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


def _recover_draft(node: dict) -> tuple[str, str]:
    """(task description, data preview) from a DRAFT prompt: the description sits between
    '# Task description' and the next section; the preview is the assistant prefix's tail."""
    text = _prompt_text(node)
    desc = ""
    m = re.search(r"# Task description\n(.*?)(?:\n# Memory\n|\n# Instructions\n)", text, re.S)
    if m:
        desc = m.group(1).strip()
    preview = ""
    m = re.search(r"examine the dataset:\n(.*)$", text, re.S)
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


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


def _v2_packet(args, journal, selected, desc, preview):
    from engine.analogy.code_tools import CodeReadingSession, is_completed
    from engine.analogy.context import ContextOptions, packet_from_search
    import yaml

    run = Path(args.run).resolve()
    logs = run / "logs"
    config_path = logs / "config.yaml"
    class ReplayConfigLoader(yaml.SafeLoader):
        pass
    ReplayConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
                                       lambda loader, node: str(Path(*loader.construct_sequence(node))))
    config = yaml.load(config_path.read_text(), Loader=ReplayConfigLoader) if config_path.exists() else {}
    config = config or {}
    config["workspace_dir"] = str(Path(args.workspace).resolve() if args.workspace else run / "workspace")
    config["log_dir"] = str(logs)
    config.setdefault("analogy", {})["context"] = vars(ContextOptions(version=2))
    config.setdefault("candidate_runtime", {"enabled": False})
    nodes = {n["id"]: _namespace(n) for n in journal["nodes"]}
    relations = journal.get("node2parent", {})
    for n in nodes.values():
        parent_id = relations.get(n.id)
        if parent_id is None and isinstance(n.parent, str):
            parent_id = n.parent
        n.parent = nodes.get(parent_id)
        n.children = []
    current = nodes[selected["id"]]

    def finished_at(node_id):
        path = logs / "candidate_results" / node_id / "execution.json"
        if not path.exists():
            path = Path(config["workspace_dir"]) / "candidate_results/candidates" / node_id / "execution.json"
        try:
            return float(json.loads(path.read_text()).get("finished_at"))
        except (OSError, ValueError, TypeError):
            return None

    cutoff = finished_at(current.id)
    # Ancestors necessarily precede this candidate. Other branches/siblings need
    # an explicit completed timestamp; creation time alone cannot prove completion.
    allowed_ids = {current.id}
    ancestor = current.parent
    while ancestor is not None and ancestor.id not in allowed_ids:
        allowed_ids.add(ancestor.id)
        ancestor = ancestor.parent
    for n in nodes.values():
        finished = finished_at(n.id)
        if cutoff is not None and finished is not None and finished <= cutoff and is_completed(n):
            allowed_ids.add(n.id)
    retained = [n for n in nodes.values() if n.id in allowed_ids]
    for n in retained:
        n.children = [child for child in retained if child.parent is n]
    branch = [n for n in retained if n.branch_id == current.branch_id]
    replay_agent = SimpleNamespace(cfg=_namespace(config), task_desc=desc, data_preview=preview,
        branch_all_nodes={current.branch_id: branch}, branch_successful_nodes={current.branch_id:
            [n for n in branch if not n.is_buggy and getattr(n, "is_valid", False)]})
    session = CodeReadingSession.from_search(replay_agent, current)
    result = packet_from_search(replay_agent, current, code_session=session)
    result.metadata["replay"] = {"mode": "historical candidate completion cutoff", "cutoff_unix": cutoff,
        "source": "registered source only when runtime enabled; no mutable runfile fallback",
        "future_children_excluded": True, "resource_deadline": "unknown in replay; never inferred from stale config"}
    return result, session


def _save_packet(args, packet, rendered=None, code_session=None):
    if not args.out:
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(packet, encoding="utf-8")
    if rendered is not None:
        out.with_suffix(".context.json").write_text(json.dumps({"metadata": rendered.metadata, "data": rendered.data,
            "code_reads": {"ledger": code_session.ledger, "anchors": code_session.anchors} if code_session else None},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory (contains logs/journal.json)")
    ap.add_argument("--node", help="node id prefix, or its step number (improve mode)")
    ap.add_argument("--corpus", help="dir with records.jsonl + manifest.json")
    ap.add_argument("--desc", help="description.md (default: recovered from the node's prompt)")
    ap.add_argument("--draft", action="store_true",
                    help="replay the draft-stage (task-structure) variant instead of a node")
    ap.add_argument("--data", help="draft mode: file with the data preview text")
    ap.add_argument("--time-limit-h", type=float, default=12.0, help="draft mode: search budget")
    ap.add_argument("--exec-timeout-h", type=float, default=6.0, help="draft mode: per-solution cap")
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 3090, 24 GiB", help="draft mode: GPU line")
    ap.add_argument("--out", help="write the full trace here (markdown)")
    ap.add_argument("--packet-only", action="store_true", help="print the packet and stop")
    ap.add_argument("--context-version", type=int, choices=(1, 2), default=1)
    ap.add_argument("--workspace", help="immutable runtime workspace (default: RUN/workspace)")
    ap.add_argument("--model", default="gpt-6-astra", help="explicit replay model; defaults to GPT-6 Astra")
    ap.add_argument("--reasoning-effort", default="high")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--report-chars", type=int, default=None)
    ap.add_argument("--max-mechanisms", type=int, default=3)
    ap.add_argument("--fulltext", action="store_true", help="enable original-paper reading")
    ap.add_argument("--fulltext-cache", default="", help="shared PDF/text cache directory")
    ap.add_argument("--fulltext-offline", action="store_true", help="read cached versions only")
    ap.add_argument("--fulltext-max-papers", type=int, default=3)
    ap.add_argument("--fulltext-read-chars", type=int, default=8000)
    ap.add_argument("--fulltext-total-chars", type=int, default=40000)
    args = ap.parse_args()
    args.max_turns = args.max_turns or (14 if args.context_version == 2 else 10)
    args.report_chars = args.report_chars or (12000 if args.context_version == 2 else 8000)
    if not args.packet_only and not args.corpus:
        ap.error("--corpus is required unless --packet-only is used")

    mode = "draft" if args.draft else "improve"
    if args.draft:
        desc = preview = ""
        if args.run:
            j = json.loads((Path(args.run) / "logs" / "journal.json").read_text(encoding="utf-8"))
            first = next((n for n in j.get("nodes", []) if n.get("stage") == "draft"), None)
            if first:
                desc, preview = _recover_draft(first)
        if args.desc:
            desc = Path(args.desc).read_text(encoding="utf-8")
        if args.data:
            preview = Path(args.data).read_text(encoding="utf-8")
        if not desc:
            print("FATAL: draft mode needs --desc (or --run with a draft node)", file=sys.stderr)
            return 1
        packet = build_task_packet(
            task_desc=desc, data_preview=preview,
            resources={"search budget": f"{args.time_limit_h:.1f} h wall clock for the whole search",
                       "per-solution execution cap": f"{args.exec_timeout_h:.1f} h",
                       "CPU cores": 8, "GPU": args.gpu},
            pretrained="")
        rendered = None
        if args.context_version == 2:
            from engine.analogy.context import build_task_packet as task_v2, ContextOptions
            rendered = task_v2(task_desc=desc, data_preview=preview, resources={
                "replay_mode": "draft with user-supplied nominal resource assumptions",
                "run_remaining_seconds": None, "search_configured_seconds": args.time_limit_h * 3600,
                "stage_configured_cap_seconds": args.exec_timeout_h * 3600,
                "gpu_user_assumption": args.gpu, "actual_candidate_budget": "unknown in offline replay"},
                pretrained="", options=ContextOptions(version=2))
            packet = rendered.text
        print(packet)
        if args.packet_only:
            _save_packet(args, packet, rendered)
            return 0
        return _run(packet, args, mode, rendered=rendered)

    if not (args.run and args.node):
        print("FATAL: --run and --node are required (or use --draft)", file=sys.stderr)
        return 1
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

    if args.context_version == 2:
        rendered, session = _v2_packet(args, j, node, desc, preview)
        print(rendered.text)
        if args.packet_only:
            _save_packet(args, rendered.text, rendered, session)
            return 0
        return _run(rendered.text, args, mode, rendered=rendered, code_session=session)

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
        _save_packet(args, packet)
        return 0
    return _run(packet, args, mode)


def _run(packet: str, args, mode: str, *, rendered=None, code_session=None) -> int:
    corpus = load_corpus(args.corpus)
    if corpus is None:
        return 1
    llm = SimpleNamespace(model=args.model, reasoning_effort=args.reasoning_effort,
                          base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                          api_key=os.environ.get("LLM_API_KEY", ""))
    if not llm.api_key:
        print("FATAL: LLM_API_KEY not set", file=sys.stderr)
        return 1
    print(f"\n=== running analogy agent: {llm.model} @ {llm.base_url}, corpus {corpus.digest} "
          f"({len(corpus)} papers) ===\n")
    from engine.analogy.context import ContextOptions
    res = run_analogy_agent(packet, corpus, llm, max_turns=args.max_turns,
                            report_char_budget=args.report_chars,
                            context_options=ContextOptions(version=args.context_version),
                            packet_metadata=rendered.metadata if rendered else None, code_session=code_session,
                            runtime_context=rendered.data.get("runtime_context", {}) if rendered else None,
                            max_mechanisms=args.max_mechanisms, mode=mode,
                            fulltext=FullTextConfig(enabled=args.fulltext, cache_dir=args.fulltext_cache,
                                offline=args.fulltext_offline, max_papers=args.fulltext_max_papers,
                                read_chars=args.fulltext_read_chars, total_chars=args.fulltext_total_chars))
    for t in res.trace:
        print(t, "\n")
    print("=== REPORT (as injected) ===\n")
    print(res.report_md or f"(empty: {res.reason})")
    print(f"\nturns {res.turns} | queries {len(res.queries)} | papers {res.paper_ids} | "
          f"tokens in/out {res.in_tokens}/{res.out_tokens} | {res.seconds:.0f}s")
    if args.out:
        _save_packet(args, packet, rendered, code_session)
        Path(args.out).with_suffix(".context.json").write_text(json.dumps({
            "packet_metadata": rendered.metadata if rendered else None,
            "packet_data": rendered.data if rendered else None, "context": res.context,
            "code_reads": res.code_reads, "model_calls": res.model_calls}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        Path(args.out).write_text(
            "# packet\n\n" + packet + "\n\n# trace\n\n" + "\n\n".join(res.trace) +
            "\n\n# report\n\n" + (res.report_md or f"(empty: {res.reason})") + "\n" +
            ("\n```json\n" + json.dumps(res.report, ensure_ascii=False, indent=2) + "\n```\n"
             if res.report else ""), encoding="utf-8")
        print(f"wrote {args.out}")
        if res.fulltext is not None:
            Path(args.out).with_suffix(".fulltext.json").write_text(
                json.dumps(res.fulltext, ensure_ascii=False, indent=2), encoding="utf-8")
    elif res.fulltext is not None:
        print("WARNING: use --out to persist the full-text reading/version manifest", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
