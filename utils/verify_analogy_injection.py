"""Verify the analogy retrieval wiring without a GPU, an API key, or a 12-hour run.

Checks, in order:

  1. config: the `analogy:` YAML block and the AnalogyConfig dataclass agree key-for-key, no
     key of the removed retrieval path survives in config.yaml, and the full YAML merges
     against the Config schema (a key present in only one place is a ConfigKeyError at
     startup — that has killed a job before).
  2. corpus + BM25: a tiny in-memory corpus ranks the lexically matching paper first, unknown
     ids are skipped by get(), and an all-stopword query returns nothing.
  3. report validation and rendering: uncited or unknown paper ids are dropped, a mechanism
     without any surviving citation is discarded, an honest empty report renders to "", and the
     rendered text has one `### ` heading per mechanism (what measure_adoption.py splits on).
  4. the agent loop, with a scripted stand-in for the OpenAI client: search -> read_abstract
     (one rejected id) -> submit_report (one good, one uncited mechanism) yields exactly one
     mechanism; a model that answers in prose is nudged once and then stops with a reason.
  5. improve_agent: with analogy off nothing is injected; with it on the report lands in
     prompt["Instructions"], which is the dict both generation paths render (full rewrite
     compiles it directly; the diff path's generate_initial_plan does prompt_base.copy()).
  6. SearchNode has the declared `analogy_report` field and it survives to_dict().

Run:  python utils/verify_analogy_injection.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MARKER = "SENTINEL_ANALOGY_MARKER"
REMOVED_KEYS = ("methodology_kb_path", "methodology_retrieval", "abstract_index_path",
                "lazy_pool", "agent_paper_filter", "filter_min_keep", "retr_token_budget",
                "retr_embedding_device", "inject_into_improve", "improve_token_budget",
                "methodology_text")

_failures = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _failures
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        _failures += 1
    return ok


# ---------------------------------------------------------------- 1. config

def check_config() -> None:
    print("\n1. config.yaml <-> AnalogyConfig")
    from omegaconf import OmegaConf
    from config import AnalogyConfig, Config

    yaml_text = (ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
    cfg = OmegaConf.load(ROOT / "config" / "config.yaml")
    yaml_keys = set(cfg.analogy.keys())
    dc_keys = {f.name for f in fields(AnalogyConfig)}
    check("analogy keys identical in YAML and dataclass", yaml_keys == dc_keys,
          f"yaml-only {sorted(yaml_keys - dc_keys)}, dataclass-only {sorted(dc_keys - yaml_keys)}")
    stale = [k for k in REMOVED_KEYS if f"\n{k}:" in yaml_text or f"\n  {k}:" in yaml_text]
    check("no key of the removed retrieval path in config.yaml", not stale, str(stale))
    check("analogy disabled by default (arm A unchanged)", cfg.analogy.enabled is False)
    # The fields prep_cfg() fills from the CLI before validating; without them the schema
    # merge fails on `data_dir: None`, which is not the failure this check is looking for.
    run_like = OmegaConf.from_dotlist([
        "data_dir=/tmp/x", "dataset_dir=/tmp", "desc_file=/tmp/d.md", "exp_id=essay",
        "exp_name=t", "log_dir=/tmp/l", "workspace_dir=/tmp/w"])

    def merge(extra):
        return OmegaConf.merge(OmegaConf.structured(Config),
                               OmegaConf.merge(cfg, run_like, OmegaConf.from_dotlist(extra)))
    try:
        merged = merge([])
        check("full YAML merges against the Config schema", True)
        check("analogy block reachable after merge", merged.analogy.max_turns == cfg.analogy.max_turns)
        d = merge(["analogy.enabled=True", "analogy.corpus_path=/corpus"])
        check("arm D overrides (as in k8s/job-*-ad-*.yaml) merge", d.analogy.enabled is True and d.analogy.corpus_path == "/corpus")
    except Exception as e:                       # noqa: BLE001
        check("full YAML merges against the Config schema", False, f"{type(e).__name__}: {e}")
    for stale in ("methodology_kb_path=/x", "coldstart.inject_into_improve=True", "analogy.bogus=1"):
        try:
            merge([stale])
            check(f"stale override rejected: {stale}", False, "accepted")
        except Exception:                        # noqa: BLE001 - ConfigKeyError is the point
            check(f"stale override rejected: {stale}", True)


# ---------------------------------------------------------------- 2. corpus

def _fake_corpus():
    from engine.analogy.corpus import PaperCorpus
    records = [
        {"id": "v/sym", "venue": "v", "title": "Frame averaging for equivariant networks",
         "tldr": "symmetrization via group averaging", "abstract": "We enforce equivariance by averaging over the group orbit."},
        {"id": "v/ord", "venue": "v", "title": "Ordinal regression with distance-aware losses",
         "tldr": "ordinal labels", "abstract": "Losses that respect the ordering of graded labels."},
        {"id": "v/uda", "venue": "v", "title": "Pseudo-labelling for unsupervised domain adaptation",
         "tldr": "source target shift", "abstract": "Relabel the source with a target-trained model."},
    ]
    return PaperCorpus(records, {"records_sha1": "deadbeef", "count": 3, "venues": {"v": 3}})


def check_corpus():
    print("\n2. corpus + BM25")
    c = _fake_corpus()
    hits = c.search("permutation equivariance symmetrization group averaging", k=3)
    check("lexical match ranks the equivariance paper first", bool(hits) and hits[0]["id"] == "v/sym",
          str([h["id"] for h in hits]))
    check("stemming: 'equivariant' query finds 'equivariance' doc",
          any(h["id"] == "v/sym" for h in c.search("equivariant", k=3)))
    check("search results carry no abstract", hits and "abstract" not in hits[0])
    check("get() skips unknown ids", [p["id"] for p in c.get(["v/ord", "nope"])] == ["v/ord"])
    check("all-stopword query returns nothing", c.search("the of and", k=3) == [])
    return c


# ---------------------------------------------------------------- 3. validation + rendering

def check_report(c):
    print("\n3. report validation and rendering")
    from engine.analogy.agent import render_report, validate_report

    raw = {"bottlenecks": [{"statement": "outputs are not swap-symmetric", "evidence": "..."}],
           "mechanisms": [
               {"title": "Swap symmetrisation", "paper_ids": ["v/sym", "v/fake", "v/ord"],
                "mechanism": "average over the orbit", "intervention": "swap A/B at train and test",
                "object_mappings": [{"source": "answer slots", "target": "group elements", "rationale": "orbit"}],
                "shared_relations": "output must be covariant under a known input transform"},
               {"title": "Uncited idea", "paper_ids": ["v/uda"], "mechanism": "m", "intervention": "i"},
               {"title": "No intervention", "paper_ids": ["v/sym"], "mechanism": "m", "intervention": ""},
           ]}
    seen = {"v/sym"}                       # only v/sym appeared in a search result this run
    clean, problems = validate_report(raw, seen, c, max_mechanisms=3)
    check("uncited / unknown ids dropped", clean["mechanisms"] and clean["mechanisms"][0]["paper_ids"] == ["v/sym"],
          str(clean))
    check("mechanism with no surviving citation discarded",
          all(m["title"] != "Uncited idea" for m in clean["mechanisms"]))
    check("mechanism without intervention discarded",
          all(m["title"] != "No intervention" for m in clean["mechanisms"]))
    check("problems are reported, not silent", len(problems) >= 2, str(problems))
    md = render_report(clean, c, budget_chars=8000)
    check("one '### ' heading per mechanism", md.count("\n### ") == len(clean["mechanisms"]) == 1, md[:200])
    check("citation rendered with title and id", "Frame averaging" in md and "`v/sym`" in md)
    check("honest empty report renders to ''", render_report({"bottlenecks": clean["bottlenecks"], "mechanisms": []}, c, 8000) == "")
    two = {"bottlenecks": [], "mechanisms": [clean["mechanisms"][0], dict(clean["mechanisms"][0], title="Second")]}
    tight = render_report(two, c, budget_chars=len(render_report({"bottlenecks": [], "mechanisms": [clean["mechanisms"][0]]}, c, 10**6)) + 10)
    check("budget drops whole mechanisms, never cuts one", tight.count("\n### ") == 1 and "### Second" not in tight)


# ---------------------------------------------------------------- 4. the loop

class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Call:
    def __init__(self, i, name, args):
        self.id = f"call_{i}"
        self.function = SimpleNamespace(name=name, arguments=json.dumps(args))


class _Resp:
    def __init__(self, msg):
        self.choices = [SimpleNamespace(message=msg)]
        self.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)


class _ScriptedClient:
    """Stand-in for openai.OpenAI: returns the scripted turns in order, records requests."""
    def __init__(self, turns):
        self._turns = list(turns)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.requests.append(params)
        if not self._turns:
            return _Resp(_Msg(content="(script exhausted)"))
        return _Resp(self._turns.pop(0))


def check_loop(c):
    print("\n4. agent loop with a scripted client")
    import openai
    from engine.analogy import agent as ag

    script = [
        _Msg(content="Bottleneck: swap asymmetry.",
             tool_calls=[_Call(1, "search_papers", {"query": "permutation equivariance symmetrization", "k": 3})]),
        _Msg(tool_calls=[_Call(2, "read_abstract", {"ids": ["v/sym", "v/uda"]})]),
        _Msg(tool_calls=[_Call(3, "submit_report", {
            "bottlenecks": [{"statement": "outputs not swap-symmetric"}],
            "mechanisms": [
                {"title": "Swap symmetrisation", "paper_ids": ["v/sym"], "mechanism": "orbit averaging",
                 "intervention": "swap TTA", "bottleneck_idx": 0},
                {"title": "Uncited", "paper_ids": ["v/uda"], "mechanism": "m", "intervention": "i"}]})]),
    ]
    client = _ScriptedClient(script)
    real = openai.OpenAI
    openai.OpenAI = lambda **kw: client
    try:
        llm = SimpleNamespace(model="gpt-5.6-terra", base_url="http://proxy/v1", api_key="k")
        res = ag.run_analogy_agent("PACKET", c, llm, max_turns=6)
    finally:
        openai.OpenAI = real
    check("loop ends with a report after submit_report", bool(res.report_md), res.reason)
    check("exactly the cited mechanism survives", res.paper_ids == ["v/sym"], str(res.paper_ids))
    check("query recorded", res.queries == ["permutation equivariance symmetrization"])
    check("read_abstract rejects ids the agent never saw",
          any("rejected ['v/uda']" in t for t in res.trace), str(res.trace))
    check("turn count is the number of assistant turns", res.turns == 3)
    first = client.requests[0]
    check("tools + tool_choice=auto sent", first.get("tool_choice") == "auto" and len(first.get("tools", [])) == 3)
    check("gpt-5: max_completion_tokens + reasoning_effort=none (tools on chat completions)",
          "max_completion_tokens" in first and first.get("extra_body", {}).get("reasoning_effort") == "none",
          str({k: v for k, v in first.items() if k != "messages"}))
    check("tool results appended with matching ids",
          any(m.get("role") == "tool" and m.get("tool_call_id") == "call_1" for m in client.requests[1]["messages"]))

    prose = _ScriptedClient([_Msg(content="I think you should tune the learning rate."),
                             _Msg(content="Still just prose.")])
    openai.OpenAI = lambda **kw: prose
    try:
        res2 = ag.run_analogy_agent("PACKET", c, SimpleNamespace(model="claude-opus-4-8", base_url="http://x/v1", api_key="k"), max_turns=6)
    finally:
        openai.OpenAI = real
    check("prose-only model is nudged once, then stops with a reason",
          res2.report_md == "" and "without submit_report" in res2.reason and len(prose.requests) == 2, res2.reason)
    check("claude: max_tokens, no extra_body (no thinking params in a multi-turn tool loop)",
          "max_tokens" in prose.requests[0] and "extra_body" not in prose.requests[0],
          str({k: v for k, v in prose.requests[0].items() if k != "messages"}))


# ---------------------------------------------------------------- 5. improve_agent wiring

def check_injection():
    print("\n5. improve_agent injection")
    from agents import improve_agent as ia
    from engine.analogy import agent as ag

    def fake_agent(enabled: bool):
        return SimpleNamespace(cfg=SimpleNamespace(analogy=SimpleNamespace(enabled=enabled)))

    real = ag.retrieve_for_node
    ag.retrieve_for_node = lambda agent, node: MARKER if agent.cfg.analogy.enabled else ""
    try:
        p_off = {"Instructions": {}}
        got_off = ia._inject_analogy(fake_agent(False), p_off, SimpleNamespace(id="n1"))
        check("analogy off: nothing injected, '' returned", got_off == "" and p_off["Instructions"] == {})
        p_on = {"Instructions": {"Existing": ["x"]}}
        got_on = ia._inject_analogy(fake_agent(True), p_on, SimpleNamespace(id="n2"))
        sec = p_on["Instructions"].get(ia.ANALOGY_SECTION)
        check("analogy on: report returned and injected under its own heading",
              got_on == MARKER and sec is not None and MARKER in "\n".join(sec))
        check("existing instructions untouched", p_on["Instructions"]["Existing"] == ["x"])
        copied = p_on.copy()                      # what generate_initial_plan does on the diff path
        check("reaches the diff/planner path (prompt_base.copy() keeps Instructions)",
              MARKER in "\n".join(copied["Instructions"][ia.ANALOGY_SECTION]))
    finally:
        ag.retrieve_for_node = real

    def boom(agent, node):
        raise RuntimeError("simulated failure")
    ag.retrieve_for_node = boom
    try:
        p = {"Instructions": {}}
        got = ia._inject_analogy(fake_agent(True), p, SimpleNamespace(id="n3"))
        check("a failing agent never propagates into improve", got == "" and p["Instructions"] == {})
    finally:
        ag.retrieve_for_node = real


# ---------------------------------------------------------------- 6. journal field

def check_node_field():
    print("\n6. SearchNode.analogy_report")
    from engine.search_node import Journal, SearchNode
    from utils import serialize
    names = {f.name for f in fields(SearchNode)}
    check("declared dataclass field", "analogy_report" in names)
    j = Journal()
    j.append(SearchNode(code="x", stage="improve", analogy_report="REPORT"))
    # The real writer (config.save_run -> serialize.dump_json) — it strips the node's lock and
    # parent links before to_dict(), which is why to_dict() alone would not be a fair test.
    dumped = json.loads(serialize.dumps_json(j))
    check("survives journal.json serialization", dumped["nodes"][0].get("analogy_report") == "REPORT")
    check("defaults to None", SearchNode(code="x", stage="improve").analogy_report is None)


def main() -> int:
    check_config()
    c = check_corpus()
    check_report(c)
    check_loop(c)
    check_injection()
    check_node_field()
    print(f"\n{'ALL CHECKS PASSED' if not _failures else f'{_failures} CHECK(S) FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
