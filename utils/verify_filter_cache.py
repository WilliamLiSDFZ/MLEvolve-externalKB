"""Verify that the agent paper filter gives every arm of a draw the SAME papers.

Why this file exists. `_agent_filter_papers` is an LLM call, and `_chat` only passes
`temperature=0` when `supports_sampling_params(model)` is true — the reasoning models this
project runs on (gpt-5*, o*, claude-opus-4-7/8, fable) are excluded from sampling params
entirely. So the filter SAMPLES. The embedding reranker it replaced did not. Without a shared
decision, arms B and C of one draw each run their own filter and can keep different paper
sets, which silently destroys the pairing the whole analysis rests on.

The fix is a disk cache, and this script is the test that the cache actually holds. Note the
stub below returns a RANDOM verdict on purpose: a deterministic stub would pass even if the
cache did nothing, which is the failure mode a test like this usually has.

    python utils/verify_filter_cache.py        # no API calls, no GPU, ~1s

Check 1 is the negative control. It must print identical=False — that is the pre-fix
behaviour, and if it ever prints True the rest of this script proves nothing.
"""
import json, random, re, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.coldstart import ondemand as od

CALLS = {"n": 0}

def fake_chat(llm_cfg, prompt, max_tokens=None):
    CALLS["n"] += 1
    idxs = [int(m.group(1)) for m in re.finditer(r"^\[(\d+)\] ", prompt, re.M)]
    assert idxs, "stub parsed no paper indices"
    return json.dumps([{"i": i,
                        "decision": random.choice(["keep", "drop"]),
                        "why": "coin flip"} for i in idxs])

od._chat = fake_chat

def make_cfg(tmp):
    return types.SimpleNamespace(
        agent=types.SimpleNamespace(code=types.SimpleNamespace(model="gpt-5.6-terra")),
        abstract_index_path=str(Path(tmp) / "output" / "abstract_index"),
        retr_query_cache_dir="",
        data_dir="", filter_min_keep=5, filter_max_keep=15, filter_batch_size=10)

def cands(n=40):
    return [(od.PaperRecord({"id": f"p{i:03d}", "title": f"Paper {i}",
                             "abstract": "abstract " * 20}), 1.0 - i * 0.01)
            for i in range(n)]

def ids(xs): return [str(r.id) for r, _ in xs]

fails = []
with tempfile.TemporaryDirectory() as tmp:
    cfg, C, Q = make_cfg(tmp), cands(), "predict toxicity from text"

    # --- negative control: no cache -> arms diverge -------------------------------
    random.seed(1); a = ids(od._agent_filter_papers(C, Q, cfg)[0])
    fc = od._filter_cache_file(C, Q, cfg)
    assert fc.exists(), "first call did not write a cache file"
    fc.unlink()                                   # simulate the pre-fix world
    random.seed(2); b = ids(od._agent_filter_papers(C, Q, cfg)[0])
    print(f"1. no cache      arm1={len(a)} arm2={len(b)}  identical={a == b}")
    if a == b:
        fails.append("negative control did not diverge — the test cannot detect the bug")

    # --- the fix: warm once, then every arm reads it ------------------------------
    for f in fc.parent.glob("*.json"): f.unlink()
    CALLS["n"] = 0
    random.seed(10); warm = ids(od._agent_filter_papers(C, Q, cfg)[0])
    n_warm = CALLS["n"]
    arms = []
    for seed in (11, 12, 13):
        random.seed(seed); arms.append(ids(od._agent_filter_papers(C, Q, cfg)[0]))
    print(f"2. warm + 3 arms  llm_calls: warm={n_warm} arms={CALLS['n'] - n_warm}  "
          f"all identical={all(x == warm for x in arms)}")
    if not all(x == warm for x in arms): fails.append("arms diverged despite a warm cache")
    if CALLS["n"] - n_warm != 0:         fails.append("a cached arm still called the LLM")

    # --- key sensitivity ----------------------------------------------------------
    cfg2 = make_cfg(tmp); cfg2.filter_max_keep = 8
    print(f"3. knob change    same key={od._filter_cache_file(C, Q, cfg2) == fc}  (want False)")
    if od._filter_cache_file(C, Q, cfg2) == fc: fails.append("filter_max_keep not in the key")
    print(f"4. query change   same key="
          f"{od._filter_cache_file(C, 'other task', cfg) == fc}  (want False)")
    if od._filter_cache_file(C, "other task", cfg) == fc: fails.append("query not in the key")
    print(f"5. candidate set  same key="
          f"{od._filter_cache_file(cands(39), Q, cfg) == fc}  (want False)")
    if od._filter_cache_file(cands(39), Q, cfg) == fc: fails.append("candidate ids not in the key")

    # --- total LLM failure must NOT be cached -------------------------------------
    for f in fc.parent.glob("*.json"): f.unlink()
    od._chat = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network down"))
    out, _ = od._agent_filter_papers(C, Q, cfg)
    print(f"6. all batches fail  survivors={len(out)} (want 40, unfiltered)  "
          f"cache written={fc.exists()} (want False)")
    if len(out) != len(C): fails.append("failure path did not pass all candidates through")
    if fc.exists():        fails.append("a failed filter got cached — would freeze into every arm")

    # --- duplicate ids --------------------------------------------------------------
    od._chat = fake_chat
    for f in fc.parent.glob("*.json"): f.unlink()
    dup = C[:20] + [(od.PaperRecord({"id": "p000", "title": "dup", "abstract": "x"}), 0.5)]
    random.seed(20); d1 = od._agent_filter_papers(dup, Q, cfg)[0]
    random.seed(21); d2 = od._agent_filter_papers(dup, Q, cfg)[0]
    print(f"7. duplicate id   warm={len(d1)} cached={len(d2)}  equal={len(d1) == len(d2)}")
    if len(d1) != len(d2): fails.append("duplicate id inflated the cached survivor list")

print("\n" + ("FAIL: " + "; ".join(fails) if fails else "all checks passed"))
sys.exit(1 if fails else 0)
