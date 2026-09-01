#!/usr/bin/env bash
# Get a competition ready for a three-arm run: download the data, check that retrieval
# actually finds something, and warm the caches so concurrent arms cannot race.
#
#   bash k8s/prepare-task.sh <EXP_ID> "<probe keywords, comma separated>"
#
# e.g.
#   bash k8s/prepare-task.sh learning-agency-lab-automated-essay-scoring-2 \
#        essay,scoring,writing,educat,rubric,student,text-regress,ordinal,grading
#   bash k8s/prepare-task.sh lmsys-chatbot-arena \
#        preference,reward,rlhf,human-feedback,llm-eval,pairwise,judge,arena,alignment,chatbot
#
# Three phases, each in the venv that owns it:
#   1 PREPARE  mlebench (MLEvolve venv) downloads and splits the dataset
#   2 PROBE    probe_retrieval.py (AKB venv) — HARD GATE, exits non-zero below the threshold
#   3 WARM     one offline cold-start (MLEvolve venv) fills the query and extraction caches
#
# Phase 3 is not an optimisation, it is a correctness requirement. On a fresh task the query
# cache is empty, so two arms launched together would each call the LLM to distil the query,
# get two DIFFERENT sampled queries, and race on the same tmp path — leaving B and C retrieving
# different knowledge and making the B-vs-C contrast meaningless. They would also extract the
# same 20 papers twice, doubling token spend and racing on the same *.md.tmp files.
set -uo pipefail

EXP_ID=${1:?Usage: bash k8s/prepare-task.sh <EXP_ID> "<keywords>"}
KEYWORDS=${2:?Usage: bash k8s/prepare-task.sh <EXP_ID> "<keywords>"}

REPO=${REPO_DIR:-/workspace/MLEvolve}
AKB=${AKB_DIR:-/workspace/Agentic_Knowledge_Base}
DATA=${DATASET_DIR:-/workspace/data/mlebench}
GATE=${GATE:-8}                      # minimum on-topic hits in the top 10
DESC="$DATA/$EXP_ID/prepared/public/description.md"

# Phase 3 calls an LLM to extract techniques from PDFs, through cfg.agent.code — which reads
# these from the environment. An interactive `kubectl exec` shell has none of them set, so
# without this the config falls back to https://api.openai.com/v1 with an empty key and every
# extraction dies with APIConnectionError after the retrieval work is already done.
#
# Default to the same shared proxy the jobs use: the cached extractions are then produced by
# the same model that will consume them, rather than by whatever happened to be configured.
export LLM_BASE_URL=${LLM_BASE_URL:-http://cliproxy:8317/v1}
export LLM_MODEL=${LLM_MODEL:-gpt-5.6-terra}

hr() { printf '%s\n' "────────────────────────────────────────────────────────────────"; }

# ── 1. dataset ──────────────────────────────────────────────────────────────────────────
hr; echo "1/3  PREPARE  $EXP_ID"; hr
if [ -f "$DESC" ]; then
    echo "already prepared: $DESC"
else
    (
        set -e
        source "$REPO/.venv/bin/activate"
        export KAGGLE_CONFIG_DIR=${KAGGLE_CONFIG_DIR:-/workspace/.kaggle}
        [ -f "$KAGGLE_CONFIG_DIR/kaggle.json" ] || {
            echo "FAIL: no kaggle.json in $KAGGLE_CONFIG_DIR"
            echo "  kubectl cp ~/.kaggle/kaggle.json <pod>:/workspace/.kaggle/kaggle.json"
            exit 1; }
        mlebench prepare -c "$EXP_ID" --data-dir "$DATA"
    )
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "$DESC" ]; then
        echo
        echo "FAIL: prepare did not produce $DESC"
        echo "  If the error mentions competition rules, accept them once in a browser:"
        echo "    https://www.kaggle.com/competitions/$EXP_ID/rules"
        exit 1
    fi
fi
du -sh "$DATA/$EXP_ID/prepared" 2>/dev/null

# ── 2. retrieval gate ───────────────────────────────────────────────────────────────────
hr; echo "2/3  PROBE   (gate: >= $GATE/10 on-topic with center=on, query=llm)"; hr
PROBE_OUT=$(cd "$AKB" && source .venv/bin/activate && \
    python scripts/probe_retrieval.py --task "$DESC" --all --keywords "$KEYWORDS" 2>&1)
echo "$PROBE_OUT" | grep -vE "^[[:space:]]*$|it/s\]|B/s\]|%\|"

HITS=$(echo "$PROBE_OUT" | sed -n 's/.*center=on[[:space:]]*query=llm.*on-topic \([0-9]\+\)\/10.*/\1/p' | head -1)
if [ -z "${HITS:-}" ]; then
    echo; echo "FAIL: could not read the center=on query=llm line from the probe output."; exit 1
fi
echo
if [ "$HITS" -lt "$GATE" ]; then
    echo "GATE FAILED: $HITS/10 on-topic, need >= $GATE. Do not launch — fix retrieval first."
    exit 1
fi
echo "GATE PASSED: $HITS/10 on-topic."
echo "  Now READ THE TITLES above. The count saturates at 10/10 on well-covered tasks and"
echo "  stops discriminating; whether the papers are actually usable is a judgement call."

# ── 3. warm the caches ──────────────────────────────────────────────────────────────────
hr; echo "3/3  WARM    query cache + on-demand extraction"; hr
echo "LLM endpoint: $LLM_BASE_URL   model: $LLM_MODEL"

# Preflight before the expensive part. Extraction downloads a PDF and makes an LLM call per
# paper; if the endpoint is unreachable that is ~20 failures reported one line at a time, after
# retrieval has already run. One 2-second call up front turns that into a clear error.
if [ -z "${LLM_API_KEY:-}" ]; then
    echo
    echo "FAIL: LLM_API_KEY is not set in this shell. Export it before running:"
    echo "    export LLM_API_KEY=\$(kubectl get secret mlevolve-llm-proxy \\"
    echo "        -o jsonpath='{.data.LLM_API_KEY}' | base64 -d)   # from your laptop"
    echo "  or paste it directly inside the pod."
    exit 1
fi
if ! ( source "$REPO/.venv/bin/activate" && python - <<'PY'
import json, os, sys, urllib.error, urllib.request
base = os.environ["LLM_BASE_URL"].rstrip("/")
req = urllib.request.Request(
    base + "/chat/completions",
    data=json.dumps({"model": os.environ["LLM_MODEL"],
                     "messages": [{"role": "user", "content": "reply OK"}],
                     "max_completion_tokens": 8}).encode(),
    headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()
    print("  preflight: endpoint reachable and answering")
except urllib.error.HTTPError as e:
    print(f"  preflight FAILED: HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
    sys.exit(1)
except Exception as e:
    print(f"  preflight FAILED: {type(e).__name__}: {e}")
    print("  If this is a Service DNS name, is the proxy up?  kubectl get deploy cliproxy")
    sys.exit(1)
PY
    ); then
    echo
    echo "FAIL: cannot reach $LLM_BASE_URL — not starting extraction."
    exit 1
fi

OUT="/workspace/injected_${EXP_ID}.txt"
(
    source "$REPO/.venv/bin/activate"
    cd "$REPO"
    EXP_ID="$EXP_ID" DESC="$DESC" AKB="$AKB" OUT="$OUT" python - <<'PY'
import os, sys
from pathlib import Path
from omegaconf import OmegaConf
sys.path.insert(0, os.environ.get("REPO_DIR", "/workspace/MLEvolve"))
from engine.coldstart.ondemand import build_lazy_guidance
from engine.coldstart.knowledge import text_digest

akb = os.environ["AKB"]
cfg = OmegaConf.load("config/config.yaml")
cfg.methodology_retrieval = "lazy"
cfg.methodology_kb_path   = f"{akb}/methodology_kb"
cfg.abstract_index_path   = f"{akb}/output/abstract_index"

# Lift the per-cold-start extraction cap for the duration of the warm-up. At its default of
# 20 against a 40-paper candidate pool, one pass can only ever cache half the pool — which is
# exactly what happened on lmsys: the warm-up reported success, then the run still logged
# "40 candidates: 15 cached, 25 missing" and both KB arms went on to extract concurrently,
# racing on the same *.md.tmp paths and producing two different technique sets. The cap
# belongs in a real run, where it bounds cost; here the whole point is to leave nothing for
# the arms to fetch.
cfg.max_extractions_per_coldstart = int(cfg.get("lazy_pool", 40))

# Repeat until nothing new lands. Some papers can never be extracted — AAAI records carry no
# resolvable PDF URL at all — so a fixed number of passes cannot be the stopping rule. Those
# permanently-missing papers are harmless: _extract_one returns before writing anything, so
# both arms skip them identically.
desc = Path(os.environ["DESC"]).read_text(encoding="utf-8")
kb = Path(cfg.methodology_kb_path)
count = lambda: len(list(kb.rglob("*_methodology.md")))

text, prev = "", -1
for attempt in range(1, 6):
    before = count()
    if before == prev:
        break
    prev = before
    text = build_lazy_guidance(desc, cfg)
    after = count()
    print(f"  pass {attempt}: {before} -> {after} cached papers (+{after - before})")
    if after == before:
        print("  no further progress — the remaining candidates have no resolvable PDF")
        break

Path(os.environ["OUT"]).write_text(text, encoding="utf-8")
print(f"\ninjected text: {len(text)} chars, digest {text_digest(text)}")
print(f"saved to {os.environ['OUT']}")
if not text.strip():
    print("WARNING: retrieval returned NOTHING — the KB arms would run as expensive baselines")
    sys.exit(1)
PY
) || { echo "FAIL: cache warm-up failed"; exit 1; }

hr
QC="$AKB/output/query_cache"
echo "query cache : $(ls "$QC" 2>/dev/null | wc -l) entries in $QC"
echo "methodology : $(find "$AKB/methodology_kb" -name '*_methodology.md' 2>/dev/null | wc -l) extracted papers"

# Verify rather than assert. Re-run retrieval with extraction disabled, which is exactly what
# each arm will see at launch, and report the split. Saying "READY" without this is how the
# lmsys run got to "15 cached, 25 missing" after a warm-up that reported success.
hr; echo "VERIFY  (read-only replay of what each arm will see at launch)"; hr
(
    source "$REPO/.venv/bin/activate"
    cd "$REPO"
    DESC="$DESC" AKB="$AKB" python - <<'PY'
import os, sys
from pathlib import Path
from omegaconf import OmegaConf
sys.path.insert(0, os.environ.get("REPO_DIR", "/workspace/MLEvolve"))
from engine.coldstart import ondemand as od
from engine.coldstart.knowledge import text_digest

akb = os.environ["AKB"]
cfg = OmegaConf.load("config/config.yaml")
cfg.methodology_retrieval = "lazy"
cfg.methodology_kb_path   = f"{akb}/methodology_kb"
cfg.abstract_index_path   = f"{akb}/output/abstract_index"
cfg.max_extractions_per_coldstart = 0          # read-only: touch nothing

desc = Path(os.environ["DESC"]).read_text(encoding="utf-8")
retr = od._load_abstract_index(Path(cfg.abstract_index_path), cfg)
query = od._build_query(desc, cfg)
hits = retr.search(query, top_k=int(cfg.lazy_pool), alpha=float(cfg.retr_alpha))
best = hits[0][1] or 1.0
cands = [(r, s) for r, s in hits if (s / best) >= float(cfg.lazy_min_score)]

# The agent filter drops most of stage-1 BEFORE anything is extracted, so checking all 40
# stage-1 candidates for extractions asks the wrong question: the ~25 the filter rejects are
# never fetched and cannot be raced on. Replay the filter here so `missing` means "papers a
# real run would actually try to extract".
stage1 = len(cands)
if bool(getattr(cfg, "agent_paper_filter", False)):
    fc = od._filter_cache_file(cands, query, cfg)
    if not fc.exists():
        print(f"  stage-1 {stage1} candidates, but the agent filter has NO cached decision")
        print(f"    {fc}")
        print("\n  NOT SAFE TO LAUNCH: the filter is an LLM call with no temperature (reasoning")
        print("  models reject sampling params), so it is not deterministic. With no cached")
        print("  decision each arm runs its own filter and can keep a DIFFERENT paper set —")
        print("  the same class of race as the un-warmed query cache. WARM should have written")
        print("  this file; re-run this script.")
        sys.exit(1)
    cands, _ = od._agent_filter_papers(cands, query, cfg)
    print(f"  stage-1 {stage1} -> filter kept {len(cands)}  (cached decision {fc.name})")

cached, missing = od._split_cached(cands, Path(cfg.methodology_kb_path))

print(f"  candidates {len(cands)}: {len(cached)} cached, {len(missing)} missing")
text = od.build_lazy_guidance(desc, cfg)
print(f"  injected   {len(text)} chars, digest {text_digest(text)}")
if missing:
    print(f"\n  {len(missing)} papers still have no extraction. Both KB arms will attempt "
          f"them concurrently at launch:")
    for rec, _ in missing[:8]:
        print(f"      {rec.id}  pdf={'yes' if od._resolve_pdf(rec) else 'NO — unextractable'}")
    if any(od._resolve_pdf(r) for r, _ in missing):
        print("\n  NOT SAFE TO LAUNCH: at least one is extractable, so the arms would race on")
        print("  it and could end up with different technique sets. Re-run this script.")
        sys.exit(1)
    print("\n  All unextractable (no resolvable PDF). Both arms will skip them identically")
    print("  without writing anything — safe to launch.")
PY
) || exit 1

hr
echo "READY. In the first minutes of each run, confirm:"
echo "    [Lazy] distilled query (cached)              <- not 'new, cached to ...'"
echo "    [Filter] decision (cached <hash>.json)       <- not 'new, cached to ...'"
echo "    [Lazy] N candidates: N cached, 0 missing     <- or only unextractable ones"
echo "    Knowledge injected at draft: ... digest <X>  <- identical X in the B and C arms"
echo
echo "The two 'cached' lines are the ones that matter. Both the query distiller and the paper"
echo "filter are LLM calls WITHOUT temperature (reasoning models reject sampling params), so"
echo "either one running fresh inside an arm means that arm may have received different papers"
echo "than its pair, and the draw is not a paired comparison."
