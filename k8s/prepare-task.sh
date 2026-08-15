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
# extraction left at the default cap so the cache is genuinely populated

text = build_lazy_guidance(Path(os.environ["DESC"]).read_text(encoding="utf-8"), cfg)
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
hr
echo "READY. Launch, then confirm in the first minutes of the run logs:"
echo "    [Lazy] distilled query (cached)          <- not 'new, cached to ...'"
echo "    [Lazy] 40 candidates: 40 cached, 0 missing"
echo "    Knowledge injected at draft: ... digest <X>   <- same X in the B and C arms"
