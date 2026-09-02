#!/usr/bin/env bash
# Get a competition ready for a multi-arm run: download and split the dataset, and check that
# the paper corpus the analogy arm (D) will search is present.
#
#   bash k8s/prepare-task.sh <EXP_ID>
#
# e.g.
#   bash k8s/prepare-task.sh learning-agency-lab-automated-essay-scoring-2
#
# There is no cache to warm any more. The old retrieval ran once at cold start with the task
# description as its query, so its two LLM calls (query distiller, paper filter) had to be
# warmed and cached before a paired launch or the arms could receive different knowledge. The
# analogy agent runs per improve node on that node's own diagnosed bottleneck: its input is the
# run's own search trajectory, which is different in every run by construction, so there is
# nothing to share between arms and nothing to pre-compute. What each node received is recorded
# in the run itself (journal.json `analogy_report`, logs/analogy/).
set -uo pipefail

EXP_ID=${1:?Usage: bash k8s/prepare-task.sh <EXP_ID>}

REPO=${REPO_DIR:-/workspace/MLEvolve}
AKB=${AKB_DIR:-/workspace/Agentic_Knowledge_Base}
DATA=${DATASET_DIR:-/workspace/data/mlebench}
DESC="$DATA/$EXP_ID/prepared/public/description.md"
CORPUS="$AKB/output/paper_corpus"

hr() { printf '%s\n' "────────────────────────────────────────────────────────────────"; }

# ── 1. dataset ──────────────────────────────────────────────────────────────────────────
hr; echo "1/2  PREPARE  $EXP_ID"; hr
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

# ── 2. paper corpus ─────────────────────────────────────────────────────────────────────
hr; echo "2/2  CORPUS   $CORPUS"; hr
if [ ! -f "$CORPUS/records.jsonl" ]; then
    echo "FAIL: no records.jsonl in $CORPUS — build it in the KB repo first:"
    echo "    cd $AKB && python scripts/6_build_paper_corpus.py"
    echo "  (zero LLM calls, seconds; re-run whenever output/ gains a venue)"
    exit 1
fi
python3 - "$CORPUS" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {}
n = sum(1 for line in (d / "records.jsonl").open(encoding="utf-8") if line.strip())
print(f"  {n} papers, sha1 {m.get('records_sha1', '?')}, built {m.get('built_at', '?')}")
for v, c in sorted((m.get("venues") or {}).items()):
    print(f"    {v:<16}{c:>7}")
PY

hr
echo "READY. Launch the A + D arms; in the first minutes of the D arm confirm:"
echo "    [analogy] corpus: N papers, sha1 <X>          <- corpus loaded at start"
echo "    KB snapshot <digest>: ... corpus sha1 <X>      <- logs/kb_snapshot.json written"
echo "and at the first improve node:"
echo "    [analogy] node <id>: N mechanism(s) from M queries ...   <- or 'no report (reason)'"
