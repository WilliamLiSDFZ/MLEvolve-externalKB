#!/bin/bash
# Run MLEvolve on a single competition task.
# Usage: bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
set -x

EXP_ID=${1:?Usage: bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]}
dataset_dir=${2:?Usage: bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]}
SERVER_ID=${3:-111}

# ── Proxy (uncomment & fill in if behind a corporate firewall) ──
# export http_proxy=http://YOUR_PROXY:PORT
# export https_proxy=http://YOUR_PROXY:PORT

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BASE_PORT=5005
GRADING_SERVER_PORT=$((BASE_PORT + SERVER_ID))
export GRADING_SERVER_PORT
export DATASET_DIR="${dataset_dir}"

# ── Launch the local grading (format-validation) server ──
# Skip for competitions that aren't in mle-bench (the server can't score them, and
# waiting on its /health just costs 30s): SKIP_GRADING_SERVER=1
if [ -n "${SKIP_GRADING_SERVER:-}" ]; then
    echo "Skipping grading server (SKIP_GRADING_SERVER set)."
else
    bash "$ROOT/launch_server.sh" "${SERVER_ID}"

    echo "Waiting for grading server on port ${GRADING_SERVER_PORT} ..."
    MAX_WAIT=30
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -s "http://127.0.0.1:${GRADING_SERVER_PORT}/health" > /dev/null 2>&1; then
            echo "Grading server ready (port ${GRADING_SERVER_PORT})."
            break
        fi
        sleep 1
        WAITED=$((WAITED + 1))
    done
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Warning: grading server may not be ready yet, proceeding anyway ..."
    fi
fi

# ── Experiment settings (env-overridable for containerized runs, see k8s/) ──
MEMORY_INDEX=${MEMORY_INDEX:-0}
start_cpu=${start_cpu:-0}
CPUS_PER_TASK=${CPUS_PER_TASK:-21}
TIME_LIMIT_SECS=${TIME_LIMIT_SECS:-43200}   # 12 hours

export MEMORY_INDEX
format_time() {
  local t=$1
  echo "$((t/3600))hrs $(((t%3600)/60))mins $((t%60))secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=500

# EXP_NAME labels the run directory. Defaults to EXP_ID; override it when running the same
# competition under different conditions (e.g. an A/B on the knowledge base) so the runs
# don't land in near-identical directories distinguishable only by timestamp.
EXP_NAME="${EXP_NAME:-${EXP_ID}}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CLOSEST_EXP_NAME="${TIMESTAMP}_${EXP_NAME}"

# ── HuggingFace cache (optional, point to a shared directory) ──
# export HF_ENDPOINT=https://huggingface.co
# export HF_DATASETS_CACHE=/path/to/hf_cache
# export HF_MODELS_CACHE=/path/to/hf_cache
# export HUGGINGFACE_HUB_CACHE=/path/to/hf_cache
# export TRANSFORMERS_CACHE=/path/to/hf_cache


# ── Run the main agent loop ──
# Data layout: defaults to the mle-bench convention
# (<dataset_dir>/<EXP_ID>/prepared/public), but DATA_DIR / DESC_FILE override it so the
# data can live anywhere (e.g. a flat dir of csv+md on a PVC).
DATA_DIR="${DATA_DIR:-${dataset_dir}/${EXP_ID}/prepared/public}"
DESC_FILE="${DESC_FILE:-${DATA_DIR}/description.md}"

# EXTRA_RUN_ARGS: optional extra OmegaConf dotlist overrides (e.g. API keys injected
# from a k8s Secret by k8s/entrypoint.sh: "agent.code.api_key=... agent.code.model=...").
# Intentionally unquoted so it expands into separate words.
CUDA_VISIBLE_DEVICES=$MEMORY_INDEX timeout --foreground --signal=TERM --kill-after=10s "${TIME_LIMIT_SECS}s" python run.py \
  exp_id="${EXP_ID}" \
  dataset_dir="${dataset_dir}" \
  data_dir="${DATA_DIR}" \
  desc_file="${DESC_FILE}" \
  exp_name="${EXP_NAME}" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  ${EXTRA_RUN_ARGS:-}

RUN_EXIT=$?

if [ $RUN_EXIT -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
elif [ $RUN_EXIT -eq 130 ]; then
  echo "Interrupted."
  exit 130
elif [ $RUN_EXIT -ne 0 ]; then
  echo "Run failed with exit code: $RUN_EXIT"
fi

# ── Post-processing: ensemble top solutions ──
echo "Running submission fusion ..."
python utils/submission_fusion_utils.py \
  --task_id "${EXP_ID}" \
  --exp_name "${CLOSEST_EXP_NAME}"
