#!/usr/bin/env bash
# Job entrypoint: activate the PVC venv, wire env-driven config overrides, run one task.
# Invoked by k8s/job-mlevolve.yaml; safe to run manually on the dev pod too:
#   EXP_ID=spooky-author-identification bash /workspace/MLEvolve/k8s/entrypoint.sh
set -euo pipefail

EXP_ID="${EXP_ID:?EXP_ID is required}"
REPO_DIR="${REPO_DIR:-/workspace/MLEvolve}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"   # per-project venv, inside the repo checkout
DATASET_DIR="${DATASET_DIR:-/workspace/data/mlevolve}"

# Data layout: DATA_DIR/DESC_FILE (if set) point straight at the data; otherwise fall back
# to the mle-bench convention <DATASET_DIR>/<EXP_ID>/prepared/public. Exported so
# run_single_task.sh picks them up.
export DATA_DIR="${DATA_DIR:-${DATASET_DIR}/${EXP_ID}/prepared/public}"
export DESC_FILE="${DESC_FILE:-${DATA_DIR}/description.md}"

echo "[entrypoint] EXP_ID=${EXP_ID}"
echo "[entrypoint] repo=${REPO_DIR} venv=${VENV_DIR}"
echo "[entrypoint] data=${DATA_DIR} desc=${DESC_FILE}"

# ── Sanity checks (fail fast with a clear message, not 10 minutes in) ──
[ -d "${REPO_DIR}" ]  || { echo "FATAL: ${REPO_DIR} missing — clone the repo onto the PVC"; exit 1; }
[ -f "${VENV_DIR}/bin/activate" ] || { echo "FATAL: venv missing — run k8s/setup-venv.sh once on the dev pod"; exit 1; }
[ -d "${DATA_DIR}" ]  || { echo "FATAL: data dir ${DATA_DIR} missing (set DATA_DIR to where train/test live)"; exit 1; }
[ -f "${DESC_FILE}" ] || { echo "FATAL: description ${DESC_FILE} missing (set DESC_FILE)"; exit 1; }

# A venv built on another machine (e.g. copied from a laptop) still has bin/activate but
# its hardcoded VIRTUAL_ENV path is wrong, so activation silently no-ops and the run falls
# back to system python — which then dies on the first missing dependency. Catch that here.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
if [ "${VIRTUAL_ENV:-}" != "${VENV_DIR}" ]; then
    echo "FATAL: venv did not activate (VIRTUAL_ENV='${VIRTUAL_ENV:-unset}', expected '${VENV_DIR}')."
    echo "       It was probably created on another machine. Rebuild it on the dev pod:"
    echo "         rm -rf ${VENV_DIR} && bash ${REPO_DIR}/k8s/setup-venv.sh"
    exit 1
fi
python - <<'PY' || { echo "FATAL: venv is missing dependencies — rerun k8s/setup-venv.sh"; exit 1; }
import sys
missing = []
for m in ("torch", "omegaconf", "pandas", "humanize", "rich", "openai"):
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    print("[entrypoint] missing modules:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
import torch
print("[entrypoint] deps OK — torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

# ── Model caches on the PVC (survive across jobs; avoid re-downloading) ──
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# ── LLM credentials: config/config.yaml reads LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
#    straight from the environment via OmegaConf's oc.env resolver. They are deliberately
#    NOT passed as command-line overrides — argv is visible in `ps` and is echoed by
#    `set -x` into the pod logs, which would leak the key. envFrom already put them here.
export EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-}"
if [ -z "${LLM_API_KEY:-}" ]; then
  echo "[entrypoint] WARN: LLM_API_KEY is empty — is the 'mlevolve-llm' Secret present in this namespace?"
else
  echo "[entrypoint] LLM_API_KEY present (${#LLM_API_KEY} chars), model=${LLM_MODEL:-<config default>}"
fi

# ── Resource alignment: run_single_task.sh reads these (env-overridable) ──
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-43200}"

cd "${REPO_DIR}"
exec bash run_single_task.sh "${EXP_ID}" "${DATASET_DIR}" "${SERVER_ID:-1}"
