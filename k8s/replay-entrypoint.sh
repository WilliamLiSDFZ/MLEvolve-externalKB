#!/usr/bin/env bash
# One replay arm, using the existing PVC checkout and environment.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/MLEvolve}"
VENV_DIR="${VENV_DIR:-/workspace/MLEvolve/.venv}"
: "${SOURCE_RUN:?SOURCE_RUN is required}"
: "${PARENT_NODE_ID:?PARENT_NODE_ID is required}"
: "${REPLAY_EXPERIMENT_ID:?REPLAY_EXPERIMENT_ID is required}"
: "${REPLAY_ARM:?REPLAY_ARM is required}"
: "${REPLAY_SEED:?REPLAY_SEED is required}"
: "${LLM_API_KEY:?The mlevolve-llm-proxy Secret must provide LLM_API_KEY}"

[[ -f "${REPO_DIR}/replay_improve.py" ]] || {
    echo "FATAL: sync feature/noise_effect to ${REPO_DIR} before submitting these Jobs" >&2
    exit 1
}
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
[[ "${VIRTUAL_ENV:-}" == "${VENV_DIR}" ]] || {
    echo "FATAL: the selected venv did not activate at ${VENV_DIR}" >&2
    exit 1
}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"
python -B utils/mlebench_patch.py --check

# The runtime image lacks compilers; install only in this new Job's container.
if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends build-essential
fi

exec timeout --signal=TERM --kill-after=120s 21600 \
    python -B replay_improve.py \
    --runs-root /workspace/MLEvolve/runs \
    --source-run "${SOURCE_RUN}" \
    --parent-node-id "${PARENT_NODE_ID}" \
    --experiment-id "${REPLAY_EXPERIMENT_ID}" \
    --arm "${REPLAY_ARM}" \
    --seed "${REPLAY_SEED}" \
    --corpus /workspace/Agentic_Knowledge_Base/output/paper_corpus
