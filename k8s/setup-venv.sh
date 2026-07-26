#!/usr/bin/env bash
# One-time setup: build the shared venv on the PVC. Run this ON THE DEV POD
# (which has the same base image as the Job, so compiled wheels match):
#   bash /workspace/MLEvolve/k8s/setup-venv.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/MLEvolve}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"   # per-project venv, inside the repo checkout

[ -d "${REPO_DIR}" ] || { echo "FATAL: ${REPO_DIR} missing — git clone the repo first"; exit 1; }

# Fail before the (long) install rather than 200 packages in: scipy==1.16.2 and friends
# require Python >= 3.11, so a 3.10 base image cannot satisfy requirements_base.txt.
PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$(printf '%s\n3.11\n' "${PYVER}" | sort -V | head -1)" != "3.11" ]; then
    echo "FATAL: Python ${PYVER} is too old — these requirements need >= 3.11."
    echo "       Use the pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime image (see k8s/job-mlevolve.yaml)."
    exit 1
fi
echo "[setup] Python ${PYVER} OK"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "[setup] creating venv at ${VENV_DIR}"
  python -m venv --system-site-packages "${VENV_DIR}"   # reuse the image's torch+cuda
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip

cd "${REPO_DIR}"

# Order matters; versions are pinned and conflict if resolved together (see CLAUDE.md).
# pip aborts the whole file on the first unsatisfiable pin, which turns cleaning up stale
# requirements into one-error-per-run whack-a-mole. So: try the fast bulk install, and if
# it fails, retry line-by-line and report EVERY bad pin at the end.
FAILED_FILE="$(mktemp)"

install_reqs() {
    local req="$1"
    echo "[setup] installing ${req} ..."
    if pip install --no-deps -q -r "${req}"; then
        return 0
    fi
    echo "[setup] bulk install of ${req} failed — retrying line by line to find the culprits"
    while IFS= read -r line; do
        line="${line%%#*}"                      # strip comments
        line="$(echo "${line}" | xargs)"        # trim whitespace
        [ -z "${line}" ] && continue
        if ! pip install --no-deps -q "${line}" 2>/dev/null; then
            echo "  [FAIL] ${line}"
            echo "${req}: ${line}" >> "${FAILED_FILE}"
        fi
    done < "${req}"
}

install_reqs requirements_base.txt
install_reqs requirements_ml.txt
install_reqs requirements_domain.txt

if [ -s "${FAILED_FILE}" ]; then
    echo ""
    echo "=============================================================="
    echo "These pins could not be installed (all of them, in one pass):"
    sed 's/^/  /' "${FAILED_FILE}"
    echo ""
    echo "Fix or remove them in the requirements files, then re-run this script."
    echo "(Already-installed packages are skipped, so re-running is fast.)"
    echo "=============================================================="
    rm -f "${FAILED_FILE}"
    exit 1
fi
rm -f "${FAILED_FILE}"

# mle-bench (grading server imports mlebench.grade / mlebench.registry)
if ! python -c "import mlebench" 2>/dev/null; then
  echo "[setup] installing mle-bench"
  pip install git+https://github.com/openai/mle-bench.git
fi

python -c "import torch, mlebench; print('[setup] OK — torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup] Done. Jobs can now use VENV_DIR=${VENV_DIR}"
