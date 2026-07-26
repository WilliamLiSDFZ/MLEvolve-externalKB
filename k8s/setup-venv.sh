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
pip install --no-deps -r requirements_base.txt
pip install --no-deps -r requirements_ml.txt
pip install --no-deps -r requirements_domain.txt

# mle-bench (grading server imports mlebench.grade / mlebench.registry)
if ! python -c "import mlebench" 2>/dev/null; then
  echo "[setup] installing mle-bench"
  pip install git+https://github.com/openai/mle-bench.git
fi

python -c "import torch, mlebench; print('[setup] OK — torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[setup] Done. Jobs can now use VENV_DIR=${VENV_DIR}"
