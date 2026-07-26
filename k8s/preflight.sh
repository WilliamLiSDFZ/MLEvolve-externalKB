#!/usr/bin/env bash
# Pre-launch check: verify everything a Job needs is in place ON THE PVC.
# Run this on the dev pod BEFORE submitting the Job:
#   bash /workspace/MLEvolve/k8s/preflight.sh
#
# Checks the same paths/vars the Job sets, so a green run here means the Job should start.
set -uo pipefail

EXP_ID="${EXP_ID:-openadmet-expansionrx}"
REPO_DIR="${REPO_DIR:-/workspace/MLEvolve}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
DATASET_DIR="${DATASET_DIR:-/workspace/data/mlevolve/openadmet}"
DATA_DIR="${DATA_DIR:-${DATASET_DIR}}"
DESC_FILE="${DESC_FILE:-${DATA_DIR}/description.md}"

fail=0
ok()   { echo "  [ OK ] $1"; }
bad()  { echo "  [FAIL] $1"; fail=1; }
warn() { echo "  [warn] $1"; }

echo "Preflight for EXP_ID=${EXP_ID}"
echo ""
echo "Repo & venv"
[ -d "${REPO_DIR}" ] && ok "repo ${REPO_DIR}" || bad "repo missing: ${REPO_DIR}"
[ -f "${REPO_DIR}/run_single_task.sh" ] && ok "run_single_task.sh present" || bad "run_single_task.sh missing"
PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
if [ "$(printf '%s\n3.11\n' "${PYVER}" | sort -V | head -1)" = "3.11" ]; then
  ok "python ${PYVER} (>= 3.11)"
else
  bad "python ${PYVER} is too old — requirements need >= 3.11; use pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime"
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
  ok "venv ${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  [ "${VIRTUAL_ENV:-}" = "${VENV_DIR}" ] && ok "venv activated" \
    || bad "venv did not activate (built on another machine?) — rm -rf ${VENV_DIR} && bash k8s/setup-venv.sh"
  python - <<'PY' 2>/dev/null && ok "python deps import (torch/omegaconf/pandas/rdkit)" || bad "python deps missing — rerun k8s/setup-venv.sh"
import torch, omegaconf, pandas  # noqa
try:
    import rdkit  # noqa
except ImportError:
    raise SystemExit("rdkit missing (needed for molecular features)")
PY
  python -c "import torch;print('  [info] cuda available:', torch.cuda.is_available())" 2>/dev/null
else
  bad "venv missing: ${VENV_DIR} — run k8s/setup-venv.sh"
fi

echo ""
echo "Data"
[ -d "${DATA_DIR}" ] && ok "data dir ${DATA_DIR}" || bad "data dir missing: ${DATA_DIR}"
for f in train.csv test.csv; do
  [ -f "${DATA_DIR}/${f}" ] && ok "${f}" || bad "${DATA_DIR}/${f} missing"
done
[ -f "${DATA_DIR}/sample_submission.csv" ] && ok "sample_submission.csv" || warn "sample_submission.csv missing (optional)"
[ -f "${DESC_FILE}" ] && ok "description ${DESC_FILE}" || bad "description missing: ${DESC_FILE}"

echo ""
echo "Config"
CFG="${REPO_DIR}/config/config.yaml"
if [ -f "${CFG}" ]; then
  ok "config.yaml present"
  if grep -qE '^\s*api_key:\s*"sk-' "${CFG}"; then
    bad "a real api_key is hardcoded in config.yaml — remove it (this file is in git); use the k8s Secret"
  else
    ok "no hardcoded api_key in config.yaml"
  fi
  grep -qE '^\s*model:\s*claude' "${CFG}" && ok "model: claude (see config.yaml)" || warn "model is not claude — check config.yaml"
else
  bad "config.yaml missing"
fi

echo ""
echo "LLM credentials"
if [ -n "${LLM_API_KEY:-}" ]; then
  ok "LLM_API_KEY present in env (Job gets this from the Secret)"
else
  warn "LLM_API_KEY not set here — fine on the dev pod, but the Job needs the Secret:"
  warn "  kubectl -n <NS> get secret mlevolve-llm   # must exist, else the run has no key"
fi

echo ""
if [ "${fail}" -eq 0 ]; then
  echo "PREFLIGHT PASSED — safe to submit the Job:"
  echo "  sed \"s/__EXP_ID__/${EXP_ID}/g\" ${REPO_DIR}/k8s/job-mlevolve.yaml | kubectl -n <NS> apply -f -"
else
  echo "PREFLIGHT FAILED — fix the [FAIL] items above before submitting."
fi
exit "${fail}"
