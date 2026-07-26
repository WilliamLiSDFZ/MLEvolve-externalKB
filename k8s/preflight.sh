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
  # Report exactly WHICH module is missing — "deps missing" alone is useless.
  # Core = the run cannot start without it. Optional = task-specific, only a warning.
  dep_out="$(python - <<'PY'
core = ["torch", "omegaconf", "pandas", "numpy", "humanize", "rich", "openai", "anthropic",
        "sklearn", "faiss", "rank_bm25"]
optional = {"rdkit": "molecular features (this competition)",
            "mlebench": "grading server (not needed when use_grading_server=False)",
            "lightgbm": "gradient boosting", "xgboost": "gradient boosting"}
miss_core, miss_opt = [], []
for m in core:
    # Show the real exception: --no-deps installs mean the failure is usually a missing
    # transitive dependency, not the module we named.
    try: __import__(m)
    except Exception as e: miss_core.append(f"{m} [{type(e).__name__}: {e}]")
for m, why in optional.items():
    try: __import__(m)
    except Exception: miss_opt.append(f"{m} ({why})")
print("CORE_MISSING=" + ",".join(miss_core))
print("OPT_MISSING=" + ",".join(miss_opt))
PY
)" || dep_out="CORE_MISSING=<python failed>"
  core_missing="$(echo "${dep_out}" | sed -n 's/^CORE_MISSING=//p')"
  opt_missing="$(echo "${dep_out}"  | sed -n 's/^OPT_MISSING=//p')"
  if [ -z "${core_missing}" ]; then
    ok "core python deps present"
  else
    bad "missing core deps: ${core_missing}  -> rerun k8s/setup-venv.sh"
  fi
  [ -n "${opt_missing}" ] && warn "optional deps absent: ${opt_missing}"
  python -c "import torch;print('  [info] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())" 2>/dev/null
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
  # config.yaml reads creds from the environment: model: ${oc.env:LLM_MODEL,<default>}
  eff_model="$(cd "${REPO_DIR}" && python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('config/config.yaml').agent.code.model)" 2>/dev/null)"
  if [ -n "${eff_model}" ]; then
    ok "effective model: ${eff_model}"
  else
    warn "could not resolve agent.code.model from config.yaml"
  fi
  grep -q 'oc.env:LLM_API_KEY' "${CFG}" && ok "api_key is read from \$LLM_API_KEY (not stored in git)" \
    || warn "config.yaml does not read api_key from the environment"
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
