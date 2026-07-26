# Running MLEvolve on NRP Nautilus (Kubernetes)

One competition = one `Job`, running on the shared PVC. The design follows the existing
dev-pod setup (`~/nautilus/devpod.yaml`): same base image, same PVC (`yuze-li-vol`
mounted at `/workspace`), requests == limits.

## Layout on the PVC

```
/workspace/
├── MLEvolve/          # this repo (git clone; pull to update — jobs run whatever is here)
│   └── .venv/         # per-project python env, built once by k8s/setup-venv.sh
├── mle-bench/data/    # datasets: <EXP_ID>/prepared/public/...
├── hf_cache/          # HuggingFace cache (auto-created; survives across jobs)
└── torch_cache/       # torch hub cache (auto-created)
```

Each project on the PVC keeps its own `.venv` inside its checkout (other projects get
their own), so dependencies never bleed across projects. Make sure `.venv/` is
git-ignored.

## One-time setup

1. **Code** — on the dev pod (`kubectl -n <NS> exec -it mlevolve-agentic-knowledge-base-dev -- bash`):

   ```bash
   cd /workspace && git clone <YOUR_GIT_REMOTE> MLEvolve   # later updates: git pull
   ```

2. **Venv** — still on the dev pod (same image as the Job, so wheels match):

   ```bash
   bash /workspace/MLEvolve/k8s/setup-venv.sh
   ```

3. **Datasets** — put mle-bench data under `/workspace/mle-bench/data/<EXP_ID>/prepared/public/`
   (prepare via mle-bench on the dev pod, or `kubectl cp` from your machine).

4. **LLM credentials** — either way works:
   - Secret (recommended): `cp k8s/secret-llm.example.yaml /tmp/s.yaml`, fill it,
     `kubectl -n <NS> apply -f /tmp/s.yaml`. The entrypoint converts it into
     `agent.code.* / agent.feedback.*` overrides.
   - Or edit `config/config.yaml` in the PVC checkout directly (the secret is optional).

## Launch a run

```bash
EXP_ID=spooky-author-identification   # must be lowercase dns-safe (it becomes the job name)
sed "s/__EXP_ID__/${EXP_ID}/g" k8s/job-mlevolve.yaml | kubectl -n <NS> apply -f -
```

Monitor:

```bash
kubectl -n <NS> get jobs -l app=mlevolve
kubectl -n <NS> logs -f job/mlevolve-${EXP_ID}
```

Outputs land on the PVC: `/workspace/MLEvolve/runs/<timestamp>_<EXP_ID>/`
(`logs/best_solution.py`, `journal.json`, `workspace/submission/`). Inspect from the dev
pod, or `kubectl cp` out.

Cleanup (jobs also auto-delete 3 days after finishing):

```bash
kubectl -n <NS> delete job mlevolve-${EXP_ID}
```

## Knobs

| where | knob | default | note |
|---|---|---|---|
| job yaml | resources | 8 CPU / 32Gi / 1 GPU | keep `CPUS_PER_TASK` env in sync with the CPU limit |
| job yaml | `activeDeadlineSeconds` | 46800 (13h) | hard kill; agent budget is 12h |
| job yaml | `nodeSelector` (commented) | any GPU | pin `nvidia.com/gpu.product` if a specific card is needed |
| entrypoint | `TIME_LIMIT_SECS` | 43200 | agent time budget passed to `run_single_task.sh` |
| entrypoint | `EXTRA_RUN_ARGS` | from secret | extra OmegaConf overrides appended to `run.py` |

Multiple competitions in parallel: launch several jobs with different `EXP_ID`s — each
gets its own pod/GPU; use distinct `SERVER_ID`s only if you ever co-locate runs in one pod
(one job per pod needs no change).

## Nautilus etiquette

- `requests == limits` everywhere (cluster policy); don't request a GPU you won't use —
  `backoffLimit: 0` avoids burning GPU time on auto-retries of a broken run.
- Long `sleep infinity` pods are for development only; batch work belongs in Jobs (this).
- Model/data caches live on the PVC so jobs don't re-download on every start.
