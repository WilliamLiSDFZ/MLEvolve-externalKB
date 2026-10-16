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

## Prepare unintended-bias data and techniques

`job-prepare-jigsaw-unintended.yaml` runs the new corpus retriever on CPU (4 CPU,
16Gi), using the existing PVC and venv. It prepares missing task data with
`mlebench prepare`, then retrieves up to 100 papers, screens all chunks with Jev,
and extracts task-specific techniques. It does not start training or inject the
results, and does not call the legacy `prepare-task.sh`.

Before submitting:

- Update `/workspace/MLEvolve` on the PVC to include `corpus_retriever` and this
  Job. Jobs execute that checkout, not the local checkout used by `kubectl`.
- Use the existing `/workspace/MLEvolve/.venv`. If it predates corpus retrieval,
  install `pymupdf4llm==0.0.27 pymupdf==1.26.3` there with `pip install --no-deps`.
- Provide `/workspace/Agentic_Knowledge_Base/output/abstract_index/` with
  `manifest.json`, `records.jsonl`, `embeddings.npy`, and access to its embedding
  model/tokenizer. Model caches use `/workspace/hf_cache`.
- Data is expected at `/workspace/data/mlebench/jigsaw-unintended-bias-in-toxicity-classification/prepared/public/`.
  If not prepared, provide `/workspace/.kaggle/kaggle.json` and accept the
  competition rules first. Existing prepared data skips downloading.
- The same namespace must contain the existing `mlevolve-llm-proxy` Secret with
  `LLM_API_KEY` and the `cliproxy` Service. The Job uses `gpt-5.6-terra`.
- Create `mlevolve-jev` from a local credentials file containing
  `TYPESAFE_API_KEY=...` (do not commit the credentials file):

  ```bash
  kubectl -n <NS> create secret generic mlevolve-jev --from-env-file=/path/to/jev.env
  ```

Submit and follow logs:

```bash
kubectl -n <NS> apply -f k8s/job-prepare-jigsaw-unintended.yaml
kubectl -n <NS> logs -f job/mlevolve-prepare-jubias
```

Startup logs show each dependency import with its elapsed time. An import taking
longer than 60 seconds prints a Python stack trace every minute until it finishes,
so a stalled startup identifies the active module instead of remaining silent.

Results replace the same PVC directory on each run:
`/workspace/MLEvolve/cache/techinique/jigsaw-unintended-bias-in-toxicity-classification/`.
Inspect `manifest.json` and `techniques.md`. A partial result exits with code 2,
so Kubernetes marks the Job failed while retaining its diagnostic output.
There are no automatic retries. The 24-hour deadline includes time waiting for
the pod to start; the Job is removed three days after it finishes, while results
remain on the PVC.

To rerun after the previous Job has stopped, delete it and submit again:

```bash
kubectl -n <NS> delete job mlevolve-prepare-jubias --ignore-not-found --wait=true
kubectl -n <NS> apply -f k8s/job-prepare-jigsaw-unintended.yaml
```

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
| job yaml | `activeDeadlineSeconds` | 86400 (24h) | wall clock **including Pending** — see below; agent budget is 12h |
| job yaml | `nodeSelector` (commented) | any GPU | pin `nvidia.com/gpu.product` if a specific card is needed |
| entrypoint | `TIME_LIMIT_SECS` | 43200 | agent time budget passed to `run_single_task.sh` |
| entrypoint | `EXTRA_RUN_ARGS` | from secret | extra OmegaConf overrides appended to `run.py` |

### Why `activeDeadlineSeconds` is 24h and not 13h

It is measured from `job.status.startTime`, which the Job controller sets when it *accepts*
the Job — not when a pod is scheduled. **Pending time is inside the deadline.** At the old
46800 (13h) the slack over the 12h agent budget was exactly one hour, so a pod that queued
longer than that got `DeadlineExceeded` mid-run.

That is worse than losing one run. A/B/C arms are applied together but schedule at different
times, so a queue-delayed arm gets a *smaller* effective compute budget than its pair — an
uncontrolled difference inside a paired comparison. `analyze_runs.py` catches the killed run
as `terminated_early -> invalid` (no ensembles but top_solutions present), which discards the
whole draw rather than silently biasing it; still, tasks that queue for hours (tf2qa) would
bleed draws for a reason unrelated to the KB.

The 12h budget is not enforced by this deadline anyway — `timeout --kill-after` inside
`run_single_task.sh` is, and it always fires. `activeDeadlineSeconds` only backstops what runs
*outside* that timeout: entrypoint setup and the unbounded `submission_fusion_utils.py` after
it. 24h = up to ~11h queued + 12h run + ~1h fusion, and keeps the backstop.

Do not lower it back to "12h + a bit". If you want a run to fail fast when the cluster is
full, check queue depth before applying rather than shrinking this number.

Multiple competitions in parallel: launch several jobs with different `EXP_ID`s — each
gets its own pod/GPU; use distinct `SERVER_ID`s only if you ever co-locate runs in one pod
(one job per pod needs no change).

## Nautilus etiquette

- `requests == limits` everywhere (cluster policy); don't request a GPU you won't use —
  `backoffLimit: 0` avoids burning GPU time on auto-retries of a broken run.
- Long `sleep infinity` pods are for development only; batch work belongs in Jobs (this).
- Model/data caches live on the PVC so jobs don't re-download on every start.
