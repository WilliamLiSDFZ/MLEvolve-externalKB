# Candidate validation and durable results

This opt-in protocol keeps the existing search stages (`draft`, `debug`, `improve`,
etc.). A checkpoint is an artifact belonging to a candidate, never another search
node. The first supported task adapter is Jigsaw Unintended Bias. Unsupported tasks
fail at startup if the feature is enabled; they do not silently use a different metric.

## Enable on a new run

Defaults are disabled and leave existing tasks on their legacy execution path.
For a **new** Jigsaw A/F comparison, give both arms these identical overrides:

```text
candidate_runtime.enabled=true
candidate_runtime.draft_budget_seconds=5400
candidate_runtime.candidate_budget_seconds=7200
```

They can be appended to the existing `EXTRA_RUN_ARGS` in a new task manifest. Do not
replace its existing model/arm overrides. No old S51–S53 manifests were changed.
The 90/120-minute values are initial experimental settings, not hardware performance
guarantees. Leaving either budget null retains `exec.timeout` for that stage.

Each budget includes model loading, smoke checks, training, validation, checkpoint
I/O and full test inference. The outer run deadline also bounds queued candidates;
waiting does not consume an independent per-candidate execution allowance but does
consume the run's total time. Training stops with a reserve of at least 900 seconds,
or 1.5 times estimated remaining validation/export cost if larger. Before the first
export, this cost is two full validations plus test inference and checkpoint saving.
Afterwards it uses measured validation, saving and the complete export duration
(checkpoint reload, verification inference, test prediction, publication and pruning).
The estimate depends on callback timings, not GPU model or memory size. A hard
timeout still ends non-cooperative or stuck candidates. One candidate retains one
GPU slot during all its training and inference work.

Create `CandidateSession.from_env()` before expensive preparation. Use
`session.remaining()` for the candidate's current allowance and `session.elapsed()`
for its execution duration. The session's deadline starts when the execution slot
is acquired and is already capped by the whole-run deadline. Never derive a candidate
deadline from `candidate_results/run.json` (whose `started_at` belongs to the whole
experiment), parent timestamps or the configured stage budget. This can otherwise
make a late-starting candidate immediately time out. Preprocessing may consult
`remaining()`; training uses the stop value from `step()` and then calls `finish()`.
Do not subtract finalization reserves again: the runtime owns that calculation.

Smoke timing first warms each prediction callback, then measures two subset sizes
(up to 128 and 512 rows by default). It estimates fixed invocation overhead and
per-row cost separately; the cold call is not extrapolated across the dataset.
Samples are evenly spaced across each partition. Negative slope/intercept estimates
fall back to the larger warmed sample's per-row rate. For a partition no larger than
the smoke sample, one full-partition call suffices. Optional timing calls are skipped
when remaining time reaches the minimum reserve. Every call still checks prediction
shape and finite probabilities, and all its time consumes the absolute deadline.

Small samples can remain misleading. If their estimate requests a budget stop before
any formal validation, and more than the minimum reserve remains, the worker first
validates and publishes one complete result. It then recalculates the reserve from
actual full-cycle timings and resumes training if time allows. This calibration only
reloads the current checkpoint, preserving the model/optimizer relationship. A
`budget_recalibrated` event records the old/new reserve and the decision. Once actual
costs require finalization, or the minimum reserve has been reached, it closes. No
deadline is extended; a blocking callback can still hit the outer hard timeout.

## Generated-code contract

The shared implementation guideline and code reviewer require this structure:

```python
from engine.candidate_runtime import CandidateSession

session = CandidateSession.from_env()
train_df, valid_df, test_df = session.split(train_df, test_df)
# Fit transforms only on train_df, then create model/optimizer/loaders.

session.bind(
    predict_validation=predict_validation,  # positional indices -> 1D probabilities
    predict_test=predict_test,              # positional indices -> 1D probabilities
    save_checkpoint=save_checkpoint,        # directory -> write inference state
    load_checkpoint=load_checkpoint,        # directory -> restore the same model
)
session.start_training(train_df["id"].astype(str).tolist())
stop = False
for epoch in range(epochs):
    for batch in train_loader:
        # zero_grad, forward, loss.backward, optimizer.step
        stop = session.step()              # AFTER a real completed update
        if stop:
            break
    if stop:
        break
result = session.finish()                   # also safe after step() returned True
# Optional consumers use the saved result; no extra inference is needed.
score = result["best_validation_score"]
submission_path = result["submission_path"]
```

The returned prediction paths are authoritative. A successful runtime export does
not require a legacy `./submission/submission.csv` or `submission_<node_id>.csv` to
exist; do not add a post-finish assertion for those paths or regenerate their contents.

Predict callbacks must preserve the requested row order, use identical validation
and test preprocessing, use inference mode and restore the previous train/eval mode.
Smoke/timing calls request arbitrary positional subsets; full validation/export calls
request the whole partition. Save callbacks
include weights, model configuration, tokenizer and fitted transforms. Load callbacks
restore the **existing** model so optimizer parameter references stay valid.
Checkpoint directories remain immutable after publication. The runtime checks that
reloading reproduces validation predictions before exporting test predictions.

The worker normally runs the smoke check after five real training updates, including
the candidate's actual backward path; an earlier validation/finalization also runs it.
Smoke scores are never ranked. Formal validation starts after approximately 15
minutes of training, then at most every 30 training minutes (or three times the most
recent full-validation duration, if longer). The training interval excludes smoke,
validation, saving and export time. These phases still consume the total execution
budget: a slow export cannot make another validation immediately due after one update.
The budget is checked again after each formal validation/export cycle, and closing
at that same update reuses its validation rather than repeating it.
It does not wait for an epoch boundary. A partial epoch is
allowed; the exact update count is recorded. A blocking estimator `fit` needs its
own callback/time limit to cooperate; AST checks cannot make an arbitrary training
function interruptible or prove that callbacks were called correctly.

The first formal checkpoint produces a full submission immediately. Later better
checkpoints are saved, with full exports at a lower cadence (one hour by default)
or during finalization. Continuing-training exports only load the current best
model state; finalization may load an older best checkpoint. An unfinished export
never replaces a previously complete snapshot. If the process dies before its first
complete export, it still has no usable submission.

`finish()` returns a dictionary for both normal and cooperative budget exits:

| Field | Meaning |
| --- | --- |
| `best_validation_score`, `maximize` | Full-validation score and direction of the selected saved checkpoint |
| `submission_path`, `validation_path` | Absolute paths to complete CSVs in that immutable snapshot |
| `node_id`, `checkpoint_id`, `snapshot_id` | Result provenance |
| `selected_optimizer_steps` | Updates completed at the selected checkpoint |
| `optimizer_steps` | Total updates completed by this candidate, including work after that checkpoint |
| `reason` | `completed` or `budget_exhausted` |
| `version`, `finished_at`, `elapsed_seconds` | Result schema version (1), finish timestamp and total execution duration |

The same metadata is stored in `worker_finished.json`. Repeated `finish()` calls
return a copy of the same result with no further validation, export or score printing.
The session itself prints `Final Validation Score`. Before finalization,
`session.best_validation_score` and the compatibility alias `session.best_score` are
read-only properties: `None` before formal validation, then the best saved full
validation score (which may still await export). Generated code and the reviewer
share this contract. Do not guess other attributes, treat the return value as a scalar,
or run another prediction/metric pass after finishing.

## Validation and metric

Before code generation, the controller builds a deterministic target-stratified
holdout from **public** train.csv (5% by default) using the experiment seed. The
same public data, seed and fraction produce the same contract across A/F runs.
Split indices, source hashes, row IDs and `jubias-continuous-auc-v1` are recorded.
All nine identity groups and all 27 bias AUC components must be defined; deterministic
split retries are only for that condition, never for selecting a better score.

Continuous predictions feed ROC-AUC; ground-truth target/identity values are
thresholded at 0.5. The local evaluator applies the 0.25 overall-AUC + 0.75 mean bias
power-mean formula. It rejects missing metric terms, NaN/Inf, wrong IDs/order,
incomplete test predictions and out-of-range probabilities. Constant predictions
from a trained model can be weak but mathematically valid; low scores alone do not
invalidate a result. A perfect score still requires the existing leakage review
when enabled. Private MLE-bench answers/scores are never used for online selection.

The fixed split is a new experimental protocol. It does not retrospectively change
the validation metrics of old experiments. Code review still checks training/data
usage; file checks and callback instrumentation do not formally prove absence of
all leakage or misuse in arbitrary generated code.

## State and recovery

`execution_status` and `artifact_status` are separate SearchNode fields. Clean and
cooperative budget exits with a valid snapshot can enter improve. A later exception
or hard timeout retains eligible snapshots for **final submission** but keeps the
node buggy and its search metric worst, so it enters debug. There is at most one
search update and one final ensemble member per candidate. The final metric always
belongs to the exact checkpoint and submission being selected.

```text
workspace/candidate_results/
  contract/                         fixed public split, labels, IDs and hashes
  run.json
  candidates/<node_id>/
    candidate.json, solution.py     registered before execution
    execution_spec.json            absolute deadlines and runtime settings
    execution.json                 exit status (may remain running after a kill)
    checkpoints/<checkpoint_id>/   immutable model/inference state and validation
    snapshots/<snapshot_id>/       complete CSV, code, provenance manifest
  exports/<generation>/            aligned best/top-K code, metrics and CSVs
  current -> exports/<generation>  single atomic publication pointer
logs/candidate_results/
  <node_id>/                       compact events, manifests and exit metadata
  summary.json, selection.json     independently recovered artifact accounting
```

All snapshot files are staged and checked before a directory rename. Checksum and
local-score verification happens again during selection/recovery. Keep two complete
snapshots per candidate by default, plus the best checkpoint awaiting export.
Incomplete temporary directories are ignored. Best/top output writers use a process
lock around both selection and atomic publication, so an older writer cannot replace
a newer winner. Existing real legacy output directories are preserved; new runs use
compatibility symlinks through `current`.

Normal finalization and `submission_fusion_utils.py` recover artifacts without relying
on journal completion. If the whole Pod was killed before either ran, use the CPU dev
pod with the updated repository, against the original PVC run directory:

```bash
python utils/recover_candidate_results.py --run-dir /path/to/runs/EXACT_RUN_NAME
python utils/submission_fusion_utils.py --runs_root /path/to/runs \
  --task_id jigsaw-unintended-bias-in-toxicity-classification --exp_name EXACT_RUN_NAME
```

Recovery never resumes training, invokes a model API, or reads private answers.
The existing post-run grader can then score the ensemble CSVs. Execution time charged
to a selected candidate includes work after the saved checkpoint, rather than counting
only time to its export. If the final execution record was lost, recovery conservatively
uses elapsed time up to its recorded deadline; the status remains unfinished rather
than inventing a confirmed exit cause.

`fetch-run.sh` now includes the selected best submission and compact runtime metadata.
It dereferences the best/top output symlinks; large model checkpoints remain on the PVC.
Run recovery **before fetching** when the Pod died before finalization. Analysis adds
artifact counters and a wide `candidate_runtime.png`: completed, budget stop, failed
with a saved result, unfinished with a saved result, and no verified result. These
counts stay separate from journal nodes and the existing experiment exclusion rules.
The new figure includes every runtime-enabled run, including excluded ones.

The initial-draft barrier is preserved: raw workers may write snapshots, but parsing,
search updates and global-memory writes wait until all initial drafts are generated.

## Validation and rollout

```bash
python utils/verify_candidate_runtime.py
python utils/verify_execution_pipeline.py
python utils/verify_best_solution.py
# In Agentic_Knowledge_Base:
python scripts/verify_candidate_runtime_analysis.py
```

Tests use synthetic public data and real CPU subprocesses, including timeouts before
publication, during a new CSV write and after a complete export. They also check metric
recomputation, split consistency, recovery without a journal, corrupted checkpoints,
concurrent writers, failure/debug routing, and one-candidate-one-ensemble accounting.
Timing regressions advance a simulated clock while running real prediction/checkpoint
and artifact verification code. They cover cold-start/fixed overhead, noisy timing,
resuming after provisional overestimation, truly expensive finalization, and excluding
runtime work from validation cadence. Real CPU subprocesses consume the returned
result after both normal completion and budget stops.
Existing execution/GPU-allocation regressions remain required. Actual GPU training and
the revised timing policy with 90/120-minute budgets require a new isolated pilot on
the cluster; the S54–S56 pilot of the previous policy exposed premature five-update
finalization and generated-code assumptions about the missing score return value.

Sync using Git to a separate checkout for new tasks; do not pull over the shared source
used by active experiment Pods. This implementation does not apply any cluster jobs.
