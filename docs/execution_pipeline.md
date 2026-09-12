# Initial-draft execution pipeline

Initial drafts are still generated and reviewed **sequentially**. Each reviewed draft is
submitted immediately to the execution pool, while the coordinator generates the next draft.
There is no longer an all-drafts-ready barrier before starting candidate Python processes.

Only raw subprocess execution crosses this boundary. Result parsing, grading, global-memory
writes, best-solution updates and journal/tree updates wait until initial generation finishes.
Consequently, draft 2/3 still see previous designs with pending outcomes; faster execution
cannot feed a score or successful memory record into their prompts. The first-draft analogy
gate remains serial and unchanged; F retains first-draft and improve injection, including the
configured full-text reader. Initial raw results are saved as `logs/executions/<node_id>.json`
even if later generation is still running. They are processed through the existing deferred-node
path once, without rerunning the candidate.

## Execution capacity

`agent.search.parallel_search_num` controls search/LLM workers (default 3).
`exec.max_parallel_run` controls execution capacity independently:

| Setting / visible devices | Concurrent candidates |
| --- | --- |
| `null` (default), one CUDA GPU | 1 |
| `null`, N CUDA GPUs | up to N, one per GPU, also capped by available CPUs |
| positive integer M, N CUDA GPUs | up to min(M, N, available CPUs) |
| `null`, CPU-only | 1 |
| positive integer M, CPU-only | up to min(M, available CPUs) |

There is no GPU model or VRAM-size allowlist: A10, A100, H100, RTX and other CUDA GPU
models follow the same rule. GPU discovery uses PyTorch's visible-device count. Each
candidate receives exactly one identifier through its **own** `CUDA_VISIBLE_DEVICES` environment;
the parent's mask is unchanged. Numeric IDs, GPU UUIDs and visible MIG instance IDs are
preserved. Inside candidate code the assigned device is `cuda:0`; code must not overwrite
the mask or assume host GPU ordinals. CPU-only candidates receive an empty CUDA mask.

This is per-experiment scheduling, relying on the cluster to allocate exclusive devices to
Pods. It does not coordinate separate experiments sharing a physical GPU, and a MIG instance
counts as one visible device. The agent's existing memory embedding process is not a candidate
and is unchanged. A single candidate can still exceed the device's memory capacity.

Execution callers wait in FIFO order when all slots are busy; they do not fail or increment
the active count while queued. Per-candidate execution timeout starts after slot acquisition.
The existing outer wall-clock budget still includes generation and queue waiting. CPU affinity
is divided across execution slots, including remainder cores, rather than search workers.
On platforms with `sched_setaffinity`, a short launcher sets the child CPU mask and
then replaces itself with the candidate's Python process. It does not prepend code
to the candidate, so module docstrings, `from __future__` imports and source line
numbers retain their normal Python behavior. GPU visibility and process groups are
preserved across the replacement.

Execution summaries report actual elapsed seconds. Only a deadline enforced by the
executor is reported as exceeding the execution time limit; a candidate that raises
`TimeoutError` earlier keeps its traceback and elapsed time without that misleading
claim. Both still follow the existing timeout/debug handling.

Process groups keep a candidate's descendants within its slot: timeout, cancellation and
completion clean them up before the slot can be reused. SIGTERM from `run_single_task.sh`'s
existing timeout unwinds the pipeline and stops active/queued executions.

## Launching a future run

No new YAML setting is required for the one-GPU Jobs: the new default automatically gives
them one execution slot while retaining three search workers. To lower a multi-GPU run to
one execution slot, pass `exec.max_parallel_run=1` through the existing CLI overrides.

`run_single_task.sh` now inherits `CUDA_VISIBLE_DEVICES` when `MEMORY_INDEX` is unset.
If both are unset, it preserves the container's full CUDA visibility. An explicitly supplied
`MEMORY_INDEX` overrides the mask; it can be a comma-separated device list, a GPU UUID, or
an empty string for CPU-only execution. Do not select devices outside the scheduler allocation.

The 2026-09-10 change was prepared **locally only**: no cluster checkout, environment, Job or
Pod was modified. Existing S51–S53 A/F experiments continue using their original runtime.
For a new run before those finish, use a separate cluster checkout and point only new Jobs
to it; do not pull this change into their shared `/workspace/MLEvolve` checkout mid-run.

## Verification

```bash
python utils/verify_execution_pipeline.py
```

The CPU-only regressions use real candidate subprocesses with mocked GPU discovery and
LLM generation. They check early execution, the initial-result barrier, preserved prior-plan
memory, first-draft-only analogy injection, budgets, no duplicate execution, single/multiple
device isolation, queueing, timeout/failure recovery, cancellation and descendant cleanup.
GPU kernels and real multi-GPU hardware still need a smoke run in an isolated future checkout.

CUDA visibility reference: [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html).
