# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MLEvolve is an agentic ML-engineering system that solves Kaggle-style / MLE-bench competitions by Monte Carlo Graph Search (MCGS) over a tree of candidate solutions, with stage-specific LLM agents generating and refining code at each node. There is no test suite, package manifest, or linter config; the codebase is run as scripts.

`AGENTS.md` is a near-verbatim copy of this file for Codex — when you change one, change both.

## Setup & commands

Dependencies install in three layers, each with `--no-deps` (versions are pinned and conflict if resolved together). Python **>= 3.11** is required (`scipy==1.16.2` and friends have no 3.10 wheels):

```bash
pip install --no-deps -r requirements_base.txt   # core: omegaconf, google-genai, openai, flask, rich
pip install --no-deps -r requirements_ml.txt      # torch 2.7.1 + ML stack
pip install --no-deps -r requirements_domain.txt  # faiss-cpu, rank-bm25, domain torch-* libs
```

`requirements_domain.txt` is the union of every mle-bench domain (vision, audio, NLP, graph, geo, chem) and includes source-only packages needing a compiler; for a single competition most of it is dead weight. `k8s/setup-venv.sh` wraps all three layers and supports `SKIP_DOMAIN=1` or `DOMAIN_ONLY="rdkit==2025.3.5 ..."`, and retries line-by-line to report *every* bad pin in one pass instead of stopping at the first.

[mle-bench](https://github.com/openai/mle-bench) must also be installed separately — `engine/validation/format_server.py` imports `mlebench.grade` / `mlebench.registry` for submission grading.

Run one competition task end-to-end (launches the grading server, runs the agent under a 12 h `timeout`, then ensembles top solutions):

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
# e.g. bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

Env vars it honors (set by `k8s/entrypoint.sh`, useful locally too): `EXP_NAME` (labels the run dir — override it when running the same competition under different conditions), `DATA_DIR` / `DESC_FILE` (point straight at the data instead of the mle-bench convention), `CPUS_PER_TASK`, `TIME_LIMIT_SECS`, `SKIP_GRADING_SERVER=1`, `EXTRA_RUN_ARGS` (extra OmegaConf overrides, appended unquoted).

Run the agent loop directly (CLI args are OmegaConf dotlist overrides of `config/config.yaml`):

```bash
python run.py exp_id=<EXP_ID> dataset_dir=<DIR> \
  data_dir=<DIR>/<EXP_ID>/prepared/public \
  desc_file=<DIR>/<EXP_ID>/prepared/public/description.md
```

Override any nested config key the same way, e.g. `agent.steps=50 agent.code.model=gpt-5 coldstart.use_coldstart=False`.

Outputs land in `runs/<timestamp>_<exp_name>/` with `logs/` (journal.json, filtered_journal.json, config.yaml, best_solution.py, injected_knowledge.md) and `workspace/` (input/, working/, submission/).

## Configuration

`config/config.yaml` is the single source of truth, loaded by `config/__init__.py:load_cfg` → merged with CLI args → validated against the `@dataclass Config` schema (the dataclasses are type hints only; the real values live in the YAML). The two model slots are `code` (generation) and `feedback` (parsing/review).

**Credentials.** `agent.code` / `agent.feedback` resolve `model` / `base_url` / `api_key` from `${oc.env:LLM_MODEL,...}` / `LLM_BASE_URL` / `LLM_API_KEY`, so exporting those three env vars configures both slots without editing the YAML. Pass keys via the **environment, never as CLI overrides** — argv is visible in `ps` and `run_single_task.sh` runs under `set -x`, which would echo the key into the pod logs. `dataset_dir` still has to be filled in (or passed as an override).

Notable behavioral switches — many double as ablation toggles:
- `agent.use_diff_mode`, `agent.use_stepwise_generation` — select code-gen strategy (see Coder below).
- `agent.use_evolution` / `use_fusion` / `use_aggregation` — the three stagnation-triggered actions.
- `agent.use_global_memory` (+ `memory_embedding_model_path`, `memory_embedding_device`) — RAG memory; **set device to `cpu` if no CUDA**, default is `cuda`.
- `agent.search.use_stagnation_detection` — set `False` for a vanilla-MCTS baseline.
- `agent.seed` (default 42) — the knob for repeat runs of one condition.
- `coldstart.use_coldstart` — knowledge-base model recommendations.
- `methodology_kb_path` + `methodology_retrieval` — literature-KB injection (see below); empty path = off.

## Architecture

**Entry & loop (`run.py`).** Loads config, builds one `AgentSearch` (the coordinator) and one `Interpreter`. Phase 1 generates `agent.initial_drafts` drafts sequentially (code only, execution deferred). Phase 2 runs a `ThreadPoolExecutor` pipeline sized to `interpreter.max_parallel_run`: it executes deferred drafts and submits new `agent.step()` tasks until `agent.steps` nodes exist, calling `save_run` after each completion. SIGINT terminates subprocesses and cancels futures. `__init__.py` exposes a thin programmatic `Experiment` wrapper around the same pieces.

**Search engine (`engine/`)** — the coordinator delegates to focused modules rather than holding all logic:
- `agent_search.py` — `AgentSearch.step()` → `_run_single_step()`. **This is the dispatch heart:** given a selected parent node it picks the agent by node state — root → `draft_agent` (or `aggregation_agent` once the draft limit is hit), buggy/invalid → `debug_agent`, healthy → `improve_agent`, *unless* the branch is stagnant after ≥ half the time budget, in which case `evolution_agent` (intra-branch) or `fusion_agent` (cross-branch) fires per `fusion_vs_evolution_prob`. Generated code is run through `code_review_agent` before execution, then `result_parse_agent` + `execution.validate_executed_node` after.
- `node_selection.py` — UCT `select` with a piecewise-decaying exploration constant, plus global top-K selection and a time-based soft explore→exploit switch (`select_with_soft_switch`).
- `evaluation.py` — `backpropagate`, `check_improvement`, reward shaping.
- `execution.py` — post-run validation (submission CSV must exist; `metric == 0.0` on a maximize task is treated as a bug).
- `executor.py` — `Interpreter` runs each candidate as a **subprocess** (avoids CUDA/fork issues) with CPU pinning and N parallel slots.
- `search_node.py` — `SearchNode` (the tree node: code, plan, metric, branch_id, stage, lock, expected-child accounting) and `Journal` (the node collection, serialized to JSON).
- `solution_manager.py` — top-K candidate tracking and best-solution persistence.
- `conditions.py` — branch/global stagnation and multi-branch-fusion trigger predicates.
- `validation/` — `format_server.py` is a standalone Flask app (started by `launch_server.sh`) that wraps mle-bench grading; `format_client.py` calls it; `quality_check.py` does submission content/format checks and LLM-assisted fixes.

Node `stage` values: `root`, `draft`, `fusion_draft`, `improve`, `debug`, `evolution`, `fusion`. Nodes are grouped into branches (`branch_id`); much of the search logic is per-branch.

**Cold-start knowledge (`engine/coldstart/`)** — two *separate* injections that share this package:
- **Pretrained-model guidance** — `knowledge.py:build_guidance_description` maps a task to recommended models via `competition_tag_classified.json` + `models_guidance_classified.json`, landing on `cfg.coldstart.description`.
- **Literature techniques** — retrieved from an external methodology KB and landing on `cfg.coldstart.methodology_text`. These are deliberately **not** concatenated into `description`: they used to be, which made `draft_agent` render paper techniques inside its "Pretrained Model Strategy → Option A" block and inherit that block's "copy the Code template EXACTLY" instruction. Keep them separate. `methodology_agent.py` serves the `vector` (technique-level BM25+FAISS over a prebuilt index), `llm` (LLM picks categories) and `static` modes; `ondemand.py` serves `lazy` — abstract-level retrieval, then on-demand PDF download + one LLM extraction per paper, cached permanently into `{methodology_kb_path}/{venue}/{category}/{stem}_methodology.md`. By default the KB reaches only the initial drafts; `coldstart.inject_into_improve` extends it to `improve_agent` and is an explicit ablation.

**Agents (`agents/`).** One module per stage (`draft_agent`, `improve_agent`, `debug_agent`, `evolution_agent`, `fusion_agent`, `aggregation_agent`, `code_review_agent`, `result_parse_agent`, `data_leakage_agent`); each exposes a `run(agent, ...)` taking the `AgentSearch` instance. `triggers.py` holds the predicates deciding when the optional agents (data-leakage check, branch fusion) fire. `result_parse_agent` also determines metric direction (minimize vs maximize) up front. Subpackages:
- `coder/` — three generation strategies dispatched adaptively: `base_coder` (single-shot plan+code), `stepwise_coder` (multi-agent data-prep → model → training), `diff_coder` (SEARCH/REPLACE patch application).
- `planner/` — `base_planner` (single-stage) and `planner_with_memory` (two-stage retrieval-augmented).
- `memory/` — `GlobalMemoryLayer`: per-task store of node experience (plan/code/metric/label) with `HybridRetriever` (BM25 + FAISS). Different agents query it differently (similar records to reinforce, dissimilar to encourage novelty). Note `draft`/`evolution`/`fusion` agents instead use the *in-tree* memory `SearchNode.fetch_child_memory()`, which is unrelated.
- `prompts/` — shared prompt templates and guidelines.

**LLM layer (`llm/`).** `query()` (with optional `FunctionSpec` function-calling) and `generate()` (streaming) dispatch by model-name prefix: `gemini*` → `gemini.py`, everything else → OpenAI-compatible `openai.py`. `model_profiles.py` holds per-family sampling params (Qwen/GPT/Kimi/DeepSeek, thinking vs non-thinking) for the OpenAI backend.

## Running on Kubernetes (`k8s/`)

Cluster runs target NRP Nautilus: one competition = one `Job` on a shared PVC mounted at `/workspace`, holding the repo checkout (`/workspace/MLEvolve` with its own `.venv`), datasets, and the HF/torch caches. `k8s/README.md` has the full runbook; the shape:

```bash
bash k8s/setup-venv.sh                      # once, ON THE DEV POD (same image as the Job)
bash k8s/preflight.sh                       # verify paths/venv/deps before submitting
python k8s/validate.py                      # shape-check the Job manifests before kubectl apply
kubectl -n <NS> apply -f k8s/job-<name>.yaml
```

`k8s/entrypoint.sh` is what the Job actually runs: it activates the venv, fails fast with actionable messages if the venv/data/deps are wrong, points the caches at the PVC, and `exec`s `run_single_task.sh`. `cliproxy-deployment.yaml` runs the single shared LLM proxy every job points at (`http://cliproxy:8317/v1`) — `replicas: 1` + `Recreate` is deliberate, since two instances refreshing the same OAuth `auth/*.json` invalidate each other's tokens mid-run.

**A/B/C experiments.** The `job-<task>-abc*.yaml` manifests launch matched arms of one comparison in a single file (base / kb / kb-variant), distinguished by `EXP_NAME` and `EXTRA_RUN_ARGS: "agent.seed=NN"`, with a distinct `SERVER_ID` per arm (grading port = 5005 + SERVER_ID). `k8s/prepare-task.sh` must run before a multi-arm launch: besides downloading data it *warms the retrieval caches*, which is a correctness requirement, not an optimization — on a cold cache two concurrent arms each distil their own query via the LLM, get different queries, and race on the same tmp paths, making the contrast meaningless.

## Analysis utilities (`utils/`)

Mostly written for the KB experiments; each script's docstring explains why it exists.
- `grade_all.py` — grade every run's ensembles into one `scores.csv` (the private answers only exist on the cluster, so this is the file that carries results to local analysis).
- `grade_local.py` — grade CSVs offline; unlike `mlebench grade-sample` it treats the bundled leaderboard as optional, so a stale `leaderboard.csv` can't lose you an already-computed score.
- `compare_arms.py` — matched-K comparison table across run directories (`LABEL=path` args); enforces comparing arms only at equal ensemble size.
- `verify_kb_injection.py` — static check of where KB text actually reaches the prompts. No GPU, no API calls.
- `dump_injected.py` — regenerate `injected_knowledge.md` for older runs, writing only when the replay's sha1 matches the digest the run logged.
- `submission_fusion_utils.py` — the post-run ensembler; `refuse_all.py` re-runs it with the conservative 9 h serial-time cap lifted.

## Gotchas

- The application logger is named `"MLEvolve"` (memory uses `"memory"`); get it via `logging.getLogger("MLEvolve")`.
- The grading server is addressed by `GRADING_SERVER_PORT = 5005 + SERVER_ID` via env var; `run_single_task.sh` launches it and waits on `/health`. Use `SKIP_GRADING_SERVER=1` for non-mle-bench tasks it can't score, or `use_grading_server=False` to disable format validation entirely.
- `data_dir` must point at the prepared public split (`<dataset_dir>/<exp_id>/prepared/public`), not the dataset root — unless you override `DATA_DIR`/`DESC_FILE`, which is how non-mle-bench tasks run (see `examples/openadmet-expansionrx/`).
- Default budgets are 500 steps / 12 h; `run_single_task.sh` additionally wraps `run.py` in a hard `timeout`, and Job manifests add `activeDeadlineSeconds` on top.
- A venv copied from another machine still has `bin/activate` but a wrong hardcoded `VIRTUAL_ENV`, so activation silently no-ops and the run falls back to system python. Rebuild it in place rather than copying. (The root `.venv/` in this checkout is a laptop stub, not a runnable env.)
- Because everything installs with `--no-deps`, a failing `import X` is usually a missing *transitive* dep of X — print the real exception, not the module name.
- Post-run ensembling is a separate step: `python utils/submission_fusion_utils.py --task_id <EXP_ID> --exp_name <timestamp>_<EXP_NAME>`.
- `EXTERNAL_MEMORY_INTERFACE.md` is a design proposal (pluggable memory backends), not implemented — `agents/memory/` has no `base.py`/`factory.py`.
