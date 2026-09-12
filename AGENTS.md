# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

MLEvolve is an agentic ML-engineering system that solves Kaggle-style / MLE-bench competitions by Monte Carlo Graph Search (MCGS) over a tree of candidate solutions, with stage-specific LLM agents generating and refining code at each node. The codebase is run as scripts; CPU regression checks live in `utils/verify_*.py`.

## Setup & commands

Dependencies install in three layers, each with `--no-deps` (versions are pinned and conflict if resolved together):

```bash
pip install --no-deps -r requirements_base.txt   # core: omegaconf, google-genai, openai, flask, rich
pip install --no-deps -r requirements_ml.txt      # torch 2.7.1 + ML stack
pip install --no-deps -r requirements_domain.txt  # faiss-cpu, rank-bm25, domain torch-* libs
```

[mle-bench](https://github.com/openai/mle-bench) must also be installed separately — `engine/validation/format_server.py` imports `mlebench.grade` / `mlebench.registry` for submission grading.

Run one competition task end-to-end (launches the grading server, runs the agent under a 12 h `timeout`, then ensembles top solutions):

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
# e.g. bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

Run the agent loop directly (CLI args are OmegaConf dotlist overrides of `config/config.yaml`):

```bash
python run.py exp_id=<EXP_ID> dataset_dir=<DIR> \
  data_dir=<DIR>/<EXP_ID>/prepared/public \
  desc_file=<DIR>/<EXP_ID>/prepared/public/description.md
```

Override any nested config key the same way, e.g. `agent.steps=50 agent.code.model=gpt-5 coldstart.use_coldstart=False`.

Outputs land in `runs/<timestamp>_<exp_id>/` with `logs/` (journal.json, filtered_journal.json, config.yaml, best_solution.py) and `workspace/` (input/, working/, submission/).

## Configuration

`config/config.yaml` is the single source of truth, loaded by `config/__init__.py:load_cfg` → merged with CLI args → validated against the `@dataclass Config` schema (the dataclasses are type hints only; the real values live in the YAML). **Must be filled before running:** `dataset_dir`, and `agent.code` / `agent.feedback` `base_url` + `api_key`. The two model slots are `code` (generation) and `feedback` (parsing/review).

Notable behavioral switches in `config.yaml` — many double as ablation toggles:
- `agent.use_diff_mode`, `agent.use_stepwise_generation` — select code-gen strategy (see Coder below).
- `agent.use_evolution` / `use_fusion` / `use_aggregation` — the three stagnation-triggered actions.
- `agent.use_global_memory` (+ `memory_embedding_model_path`, `memory_embedding_device`) — RAG memory; **set device to `cpu` if no CUDA**, default is `cuda`.
- `agent.search.use_stagnation_detection` — set `False` for a vanilla-MCTS baseline.
- `coldstart.use_coldstart` — knowledge-base model recommendations.
- `analogy.enabled` + `analogy.corpus_path` — improve-stage analogy retrieval over the paper corpus (arm D); off by default.

## Architecture

**Entry & loop (`run.py`, `engine/pipeline.py`).** Loads config, builds one `AgentSearch` and one `Interpreter`. Initial drafts are generated sequentially and immediately queued for raw execution; parsing/grading/tree and global-memory updates wait until all initial drafts are generated, preserving their pending-result context. Search workers use `agent.search.parallel_search_num`; execution capacity is independent (`exec.max_parallel_run: null` auto-detects one slot per visible CUDA device, CPU-only defaults to one). Each candidate sees only its assigned GPU. Full slots queue callers. SIGINT/SIGTERM stop queued work and active candidate process groups. See `docs/execution_pipeline.md`; CPU-only regression command: `python utils/verify_execution_pipeline.py`. `__init__.py` exposes a thin sequential programmatic `Experiment` wrapper around the same agent/interpreter.

**Search engine (`engine/`)** — the coordinator delegates to focused modules rather than holding all logic:
- `agent_search.py` — `AgentSearch.step()` → `_run_single_step()`. **This is the dispatch heart:** given a selected parent node it picks the agent by node state — root → `draft_agent` (or `aggregation_agent` once the draft limit is hit), buggy/invalid → `debug_agent`, healthy → `improve_agent`, *unless* the branch is stagnant after ≥ half the time budget, in which case `evolution_agent` (intra-branch) or `fusion_agent` (cross-branch) fires per `fusion_vs_evolution_prob`. Generated code is run through `code_review_agent` before execution, then `result_parse_agent` + `execution.validate_executed_node` after.
- `node_selection.py` — UCT `select` with a piecewise-decaying exploration constant, plus global top-K selection and a time-based soft explore→exploit switch (`select_with_soft_switch`).
- `evaluation.py` — `backpropagate`, `check_improvement`, reward shaping.
- `execution.py` — post-run validation (submission CSV must exist; `metric == 0.0` on a maximize task is treated as a bug).
- `executor.py` — `Interpreter` runs each candidate as a **subprocess** (avoids CUDA/fork issues) with CPU pinning and N parallel slots.
- `search_node.py` — `SearchNode` (the tree node: code, plan, metric, branch_id, stage, lock, expected-child accounting) and `Journal` (the node collection, serialized to JSON).
- `solution_manager.py` — top-K candidate tracking and best-solution persistence.
- `conditions.py` — branch/global stagnation and multi-branch-fusion trigger predicates.
- `coldstart/` — maps a task to recommended pretrained models via `competition_tag_classified.json` + `models_guidance_classified.json`; `kb_snapshot.py` records the paper corpus a D run could search.
- `analogy/` — literature retrieval at improve and optionally first draft (arm F): `corpus.py` builds BM25 over the KB repo's `output/paper_corpus/records.jsonl`; `agent.py` preserves the legacy loop, while `observed_loop.py` handles context v2 and GPT-6. `context.py` separates plans from source/runtime facts; `code_tools.py` exposes frozen allow-listed source index/read/diff tools; `report_v2.py` checks visible evidence and keeps complete mechanism blocks. Existing abstract/full-text reading remains supported. `agents/analogy_handoff.py` records explicit planner selection and passes the complete selected mechanism to the coder, with child diff/provenance in `logs/analogy/handoffs/`. Optional analogy failures are traced and return no report. See `docs/analogy_context_v2.md`.
- `validation/` — `format_server.py` is a standalone Flask app (started by `launch_server.sh`) that wraps mle-bench grading; `format_client.py` calls it; `quality_check.py` does submission content/format checks and LLM-assisted fixes.

Node `stage` values: `root`, `draft`, `fusion_draft`, `improve`, `debug`, `evolution`, `fusion`. Nodes are grouped into branches (`branch_id`); much of the search logic is per-branch.

**Candidate runtime (`engine/candidate_runtime/`, opt-in).** `candidate_runtime.enabled`
adds fixed public-data validation, cooperative budgets, immutable model/prediction snapshots
and recovery independent of journal completion. First adapter: Jigsaw Unintended Bias.
Snapshots are not new search nodes. Execution errors still route to debug while already
verified snapshots remain eligible for final submissions. See `docs/candidate_runtime.md`;
run `python utils/verify_candidate_runtime.py`. Default off; do not enable it silently for
old tasks. Keep configuration keys in both CandidateRuntimeConfig and config.yaml.

**Agents (`agents/`).** One module per stage (`draft_agent`, `improve_agent`, `debug_agent`, `evolution_agent`, `fusion_agent`, `aggregation_agent`, `code_review_agent`, `result_parse_agent`, `data_leakage_agent`); each exposes a `run(agent, ...)` taking the `AgentSearch` instance. `result_parse_agent` also determines metric direction (minimize vs maximize) up front. Subpackages:
- `coder/` — three generation strategies dispatched adaptively: `base_coder` (single-shot plan+code), `stepwise_coder` (multi-agent data-prep → model → training), `diff_coder` (SEARCH/REPLACE patch application).
- `planner/` — `base_planner` (single-stage) and `planner_with_memory` (two-stage retrieval-augmented).
- `memory/` — `GlobalMemoryLayer`: per-task store of node experience (plan/code/metric/label) with `HybridRetriever` (BM25 + FAISS). Different agents query it differently (similar records to reinforce, dissimilar to encourage novelty).
- `prompts/` — shared prompt templates and guidelines.

**LLM layer (`llm/`).** `query()` (with optional `FunctionSpec`) and streaming `generate()` dispatch `gemini*` to `gemini.py` and other models to `openai.py`. GPT-6 uses `responses.py`, including full tool/opaque-state replay, explicit code/feedback roles, bounded transport retries and terminal errors; older model routes remain intact. Active defaults are `gpt-6-astra/high`. Keep all runtime generative calls on configured slots and propagate `transport_retry_exhausted` instead of restarting outer generation loops. `utils/llm_preflight.py` records effective configuration; `telemetry.py` records safe per-call metadata without credentials or opaque reasoning contents.

## Gotchas

- The application logger is named `"MLEvolve"` (memory uses `"memory"`); get it via `logging.getLogger("MLEvolve")`.
- The grading server is addressed by `GRADING_SERVER_PORT = 5005 + SERVER_ID` via env var; `run_single_task.sh` launches it and waits on `/health`. Disable format validation entirely with `use_grading_server=False`.
- `data_dir` must point at the prepared public split (`<dataset_dir>/<exp_id>/prepared/public`), not the dataset root.
- Default budgets are 500 steps / 12 h; `run_single_task.sh` additionally wraps `run.py` in a hard `timeout`.
- Post-run ensembling is a separate step: `python utils/submission_fusion_utils.py --task_id <EXP_ID> --exp_name <timestamp>_<EXP_ID>`.
