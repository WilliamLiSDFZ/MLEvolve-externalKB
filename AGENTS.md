# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

MLEvolve is an agentic ML-engineering system that solves Kaggle-style / MLE-bench competitions by Monte Carlo Graph Search (MCGS) over a tree of candidate solutions, with stage-specific LLM agents generating and refining code at each node. There is no test suite, package manifest, or linter config; the codebase is run as scripts.

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
- `coldstart/` — maps a task to recommended pretrained models via `competition_tag_classified.json` + `models_guidance_classified.json`.
- `validation/` — `format_server.py` is a standalone Flask app (started by `launch_server.sh`) that wraps mle-bench grading; `format_client.py` calls it; `quality_check.py` does submission content/format checks and LLM-assisted fixes.

Node `stage` values: `root`, `draft`, `fusion_draft`, `improve`, `debug`, `evolution`, `fusion`. Nodes are grouped into branches (`branch_id`); much of the search logic is per-branch.

**Agents (`agents/`).** One module per stage (`draft_agent`, `improve_agent`, `debug_agent`, `evolution_agent`, `fusion_agent`, `aggregation_agent`, `code_review_agent`, `result_parse_agent`, `data_leakage_agent`); each exposes a `run(agent, ...)` taking the `AgentSearch` instance. `result_parse_agent` also determines metric direction (minimize vs maximize) up front. Subpackages:
- `coder/` — three generation strategies dispatched adaptively: `base_coder` (single-shot plan+code), `stepwise_coder` (multi-agent data-prep → model → training), `diff_coder` (SEARCH/REPLACE patch application).
- `planner/` — `base_planner` (single-stage) and `planner_with_memory` (two-stage retrieval-augmented).
- `memory/` — `GlobalMemoryLayer`: per-task store of node experience (plan/code/metric/label) with `HybridRetriever` (BM25 + FAISS). Different agents query it differently (similar records to reinforce, dissimilar to encourage novelty).
- `prompts/` — shared prompt templates and guidelines.

**LLM layer (`llm/`).** `query()` (with optional `FunctionSpec` function-calling) and `generate()` (streaming) dispatch by model-name prefix: `gemini*` → `gemini.py`, everything else → OpenAI-compatible `openai.py`. `model_profiles.py` holds per-family sampling params (Qwen/GPT/Kimi/DeepSeek, thinking vs non-thinking) for the OpenAI backend.

## Gotchas

- The application logger is named `"MLEvolve"` (memory uses `"memory"`); get it via `logging.getLogger("MLEvolve")`.
- The grading server is addressed by `GRADING_SERVER_PORT = 5005 + SERVER_ID` via env var; `run_single_task.sh` launches it and waits on `/health`. Disable format validation entirely with `use_grading_server=False`.
- `data_dir` must point at the prepared public split (`<dataset_dir>/<exp_id>/prepared/public`), not the dataset root.
- Default budgets are 500 steps / 12 h; `run_single_task.sh` additionally wraps `run.py` in a hard `timeout`.
- Post-run ensembling is a separate step: `python utils/submission_fusion_utils.py --task_id <EXP_ID> --exp_name <timestamp>_<EXP_ID>`.