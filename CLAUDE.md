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

Outputs land in `runs/<timestamp>_<exp_name>/` with `logs/` (journal.json, filtered_journal.json, config.yaml, best_solution.py, kb_snapshot.json, `analogy/` — one trace per improve node plus `index.jsonl`) and `workspace/` (input/, working/, submission/).

## Configuration

`config/config.yaml` is the single source of truth, loaded by `config/__init__.py:load_cfg` → merged with CLI args → validated against the `@dataclass Config` schema in `config/__init__.py`. The YAML holds the values, but the dataclass is **not** just type hints: `OmegaConf.merge` validates against it, so a top-level key present only in the YAML raises `ConfigKeyError` inside `load_cfg` and kills the run before it writes anything. Adding a config key means adding it in *both* places — `agent_paper_filter` once shipped in the YAML alone and killed a job at startup. `python utils/verify_analogy_injection.py` checks this (section 1) without needing a GPU or API keys; run it after touching either file. The two model slots are `code` (generation) and `feedback` (parsing/review).

**Credentials.** `agent.code` / `agent.feedback` resolve `model` / `base_url` / `api_key` from `${oc.env:LLM_MODEL,...}` / `LLM_BASE_URL` / `LLM_API_KEY`, so exporting those three env vars configures both slots without editing the YAML. Pass keys via the **environment, never as CLI overrides** — argv is visible in `ps` and `run_single_task.sh` runs under `set -x`, which would echo the key into the pod logs. `dataset_dir` still has to be filled in (or passed as an override).

Notable behavioral switches — many double as ablation toggles:
- `agent.use_diff_mode`, `agent.use_stepwise_generation` — select code-gen strategy (see Coder below).
- `agent.use_evolution` / `use_fusion` / `use_aggregation` — the three stagnation-triggered actions.
- `agent.use_global_memory` (+ `memory_embedding_model_path`, `memory_embedding_device`) — RAG memory; **set device to `cpu` if no CUDA**, default is `cuda`.
- `agent.search.use_stagnation_detection` — set `False` for a vanilla-MCTS baseline.
- `agent.seed` (default 42) — the knob for repeat runs of one condition.
- `coldstart.use_coldstart` — knowledge-base model recommendations.
- `analogy.enabled` + `analogy.corpus_path` — improve-stage analogy retrieval (arm D, see below); off by default.

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

**Cold-start knowledge (`engine/coldstart/`)** — pretrained-model guidance only: `knowledge.py:build_guidance_description` maps a task to recommended models via `competition_tag_classified.json` + `models_guidance_classified.json`, landing on `cfg.coldstart.description`. It also calls `kb_snapshot.py`, which for arm D writes `logs/kb_snapshot.json` (venues, paper counts and `records_sha1` of the corpus the run could search; a missing file means arm A, a file containing `"error"` means the snapshot crashed). Nothing literature-related is injected at draft time any more — the old cold-start retrieval (`methodology_agent.py`, `ondemand.py`, `coldstart.inject_into_improve`) was removed on 2026-09-02; arms B/C in `results/` were produced by it and are read by the KB repo's `analyze_runs.py` from their old `config.yaml` keys.

**Analogy retrieval (`engine/analogy/`)** — the literature path, run **per improve node** (arm D, `analogy.enabled`). Design: `Agentic_Knowledge_Base/docs/analogy_bm25_agent_design.md`. `corpus.py` loads the KB repo's `output/paper_corpus/records.jsonl` (title + tldr + abstract, no preprocessing, no embeddings) and builds BM25 once per process (Porter-stemmed, stopworded; ~1–2 min for 38k papers, preloaded in `AgentSearch.__init__`). `agent.py` is an OpenAI tools loop (`search_papers` / `read_abstract` / `submit_report`, `analogy.max_turns` cap) whose prompt follows arXiv 2605.11258: diagnose ≤3 bottlenecks of the node being improved, rewrite each as 3–6-term queries in *other subfields'* vocabulary, search, read, and map the mechanisms back as concrete interventions. **BM25 does none of the analogy — the LLM's query rewriting does; BM25 only looks the words up.** A mechanism may cite only paper ids that appeared in that run's search results (validated, else dropped), so the report cannot invent citations. `improve_agent._inject_analogy` puts the rendered report (one `### ` block per mechanism, ≤ `analogy.report_char_budget` chars) under its own heading in `prompt["Instructions"]` — the dict both generation paths render — and stores it on the child node as `SearchNode.analogy_report`, so `journal.json` records what each node saw. Per-node traces go to `logs/analogy/<parent>_<n>.md` + `index.jsonl`. Nothing in this package may end a run: every failure path logs and returns an empty report. There is no cross-arm cache to warm: the query is the run's own trajectory, different in every run by construction.

Diagnostics here follow one rule the hard way: **they must not be able to end a run, and that includes failing to import.** `write_kb_snapshot` is imported *inside* the try block in `knowledge.py` — when it wasn't, a deploy missing `kb_snapshot.py` raised ImportError at cold start in a function every arm calls, and essay s47 died with `BackoffLimitExceeded` before writing a single node. `_inject_analogy` imports `engine.analogy.agent` inside its try for the same reason.

**Agents (`agents/`).** One module per stage (`draft_agent`, `improve_agent`, `debug_agent`, `evolution_agent`, `fusion_agent`, `aggregation_agent`, `code_review_agent`, `result_parse_agent`, `data_leakage_agent`); each exposes a `run(agent, ...)` taking the `AgentSearch` instance. `improve_agent` is the only stage that consults the analogy agent (see above); draft/debug/evolution/fusion do not. `triggers.py` holds the predicates deciding when the optional agents (data-leakage check, branch fusion) fire. `result_parse_agent` also determines metric direction (minimize vs maximize) up front. Subpackages:
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

**A/D experiments.** The `job-<task>-ad-sNN.yaml` manifests launch the matched arms of one comparison in a single file (`base` / `ana`), distinguished by `EXP_NAME` and `EXTRA_RUN_ARGS` (`agent.seed=NN`; the D arm adds `analogy.enabled=True analogy.corpus_path=/workspace/Agentic_Knowledge_Base/output/paper_corpus`), with a distinct `SERVER_ID` per arm (grading port = 5005 + SERVER_ID). `k8s/prepare-task.sh <EXP_ID>` downloads the data and checks the corpus exists; that is all it does now. The old script also warmed two LLM-call caches so that the B and C arms would receive byte-identical knowledge — the analogy agent has no such shared state (its input is each node's own diagnosis), so there is nothing to warm and nothing for arms to race on. In the first minutes of a D run confirm `[analogy] corpus: N papers, sha1 ...` and the `KB snapshot` line; at the first improve node, `[analogy] node ...: N mechanism(s) ...` or `no report (reason)`. The older `job-<task>-abc*.yaml` files (essay, jigsaw, jigsaw-unintended, lmsys) are kept as the record of how the B/C runs were launched; they reference config keys this branch no longer has and will not start. tf2qa's were replaced outright by `job-tf2qa-ad-s4{2,3,4}.yaml` (same seeds, same grading ports).

## Analysis utilities (`utils/`)

Mostly written for the KB experiments; each script's docstring explains why it exists.
- `grade_all.py` — grade every run's ensembles into one `scores.csv` (the private answers only exist on the cluster, so this is the file that carries results to local analysis).
- `grade_local.py` — grade CSVs offline; unlike `mlebench grade-sample` it treats the bundled leaderboard as optional, so a stale `leaderboard.csv` can't lose you an already-computed score.
- `compare_arms.py` — matched-K comparison table across run directories (`LABEL=path` args); enforces comparing arms only at equal ensemble size.
- `verify_analogy_injection.py` — offline check of the analogy wiring: config YAML/dataclass agreement, BM25 on a toy corpus, report validation (citation rule, budget), the tools loop against a scripted client, injection into the improve prompt, and the `analogy_report` journal field. No GPU, no API calls, ~2 s.
- `replay_analogy.py` — rebuild the packet for one node of an existing run from `journal.json` and run the agent against a corpus with the LLM in the environment. The fast loop for prompt/tokenizer changes (design doc §6.2); `--packet-only` needs no key.
- `submission_fusion_utils.py` — the post-run ensembler; `refuse_all.py` re-runs it with the conservative 9 h serial-time cap lifted.

## Gotchas

- The application logger is named `"MLEvolve"` (memory uses `"memory"`); get it via `logging.getLogger("MLEvolve")`.
- The grading server is addressed by `GRADING_SERVER_PORT = 5005 + SERVER_ID` via env var; `run_single_task.sh` launches it and waits on `/health`. Use `SKIP_GRADING_SERVER=1` for non-mle-bench tasks it can't score, or `use_grading_server=False` to disable format validation entirely.
- `data_dir` must point at the prepared public split (`<dataset_dir>/<exp_id>/prepared/public`), not the dataset root — unless you override `DATA_DIR`/`DESC_FILE`, which is how non-mle-bench tasks run (see `examples/openadmet-expansionrx/`).
- Default budgets are 500 steps / 12 h, enforced by the `timeout --kill-after` inside `run_single_task.sh`; a single node is capped at 6 h (`exec.timeout: 21600`, lowered from 9 h on 2026-09-04 after stuck nodes ate most of the 12 h). A/D job manifests pin `nvidia.com/gpu.product` to >= 24 GB cards and `k8s/entrypoint.sh` installs `build-essential` for `torch.compile` — see `k8s/README.md` "GPU type". Job manifests set `activeDeadlineSeconds: 86400` (24 h) as a backstop for what runs *outside* that timeout (entrypoint setup, the unbounded fusion step). It is measured from when the Job is **accepted**, so queue time counts against it — at the old 13 h a pod that queued for over an hour was killed mid-run, which silently hands paired arms different effective budgets. Don't lower it; check queue depth instead.
- A venv copied from another machine still has `bin/activate` but a wrong hardcoded `VIRTUAL_ENV`, so activation silently no-ops and the run falls back to system python. Rebuild it in place rather than copying. (The root `.venv/` in this checkout is a laptop stub, not a runnable env.)
- Because everything installs with `--no-deps`, a failing `import X` is usually a missing *transitive* dep of X — print the real exception, not the module name.
- Post-run ensembling is a separate step: `python utils/submission_fusion_utils.py --task_id <EXP_ID> --exp_name <timestamp>_<EXP_NAME>`.
- `kb_snapshot.json` and `injected_knowledge.md` at the repo **root** are stray artifacts from a local smoke test (the snapshot points at `/tmp/fake_kb`), committed in `59d5eaf`. The real ones are per-run under `runs/<run>/logs/`, which is gitignored.
- `EXTERNAL_MEMORY_INTERFACE.md` is a design proposal (pluggable memory backends), not implemented — `agents/memory/` has no `base.py`/`factory.py`.

## Coding rules

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
