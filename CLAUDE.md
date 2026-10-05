# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MLEvolve is an agentic ML-engineering system that solves Kaggle-style / MLE-bench competitions by Monte Carlo Graph Search (MCGS) over a tree of candidate solutions, with stage-specific LLM agents generating and refining code at each node. There is no package manifest or linter config; the codebase is run as scripts. The test suite is the set of CPU-only regression scripts `utils/verify_*.py` (see "Verification").

`AGENTS.md` is a near-verbatim copy of this file for Codex — when you change one, change both (only the header line differs).

## Setup & commands

Dependencies install in layers, each with `--no-deps` (versions are pinned and conflict if resolved together). Python **>= 3.11** is required (`scipy==1.16.2` and friends have no 3.10 wheels):

```bash
pip install --no-deps -r requirements_base.txt      # core: omegaconf, openai==1.66.3, google-genai, flask, rich, rank-bm25, nltk
pip install --no-deps -r requirements_ml.txt        # torch 2.7.1 + ML stack
pip install --no-deps -r requirements_domain.txt    # faiss-cpu, domain torch-* libs
pip install --no-deps -r requirements_fulltext.txt  # optional: pymupdf stack for analogy.fulltext (CPU paper reader)
```

`requirements_domain.txt` is the union of every mle-bench domain (vision, audio, NLP, graph, geo, chem) and includes source-only packages needing a compiler; for a single competition most of it is dead weight. `k8s/setup-venv.sh` wraps all layers and supports `SKIP_DOMAIN=1`, `DOMAIN_ONLY="rdkit==2025.3.5 ..."` and `FULLTEXT=1`, and retries line-by-line to report *every* bad pin in one pass instead of stopping at the first.

[mle-bench](https://github.com/openai/mle-bench) must also be installed separately — `engine/validation/format_server.py` imports `mlebench.grade` / `mlebench.registry` for submission grading. For Jigsaw Unintended Bias the installed grader must carry the versioned continuous-AUC fix in `patches/mlebench/` (`python utils/mlebench_patch.py --check` / `--apply`; `k8s/setup-venv.sh` applies it, `k8s/entrypoint.sh` checks it at jubias Job start, and the scoring utilities refuse to grade jubias without it). See `patches/mlebench/README.md`.

Run one competition task end-to-end (launches the grading server, runs the agent under a 12 h `timeout`, then ensembles top solutions):

```bash
bash run_single_task.sh <EXP_ID> <DATASET_DIR> [SERVER_ID]
# e.g. bash run_single_task.sh denoising-dirty-documents /mle-bench/data 1
```

Env vars it honors (set by `k8s/entrypoint.sh`, useful locally too): `EXP_NAME` (labels the run dir — override it when running the same competition under different conditions), `DATA_DIR` / `DESC_FILE` (point straight at the data instead of the mle-bench convention), `CPUS_PER_TASK`, `TIME_LIMIT_SECS`, `SKIP_GRADING_SERVER=1`, `EXTRA_RUN_ARGS` (extra OmegaConf overrides, appended unquoted), `MEMORY_INDEX` (if *set*, becomes `CUDA_VISIBLE_DEVICES`; empty string = CPU-only; unset = inherit the container's mask). It also exports `MLEVOLVE_RUN_DEADLINE` (absolute epoch) so `run.py` and the executor observe the same wall-clock deadline.

Run the agent loop directly (CLI args are OmegaConf dotlist overrides of `config/config.yaml`):

```bash
python run.py exp_id=<EXP_ID> dataset_dir=<DIR> \
  data_dir=<DIR>/<EXP_ID>/prepared/public \
  desc_file=<DIR>/<EXP_ID>/prepared/public/description.md
```

Override any nested config key the same way, e.g. `agent.steps=50 agent.code.model=gpt-5 coldstart.use_coldstart=False`.

Outputs land in `runs/<timestamp>_<exp_name>/` with `logs/` (journal.json, filtered_journal.json, config.yaml, best_solution.py, kb_snapshot.json, llm_preflight.json, llm_calls.jsonl, `executions/<node>.json` raw initial-draft results, `analogy/` traces, `candidate_results/` when the candidate runtime is on) and `workspace/` (input/, working/, submission/, and `candidate_results/` when the runtime is on).

### Verification

Every `utils/verify_*.py` runs on CPU with no API key, cluster or GPU (the two `*_proxy.py` probes are the exception: they make real calls and need `LLM_API_KEY` / `LLM_BASE_URL`). Most are `unittest` modules, so one test runs with `python -m unittest utils.verify_execution_pipeline.<Class>.<test>` and the whole file with `python utils/verify_execution_pipeline.py`. Run the ones matching what you touched:

| touched | run |
|---|---|
| `config/` (YAML or dataclass) | `verify_analogy_injection.py` (section 1 checks YAML/dataclass agreement), `verify_gpt6_configuration.py` |
| `engine/pipeline.py`, `engine/executor.py`, `engine/gpu_devices.py` | `verify_execution_pipeline.py`, `verify_best_solution.py` |
| `engine/candidate_runtime/` | `verify_candidate_runtime.py`, `verify_runtime_diagnostics.py`, `verify_best_solution.py` |
| `engine/analogy/`, `agents/analogy_handoff.py`, `agents/improve_agent.py`, `agents/draft_agent.py` | `verify_analogy_injection.py`, `verify_analogy_context.py`, `verify_analogy_observation.py`, `verify_analogy_handoff.py`, `verify_analogy_report_delivery.py`, `verify_analogy_submission_loop.py`, `verify_analogy_fulltext.py` |
| `llm/` | `verify_gpt6_transport.py`, `verify_gpt6_configuration.py`, `verify_review_contracts.py` |
| `agents/code_review_agent.py`, `agents/result_parse_agent.py` | `verify_review_contracts.py` |
| `patches/mlebench/`, `utils/mlebench_patch.py` | `verify_mlebench_grading.py --candidate` (tests the patch without touching the install) |

## Configuration

`config/config.yaml` is the single source of truth, loaded by `config/__init__.py:load_cfg` → merged with CLI args → validated against the `@dataclass Config` schema in `config/__init__.py`. The YAML holds the values, but the dataclass is **not** just type hints: `OmegaConf.merge` validates against it, so a top-level key present only in the YAML raises `ConfigKeyError` inside `load_cfg` and kills the run before it writes anything. Adding a config key means adding it in *both* places — `agent_paper_filter` once shipped in the YAML alone and killed a job at startup. The nested blocks `analogy.context` / `analogy.code_tools` / `analogy.fulltext` / `candidate_runtime` have their dataclasses in `engine/analogy/{context,code_tools,fulltext}.py` and `engine/candidate_runtime/config.py`, which `config/__init__.py` imports at module load — so those four modules must stay importable without torch, an API key or a corpus. `python utils/verify_analogy_injection.py` checks the YAML/dataclass agreement without needing a GPU or API keys; run it after touching either side. The two model slots are `code` (generation; the analogy agent also uses it) and `feedback` (parsing/review).

**Credentials.** `agent.code` / `agent.feedback` resolve `model` / `reasoning_effort` / `base_url` / `api_key` from `${oc.env:LLM_MODEL,gpt-5.6-sol}` / `LLM_REASONING_EFFORT` / `LLM_BASE_URL` / `LLM_API_KEY`, so exporting those env vars configures both slots without editing the YAML. Pass keys via the **environment, never as CLI overrides** — argv is visible in `ps` and `run_single_task.sh` runs under `set -x`, which would echo the key into the pod logs. `dataset_dir` still has to be filled in (or passed as an override).

**Preflight.** `utils/llm_preflight.py:record_configuration` runs at the top of `run.py` (and `Experiment.__init__`): it writes `logs/llm_preflight.json` (effective slots, endpoint type, SDK version, analogy input cap) and raises if `MLEVOLVE_REQUIRED_MODEL` (with `MLEVOLVE_REQUIRED_REASONING_EFFORT`, default `high`) is set and any slot disagrees, or if a required model is set and `analogy.context.version != 2`. Job manifests set it so a stale Secret or dotlist override cannot silently change the model of a paired arm. `MLEVOLVE_REQUIRE_GPT6=1` is the older guard (forces `gpt-6-astra`/high) kept for historical reproduction. The preflight validates configuration only, not endpoint availability.

Notable behavioral switches — many double as ablation toggles:
- `agent.use_diff_mode`, `agent.use_stepwise_generation` — select code-gen strategy (see Coder below).
- `agent.use_evolution` / `use_fusion` / `use_aggregation` — the three stagnation-triggered actions.
- `agent.use_global_memory` (+ `memory_embedding_model_path`, `memory_embedding_device`) — RAG memory; **set device to `cpu` if no CUDA**, default is `cuda`.
- `agent.search.use_stagnation_detection` — set `False` for a vanilla-MCTS baseline.
- `agent.search.parallel_search_num` (default 3) — search/LLM workers. `exec.max_parallel_run` (default `null`) — execution slots; `null` means `parallel_search_num` candidates train concurrently, sharing the visible GPU(s) (the original behaviour, restored 2026-10-05); a positive int sets the count explicitly (the Jigsaw A/F Jobs pass `1`). Both are capped by available CPUs.
- `agent.seed` (default 42) — the knob for repeat runs of one condition.
- `coldstart.use_coldstart` — knowledge-base model recommendations.
- `analogy.enabled` + `analogy.corpus_path` — analogy retrieval (see below); off by default. `analogy.improve` (default True) runs the agent at every improve node (arm D); `analogy.draft` (default False) runs it once on the task before the FIRST draft (arm E when `improve=False`; F when both). `analogy.context.version` (default 2) selects the context/report contract; `analogy.code_tools.enabled` exposes read-only candidate-source tools at improve; `analogy.fulltext.enabled` lets the agent read paper PDFs.
- `candidate_runtime.enabled` (default False) — cooperative candidate execution with fixed public validation and durable snapshots; Jigsaw Unintended Bias only, see below.

## Architecture

**Entry & loop (`run.py` → `engine/pipeline.py:run_search_pipeline`).** `run.py` loads config, records the LLM preflight, builds cold-start guidance, prepares the workspace (and the candidate-runtime contract when enabled), then builds one `AgentSearch` (the coordinator) and one `Interpreter`. The pipeline generates `agent.initial_drafts` drafts **sequentially** (`agent.step(..., execute_immediately=False)`), and submits each reviewed draft to an execution pool immediately — raw subprocess execution overlaps with generating the next draft, but parsing, grading, tree updates and global-memory writes wait until all initial drafts are generated, so draft 2/3 still see earlier designs as pending. Raw results are persisted to `logs/executions/<node_id>.json` and then consumed once via `agent.execute_deferred_node`. Phase 2 is a `ThreadPoolExecutor` of `parallel_search_num` workers calling `agent.step()` until `agent.steps` nodes exist, calling `save_run` after each completion. SIGINT/SIGTERM (the latter from `run_single_task.sh`'s `timeout`) raise through the pipeline, which terminates candidate process groups and cancels queued work. An LLM error flagged `transport_retry_exhausted` (see LLM layer) is re-raised and ends the run rather than being rescheduled forever. `__init__.py` exposes a thin sequential programmatic `Experiment` wrapper around the same pieces. Design notes: `docs/execution_pipeline.md`.

**Search engine (`engine/`)** — the coordinator delegates to focused modules rather than holding all logic:
- `agent_search.py` — `AgentSearch.step()` → `_run_single_step()`. **This is the dispatch heart:** given a selected parent node it picks the agent by node state — root → `draft_agent` (or `aggregation_agent` once the draft limit is hit), buggy/invalid → `debug_agent`, healthy → `improve_agent`, *unless* the branch is stagnant after ≥ half the time budget, in which case `evolution_agent` (intra-branch) or `fusion_agent` (cross-branch) fires per `fusion_vs_evolution_prob`. Generated code is run through `code_review_agent` before execution, then `result_parse_agent` + `execution.validate_executed_node` after. `__init__` preloads the analogy corpus (arm D) and the global memory; both failures are logged, never fatal.
- `node_selection.py` — UCT `select` with a piecewise-decaying exploration constant, plus global top-K selection and a time-based soft explore→exploit switch (`select_with_soft_switch`).
- `evaluation.py` — `backpropagate`, `check_improvement`, reward shaping.
- `execution.py` — post-run validation (submission CSV must exist; `metric == 0.0` on a maximize task is treated as a bug).
- `executor.py` — `Interpreter` runs each candidate as a **subprocess** (avoids CUDA/fork issues) in its own process group, with CPU affinity divided across execution slots. Candidates inherit the parent's `CUDA_VISIBLE_DEVICES` unchanged (no per-candidate mask; concurrent candidates share the GPU); `gpu_devices.py` only reports the visible IDs for the analogy resource context and never blocks startup. Callers wait FIFO when all slots are busy; the per-candidate timeout starts at slot acquisition.
- `search_node.py` — `SearchNode` (the tree node: code, plan, metric, branch_id, stage, lock, expected-child accounting, `analogy_report`, `analogy_adoption`, and the runtime fields `execution_status` / `artifact_status` / `best_snapshot_id` / `artifact_metric`) and `Journal` (the node collection, serialized to JSON).
- `solution_manager.py` — top-K candidate tracking and best-solution persistence.
- `conditions.py` — branch/global stagnation and multi-branch-fusion trigger predicates.
- `validation/` — `format_server.py` is a standalone Flask app (started by `launch_server.sh`) that wraps mle-bench grading; `format_client.py` calls it; `quality_check.py` does submission content/format checks and LLM-assisted fixes.

Node `stage` values: `root`, `draft`, `fusion_draft`, `improve`, `debug`, `evolution`, `fusion`. Nodes are grouped into branches (`branch_id`); much of the search logic is per-branch.

**Cold-start knowledge (`engine/coldstart/`)** — pretrained-model guidance only: `knowledge.py:build_guidance_description` maps a task to recommended models via `competition_tag_classified.json` + `models_guidance_classified.json`, landing on `cfg.coldstart.description`. It also calls `kb_snapshot.py`, which for arm D writes `logs/kb_snapshot.json` (venues, paper counts and `records_sha1` of the corpus the run could search; a missing file means arm A, a file containing `"error"` means the snapshot crashed). Nothing literature-related is injected at draft time any more — the old cold-start retrieval (`methodology_agent.py`, `ondemand.py`, `coldstart.inject_into_improve`) was removed on 2026-09-02; arms B/C in `results/` were produced by it and are read by the KB repo's `analyze_runs.py` from their old `config.yaml` keys.

**Analogy retrieval (`engine/analogy/`)** — the literature path, run **per improve node** (arm D, `analogy.enabled`). Design: `Agentic_Knowledge_Base/docs/analogy_bm25_agent_design.md` (v1) and `docs/analogy_context_v2.md` (v2). `corpus.py` loads the KB repo's `output/paper_corpus/records.jsonl` (title + tldr + abstract, no preprocessing, no embeddings) and builds BM25 once per process (Porter-stemmed, stopworded; ~1–2 min for 38k papers, preloaded in `AgentSearch.__init__`). `agent.py:run_analogy_agent` is one episode: a tools loop (`search_papers` / `read_abstract` / `submit_report`, plus `read_paper` with `fulltext` and the three code tools with `code_tools`, `analogy.max_turns` cap) whose prompt follows arXiv 2605.11258: diagnose ≤3 bottlenecks of the node being improved, rewrite each as 3–6-term queries in *other subfields'* vocabulary, search, read, and map the mechanisms back as concrete interventions. **BM25 does none of the analogy — the LLM's query rewriting does; BM25 only looks the words up.** A mechanism may cite only paper ids that appeared in that run's search results (validated, else dropped), so the report cannot invent citations.
- **Two loops.** `agent.py` keeps the legacy v1 Chat-completions loop; when `analogy.context.version >= 2` *or* the model routes through Responses, `run_analogy_agent` delegates to `observed_loop.py`. v2 pieces: `context.py` builds the versioned packet and separates *plans* (intent) from immutable source and runtime facts, with per-section char budgets and a conservative token-counted input cap that reserves a final-report turn; `code_tools.py` exposes `candidate_code_index` / `read_candidate_code` / `diff_candidate_code` over a frozen allow-list of node ids (current, parent, completed same-branch), parsing AST without importing candidate code and checking runtime source against its registered SHA256; `report_v2.py` validates evidence references (`runtime_evidence` paths resolve only against the displayed runtime object), drops a mechanism whole if any citation is invalid, keeps independently valid ones (`delivery_status` ∈ accepted_complete / accepted_partial / abstained / failed) and renders atomically within `report_char_budget`; `analogy.context.report_reserve_turns` forces a first submission by turn 12 of 14 with bounded repair turns. `fulltext.py` / `fulltext_worker.py` read paper PDFs in a bounded CPU subprocess with an immutable cache (`analogy.fulltext.cache_dir`, shared PVC on the cluster) — no PDF imports in the agent process.
- **Injection and handoff.** `improve_agent._inject_analogy` puts the rendered report (one `### ` block per mechanism) under its own heading in `prompt["Instructions"]` — the dict both generation paths render — and stores it on the child node as `SearchNode.analogy_report`, so `journal.json` records what each node saw. With a v2 structured report, `agents/analogy_handoff.py` makes the planner declare one `mechanism_id` (or reject all) with a reason, passes the complete selected mechanism to the coder (also on diff regeneration), stores `SearchNode.analogy_adoption` on the child and writes parent-vs-child diffs plus execution provenance to `logs/analogy/handoffs/<child_id>.json`. Selection is a declaration, not proof that the code implements the mechanism or that it caused a score change.
- **Artifacts.** Per-node traces `logs/analogy/<parent>_<n>.md` (packet, every tool call, report), `<same>.context.json` (truncation metadata, source-query ledger, per-turn budgets, `submission_attempts`), and one line per episode in `index.jsonl` (`delivery_status`, `failure_kind`, first-submit turn, mechanism count). Nothing in this package may end a run: every failure path logs and returns an empty report. There is no cross-arm cache to warm: the query is the run's own trajectory, different in every run by construction.

**Draft-stage variant (arm E, 2026-09-06; design `Agentic_Knowledge_Base/docs/analogy_draft_injection_design.md`).** `analogy.draft=True` makes `draft_agent._inject_analogy_draft` call `engine.analogy.agent.retrieve_for_draft` once per run, for the FIRST draft only (virtual root has no child and none in flight; initial drafts are generated sequentially). Same corpus, tools, validation and budget; `mode="draft"` swaps the prompt (`SYSTEM_PROMPT_DRAFT`: abstract the TASK's structure — metric/label/evaluation relations — not a node's bottleneck; interventions are design commitments for a simple first solution) and the report wording (`REPORT_HEADING_DRAFT`). The packet (`build_task_packet`) is description + data preview + resource budget + offline pretrained models. The report is stored on the draft node's `analogy_report`, traced in `logs/analogy/draft_001.md`, and the index line carries `"stage": "draft"`. Drafts 2–3 and all later nodes get the arm-A prompt; with `analogy.improve=False` (arm E) `_inject_analogy` returns immediately. Arm F = draft + improve (the current Jigsaw A/F experiments), which also turns on full text and context v2 source tools.

Diagnostics here follow one rule the hard way: **they must not be able to end a run, and that includes failing to import.** `write_kb_snapshot` is imported *inside* the try block in `knowledge.py` — when it wasn't, a deploy missing `kb_snapshot.py` raised ImportError at cold start in a function every arm calls, and essay s47 died with `BackoffLimitExceeded` before writing a single node. `_inject_analogy` imports `engine.analogy.agent` inside its try for the same reason.

**Candidate runtime (`engine/candidate_runtime/`, opt-in, `candidate_runtime.enabled`).** A cooperative execution protocol for generated code: fixed target-stratified public holdout built from `train.csv` with the experiment seed (`jigsaw.py`, the only task adapter — other tasks fail at startup when enabled), early and periodic full validation, cooperative time budgets (`draft_budget_seconds` / `candidate_budget_seconds` INCLUDE validation and export; `null` falls back to `exec.timeout`), immutable checkpoint/snapshot directories, and durable best/top-K exports independent of journal completion. Snapshots are artifacts of a candidate, never new search nodes. The generated-code contract (`prompt.py`, enforced by `integration.check_protocol` AST check before execution and by the code reviewer) is `CandidateSession.from_env()` → `split` → `bind(predict_validation, predict_test, save_checkpoint, load_checkpoint)` → `start_training` → `step()` after each real optimizer update (returns True to stop) → `finish()`; the worker reads its spec from `MLEVOLVE_CANDIDATE_SPEC`. State lives in `workspace/candidate_results/` (contract, per-candidate checkpoints/snapshots, `exports/<generation>`, atomic `current ->` pointer) and `logs/candidate_results/` (events, `summary.json`, `selection.json`); `diagnostics.py` derives read-only runtime facts and Jigsaw subgroup AUCs from already-selected public predictions for the analogy packet. `SearchNode.execution_status` and `artifact_status` are separate: a hard timeout after a verified snapshot keeps the node buggy (→ debug) while the snapshot stays eligible for the final ensemble. When enabled, `save_run` does **not** write `logs/best_solution.py`; the best solution is `workspace/candidate_results/current/best_solution/solution.py`. If a Pod dies before finalization: `python utils/recover_candidate_results.py --run-dir <run>` then `submission_fusion_utils.py`. Full protocol: `docs/candidate_runtime.md`. Do not enable it silently for old tasks.

**Agents (`agents/`).** One module per stage (`draft_agent`, `improve_agent`, `debug_agent`, `evolution_agent`, `fusion_agent`, `aggregation_agent`, `code_review_agent`, `result_parse_agent`, `data_leakage_agent`); each exposes a `run(agent, ...)` taking the `AgentSearch` instance. `improve_agent` consults the analogy agent at every improve node (arm D) and `draft_agent` consults it once, for the first draft, when `analogy.draft` is on (arm E); debug/evolution/fusion never do. `analogy_handoff.py` is the planner-selection/coder-handoff/provenance layer described above. `triggers.py` holds the predicates deciding when the optional agents (data-leakage check, branch fusion) fire. `result_parse_agent` also determines metric direction (minimize vs maximize) up front; it and `code_review_agent` use structured schemas that `utils/verify_review_contracts.py` pins. Subpackages:
- `coder/` — three generation strategies dispatched adaptively: `base_coder` (single-shot plan+code), `stepwise_coder` (multi-agent data-prep → model → training), `diff_coder` (SEARCH/REPLACE patch application).
- `planner/` — `base_planner` (single-stage) and `planner_with_memory` (two-stage retrieval-augmented).
- `memory/` — `GlobalMemoryLayer`: per-task store of node experience (plan/code/metric/label) with `HybridRetriever` (BM25 + FAISS). Different agents query it differently (similar records to reinforce, dissimilar to encourage novelty). Note `draft`/`evolution`/`fusion` agents instead use the *in-tree* memory `SearchNode.fetch_child_memory()`, which is unrelated.
- `prompts/` — shared prompt templates and guidelines (`impl_guideline.py` carries the candidate-runtime contract when enabled).

**LLM layer (`llm/`).** `query()` (with optional `FunctionSpec` function-calling) and `generate()` (streaming) dispatch by model-name prefix: `gemini*` → `gemini.py`, everything else → OpenAI-compatible `openai.py`. Both take an explicit `role` (`code` | `feedback`) so the slot is chosen even when both slots name the same model. Inside the OpenAI path, `responses.py:uses_responses` routes `gpt-6*` and `gpt-5.6-sol` to the **Responses API** through the SDK's generic JSON/SSE endpoint (the pinned `openai==1.66.3` predates these models; raw dicts preserve every output item and tool `call_id`, including opaque reasoning state, for replay on the next turn; no incompatible sampling params are sent). The adapter owns retries: transient errors get ≤3 attempts, then it raises `ResponsesError` with `transport_retry_exhausted=True`, which callers must propagate (the pipeline aborts the run) rather than retry in an outer loop — `should_retry_outer` is the check for legacy retry loops. `telemetry.py` appends safe per-call metadata (model, usage, elapsed, error category; never prompts, credentials or reasoning) to `logs/llm_calls.jsonl`. `model_profiles.py` holds per-family sampling params (Qwen/GPT/Kimi/DeepSeek/Claude, thinking vs non-thinking) for the legacy Chat path. Default model is `gpt-5.6-sol` at `reasoning_effort: high`; older model overrides keep their legacy route.

## Running on Kubernetes (`k8s/`)

Cluster runs target NRP Nautilus: one competition = one `Job` on a shared PVC mounted at `/workspace`, holding the repo checkout (`/workspace/MLEvolve` with its own `.venv`), datasets, and the HF/torch caches. `k8s/README.md` has the full runbook; the shape:

```bash
bash k8s/setup-venv.sh                      # once, ON THE DEV POD (same image as the Job)
bash k8s/preflight.sh                       # verify paths/venv/deps before submitting
python k8s/validate.py                      # shape-check the Job manifests before kubectl apply
kubectl -n <NS> apply -f k8s/job-<name>.yaml
```

`k8s/entrypoint.sh` is what the Job actually runs: it activates the venv, fails fast with actionable messages if the venv/data/deps are wrong (and, for arm D, if `rank_bm25`/`nltk` are missing — without nltk the tokenizer silently runs unstemmed, a different retrieval arm), checks the jubias grader patch, installs `build-essential` for `torch.compile`, points the caches at the PVC, and `exec`s `run_single_task.sh`. `cliproxy-deployment.yaml` runs the single shared LLM proxy every job points at (`http://cliproxy:8317/v1`) — `replicas: 1` + `Recreate` is deliberate, since two instances refreshing the same OAuth `auth/*.json` invalidate each other's tokens mid-run. `cliproxy-smoke.sh` / `cliproxy-loadtest.sh` (with `job-cliproxy-*.yaml`) exercise it; `utils/verify_sol_proxy.py` (and the older `verify_gpt6_proxy.py` + `job-gpt6-proxy-test.yaml`) probe the exact experiment model through the real wrappers before a batch.

**Paired-arm experiments.** Every comparison launches its matched arms from one manifest (`base` / `ana` for A/D, `base` / `anad` for A/E, `base` / `anaf` for A/F), distinguished by `EXP_NAME` and `EXTRA_RUN_ARGS` (`agent.seed=NN` plus the treatment overrides), with a distinct `SERVER_ID` per arm (grading port = 5005 + SERVER_ID). `k8s/validate.py` enforces the invariants that keep the pair valid. Within one task, seed ranges are deliberately disjoint across experiment families (jubias: A/D 45–47, A/E 48–50, A/F 51–66; essay: A/D 49, A/E 50) so the KB repo's `analyze_runs.py` never merges two baselines into one draw.
- `job-<task>-ad-sNN.yaml` — A/D (D adds `analogy.enabled=True analogy.corpus_path=/workspace/Agentic_Knowledge_Base/output/paper_corpus`).
- `job-<task>-ae-sNN.yaml` — A/E (`analogy.draft=True analogy.improve=False`); its first minutes should show `[analogy] draft: N mechanism(s) ...` and `[draft] injected N chars ...` before `Draft 1/3 code generated`.
- `job-jigsaw-unintended-af-sNN.yaml` — A/F on Jigsaw with the candidate runtime (`candidate_runtime.enabled=True draft_budget_seconds=5400 candidate_budget_seconds=7200`, `exec.max_parallel_run=1`), GPT-5.6 Sol/high pinned by `LLM_MODEL` + `MLEVOLVE_REQUIRED_MODEL`; F adds first-draft + improve analogy, full text and context v2/source tools. New seeds are rendered from `job-jigsaw-unintended-af-sol.template.yaml` (`sed 's/__SEED__/NN/g'`), then the `SERVER_ID`s are bumped; `job-jigsaw-unintended-af-gpt6.template.yaml` (with `MLEVOLVE_REQUIRE_GPT6=1`) is the historical GPT-6 Astra variant. S51–S56 ran earlier runtime policies; S57+ are Sol; S60+ require the 2026-09-13 report-delivery (P0) code. Keep Sol, GPT-6 and earlier Terra batches separate when estimating effects.
- The older `job-<task>-abc*.yaml` files (essay, jigsaw, jigsaw-unintended, lmsys) are kept as the record of how the B/C runs were launched; they reference config keys this branch no longer has and will not start.

`k8s/prepare-task.sh <EXP_ID>` downloads the data and checks the corpus exists; there is no LLM cache to warm (the analogy agent's input is each node's own diagnosis). In the first minutes of a D/F run confirm `[analogy] corpus: N papers, sha1 ...` and the `KB snapshot` line; at the first improve node, `[analogy] node ...: N mechanism(s) ...` or `no report (reason)`.

**Jobs run whatever is in the PVC checkout.** Sync a reviewed commit via Git before launching. If experiments are active, use a separate checkout and point only the new Job at it (change both the entrypoint path and `REPO_DIR`); never `git pull` over the shared `/workspace/MLEvolve` mid-run.

## Analysis utilities (`utils/`)

Mostly written for the KB experiments; each script's docstring explains why it exists.
- `grade_all.py` — grade every run's ensembles into one `scores.csv` (the private answers only exist on the cluster, so this is the file that carries results to local analysis); adds `metric_version` / `grader_sha256` columns so corrected jubias scores are never mixed with legacy ones.
- `grade_local.py` — grade CSVs offline; unlike `mlebench grade-sample` it treats the bundled leaderboard as optional, so a stale `leaderboard.csv` can't lose you an already-computed score.
- `compare_arms.py` — matched-K comparison table across run directories (`LABEL=path` args); enforces comparing arms only at equal ensemble size.
- `replay_analogy.py` — rebuild the packet for one node of an existing run from `journal.json` and run the agent against a corpus with the LLM in the environment. The fast loop for prompt/tokenizer changes; `--packet-only` needs no key or corpus; `--context-version 2 --workspace <run>/workspace` is needed for source tools (a fetch without the workspace cannot support them); `--draft --desc description.md` (or `--draft --run <dir>`) replays the arm-E task-structure variant without any node.
- `submission_fusion_utils.py` — the post-run ensembler (also recovers candidate-runtime artifacts without a journal); `refuse_all.py` re-runs it with the conservative 9 h serial-time cap lifted.
- `mlebench_patch.py` / `verify_mlebench_grading.py` — apply/check/test the versioned jubias grader fix.
- `recover_candidate_results.py` — rebuild `candidate_results` selection for a run whose Pod died before finalization (no training, no LLM).
- `llm_preflight.py` — see Configuration.

## Gotchas

- The application logger is named `"MLEvolve"` (memory uses `"memory"`); get it via `logging.getLogger("MLEvolve")`.
- The grading server is addressed by `GRADING_SERVER_PORT = 5005 + SERVER_ID` via env var; `run_single_task.sh` launches it and waits on `/health`. Use `SKIP_GRADING_SERVER=1` for non-mle-bench tasks it can't score, or `use_grading_server=False` to disable format validation entirely.
- `data_dir` must point at the prepared public split (`<dataset_dir>/<exp_id>/prepared/public`), not the dataset root — unless you override `DATA_DIR`/`DESC_FILE`, which is how non-mle-bench tasks run (see `examples/openadmet-expansionrx/`).
- Default budgets are 500 steps / 12 h, enforced by the `timeout --kill-after` inside `run_single_task.sh`; a single node is capped at 6 h (`exec.timeout: 21600`, lowered from 9 h on 2026-09-04 after stuck nodes ate most of the 12 h). A/D, A/E and A/F job manifests pin `nvidia.com/gpu.product` to >= 24 GB cards and `k8s/entrypoint.sh` installs `build-essential` for `torch.compile` — see `k8s/README.md` "GPU type". Job manifests set `activeDeadlineSeconds: 86400` (24 h) as a backstop for what runs *outside* that timeout (entrypoint setup, the unbounded fusion step). It is measured from when the Job is **accepted**, so queue time counts against it — at the old 13 h a pod that queued for over an hour was killed mid-run, which silently hands paired arms different effective budgets. Don't lower it; check queue depth instead.
- A venv copied from another machine still has `bin/activate` but a wrong hardcoded `VIRTUAL_ENV`, so activation silently no-ops and the run falls back to system python. Rebuild it in place rather than copying. (The root `.venv/` in this checkout is a laptop stub, not a runnable env.)
- Because everything installs with `--no-deps`, a failing `import X` is usually a missing *transitive* dep of X — print the real exception, not the module name.
- Post-run ensembling is a separate step: `python utils/submission_fusion_utils.py --task_id <EXP_ID> --exp_name <timestamp>_<EXP_NAME>`.
- A `ResponsesError` from `llm/responses.py` is terminal by design: do not wrap Responses calls in another retry loop, and do not catch-and-continue in the pipeline — a swallowed terminal transport error reschedules the same failing step forever while holding a GPU slot.
- `kb_snapshot.json` and `injected_knowledge.md` at the repo **root** are stray artifacts from a local smoke test (the snapshot points at `/tmp/fake_kb`), committed in `59d5eaf`. The real ones are per-run under `runs/<run>/logs/`, which is gitignored. `example_analogy.md` / `example_analogy_draft.md` at the root are reference traces of one improve-node and one draft episode. `analogy_exports/` is a downloaded, checksummed dump of the S60–S62 `logs/analogy/` trees plus tool-usage analysis (2026-09-17), not code.
- `k8s/README.md`'s "Launch a run" section still says new experiments default to GPT-6 Astra; the config, the Sol template and `docs/analogy_context_v2.md` are the current truth (GPT-5.6 Sol/high).
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
