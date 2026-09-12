# Analogy context v2 and Responses experiments

Context v2 lets the existing analogy loop inspect the executed candidate before it
proposes a paper-derived intervention. Plans are labelled as intent; immutable code,
selected snapshots and public validation observations provide separate evidence.
There is no additional search-node type.

## Defaults and budgets

The active `config/config.yaml` uses `gpt-5.6-sol`, reasoning effort `high`, for code
and feedback. Analogy uses the code slot. `analogy.context.version=2` enables the
new context/report contract. Analogy itself remains opt-in. Old saved configurations
without `analogy.context` retain v1; historical Job files have not been rewritten.

| Setting | Default |
|---|---:|
| Initial packet | 80,000 characters |
| Plan / implementation index / execution analysis | 12,000 / 6,000 / 5,000 characters |
| Cleaned raw log head / tail | 2,000 / 6,000 characters |
| Runtime facts and public diagnostics | 12,000 characters |
| Code tool calls / total serialized output | 10 / 120,000 characters |
| Code read or diff response | 20,000 characters; read at most 400 lines |
| Full-text papers / reads / total text | 3 / 12 / 40,000 characters |
| Analogy model turns / final report | 14 / 12,000 characters |
| Per-response output budget | 16,384 tokens |
| Accumulated conversation input per request | at most 196,608 tokens |

The conversation cap includes messages, tools, tool outputs and response items.
It is also bounded by `endpoint_context_tokens` minus output and safety reserves.
Token counting is conservative and its method is recorded; billed input tokens across
all calls are a separate quantity. Near the limit the loop reserves a final report
turn. It removes complete low-priority mechanisms instead of cutting a mechanism
halfway through. Missing or invalid evidence can result in an empty report.

All these limits are configurable under `analogy.context`, `analogy.code_tools`,
`analogy.fulltext`, or the existing analogy options. `analogy.code_tools.enabled=False`
disables model-initiated source queries. `analogy.context.version=1` restores the old
packet/report contract; the selected model still determines its API transport.

## What the agent can inspect

The three new tools are `candidate_code_index`, `read_candidate_code`, and
`diff_candidate_code`. They take node IDs and symbols/line ranges, never arbitrary
paths. They can inspect the current executed candidate, its direct parent and the
completed same-branch candidates listed in the packet. Source is frozen at episode
start. Runtime-enabled candidates use registered `solution.py` plus its verified
SHA256; missing or mismatched source is reported, without a mutable-runfile fallback.
Runtime-off tasks use a frozen copy of `SearchNode.code`.

The index parses AST without importing/executing candidate code. Reads and diffs are
paged, including column offsets for unusually long lines. Replies and the ledger
include exact returned text, node identity, source hash and remaining budget. Report
code references must match source lines actually exposed to the model.

Runtime facts include the selected snapshot/checkpoint and training updates,
worker termination, validation history, calibration/export costs and resource
availability. Unknown values stay unknown. Resource references use
`runtime_context.resources`; other runtime references identify fields in the visible
runtime packet. A reference proves that evidence was available, not that the model's
causal interpretation is correct.

Jigsaw additionally derives overall and identity subgroup/BPSN/BNSP AUCs and sample
counts from the already-selected public validation predictions. Predictions, labels,
IDs and contract hashes are checked. Parent deltas require the same contract. The
derived cache is under `logs/candidate_results/diagnostics/`. This does not reload
model weights, rerun inference, read private test labels or change the scalar score.
Other tasks retain source/context tools and explicitly lack Jigsaw diagnostics;
the candidate runtime adapter itself still supports Jigsaw only.

## Report, planner and coder

The report separates `observed_facts`, `hypotheses`, and `unknowns`. Each mechanism
retains paper evidence, source/runtime references, assumptions, code targets,
constraints, a validation plan and rejection criterion.

When a v2 report is available, the planner may select one `mechanism_id` or reject
all, with a reason and concrete adaptation. Missing or malformed declarations are
recorded as `unknown`. The selected mechanism is passed separately to the coder,
including on diff regeneration. The child stores `analogy_adoption`; complete parent
versus child diffs and execution provenance are saved under
`logs/analogy/handoffs/<child_id>.json`. Source hashes are refreshed after code review
when execution returns. Final scores/validity are read from that child's journal;
the record does not automatically claim that analogy caused a score change.

## Responses transport and records

`llm/responses.py` handles GPT-5.6 Sol and GPT-6 Responses JSON and SSE. It preserves every output
item and matching tool `call_id` for subsequent turns, including opaque reasoning
state when the endpoint supplies it. It sends no incompatible sampling parameters.
The pinned `openai==1.66.3` remains sufficient through the SDK's generic JSON endpoint
using typed dictionaries; no shared environment upgrade is required.

Code/feedback roles are selected explicitly even when both use the same model name.
Transient transport failures receive at most three attempts; terminal, mismatched,
refused or incomplete responses fail explicitly rather than silently changing models
or handing partial output to the caller. Optional analogy failure still leaves its
trace and lets the existing search continue without a report.

Records to inspect:

- `logs/llm_preflight.json`: effective slots, effort, endpoint type/host, SDK and budgets.
  This validates configuration, not remote availability.
- `logs/llm_calls.jsonl`: general generation/feedback calls, returned model and usage.
- `logs/analogy/*.context.json`: packet/truncation metadata, source-query ledger,
  evidence anchors, per-turn budgets and model-call usage.
- Existing analogy packet, trace, report and full-text manifests remain available.

Credentials and opaque reasoning contents are not written to these telemetry files.

## Run new experiments

Use `k8s/job-jigsaw-unintended-af-sol.template.yaml` for a new A/F seed. It keeps
S56 hardware and 90-minute draft / 120-minute other-candidate budgets. Both arms use
GPT-5.6 Sol/high; F adds first-draft and improve analogy, full-text reading and context v2.
`MLEVOLVE_REQUIRED_MODEL=gpt-5.6-sol` requires Sol/high and context v2, rejecting stale overrides before
generation. Existing Secrets do not need their model value changed because the new
Job explicitly supplies it.

```bash
sed 's/__SEED__/57/g' k8s/job-jigsaw-unintended-af-sol.template.yaml > /tmp/jubias-gpt56sol-s57.yaml
kubectl --context nautilus -n ecepxie apply -f /tmp/jubias-gpt56sol-s57.yaml
```

Synchronize the approved commit through Git before applying. If experiments are
running, use a separate fixed checkout and change both the Job entrypoint path and
`REPO_DIR`; do not pull over a shared checkout used by active workers. Keep Sol results distinct from GPT-6 and earlier Terra batches when estimating treatment effects.
The S57–S59 manifests now use `gpt56sol` in Job/output names. The explicit GPT-6
template and its `MLEVOLVE_REQUIRE_GPT6=1` guard remain available for historical reproduction.

## Verification and replay

The 2026-09-11 implementation passed 140 offline tests, 20 historical packet checks,
three live SDK wrapper checks and three corrected GPT-6 analogy replays. Live testing
also fixed a conservative byte-counting failure that had prevented report correction:
an unchanged measured prefix now uses API usage plus margin; only appended items are
estimated by bytes. Limits and opaque-state replay remain intact. See the
[validation record](../../Agentic_Knowledge_Base/results/9.11/gpt6_context_v2_validation/REPORT.md).

CPU verification scripts are `verify_analogy_context.py`,
`verify_runtime_diagnostics.py`, `verify_analogy_observation.py`,
`verify_analogy_handoff.py`, `verify_gpt6_transport.py`, and
`verify_gpt6_configuration.py` under `utils/`. Existing full-text, injection,
candidate-runtime, execution-pipeline and best-result regressions remain applicable.

```bash
python utils/replay_analogy.py --run /path/to/run --workspace /path/to/run/workspace \
  --node NODE_PREFIX --context-version 2 --packet-only --out /tmp/packet.md

python utils/replay_analogy.py --run /path/to/run --workspace /path/to/run/workspace \
  --node NODE_PREFIX --context-version 2 --corpus /path/to/paper_corpus \
  --model gpt-5.6-sol --reasoning-effort high --fulltext \
  --fulltext-cache /path/to/paper_fulltext --out /tmp/replay.md
```

Live replay reads `LLM_BASE_URL` and `LLM_API_KEY`. It excludes nodes known to finish
after the selected candidate; absent completion timestamps cannot establish a valid
historical sibling. `--packet-only` needs neither credentials nor a corpus. Fetches
that omit immutable workspace source cannot support source tools: point to the actual
workspace instead. Replay can write a derived diagnostic cache into the run log
directory; copy `logs/config.yaml` and `logs/journal.json` to a temporary run directory
and pass the original read-only workspace when preserving historical artifacts.
`--fulltext-offline` restricts reading to the existing verified paper cache.

## Sol migration verification

Sol uses the same Responses path as GPT-6, including `high` reasoning during
function/tool calls, structured streaming output, opaque state replay, safe telemetry
and bounded transient retries. Older Terra and other legacy model overrides keep
their existing route; context v2 and source/full-text tools do not require a rollback.
The default 14-turn and input/output budgets, candidate runtime, one-GPU execution
policy and A/F treatments are unchanged. Model/API switching is not a guarantee
against shared-proxy rate limits.

A CPU-only live SDK probe uses the exact Sol model with no fallback:

```bash
python utils/verify_sol_proxy.py --output-dir /tmp/sol-proxy-test
```

Provide `LLM_API_KEY` / `LLM_BASE_URL` through the environment, or `--config` pointing
to an existing run's saved configuration. The probe uses six short model requests
on success, synthetic tools only, and records text/JSON/feedback/multi-turn results.
It changes neither shared configuration nor experiments. The old standalone
`verify_gpt6_proxy.py` / GPT-6 probe Job retain their original target.

Official model capabilities: [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol).
