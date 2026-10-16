# Manual task-specific corpus preparation

Run this command in the dev pod before an experiment. MLEvolve cold start now
only supplies pretrained-model recommendations: it does not retrieve papers,
extract techniques, or automatically inject these prepared results.

## Run

For a CPU-only Kubernetes Job that prepares the unintended-bias task, see
[`k8s/README.md`](../k8s/README.md#prepare-unintended-bias-data-and-techniques) and
[`job-prepare-jigsaw-unintended.yaml`](../k8s/job-prepare-jigsaw-unintended.yaml).

From the MLEvolve repository with its existing retrieval environment activated:

```bash
export TYPESAFE_API_KEY=...
export LLM_API_KEY=...
# Optional, using the same agent.code configuration as MLEvolve:
export LLM_BASE_URL=...
export LLM_MODEL=...

python -m corpus_retriever \
  exp_id=jigsaw-unintended-bias-in-toxicity-classification \
  desc_file=/path/to/prepared/public/description.md \
  data_dir=/path/to/prepared/public \
  abstract_index_path=/path/to/Agentic_Knowledge_Base/output/abstract_index
```

The command uses `config/config.yaml` plus existing `key=value` overrides.
`goal=...` and optional `eval=...` can replace `desc_file`. It does not call
`prep_cfg`, create a training workspace, run MLE-bench, or require `prepare-task.sh`.
Do not put API keys in command arguments or saved task files.

The existing abstract index and its embedding model are required. The model
must expose a fast tokenizer with character offsets. Dependencies are the
existing MLEvolve retrieval/model packages (`sentence-transformers`, FAISS,
`rank-bm25`, `httpx`, and the configured LLM backend) plus `pymupdf4llm`, also used
by the former PDF extraction path. Its two required packages are now pinned in
`requirements_base.txt`. To update an existing dev environment, run:

```bash
pip install --no-deps pymupdf4llm==0.0.27 pymupdf==1.26.3
```

No TypeSafe SDK is required.

## Processing

1. Read the full task description and bounded metadata/CSV previews from the
   supplied public-data directory. Omitted data and resources remain unknown.
2. Reuse centered hybrid abstract search with `lazy_pool=100` and the existing
   `lazy_min_score` relative cutoff. Fewer than 100 candidates may remain.
3. Download each unique candidate once and parse the full PDF without the old
   64,000-character cutoff. Ask Jev about **every chunk**, including chunks after
   the first positive.
4. Run `CorpusTechniqueAgent` on every retained paper, without the old 15-paper
   ceiling or 20-paper extraction cap. It reads all selected chunks and can
   request any other chunk of the same paper through `read_full_text`.
5. Save task-specific proposals and source evidence. The original cross-task
   methodology cache is neither reused for extraction nor overwritten.

PDF download/parsing and tokenization run on the main thread. A single pool of
three workers screens and extracts up to three papers at once; each paper keeps
its own agent state and reads chunks sequentially. Preparation and model calls
overlap without queuing the entire corpus in memory. Results are aggregated in
retrieval order, regardless of completion order. Logs identify the paper/chunk,
stage, reading action, and completed-paper count. Ordinary paper failures remain
isolated; fatal errors stop submission and wait for running workers before
cleaning up the temporary output, preserving the previous task result.

Jev defaults to pinned `jev-1.13.0` with a Noul threshold of `0.5`. Override with
`corpus_retriever.jev_model=...` or `corpus_retriever.jev_threshold=...`.
The threshold has not been calibrated for this task.

`corpus_retriever.chunk_tokens=10000` and `corpus_retriever.chunk_overlap=200`
use the **index model's local tokenizer**, not an officially supplied Jev
tokenizer. Original text and character offsets are preserved. Each chunk is
counted again after splitting. Task, metadata, and prompt text are additional
request content. Context errors remain unresolved, rather than negative
decisions. Lower the chunk size for a particularly long task description.

The technique agent reads one chunk per turn with bounded notes and cumulative
techniques. Current bounds: 100,000 input characters, 6,000 note characters,
40,000 result characters, and 32 additional reading calls beyond selected chunks.
These are character budgets, not provider token counts. Exceeding a bound or
returning malformed evidence produces `incomplete`, never a truncated success.
Jev selection does not require the agent to produce a technique. The agent judges
whether a chunk offers a concrete, plausible task experiment or strengthens an
existing technique. It may skip content without useful contributions by recording
the reason in its notes and choosing `next_selected` (or `finish` on the last
chunk), while carrying forward earlier useful techniques. All selected chunks
are still read, and skipped chunks can be revisited. An entire paper yielding
zero useful techniques is a successful empty result, not an extraction failure.
Skipping uses the existing model turn; it does not avoid that chunk's input cost.
Evidence must reference a chunk actually read and contain a nonempty quote.
Quote text that does not exactly match the chunk produces a log warning only:
the quote is preserved and extraction continues through the remaining chunks.
Evidence therefore still needs review; neither textual fidelity nor support for
the model's claim is guaranteed.

Prompts are in `jev_reranker.py:JEV_PROMPT/JEV_CRITERIA` and
`technique_agent.py:SYSTEM_PROMPT`; their exact text is saved with each result.

## Results and replacement

The stable destination is **`MLEvolve/cache/techinique/<exp_id>/`**, with the
requested spelling and **no run ID**:

```text
<exp_id>/
  manifest.json              # settings, counts, statuses and source fingerprints
  task_context.json          # exact task description and data preview
  prompts.json               # screening and extraction prompts
  techniques.json            # aggregate from fully completed papers
  techniques.md              # readable version of that aggregate
  papers/<paper_key>/
    paper.json               # title, abstract, source, retrieval score and status
    fulltext.md               # complete parsed text, also used by the reader
    screening.json            # chunks, scores, errors and selected chunk IDs
    techniques.json           # adaptations, evidence, reading trace and errors
```

Repeating the command performs retrieval/screening/extraction again and replaces
the same task directory, removing stale papers. Processing uses a temporary
directory, not a versioned output. Fatal errors such as invalid credentials or
missing dependencies leave the previous result unchanged. An entirely failed
rerun also preserves it. A local file lock rejects concurrent preparation of
the same task.

A finished run with some unresolved papers publishes a clearly marked `partial`
result, replacing the prior directory. Incomplete papers retain diagnostic
files, but their techniques are excluded from the aggregate. Inspect
`manifest.json` and the per-paper records before using a result.

Exit codes: **0** = complete (possibly no useful techniques), **2** = partial
result saved, **1** = fatal failure. Normal all-negative screening stays empty;
there is no minimum-paper top-up or fallback to unfiltered papers.

No injection policy is selected here. Adaptations are experiment proposals,
not measured task improvements. Missing PDF text, including image-only
equations/figures or separately hosted supplements, is not recovered by this reader.

## Offline checks

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m corpus_retriever --help
```

Tests use fake model responses and temporary fixtures, without paid API calls.
