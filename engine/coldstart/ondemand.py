"""Lazy methodology retrieval: abstract-level search + on-demand deep extraction.

Flow (methodology_retrieval: "lazy"):
  1. Query the ABSTRACT index (built by the KB repo's 6_build_abstract_index.py) with the
     task description — LOW threshold, high recall.
  2. Split candidates into cached (a *_methodology.md already exists under
     methodology_kb_path) vs missing.
  3. Extract at most `max_extractions_per_coldstart` missing papers NOW (download PDF →
     pymupdf → one LLM call each, small thread pool), writing results into the STANDARD
     methodology_kb layout: {methodology_kb_path}/{venue}/{category}/{stem}_methodology.md
     — permanent cache, shared with the batch pipeline (plugin A skips existing files).
  4. Inject the [POSITIVE] sections of all available papers, ordered by retrieval score,
     under a token budget.

Cost: the abstract index is free (local embeddings); per task you pay only for the capped
extractions, which amortize to zero as the cache warms.
"""
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("MLEvolve")

_ABS_CACHE: dict = {}   # abstract_index_path -> retriever
_PDF_IMPORT_WARNED = False   # log the PDF-extractor import failure once, not per paper

EXTRACT_PROMPT = """Extract techniques from this ML/NLP paper. For each technique or design choice, identify whether it had a positive, negative, or neutral effect on results.

Return JSON with a "techniques" array. Each item:
- name: short technique name
- description: what it is
- effect: "positive" | "negative" | "neutral"
- delta: quantitative change if mentioned (e.g. "+2.3 F1"), or descriptive ("outperforms baseline")
- evidence: direct quote from paper supporting the claim
- condition: when/where this applies

Paper text:
{text}

Return only valid JSON: {{"techniques": [...]}}"""


class PaperRecord:
    def __init__(self, d: dict):
        self.__dict__.update(d)

    def __repr__(self) -> str:
        return f"<Paper {getattr(self, 'id', '?')}>"


class _CenteredEmbedding:
    """Embedding model wrapper that mean-centers vectors before use.

    Sentence embeddings of a homogeneous corpus (here: ~12k ML paper abstracts) are highly
    anisotropic — every vector shares a large common component ("this is an ML paper"), so
    raw cosine similarity is dominated by that component and barely discriminates topic.
    Measured on this corpus: top-10 similarity spread was 0.017 and only 5/10 hits were
    on-topic; after centering, spread 0.048 and 8/10 on-topic.

    Subtracting the corpus mean from BOTH the indexed vectors and the query removes that
    shared direction. Wrapping the model (rather than changing HybridRetriever) keeps the
    centering applied to queries the retriever encodes internally, and leaves the BM25 half
    untouched.
    """

    def __init__(self, base: Any, mean_vec):
        self.base = base
        self.mean = mean_vec
        self.dimension = base.dimension
        self.model_type = getattr(base, "model_type", "local")

    def encode(self, texts, show_progress_bar: bool = False):
        import numpy as np
        v = np.asarray(self.base.encode(texts, show_progress_bar=show_progress_bar),
                       dtype="float32") - self.mean
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


# ------------------------------------------------------------------ index loading

def _load_abstract_index(index_dir: Path, cfg: Any):
    key = str(index_dir)
    if key in _ABS_CACHE:
        return _ABS_CACHE[key]

    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi
    from agents.memory.embedding_models import EmbeddingModel
    from agents.memory.retriever import HybridRetriever

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [PaperRecord(json.loads(l))
               for l in (index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    texts = [r.embed_text for r in records]

    emb = EmbeddingModel(model_type="local", model_name=manifest["embedding_model"],
                         device=getattr(cfg, "retr_embedding_device", "cpu"))
    vecs = np.load(index_dir / "embeddings.npy").astype("float32")
    if vecs.shape[0] != len(records) or vecs.shape[1] != emb.dimension:
        raise ValueError(f"abstract index mismatch: vecs={vecs.shape} records={len(records)} "
                         f"dim={emb.dimension} ({manifest['embedding_model']})")

    # Mean-center the dense half (see _CenteredEmbedding). Computed from the loaded
    # vectors, so no index rebuild is needed and existing indexes stay compatible.
    centered = bool(getattr(cfg, "retr_center_embeddings", True))
    if centered:
        mean_vec = vecs.mean(axis=0)
        vecs = vecs - mean_vec
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        emb = _CenteredEmbedding(emb, mean_vec)

    retr = HybridRetriever(emb)
    retr.records, retr.texts, retr.vectors = records, texts, vecs
    retr.vector_index = faiss.IndexFlatL2(retr.dimension)
    retr.vector_index.add(vecs)
    retr.bm25 = BM25Okapi([t.lower().split() for t in texts])
    logger.info(f"[Lazy] Loaded abstract index: {len(records)} papers "
                f"(model={manifest['embedding_model']}, centered={centered})")
    _ABS_CACHE[key] = retr
    return retr


# ------------------------------------------------------------------ extraction

def _resolve_pdf(rec: PaperRecord) -> str:
    pdf = (getattr(rec, "pdf_url", "") or "").strip()
    if pdf:
        return pdf
    src = getattr(rec, "source", "") or ""
    if "aclanthology.org" in src:
        return src.rstrip("/") + ".pdf"
    if "openreview.net" in src:
        m = re.search(r"[?&]id=([^&]+)", src)
        if m:
            return f"https://openreview.net/pdf?id={m.group(1)}"
    return ""


def _download(url: str, dest: Path, timeout: int = 30, retries: int = 3) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:
            if i < retries - 1:
                time.sleep(2 ** i)
    return False


def _chat(llm_cfg: Any, user_msg: str, max_tokens: int = 4096) -> str:
    model = llm_cfg.model or ""
    if model.lower().startswith("glm"):
        import anthropic
        client = anthropic.Anthropic(api_key=llm_cfg.api_key,
                                     base_url=llm_cfg.base_url or None, timeout=1200.0)
        resp = client.messages.create(model=model, temperature=0,
                                      max_tokens=max_tokens,
                                      messages=[{"role": "user", "content": user_msg}])
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text") or ""

    # Reuse the central per-model rules rather than re-deriving them here: OpenAI
    # reasoning models (GPT-5, o-series) reject `max_tokens` AND sampling params.
    from llm.model_profiles import supports_sampling_params, uses_max_completion_tokens
    from openai import OpenAI

    params: dict = {
        "model": model,
        "messages": [{"role": "user", "content": user_msg}],
        ("max_completion_tokens" if uses_max_completion_tokens(model) else "max_tokens"): max_tokens,
    }
    if supports_sampling_params(model):
        params["temperature"] = 0

    client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url or None)
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content or ""


FILTER_PROMPT = """You are selecting research papers whose METHODS could actually be applied to a
specific machine-learning competition.

COMPETITION
{task}

AVAILABLE DATA
{data}

PAPERS
{papers}

For each paper decide:
  "keep"       - its method could be applied to this competition with the data described above
  "irrelevant" - the topic only overlaps superficially; the method is about something else
  "infeasible" - the method is relevant but assumes something this competition does not have
                 (another modality, extra annotations, a different label structure, an external
                 corpus, human-in-the-loop, orders of magnitude more compute)

Be strict on "infeasible". A method needing images, human annotations, or labels the competition
does not provide is infeasible however well the topic matches.

Return ONLY a JSON array, one object per paper, same order, no other text:
[{{"i": 1, "decision": "keep|irrelevant|infeasible", "why": "<=15 words"}}]"""


def _describe_data(cfg: Any) -> str:
    """A short description of what the competition actually provides.

    This block is what makes an "infeasible" judgement possible at all. Without it the model
    cannot know that jigsaw is text-only with six binary labels, and so cannot rule out a CLIP
    method however obviously inapplicable it is. File names and extensions establish modality
    more reliably than prose does, so both are used.
    """
    parts = []
    data_dir = Path(str(getattr(cfg, "data_dir", "") or ""))
    if data_dir.is_dir():
        try:
            entries = sorted(p.name for p in list(data_dir.iterdir())[:40])
            exts = sorted({p.suffix.lower() for p in data_dir.rglob("*")
                           if p.is_file() and p.suffix} - {""})
            parts.append("Files in the data directory: " + ", ".join(entries[:25]))
            if exts:
                parts.append("File types present: " + ", ".join(sorted(exts)[:15]))
        except Exception as e:                      # pragma: no cover - listing must not fail us
            logger.debug(f"[Filter] could not list data_dir: {e}")
    return "\n".join(parts) if parts else "(not available - judge from the competition text alone)"


def _agent_filter_papers(candidates: List[Tuple[PaperRecord, float]], task_query: str,
                         cfg: Any) -> Tuple[List[Tuple[PaperRecord, float]], List[dict]]:
    """Drop papers whose methods this competition cannot use, BEFORE paying to extract them.

    Runs on title + abstract, so it costs one small call per batch and removes papers before any
    PDF is downloaded — cheaper than the reranker it replaces, which filtered only after up to 20
    full-text extractions had already been paid for.

    Returns (survivors, decisions). Never raises: on any failure the full candidate list is
    returned unchanged, because a filter that can end a 12-hour run is worse than no filter.
    """
    if not candidates:
        return candidates, []
    llm_cfg = cfg.agent.code
    batch = max(1, int(getattr(cfg, "filter_batch_size", 10)))
    data_desc = _describe_data(cfg)
    decisions: List[dict] = []

    for start in range(0, len(candidates), batch):
        chunk = candidates[start:start + batch]
        listing = "\n\n".join(
            f"[{start + i + 1}] {getattr(r, 'title', '?')}\n"
            f"    {(getattr(r, 'abstract', '') or '')[:1200]}"
            for i, (r, _) in enumerate(chunk))
        prompt = FILTER_PROMPT.format(task=task_query, data=data_desc, papers=listing)
        try:
            raw = _chat(llm_cfg, prompt, max_tokens=200 * len(chunk) + 300)
            m = re.search(r"\[.*\]", raw, re.S)
            if not m:
                raise ValueError("no JSON array in reply")
            for item in json.loads(m.group(0)):
                idx = int(item.get("i", 0)) - 1
                if 0 <= idx < len(candidates):
                    decisions.append({"i": idx,
                                      "decision": str(item.get("decision", "keep")).lower(),
                                      "why": str(item.get("why", ""))[:120]})
        except Exception as e:
            logger.warning(f"[Filter] batch {start // batch + 1} failed ({type(e).__name__}: {e})"
                           f" — keeping its papers unfiltered")
            for i in range(len(chunk)):
                decisions.append({"i": start + i, "decision": "keep", "why": "filter failed"})

    # If EVERY batch failed, the filter did not run. Degrade to the previous behaviour — the full
    # candidate list, untouched — rather than to "top N by stage-1 score", which is a third
    # behaviour nobody chose and would silently change what the arm receives.
    if decisions and all(d["why"] == "filter failed" for d in decisions):
        logger.warning("[Filter] every batch failed — passing all %d candidates through unfiltered",
                       len(candidates))
        return candidates, decisions

    verdict = {d["i"]: d for d in decisions}
    kept_idx = [i for i in range(len(candidates))
                if verdict.get(i, {}).get("decision", "keep") == "keep"]

    # Floor: an empty injection turns the KB arm into an expensive baseline, which looks like a
    # null result rather than a broken filter.
    floor = int(getattr(cfg, "filter_min_keep", 5))
    if len(kept_idx) < floor:
        topped = [i for i in range(len(candidates)) if i not in kept_idx][:floor - len(kept_idx)]
        logger.warning(f"[Filter] agent kept only {len(kept_idx)}; topping up with "
                       f"{len(topped)} highest-scoring papers to reach the floor of {floor}")
        kept_idx = sorted(kept_idx + topped)

    # Ceiling: prompt size must not depend on how strict the agent happened to be.
    ceiling = int(getattr(cfg, "filter_max_keep", 15))
    if ceiling > 0 and len(kept_idx) > ceiling:
        kept_idx = sorted(kept_idx)[:ceiling]       # candidates are already in stage-1 order

    counts = Counter(d["decision"] for d in decisions)
    logger.info(f"[Filter] {len(candidates)} candidates -> {len(kept_idx)} kept "
                f"(agent: {dict(counts)})")
    return [candidates[i] for i in kept_idx], decisions


def _write_filter_log(cfg: Any, candidates: List[Tuple[PaperRecord, float]],
                      decisions: List[dict], kept: List[Tuple[PaperRecord, float]]) -> None:
    """Record every decision next to injected_knowledge.md.

    Without this, "the filter dropped the wrong papers" is unfalsifiable. This project has twice
    been misled by a diagnostic that recorded nothing (see UPDATELOG on `\\b429\\b` and on reading
    best_solution.py), so the filter writes down what it did.
    """
    try:
        log_dir = Path(getattr(cfg, "log_dir", "") or ".")
        log_dir.mkdir(parents=True, exist_ok=True)
        kept_ids = {getattr(r, "id", "") for r, _ in kept}
        verdict = {d["i"]: d for d in decisions}
        lines = [f"# Paper filter — {len(kept)} of {len(candidates)} kept", ""]
        for i, (rec, score) in enumerate(candidates):
            d = verdict.get(i, {"decision": "?", "why": ""})
            mark = "KEPT" if getattr(rec, "id", "") in kept_ids else d["decision"].upper()
            lines.append(f"- [{mark}] ({score:.3f}) {getattr(rec, 'title', '?')[:110]}")
            if d["why"]:
                lines.append(f"      {d['why']}")
        (log_dir / "paper_filter.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:                          # pragma: no cover
        logger.warning(f"[Filter] could not write paper_filter.md: {e}")


def _extract_techniques(text: str, llm_cfg: Any, retries: int = 3) -> list:
    last = None
    for i in range(retries):
        try:
            raw = _chat(llm_cfg, EXTRACT_PROMPT.format(text=text))
            raw = re.sub(r"^```json\s*|```$", "", raw.strip(), flags=re.MULTILINE)
            return json.loads(raw).get("techniques", [])
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(2 ** i + 1)
    raise last


def _render_methodology(title: str, source: str, techniques: list) -> str:
    lines = [f"# {title}\n", f"**Source**: {source}\n"]
    for t in techniques:
        if not t.get("name"):
            continue
        lines.append(f"## [{t.get('effect', 'neutral').upper()}] {t['name']}")
        lines.append(f"{t.get('description', '')}\n")
        lines.append(f"**Delta**: {t.get('delta', 'N/A')}")
        lines.append(f"**Condition**: {t.get('condition', 'N/A')}\n")
        lines.append(f"**Evidence**: \"{t.get('evidence', '')}\"\n")
    return "\n".join(lines)


def _cache_path(kb_root: Path, rec: PaperRecord) -> Path:
    stem = str(getattr(rec, "id", "")).split("/")[-1]
    return kb_root / rec.venue / rec.category / f"{stem}_methodology.md"


def _extract_one(rec: PaperRecord, kb_root: Path, llm_cfg: Any) -> bool:
    """Extract one paper into the methodology cache. Returns True on success."""
    try:
        import pymupdf4llm
    except Exception as e:
        # Report the REAL exception: with --no-deps installs the failure is usually a
        # missing transitive dep (pymupdf4llm needs pymupdf), not the package we named.
        global _PDF_IMPORT_WARNED
        if not _PDF_IMPORT_WARNED:
            _PDF_IMPORT_WARNED = True
            logger.warning(
                "[Lazy] cannot import pymupdf4llm (%s: %s) — extraction disabled for this run. "
                "Fix with:  pip install pymupdf4llm   (without --no-deps, so pymupdf comes too)",
                type(e).__name__, e,
            )
        return False

    pdf_url = _resolve_pdf(rec)
    if not pdf_url:
        return False
    out_file = _cache_path(kb_root, rec)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        pdf_path = Path(tf.name)
    try:
        if not _download(pdf_url, pdf_path):
            return False
        text = pymupdf4llm.to_markdown(str(pdf_path))[:64000]
        techniques = _extract_techniques(text, llm_cfg)
    except Exception as e:
        logger.warning(f"[Lazy] extraction failed for {rec.id}: {type(e).__name__}: {e}")
        return False
    finally:
        pdf_path.unlink(missing_ok=True)

    # atomic write; a concurrent run writing the same paper is harmless (same content)
    tmp = out_file.with_suffix(".md.tmp")
    tmp.write_text(_render_methodology(getattr(rec, "title", rec.id),
                                       getattr(rec, "source", ""), techniques),
                   encoding="utf-8")
    os.replace(tmp, out_file)
    return True


# ------------------------------------------------------------------ assembly

def _split_cached(candidates: List[Tuple[PaperRecord, float]], kb_root: Path):
    cached, missing = [], []
    for rec, score in candidates:
        (cached if _cache_path(kb_root, rec).exists() else missing).append((rec, score))
    return cached, missing


_POSITIVE_RE = re.compile(r'^## \[POSITIVE\] (.+?)$\n(.*?)(?=^## \[|^# [^#]|\Z)',
                          re.MULTILINE | re.DOTALL)


def _split_techniques(available: List[Tuple[PaperRecord, float]], kb_root: Path) -> List[dict]:
    """Flatten available papers into individual [POSITIVE] technique entries."""
    techniques = []
    for rec, paper_score in available:
        try:
            content = _cache_path(kb_root, rec).read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _POSITIVE_RE.finditer(content):
            techniques.append({
                "tech_title": m.group(1).strip(),
                "body": m.group(2).strip(),
                "paper_title": getattr(rec, "title", rec.id),
                "paper_id": rec.id,
                "paper_score": paper_score,
            })
    return techniques


def _rerank_techniques(task_desc: str, techniques: List[dict], emb_model: Any,
                       cfg: Any) -> List[dict]:
    """Second-stage retrieval at TECHNIQUE granularity: embed each extracted technique
    and rank by similarity to the task. This is where precision is recovered — the
    abstract stage is recall-oriented and paper-level, so a relevant paper's irrelevant
    techniques must be filtered here."""
    import numpy as np

    if not techniques:
        return []
    texts = [f"{t['tech_title']}\n{t['body'][:800]}" for t in techniques]
    tv = np.asarray(emb_model.encode(texts), dtype="float32")
    qv = np.asarray(emb_model.encode([task_desc.strip()]), dtype="float32")[0]
    sims = (tv @ qv) / (np.linalg.norm(tv, axis=1) * np.linalg.norm(qv) + 1e-8)
    for t, s in zip(techniques, sims):
        t["tech_score"] = float(s)

    ranked = sorted(techniques, key=lambda t: t["tech_score"], reverse=True)
    best = ranked[0]["tech_score"] or 1.0
    min_rel = float(getattr(cfg, "lazy_tech_min_score", 0.3))
    kept = [t for t in ranked if (t["tech_score"] / best) >= min_rel]

    seen, dedup = set(), []
    for t in kept:
        key = re.sub(r"[^a-z0-9]", "", t["tech_title"].lower())[:60]
        if key and key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    return dedup[:int(getattr(cfg, "lazy_tech_top_n", 12))]


def _assemble_techniques(selected: List[dict], budget_chars: int) -> str:
    sections, used = [], 0
    for t in selected:
        block = (f"### {t['tech_title']}\n"
                 f"*(from: {t['paper_title']})*\n\n{t['body']}")
        if sections and used + len(block) > budget_chars:
            break
        sections.append(block)
        used += len(block)
    if not sections:
        return ""
    return (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following actionable techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(sections)
    )


def _assemble(available: List[Tuple[PaperRecord, float]], kb_root: Path,
              budget_chars: int) -> str:
    from engine.coldstart.knowledge import _extract_positive_sections

    sections, used, seen_titles = [], 0, set()
    for rec, _score in available:                      # already ordered by score desc
        tkey = re.sub(r"[^a-z0-9]", "", str(getattr(rec, "title", "")).lower())[:60]
        if tkey and tkey in seen_titles:
            continue
        path = _cache_path(kb_root, rec)
        try:
            positives = _extract_positive_sections(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not positives:
            continue
        block = f"### {getattr(rec, 'title', rec.id)}\n\n" + "\n\n".join(positives)
        if sections and used + len(block) > budget_chars:
            break
        seen_titles.add(tkey)
        sections.append(block)
        used += len(block)

    if not sections:
        return ""
    return (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following actionable techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(sections)
    )


# ------------------------------------------------------------------ query building

# Fallback cap when no distilled query is available: the raw description, truncated.
_QUERY_MAX_CHARS = 2500

_DISTILL_PROMPT = """Below is a machine-learning competition description. Write a single
compact paragraph (50-80 words) that will be used as a SEARCH QUERY to find relevant
research papers.

Describe only the machine-learning problem:
- input data type and scale
- task type (e.g. multi-class classification, multi-task regression)
- evaluation metric
- modelling techniques and data characteristics likely to matter

Exclude everything else: prizes, timelines, eligibility, submission file formats, file
lists, citations, and narrative/flavour text. Write plain prose, no headings or bullets.

Competition description:
{desc}

Search query:"""


def _distill_query(task_desc: str, cfg: Any) -> str:
    """Compress the description into a short ML-task statement via one LLM call.

    Why not rules: a heading-based extractor was tried and does NOT generalise. Competition
    descriptions differ wildly — for OpenADMET the signal sat in a trailing "data
    characteristics" section, while for spooky-author-identification it sat in "Evaluation"
    (which the rules discarded), leaving only horror-story flavour text. Measured top-10
    on-topic hits: raw description 2/10, rule-extracted 0/10, hand-written summary 9/10.
    The query is the whole ballgame, so it is worth one LLM call to get it right.

    The result is cached on disk keyed by a hash of the description, so a task is distilled
    once and every later run — including both arms of an A/B — reuses the identical query.
    """
    cache_dir = Path(getattr(cfg, "retr_query_cache_dir", "") or
                     (Path(getattr(cfg, "abstract_index_path", ".")) .parent / "query_cache"))
    key = hashlib.sha1(task_desc.encode("utf-8")).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.txt"

    if cache_file.exists():
        cached = cache_file.read_text(encoding="utf-8").strip()
        if cached:
            logger.info(f"[Lazy] distilled query (cached): {cached[:120]}...")
            return cached

    query = _chat(cfg.agent.code, _DISTILL_PROMPT.format(desc=task_desc[:12000]),
                  max_tokens=300).strip()
    if len(query) < 40:
        raise ValueError(f"distilled query too short: {query!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(query, encoding="utf-8")
    os.replace(tmp, cache_file)
    logger.info(f"[Lazy] distilled query (new, cached to {cache_file.name}): {query[:120]}...")
    return query


def _build_query(task_desc: str, cfg: Any) -> str:
    """Build the retrieval query. Mode: "llm" (default, distilled + cached) or "raw"."""
    mode = str(getattr(cfg, "retr_query_mode", "llm")).lower()
    if mode == "llm":
        try:
            return _distill_query(task_desc, cfg)
        except Exception as e:
            logger.warning(f"[Lazy] query distillation failed ({type(e).__name__}: {e}); "
                           f"falling back to the raw description")
    return task_desc.strip()[:_QUERY_MAX_CHARS]


# ------------------------------------------------------------------ entry point

def build_lazy_guidance(task_desc: str, cfg: Any) -> str:
    index_dir = Path(getattr(cfg, "abstract_index_path", "") or "")
    if not (index_dir / "manifest.json").exists():
        logger.warning(f"[Lazy] no abstract index at {index_dir} — skipping lazy retrieval")
        return ""
    kb_root = Path(getattr(cfg, "methodology_kb_path", "") or "")
    if not str(kb_root):
        logger.warning("[Lazy] methodology_kb_path unset — nowhere to cache; skipping")
        return ""

    retr = _load_abstract_index(index_dir, cfg)
    pool = int(getattr(cfg, "lazy_pool", 40))
    query = _build_query(task_desc, cfg)
    logger.info(f"[Lazy] query: {len(query)} chars (from {len(task_desc)} char description)")
    hits = retr.search(query, top_k=pool,
                       alpha=float(getattr(cfg, "retr_alpha", 0.5)))
    if not hits:
        return ""

    # LOW relative threshold: recall over precision — extraction cost is capped anyway,
    # and precision is recovered at assembly (score order + budget).
    best = hits[0][1] or 1.0
    min_rel = float(getattr(cfg, "lazy_min_score", 0.05))
    candidates = [(r, s) for r, s in hits if (s / best) >= min_rel]

    # STAGE 2 (new): let an LLM read title+abstract and drop what this competition cannot use,
    # BEFORE any PDF is fetched. Embedding similarity cannot make this call — a multimodal method
    # is genuinely near a text method in embedding space; only a reader knows the dataset has no
    # images. Placing it here also makes it cheaper than the reranker it replaces, which filtered
    # after up to 20 extractions had already been paid for.
    use_agent_filter = bool(getattr(cfg, "agent_paper_filter", True))
    if use_agent_filter:
        before = len(candidates)
        candidates, decisions = _agent_filter_papers(candidates, query, cfg)
        _write_filter_log(cfg, [c for c in hits if (c[1] / best) >= min_rel], decisions,
                          candidates)
        if not candidates:
            logger.warning("[Filter] nothing survived; falling back to unfiltered candidates")
            candidates = [(r, s) for r, s in hits if (s / best) >= min_rel]
        logger.info(f"[Lazy] agent filter: {before} -> {len(candidates)} papers")

    cached, missing = _split_cached(candidates, kb_root)
    cap = int(getattr(cfg, "max_extractions_per_coldstart", 20))
    to_extract = missing[:cap]
    logger.info(f"[Lazy] {len(candidates)} candidates: {len(cached)} cached, "
                f"{len(missing)} missing, extracting {len(to_extract)} (cap={cap})")

    extracted_ok = []
    if to_extract:
        workers = int(getattr(cfg, "lazy_extract_workers", 4))
        llm_cfg = cfg.agent.code
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_extract_one, rec, kb_root, llm_cfg): (rec, score)
                    for rec, score in to_extract}
            for f in as_completed(futs):
                rec, score = futs[f]
                try:
                    if f.result():
                        extracted_ok.append((rec, score))
                except Exception as e:
                    logger.warning(f"[Lazy] worker error for {rec.id}: {e}")
        logger.info(f"[Lazy] extracted {len(extracted_ok)}/{len(to_extract)} papers")

    available = sorted(cached + extracted_ok, key=lambda x: x[1], reverse=True)
    budget_chars = int(getattr(cfg, "retr_token_budget", 6000)) * 4

    # Final selection — two flavours, chosen by lazy_technique_rerank:
    #   True  (default): second-stage TECHNIQUE-level retrieval — embed each extracted
    #          [POSITIVE] technique (same model as the abstract index, already loaded)
    #          and rank by similarity to the task; only the truly relevant techniques
    #          get injected, regardless of which paper they came from.
    #   False: paper-level — inject each candidate paper's [POSITIVE] sections wholesale,
    #          ordered by the abstract-retrieval score (stage-1 score only).
    # When the agent filter ran, precision has already been recovered at the PAPER level and the
    # review's instruction applies: inject every surviving paper's techniques whole, no second
    # filter. The reranker stays reachable with agent_paper_filter=False so the previous design
    # remains runnable — every result to date was produced with it, and being able to re-run it
    # is what makes this change measurable rather than merely different.
    if use_agent_filter:
        techniques = _split_techniques(available, kb_root)
        logger.info(f"[Lazy] injecting ALL {len(techniques)} techniques from "
                    f"{len(available)} filtered papers (budget {budget_chars} chars)")
        return _assemble_techniques(techniques, budget_chars)

    if bool(getattr(cfg, "lazy_technique_rerank", True)):
        techniques = _split_techniques(available, kb_root)
        # Rerank against the same focused query — the full description dilutes this
        # stage too (and retr.embedding_model applies the same centering as the index).
        selected = _rerank_techniques(query, techniques, retr.embedding_model, cfg)
        logger.info(f"[Lazy] technique rerank: {len(techniques)} techniques from "
                    f"{len(available)} papers -> {len(selected)} selected")
        return _assemble_techniques(selected, budget_chars)

    return _assemble(available, kb_root, budget_chars)
