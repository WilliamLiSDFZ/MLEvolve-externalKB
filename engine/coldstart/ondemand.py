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
import json
import logging
import os
import re
import tempfile
import time
import urllib.request
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

# Sections of a competition description that carry no retrieval signal: they describe
# submission mechanics, not the ML problem, and their generic wording dilutes both the
# dense vector and the BM25 term statistics.
_QUERY_DROP_HEADINGS = (
    "submission file", "file descriptions", "citation", "prizes", "timeline",
    "getting started", "required submission format", "task and metric alignment",
    # Metric formulas are long and describe scoring plumbing; knowing the loss is called
    # "MA-RAE" does not help find papers about the underlying ML problem.
    "evaluation",
)
# Generous enough to reach the data-characteristics section, which usually sits at the END
# of a description yet carries the most retrieval signal (label sparsity, task structure).
_QUERY_MAX_CHARS = 2500


def _build_query(task_desc: str, cfg: Any) -> str:
    """Build the retrieval query from the task description.

    Embedding the WHOLE description (submission format, file lists, citation, ...) yields a
    diffuse "average" query. Measured on this corpus: with the full description, lexical
    retrieval returned mostly off-topic papers; with a short task-focused query it returned
    10/10 on-topic. So keep only the sections describing the ML problem, and cap the length.

    Rule-based on purpose — no extra LLM call, deterministic and reproducible, which matters
    for A/B runs. Falls back to plain truncation if the headings don't parse as expected.
    """
    if not bool(getattr(cfg, "retr_focus_query", True)):
        return task_desc.strip()

    kept, skipping = [], False
    for line in task_desc.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            skipping = any(h in heading for h in _QUERY_DROP_HEADINGS)
            if not skipping:
                kept.append(stripped.lstrip("#").strip())
            continue
        if skipping:
            continue
        # Drop code fences / table rows / rules — layout, not meaning.
        if stripped.startswith(("```", "|", "---", "===")):
            continue
        if stripped:
            kept.append(stripped)

    query = " ".join(kept).strip()
    if len(query) < 100:                       # parsing produced nothing usable
        query = task_desc.strip()
    return query[:_QUERY_MAX_CHARS]


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
    if bool(getattr(cfg, "lazy_technique_rerank", True)):
        techniques = _split_techniques(available, kb_root)
        # Rerank against the same focused query — the full description dilutes this
        # stage too (and retr.embedding_model applies the same centering as the index).
        selected = _rerank_techniques(query, techniques, retr.embedding_model, cfg)
        logger.info(f"[Lazy] technique rerank: {len(techniques)} techniques from "
                    f"{len(available)} papers -> {len(selected)} selected")
        return _assemble_techniques(selected, budget_chars)

    return _assemble(available, kb_root, budget_chars)
