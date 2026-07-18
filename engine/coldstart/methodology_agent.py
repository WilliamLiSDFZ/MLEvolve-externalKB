"""Methodology search agent for cold-start knowledge injection.

Two selection modes, dispatched by cfg.methodology_retrieval:

- "vector" (default): technique-level hybrid semantic retrieval. Embeds the task
  description and matches it against one record per distilled insight, using the
  HybridRetriever (BM25 + FAISS + RRF) already in agents/memory/. Requires a prebuilt
  index at {methodology_kb_path}/index/ (see the KB repo's build_retrieval_index.py).
  Falls back to "llm" when no index is present.
- "llm": the original behaviour — ask an LLM to pick up to 5 categories by name, then
  read their HIGH-confidence references.

Returns plain guidance text (no adoption-tracking side-channel).
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Tuple

logger = logging.getLogger("MLEvolve")

# methodology_kb_path -> HybridRetriever (built once per run)
_INDEX_CACHE: dict = {}

_CONFIDENCE_WEIGHT = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}


def _scan_categories(kb_base: Path) -> List[str]:
    """Return category path strings relative to kb_base.

    Supports two layouts:
    - Flat:   kb_base/category/            (e.g. experience_kb/small-data-transformer-finetuning/)
    - Nested: kb_base/venue-year/category/ (e.g. paperinsight/naacl-2024/efficient-training/)
    """
    categories: List[str] = []
    for entry in sorted(kb_base.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Flat layout: entry itself contains insight.md
        if (entry / "insight.md").exists():
            categories.append(entry.name)
            continue
        # Nested layout: entry is a venue-year dir containing category subdirs
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            if (sub / "insight.md").exists():
                categories.append(f"{entry.name}/{sub.name}")
    return categories


def _match_categories_with_llm(task_desc: str, categories: List[str], cfg: Any) -> List[str]:
    """Ask the LLM which categories are relevant. Returns up to 5 matches."""
    cat_list = "\n".join(f"- {c}" for c in categories)
    user_msg = (
        f"Task description (first 1500 chars):\n{task_desc[:1500]}\n\n"
        f"Available research categories:\n{cat_list}\n\n"
        "Select up to 5 most relevant categories for this task. "
        "Output ONLY the selected category names, one per line, exactly as shown. No explanation."
    )
    try:
        if (cfg.model or "").lower().startswith("glm"):
            # GLM via the Anthropic-compatible endpoint (Zhipu).
            import anthropic
            client = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=1200.0)
            resp = client.messages.create(
                model=cfg.model,
                temperature=0,
                max_tokens=256,
                system="You are a research category selector. Output only category names, one per line.",
                messages=[{"role": "user", "content": user_msg}],
            )
            response = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text") or ""
        else:
            from openai import OpenAI
            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
            resp = client.chat.completions.create(
                model=cfg.model,
                temperature=0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": "You are a research category selector. Output only category names, one per line."},
                    {"role": "user", "content": user_msg},
                ],
            )
            response = resp.choices[0].message.content or ""
        matched = []
        for line in response.strip().splitlines():
            line = line.strip().lstrip("- ")
            if line in categories:
                matched.append(line)
        logger.info(f"[MethodologyAgent] LLM matched {len(matched)} categories: {matched}")
        return matched[:5]
    except Exception as e:
        logger.warning(f"[MethodologyAgent] LLM matching failed: {e}")
        return []


def _strip_ref_noise(text: str) -> str:
    """Remove Papers & Evidence and Delta sections + frontmatter from reference content."""
    text = re.sub(r"## Papers & Evidence.*?(?=\n## |\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*Delta\*\*:.*?\n", "", text)
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _read_high_confidence_references(cat_dir: Path) -> str:
    """Read insight.md, find HIGH-confidence rows, read their reference files."""
    insight_file = cat_dir / "insight.md"
    if not insight_file.exists():
        return ""

    insight_text = insight_file.read_text(encoding="utf-8")
    refs_dir = cat_dir / "references"
    ref_contents = []

    in_table = False
    for line in insight_text.splitlines():
        line = line.strip()
        if line.startswith("| # |") or line.startswith("|---|"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 5:
                continue
            if cells[3].upper() != "HIGH":
                continue
            ref_hint = cells[4].strip()
            ref_name = ref_hint.rsplit("/", 1)[-1]
            ref_path = refs_dir / ref_name
            if not ref_path.exists() and refs_dir.exists():
                # Fuzzy match by slug prefix
                slug = re.sub(r"[^a-z0-9-]", "", cells[1].lower().replace(" ", "-"))[:30]
                candidates = [p for p in refs_dir.glob("*.md") if slug[:15] in p.stem]
                ref_path = candidates[0] if candidates else None

            if ref_path and Path(ref_path).exists():
                try:
                    raw = Path(ref_path).read_text(encoding="utf-8")
                    ref_contents.append(_strip_ref_noise(raw))
                except Exception:
                    continue

    return "\n\n---\n\n".join(ref_contents)


def _build_via_llm_categories(task_desc: str, kb_base: Path, llm_cfg: Any) -> str:
    """Legacy path: LLM picks up to 5 categories by name → their HIGH-confidence refs."""
    categories = _scan_categories(kb_base)
    if not categories:
        logger.info("[MethodologyAgent] No categories found")
        return ""

    logger.info(f"[MethodologyAgent] Scanning {len(categories)} categories...")
    matched = _match_categories_with_llm(task_desc, categories, llm_cfg)
    if not matched:
        logger.info("[MethodologyAgent] No relevant categories matched")
        return ""

    all_sections = []
    for cat_path in matched:
        content = _read_high_confidence_references(kb_base / cat_path)
        if content:
            all_sections.append(f"### [{cat_path}]\n\n{content}")
            logger.info(f"[MethodologyAgent] Added references from {cat_path}")

    return _render(all_sections)


# ------------------------------------------------------------ vector retrieval

class InsightRecord:
    """One indexed insight (a distilled technique). Attrs come from records.jsonl."""

    def __init__(self, d: dict):
        self.__dict__.update(d)

    def __repr__(self) -> str:
        return f"<Insight {getattr(self, 'id', '?')}>"


def _load_index(kb_base: Path, cfg: Any):
    """Load {kb}/index/ into a HybridRetriever (cached per run). Returns None if absent."""
    key = str(kb_base)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    idx = kb_base / "index"
    manifest_path = idx / "manifest.json"
    if not manifest_path.exists():
        return None

    import numpy as np
    import faiss
    from rank_bm25 import BM25Okapi
    from agents.memory.embedding_models import EmbeddingModel
    from agents.memory.retriever import HybridRetriever

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        InsightRecord(json.loads(line))
        for line in (idx / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [r.embed_text for r in records]

    # The manifest is the contract: query MUST use the model the index was built with.
    emb_model = EmbeddingModel(
        model_type="local",
        model_name=manifest["embedding_model"],
        device=getattr(cfg, "retr_embedding_device", "cpu"),
    )
    retr = HybridRetriever(emb_model)

    vecs = np.load(idx / "embeddings.npy").astype("float32")
    if vecs.shape[0] != len(records):
        raise ValueError(f"index rows ({vecs.shape[0]}) != records ({len(records)})")
    if vecs.shape[1] != retr.dimension:
        raise ValueError(
            f"embedding dim mismatch: index={vecs.shape[1]} model={retr.dimension} "
            f"({manifest['embedding_model']}) — rebuild the index with this model"
        )

    retr.records, retr.texts, retr.vectors = records, texts, vecs
    retr.vector_index = faiss.IndexFlatL2(retr.dimension)
    retr.vector_index.add(vecs)
    retr.bm25 = BM25Okapi([t.lower().split() for t in texts])

    logger.info(
        f"[MethodologyAgent] Loaded index: {len(records)} insights, "
        f"model={manifest['embedding_model']}, dim={retr.dimension}"
    )
    _INDEX_CACHE[key] = retr
    return retr


def _build_query(task_desc: str, cfg: Any) -> str:
    """v1: the task description itself (not truncated)."""
    return task_desc.strip()


def _select(hits: List[Tuple[Any, float]], cfg: Any) -> List[Tuple[Any, float]]:
    """Confidence-weight → relative-score threshold → drop LOW → dedup → top_n + budget."""
    if not hits:
        return []

    scored = []
    for rec, score in hits:
        w = _CONFIDENCE_WEIGHT.get(str(getattr(rec, "confidence", "")).upper(), 0.7)
        scored.append((rec, score * w))
    scored.sort(key=lambda x: x[1], reverse=True)

    # RRF scores are tiny and unnormalized; threshold relative to the best hit.
    best = scored[0][1] or 1.0
    min_score = float(getattr(cfg, "retr_min_score", 0.15))
    kept = [(r, s) for r, s in scored if (s / best) >= min_score]

    non_low = [(r, s) for r, s in kept if str(getattr(r, "confidence", "")).upper() != "LOW"]
    if non_low:
        kept = non_low

    seen, dedup = set(), []
    for r, s in kept:
        key = re.sub(r"[^a-z0-9]", "", str(getattr(r, "title", "")).lower())[:60]
        if key and key in seen:
            continue
        seen.add(key)
        dedup.append((r, s))

    top_n = int(getattr(cfg, "retr_top_n", 10))
    budget_chars = int(getattr(cfg, "retr_token_budget", 6000)) * 4  # ~4 chars/token
    out, used = [], 0
    for r, s in dedup[:top_n]:
        size = len(getattr(r, "guidance_text", "") or "")
        if out and used + size > budget_chars:
            break
        out.append((r, s))
        used += size
    return out


def _build_via_vector(task_desc: str, kb_base: Path, cfg: Any) -> str:
    retr = _load_index(kb_base, cfg)
    if retr is None:
        return ""

    query = _build_query(task_desc, cfg)
    pool = int(getattr(cfg, "retr_pool", 30))
    alpha = float(getattr(cfg, "retr_alpha", 0.5))
    hits = retr.search(query, top_k=pool, alpha=alpha)
    selected = _select(hits, cfg)
    if not selected:
        logger.info("[MethodologyAgent] No insight cleared the relevance threshold")
        return ""

    logger.info(
        "[MethodologyAgent] Retrieved %d insights: %s",
        len(selected), [getattr(r, "id", "?") for r, _ in selected],
    )
    sections = [
        f"### [{getattr(r, 'category', '?')}] {getattr(r, 'title', '')} "
        f"(confidence: {getattr(r, 'confidence', '?')})\n\n{r.guidance_text}"
        for r, _ in selected
    ]
    return _render(sections)


def _render(sections: List[str]) -> str:
    if not sections:
        return ""
    return (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following detailed techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(sections)
    )


def build_methodology_guidance(task_desc: str, methodology_kb_path: str, cfg: Any) -> str:
    """Select methodology knowledge for this task and render it as prompt guidance.

    cfg is the FULL config: retrieval knobs are top-level, the LLM fallback uses
    cfg.agent.code.
    """
    kb_base = Path(methodology_kb_path)
    if not kb_base.exists():
        logger.info("[MethodologyAgent] methodology_kb_path not found, skipping")
        return ""

    mode = str(getattr(cfg, "methodology_retrieval", "vector")).lower()
    llm_cfg = cfg.agent.code

    if mode == "vector":
        try:
            text = _build_via_vector(task_desc, kb_base, cfg)
            if text:
                return text
            if (kb_base / "index" / "manifest.json").exists():
                return ""  # index present, nothing relevant cleared the bar
            logger.warning(
                "[MethodologyAgent] No index at %s/index — falling back to LLM category match. "
                "Build one with the KB repo's build_retrieval_index.py.", kb_base,
            )
        except Exception as e:
            logger.warning(f"[MethodologyAgent] Vector retrieval failed ({e}); falling back to LLM")

    return _build_via_llm_categories(task_desc, kb_base, llm_cfg)
