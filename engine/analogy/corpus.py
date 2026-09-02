"""Paper corpus + BM25 search for the analogy agent.

The corpus is `output/paper_corpus/records.jsonl` + `manifest.json` from the KB repo
(`scripts/6_build_paper_corpus.py`): one record per paper with title, tldr and abstract. There
is no preprocessing on the KB side and no embedding anywhere — the agent's queries are short
mechanism-vocabulary strings, and lexical matching over title+tldr+abstract is the whole
retrieval (design: `Agentic_Knowledge_Base/docs/analogy_bm25_agent_design.md` §4.1).

BM25 is built at load time (rank_bm25.BM25Okapi, default k1/b). Measured on 12.8k papers:
tokenize ~25 s with Porter stemming, build <1 s, one query ~50 ms. Loading happens once per
process (`load_corpus` caches per path, under a lock, because improve nodes run in parallel
threads); after that `search`/`get` are read-only and thread-safe.

Tokenizer: lowercase -> [a-z0-9]+ -> drop stopwords (function words plus paper boilerplate) ->
Porter stem. Stemming is what lets "equivariant" find "equivariance"; if nltk is missing the
tokenizer degrades to no stemming on BOTH documents and queries, and logs once. The title is
repeated once in the document text as a cheap title boost.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("MLEvolve")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words plus the words that appear in nearly every ML abstract. Dropping the second
# group shortens documents without losing signal (their IDF is ~0 anyway) and, more importantly,
# keeps a long query from being diluted by them.
STOPWORDS = frozenset("""
a an the of to in on for with by and or as at from is are was were be been being it its we our
they their which who whom what where when how than then there here also into via using use used
based can could may might will would should must not no nor do does did done such each any all
both more most much many few less least very between among within without through over under
across per about after before during while this that these those has have had having if else
one two three first second i ii iii etc et al
paper propose proposed proposes present presents presented method methods approach approaches
results result show shows shown novel new existing work works model models data however
demonstrate demonstrates extensive experiments experimental performance state art outperform
outperforms achieve achieves achieved significantly
""".split())


def _make_stemmer() -> Callable[[str], str]:
    try:
        from nltk.stem import PorterStemmer
    except Exception as e:  # nltk is in requirements_domain.txt but must not be load-bearing
        logger.warning("[analogy] nltk PorterStemmer unavailable (%s: %s) — tokenizing without "
                       "stemming (queries and documents alike)", type(e).__name__, e)
        return lambda w: w
    stem = PorterStemmer().stem
    memo: Dict[str, str] = {}

    def cached(w: str) -> str:               # vocabulary << token count; Porter is the slow part
        s = memo.get(w)
        if s is None:
            s = memo[w] = stem(w)
        return s
    return cached


class PaperCorpus:
    def __init__(self, records: List[dict], manifest: dict):
        self.records = records
        self.manifest = manifest
        self.by_id: Dict[str, dict] = {r["id"]: r for r in records}
        self._stem = _make_stemmer()
        t0 = time.time()
        docs = [self.tokenize(f"{r['title']} {r['title']} {r.get('tldr', '')} {r.get('abstract', '')}")
                for r in records]
        t1 = time.time()
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(docs)
        logger.info("[analogy] corpus: %d papers, sha1 %s (tokenize %.1fs, bm25 %.1fs)",
                    len(records), self.digest, t1 - t0, time.time() - t1)

    # ------------------------------------------------------------------ identity

    @property
    def digest(self) -> str:
        return str(self.manifest.get("records_sha1", "?"))

    @property
    def venues(self) -> Dict[str, int]:
        return dict(self.manifest.get("venues", {}))

    # ------------------------------------------------------------------ tools

    def tokenize(self, text: str) -> List[str]:
        return [self._stem(w) for w in _TOKEN_RE.findall(text.lower())
                if w not in STOPWORDS and len(w) > 1]

    def search(self, query: str, k: int = 10) -> List[dict]:
        """Top-k papers for a query: [{id, venue, title, tldr, score}]. No abstract, on purpose:
        the agent asks for those separately, so a broad search does not flood its context."""
        toks = self.tokenize(query)
        if not toks:
            return []
        import numpy as np
        scores = self.bm25.get_scores(toks)
        k = max(1, min(int(k), len(scores)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        out = []
        for i in top:
            if scores[i] <= 0:
                break
            r = self.records[int(i)]
            out.append({"id": r["id"], "venue": r["venue"], "title": r["title"],
                        "tldr": (r.get("tldr") or "")[:300], "score": round(float(scores[i]), 2)})
        return out

    def get(self, ids: List[str]) -> List[dict]:
        """Full abstracts for known ids, in the order given; unknown ids are skipped."""
        out = []
        for pid in ids:
            r = self.by_id.get(str(pid))
            if r is not None:
                out.append({"id": r["id"], "venue": r["venue"], "title": r["title"],
                            "abstract": r.get("abstract", "")})
        return out

    def __contains__(self, pid: object) -> bool:
        return pid in self.by_id

    def __len__(self) -> int:
        return len(self.records)


def read_corpus_dir(corpus_dir: Path) -> tuple[List[dict], dict]:
    records = [json.loads(line)
               for line in (corpus_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return records, manifest


_CACHE: Dict[str, PaperCorpus] = {}
_LOCK = threading.Lock()


def load_corpus(corpus_dir: str | Path) -> Optional[PaperCorpus]:
    """Load (once per path) or return the cached corpus. Returns None if the path has no
    records.jsonl — the caller treats that as "analogy disabled", never as a fatal error."""
    key = str(corpus_dir)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        p = Path(key)
        if not (p / "records.jsonl").exists():
            logger.warning("[analogy] no records.jsonl under %s — analogy retrieval disabled", p)
            return None
        records, manifest = read_corpus_dir(p)
        corpus = PaperCorpus(records, manifest)
        _CACHE[key] = corpus
        return corpus
