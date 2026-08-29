"""Record what the knowledge base contained when a run started.

`injected_knowledge.md` records what a run RECEIVED. This records what it could have received —
the venues, years and paper counts in the retrieval pool. Two runs can be handed byte-identical
techniques while the corpus underneath them differs, and without this there is no way to notice.

The gap this closes is specific: `output/` and `methodology_kb/` are committed to git, but
`output/abstract_index/` — the artifact retrieval actually queries — is not versioned anywhere.
It also grows during normal operation, because lazy mode caches on-demand extractions back into
`methodology_kb/`. So the composition of the knowledge base at any past moment is unrecoverable
after the fact. Writing it at cold start is the only point where it is cheap and certain.

Output: `<run>/logs/kb_snapshot.json`, a few KB.

    {
      "captured_at": "...", "digest": "ab12cd34",
      "abstract_index": {"embedding_model": ..., "count": 23166,
                         "venues": {"aaai-2024": 5097, "acl-2024": 3526, ...}},
      "methodology_kb": {"total_extracted": 243, "extracted_papers": {"naacl-2024": 223, ...}}
    }

`digest` is sha1[:8] over the venue maps and index identity, so "did these two runs see the same
corpus?" is one glance, the same way the injected-knowledge digest works.

Two things about the semantics, both of which will otherwise be misread later:

* **This is the state BEFORE the run's own extractions.** Lazy mode writes into `methodology_kb`
  as it goes, so the end-of-run directory will not match this file. That is intended — the
  snapshot describes the pool the run started from — but anyone comparing the two will think the
  snapshot is wrong unless it says so, which is why `note` is embedded in the file itself.
* **Arms of one draw can legitimately differ.** `methodology_kb` is shared mutable state; an arm
  launched thirty minutes later sees whatever the earlier arms just cached. Comparing digests
  across arms of a draw is therefore a real check, not a formality — that is exactly the failure
  that invalidated the essay seed-42 draw.

A failed snapshot writes a file containing `"error"` rather than no file at all, so that a
missing file unambiguously means "this run had no knowledge base" (arm A) rather than "the
snapshot crashed".
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")

# methodology_kb/paperinsight/ is cross-paper synthesis, not a venue directory.
NON_VENUE_DIRS = {"paperinsight", "index", ".git"}


def _index_summary(index_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(index_path)}
    manifest = index_path / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            for k in ("level", "embedding_model", "dim", "count", "built_at", "schema_version"):
                if k in m:
                    out[k] = m[k]
        except Exception as e:
            out["manifest_error"] = f"{type(e).__name__}: {e}"

    records = index_path / "records.jsonl"
    if records.exists():
        venues: Counter = Counter()
        bad = 0
        with records.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    venues[json.loads(line).get("venue", "?")] += 1
                except json.JSONDecodeError:
                    bad += 1
        out["venues"] = dict(sorted(venues.items()))
        out["venue_count"] = len(venues)
        out["record_count"] = sum(venues.values())
        if bad:
            out["unparseable_records"] = bad
    return out


def _methodology_summary(kb_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(kb_path)}
    if not kb_path.is_dir():
        out["missing"] = True
        return out
    per_venue: dict[str, int] = {}
    for d in sorted(p for p in kb_path.iterdir() if p.is_dir()):
        if d.name in NON_VENUE_DIRS:
            continue
        per_venue[d.name] = sum(1 for _ in d.glob("*/*_methodology.md"))
    out["extracted_papers"] = per_venue
    out["total_extracted"] = sum(per_venue.values())
    out["has_paperinsight"] = (kb_path / "paperinsight").is_dir()
    return out


def write_kb_snapshot(cfg: Any) -> Path | None:
    """Write <log_dir>/kb_snapshot.json. Returns the path, or None if there is no KB to describe.

    Never raises: a diagnostic must not be able to end a 12-hour run.
    """
    try:
        kb_path = str(getattr(cfg, "methodology_kb_path", "") or "")
        index_path = str(getattr(cfg, "abstract_index_path", "") or "")
        if not kb_path and not index_path:
            return None                     # arm A: nothing to snapshot, absence says so

        log_dir = Path(getattr(cfg, "log_dir", "") or ".")
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / "kb_snapshot.json"

        snap: dict[str, Any] = {
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": ("State BEFORE this run's own on-demand extractions. Lazy mode caches into "
                     "methodology_kb during the run, so the end-of-run directory will contain "
                     "more than this file lists."),
            "retrieval_mode": str(getattr(cfg, "methodology_retrieval", "") or ""),
        }
        if index_path:
            snap["abstract_index"] = _index_summary(Path(index_path))
        if kb_path:
            snap["methodology_kb"] = _methodology_summary(Path(kb_path))

        # Digest over corpus composition + index identity only — deliberately excludes
        # captured_at, so two runs over an unchanged corpus produce the same digest.
        material = {
            "venues": snap.get("abstract_index", {}).get("venues"),
            "embedding_model": snap.get("abstract_index", {}).get("embedding_model"),
            "dim": snap.get("abstract_index", {}).get("dim"),
            "extracted": snap.get("methodology_kb", {}).get("extracted_papers"),
        }
        snap["digest"] = hashlib.sha1(
            json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:8]

        target.write_text(json.dumps(snap, indent=2, sort_keys=False), encoding="utf-8")

        ai = snap.get("abstract_index", {})
        mk = snap.get("methodology_kb", {})
        logger.info(
            "KB snapshot %s: %s venue(s), %s indexed papers, %s extracted -> %s",
            snap["digest"], ai.get("venue_count", "?"), ai.get("record_count", "?"),
            mk.get("total_extracted", "?"), target.name)
        if ai.get("venues"):
            logger.info("KB venues: %s", ", ".join(
                f"{v}({n})" for v, n in ai["venues"].items()))
        return target
    except Exception as e:  # pragma: no cover - diagnostics must never break a run
        logger.warning("KB snapshot failed: %s: %s", type(e).__name__, e)
        try:
            log_dir = Path(getattr(cfg, "log_dir", "") or ".")
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "kb_snapshot.json").write_text(
                json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=2), encoding="utf-8")
        except Exception:
            pass
        return None
