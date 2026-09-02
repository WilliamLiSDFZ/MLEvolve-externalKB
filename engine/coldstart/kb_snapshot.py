"""Record what the paper corpus contained when a run started.

Each improve node's `analogy_report` (in journal.json) records what that node RECEIVED. This
records what the run could have retrieved from — the venues and paper counts in the corpus the
analogy agent searched, and the corpus identity (`records_sha1` from the corpus manifest, which
`scripts/6_build_paper_corpus.py` writes over the exact bytes of records.jsonl). Two runs can be
handed similar-looking suggestions while the corpus underneath them differs; without this there
is no way to notice, because `output/paper_corpus/` is not versioned anywhere.

Output: `<run>/logs/kb_snapshot.json`, a few hundred bytes.

    {
      "captured_at": "...", "digest": "ab12cd34",
      "corpus": {"path": ..., "count": 38273, "records_sha1": "...", "built_at": "...",
                 "venues": {"aaai-2024": 2813, "acl-2024": 1962, ...}}
    }

`digest` is sha1[:8] over the venue map and the corpus sha1 — deliberately excludes
captured_at, so two runs over an unchanged corpus produce the same digest and "did these two runs
search the same papers?" is one glance.

A failed snapshot writes a file containing `"error"` rather than no file at all, so that a
missing file unambiguously means "this run had no corpus" (arm A) rather than "the snapshot
crashed".
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


def _corpus_summary(corpus_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(corpus_dir)}
    manifest = corpus_dir / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            for k in ("level", "count", "records_sha1", "built_at", "schema_version", "venues"):
                if k in m:
                    out[k] = m[k]
        except Exception as e:
            out["manifest_error"] = f"{type(e).__name__}: {e}"

    # Count the records themselves rather than trusting the manifest: the two can disagree
    # after a partial rebuild, and the records are what the agent actually searches.
    records = corpus_dir / "records.jsonl"
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
    else:
        out["missing"] = True
    return out


def write_kb_snapshot(cfg: Any) -> Path | None:
    """Write <log_dir>/kb_snapshot.json. Returns the path, or None if there is no corpus to
    describe (analogy disabled or no corpus path).

    Never raises: a diagnostic must not be able to end a 12-hour run.
    """
    try:
        acfg = getattr(cfg, "analogy", None)
        corpus_path = str(getattr(acfg, "corpus_path", "") or "") if acfg is not None else ""
        if not corpus_path or not bool(getattr(acfg, "enabled", False)):
            return None                     # arm A: nothing to snapshot, absence says so

        log_dir = Path(getattr(cfg, "log_dir", "") or ".")
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / "kb_snapshot.json"

        snap: dict[str, Any] = {
            "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "retrieval_mode": "analogy",
            "corpus": _corpus_summary(Path(corpus_path)),
        }
        material = {"venues": snap["corpus"].get("venues"),
                    "records_sha1": snap["corpus"].get("records_sha1")}
        snap["digest"] = hashlib.sha1(
            json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:8]

        target.write_text(json.dumps(snap, indent=2, sort_keys=False), encoding="utf-8")

        c = snap["corpus"]
        logger.info("KB snapshot %s: %s venue(s), %s papers, corpus sha1 %s -> %s",
                    snap["digest"], c.get("venue_count", "?"), c.get("record_count", "?"),
                    c.get("records_sha1", "?"), target.name)
        if c.get("venues"):
            logger.info("KB venues: %s", ", ".join(f"{v}({n})" for v, n in c["venues"].items()))
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
