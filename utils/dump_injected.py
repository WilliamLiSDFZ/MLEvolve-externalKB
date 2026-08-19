"""Recover `injected_knowledge.md` for runs that finished before it was written.

Runs launched before 2026-08-19 recorded only an elided preview of the injected techniques, so
what knowledge they received is not in the run directory. Retrieval is deterministic given the
task description, the abstract index, the cached distilled query and the extracted papers — all
of which live on the pod — so the text can be regenerated.

The catch is that the knowledge base has grown since those runs. A regenerated text is only the
one that run actually saw if the corpus and query are unchanged, and there is no way to tell by
inspection. So this script never trusts the replay: it compares sha1[:8] of the regenerated text
against the `digest` the run logged at the time, and writes the file ONLY on a match.

    python utils/dump_injected.py --runs /workspace/MLEvolve/runs           # report only
    python utils/dump_injected.py --runs /workspace/MLEvolve/runs --write   # write on match

Expect partial success. A mismatch is not a failure of this script — it means the KB moved, and
that run genuinely cannot be measured. Say so in the writeup rather than substituting today's
retrieval for what the run saw; that would silently score nodes against techniques they were
never given.

Runs no new LLM calls: extraction is capped at 0 so only already-extracted papers are used. If
the distilled query is not in the cache the replay is skipped rather than paying for a new one.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LOG_DIGEST = re.compile(r"Knowledge injected at draft: (\d+) chars, digest ([0-9a-f]+)")


class _TagIgnoringLoader(yaml.SafeLoader):
    """config.yaml embeds `!!python/object/apply:pathlib.PosixPath`, which SafeLoader rejects and
    unsafe_load would execute."""


def _reconstruct(loader, suffix, node):
    """Rebuild pathlib paths; map every other unknown tag to None.

    Mapping *everything* to None is the obvious implementation and it silently destroyed the
    field this script depends on: `desc_file` is stored as a tagged sequence of path components,
    so it became None, `Path(str(None or ""))` became `Path('.')`, and `.exists()` is true for
    '.', so every run failed with `IsADirectoryError: Is a directory: '.'`. Reconstructing the
    path is both correct and cheaper than the failure it prevents.
    """
    if "pathlib" in suffix and isinstance(node, yaml.SequenceNode):
        parts = [str(p) for p in loader.construct_sequence(node)]
        return str(Path(*parts)) if parts else None
    return None


_TagIgnoringLoader.add_multi_constructor("", _reconstruct)


def logged_digest(run: Path) -> tuple[int, str] | None:
    log = run / "logs" / "MLEvolve.log"
    if not log.exists():
        return None
    m = LOG_DIGEST.search(log.read_text(errors="replace"))
    return (int(m.group(1)), m.group(2)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/workspace/MLEvolve/runs")
    ap.add_argument("--write", action="store_true", help="write the file when the digest matches")
    ap.add_argument("--only", default=None, help="substring filter on run names")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from engine.coldstart.knowledge import build_guidance_description, text_digest

    root = Path(args.runs)
    ok = mismatch = skipped = 0
    print(f"{'run':<44}{'logged':>16}{'replayed':>16}  result")
    print("-" * 100)

    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if args.only and args.only not in run.name:
            continue
        if (run / "logs" / "injected_knowledge.md").exists():
            continue
        want = logged_digest(run)
        if not want:
            continue                       # not a KB arm, or never logged an injection
        want_chars, want_digest = want

        cfg_path = run / "logs" / "config.yaml"
        try:
            raw = yaml.load(cfg_path.read_text(errors="replace"), Loader=_TagIgnoringLoader) or {}
            cfg = OmegaConf.create(raw)
            desc_str = str(cfg.get("desc_file") or "").strip()
            # is_file(), not exists(): Path("") is Path('.'), and '.' exists.
            desc = Path(desc_str) if desc_str else None
            if desc is None or not desc.is_file():
                print(f"{run.name:<44}{want_digest:>16}{'-':>16}  "
                      f"SKIP: description not readable ({desc_str or 'desc_file unset'})")
                skipped += 1
                continue
            task_desc = desc.read_text(errors="replace")
            # Cache-only: never pay for extraction or a new distillation during a replay.
            cfg.max_extractions_per_coldstart = 0
            cfg.coldstart.methodology_text = ""
            build_guidance_description(cfg, task_desc=task_desc)
            got = str(cfg.coldstart.methodology_text or "")
        except Exception as e:
            print(f"{run.name:<44}{want_digest:>16}{'-':>16}  SKIP: {type(e).__name__}: {e}")
            skipped += 1
            continue

        got_digest = text_digest(got)
        if got_digest == want_digest and len(got) == want_chars:
            ok += 1
            note = "MATCH"
            if args.write:
                (run / "logs" / "injected_knowledge.md").write_text(got, encoding="utf-8")
                note = "MATCH -> written"
            print(f"{run.name:<44}{want_digest:>16}{got_digest:>16}  {note}")
        else:
            mismatch += 1
            print(f"{run.name:<44}{want_digest:>16}{got_digest:>16}  "
                  f"MISMATCH ({want_chars} vs {len(got)} chars) — KB moved, not recoverable")

    print(f"\n{ok} recovered, {mismatch} unrecoverable, {skipped} skipped")
    if mismatch and args.write:
        print("Mismatched runs were deliberately NOT written. Report them as unmeasurable;\n"
              "substituting today's retrieval would score nodes against techniques the run\n"
              "never received.")
    if not args.write and ok:
        print("Re-run with --write to save the recovered files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
