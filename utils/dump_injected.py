"""Verify saved literature injections from historical runs without replaying retrieval.

Automatic coldstart retrieval has been retired. Missing historical injections cannot
be reconstructed by this version, and task-specific retrieval caches are not evidence
of what an old experiment received. This command never modifies run artifacts.

    python utils/dump_injected.py --runs /workspace/MLEvolve/runs
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from engine.coldstart.knowledge import text_digest

LOG_DIGEST = re.compile(r"Knowledge injected at draft: (\d+) chars, digest ([0-9a-f]+)")


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
    ap.add_argument("--write", action="store_true",
                    help="retired option; historical retrieval replay is no longer supported")
    ap.add_argument("--only", default=None, help="substring filter on run names")
    args = ap.parse_args()

    if args.write:
        print("--write is retired: this command only verifies saved files; nothing will be written.")
    root = Path(args.runs)
    ok = mismatch = skipped = 0
    print(f"{'run':<44}{'logged':>16}{'saved':>16}  result")
    print("-" * 100)

    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if args.only and args.only not in run.name:
            continue
        want = logged_digest(run)
        if not want:
            continue
        want_chars, want_digest = want
        target = run / "logs" / "injected_knowledge.md"
        if not target.is_file():
            skipped += 1
            print(f"{run.name:<44}{want_digest:>16}{'-':>16}  "
                  "SKIP: historical replay is no longer supported; no saved injection")
            continue
        saved = target.read_text(errors="replace")
        have = text_digest(saved)
        if have == want_digest and len(saved) == want_chars:
            ok += 1
            print(f"{run.name:<44}{want_digest:>16}{have:>16}  verified")
        else:
            mismatch += 1
            print(f"{run.name:<44}{want_digest:>16}{have:>16}  "
                  "MISMATCH: saved content is not the logged injection; file left unchanged")

    print(f"\n{ok} verified, {mismatch} mismatched, {skipped} unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
