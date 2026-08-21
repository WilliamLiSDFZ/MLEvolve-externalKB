"""Re-run submission fusion for every run with the time cap lifted, keeping the original results.

Why this exists. `submission_fusion_utils.py` stops adding candidates once their cumulative
execution time exceeds `max_total_time_hours = 9.0`:

    if t_total > t_max:
        print(f"[WARN] Total time {t_total}s > {t_max}s limit, stopping.")
        break

That cap sums candidate training times SERIALLY, but the candidates were trained in PARALLEL
during the run (`parallel_search_num: 3`), so it is conservative by roughly that factor. Its
practical effect is to throttle ensemble size by how fast an arm's solutions happened to train,
which is a confound aligned with the treatment rather than a property of solution quality: on
this corpus it binds on about half the usable runs, and one arm accumulated 11 valid candidates
yet was allowed to fuse only 2.

It is also not applied consistently. `20260817_022101_lmsys-kb` has ensembles at 9.87 h, 10.09 h
and 15.24 h — all past the cap, all written — while contemporaneous runs were cut at 8.5 h. So
the ensemble sizes currently in the corpus are not the product of one rule.

This script re-runs fusion for every run into `workspace/ensembles_uncapped/`, leaving
`workspace/ensembles_csv/` byte-for-byte alone, so the two can be graded and compared. Whether
lifting the cap changes the ranking of the arms is then a measurable question about the
evaluation protocol rather than an invisible assumption inside it.

    python utils/refuse_all.py --runs /workspace/MLEvolve/runs            # dry run
    python utils/refuse_all.py --runs /workspace/MLEvolve/runs --apply

Cheap: fusion re-averages `submission.csv` files that already exist in `top_solution/`. Nothing
is retrained; expect seconds per run.

Two things to keep in mind when using the output:

1. An uncapped ensemble whose members total 15 h of training could not be retrained inside a
   12 h budget on one GPU. MLE-bench grades the submitted CSV and does not require retraining,
   so this is legitimate — but if the numbers are reported, say the cap was lifted.
2. Having K = 1..N available for every run makes it possible to report whichever K looks best.
   Decide which K to report BEFORE looking, or report the whole curve. The existing cap was at
   least a rule fixed in advance; lifting it moves that discipline onto you.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

ARM_SUFFIXES = ("-kbimp", "-kbfix", "-kb", "-base")
OUT_SUBDIR = "ensembles_uncapped"

# Lift the serial-time cap far beyond any real run, and let the sweep consider every candidate
# the run kept rather than the default six.
UNCAPPED_HOURS = 10_000.0
UNCAPPED_CANDIDATES = 12


class _TagIgnoringLoader(yaml.SafeLoader):
    """config.yaml embeds !!python/object/apply:pathlib.PosixPath."""


def _reconstruct(loader, suffix, node):
    if "pathlib" in suffix and isinstance(node, yaml.SequenceNode):
        parts = [str(p) for p in loader.construct_sequence(node)]
        return str(Path(*parts)) if parts else None
    return None


_TagIgnoringLoader.add_multi_constructor("", _reconstruct)


def competition_of(run: Path) -> str:
    cfg = run / "logs" / "config.yaml"
    if not cfg.exists():
        return ""
    try:
        d = yaml.load(cfg.read_text(errors="replace"), Loader=_TagIgnoringLoader) or {}
    except Exception:
        return ""
    exp = str(d.get("exp_id") or "")
    for suf in ARM_SUFFIXES:
        if exp.endswith(suf):
            return exp[: -len(suf)]
    return exp


def ks_in(d: Path) -> list[int]:
    import re
    out = []
    if d.is_dir():
        for f in d.glob("*.csv"):
            m = re.search(r"top(\d+)ens", f.name)
            if m:
                out.append(int(m.group(1)))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/workspace/MLEvolve/runs")
    ap.add_argument("--apply", action="store_true", help="actually run fusion (default: dry run)")
    ap.add_argument("--only", default=None, help="substring filter on run names")
    ap.add_argument("--force", action="store_true",
                    help="redo runs that already have an uncapped directory")
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.is_dir():
        sys.exit(f"no such runs directory: {root}")

    todo = []
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        if args.only and args.only not in run.name:
            continue
        ws = run / "workspace"
        top = ws / "top_solution"
        if not top.is_dir() or not any(top.iterdir()):
            continue                       # nothing to fuse; preempted or never got there
        comp = competition_of(run)
        if not comp:
            print(f"[skip] {run.name}: cannot determine competition")
            continue
        if (ws / OUT_SUBDIR).is_dir() and not args.force:
            print(f"[have] {run.name}: {OUT_SUBDIR} exists, use --force to redo")
            continue
        todo.append((run, comp))

    print(f"\n{len(todo)} run(s) to re-fuse into {OUT_SUBDIR}/\n")
    print(f"{'run':<42}{'capped K':>22}  competition")
    print("-" * 100)
    for run, comp in todo:
        print(f"{run.name:<42}{str(ks_in(run / 'workspace' / 'ensembles_csv')):>22}  {comp}")
    if not args.apply:
        print("\nDry run. Re-run with --apply. Existing ensembles_csv/ is never modified.")
        return 0

    ok = failed = 0
    for run, comp in todo:
        print(f"\n=== {run.name} ===")
        cmd = [sys.executable, str(REPO / "utils" / "submission_fusion_utils.py"),
               "--task_id", comp,
               "--exp_name", run.name,
               "--runs_root", str(root),
               "--max_total_hours", str(UNCAPPED_HOURS),
               "--max_candidates", str(UNCAPPED_CANDIDATES),
               "--out_subdir", OUT_SUBDIR]
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        tail = [ln for ln in r.stdout.splitlines() if ln.startswith(("Saved:", "Ensembling", "[WARN]"))]
        for ln in tail[-6:]:
            print(f"  {ln}")
        if r.returncode != 0:
            failed += 1
            print(f"  FAILED (exit {r.returncode}): {(r.stderr or '').strip().splitlines()[-1:]}")
            continue
        ok += 1
        before = ks_in(run / "workspace" / "ensembles_csv")
        after = ks_in(run / "workspace" / OUT_SUBDIR)
        print(f"  K: capped {before}  ->  uncapped {after}")

    print(f"\n{ok} re-fused, {failed} failed")
    print(f"Original ensembles_csv/ untouched. Grade both with utils/grade_all.py, which now\n"
          f"emits a `variant` column (capped | uncapped).")
    print("Then check the new files for NaN: fusion pads missing ids via _align_submission and\n"
          "quality_check.py has no NaN guard, so more fusion means more exposure to that bug.\n"
          "analyze_runs.py already scans for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
