"""Grade submission CSVs offline against mle-bench's private answers.

Why this exists instead of `mlebench grade-sample`: that CLI computes the score and then
immediately ranks it against the competition's bundled `leaderboard.csv`. Several of those
bundled leaderboards no longer have a `score` column (the Kaggle API renamed fields after
they were generated), so the CLI dies in `rank_score` *after* the score is already known:

    AssertionError: Leaderboard must have a `score` column.

This script does the same grading call and treats the leaderboard as optional — you always
get the score, and medal thresholds only if the leaderboard happens to be usable.

Usage:
    python utils/grade_local.py -c spooky-author-identification \\
        --data-dir /workspace/data/mlebench \\
        runs/20260804_062305_spooky-kb/workspace/ensembles_csv/*.csv

    # whole directory, ordered by wall-clock, with a budget cutoff:
    python utils/grade_local.py -c spooky-author-identification \\
        --data-dir /workspace/data/mlebench \\
        --cutoff-hours 8.6 \\
        runs/20260804_062305_spooky-kb/workspace/submission/

`elapsed` is measured from the oldest file in the batch, so grading a run's whole
`submission/` directory yields a score-vs-time curve. --cutoff-hours drops everything after
the cutoff, which is how two runs with unequal wall-clock budgets (e.g. one OOM-killed early)
can still be compared at a matched budget.
"""

import argparse
import sys
from pathlib import Path

# Candidate names for the leaderboard's score column, in preference order. The bundled files
# were dumped straight from `kaggle.api.competition_leaderboard_view`, whose row attributes
# have shifted between client versions.
_SCORE_COL_CANDIDATES = ("score", "submissionScore", "displayScore", "publicScore", "_score")


def _load_leaderboard(competition):
    """Return a leaderboard DataFrame with a usable `score` column, or None."""
    try:
        import pandas as pd

        from mlebench.data import get_leaderboard

        lb = get_leaderboard(competition)
        if "score" in lb.columns:
            return lb
        for col in _SCORE_COL_CANDIDATES:
            if col in lb.columns:
                lb = lb.rename(columns={col: "score"})
                return lb
        # Fall back to the sole numeric column, if there is exactly one.
        numeric = [c for c in lb.columns if pd.api.types.is_numeric_dtype(lb[c])]
        if len(numeric) == 1:
            return lb.rename(columns={numeric[0]: "score"})
        print(f"  (leaderboard unusable; columns = {list(lb.columns)} — medals skipped)",
              file=sys.stderr)
    except Exception as e:
        print(f"  (leaderboard unavailable: {e} — medals skipped)", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="submission .csv files and/or directories of them")
    ap.add_argument("-c", "--competition", required=True, help="mle-bench competition id")
    ap.add_argument("--data-dir", required=True, help="mle-bench data root (has <comp>/prepared/)")
    ap.add_argument("--cutoff-hours", type=float, default=None,
                    help="drop files modified more than N hours after the earliest one")
    args = ap.parse_args()

    from mlebench.registry import registry
    from mlebench.utils import load_answers, read_csv

    competition = registry.set_data_dir(Path(args.data_dir)).get_competition(args.competition)
    answers = load_answers(competition.answers)

    files = []
    for p in args.paths:
        path = Path(p)
        files.extend(sorted(path.glob("*.csv")) if path.is_dir() else [path])
    files = [f for f in files if f.is_file()]
    if not files:
        sys.exit("No CSV files found.")

    files.sort(key=lambda f: f.stat().st_mtime)
    t0 = files[0].stat().st_mtime
    if args.cutoff_hours is not None:
        before = len(files)
        files = [f for f in files if (f.stat().st_mtime - t0) / 3600 <= args.cutoff_hours]
        print(f"cutoff {args.cutoff_hours}h: kept {len(files)}/{before} files\n")

    lb = _load_leaderboard(competition)
    lower_better = None
    if lb is not None:
        try:
            lower_better = competition.grader.is_lower_better(lb)
        except Exception:
            pass

    rows = []
    for f in files:
        elapsed = (f.stat().st_mtime - t0) / 3600
        try:
            score = competition.grader(read_csv(f), answers)
        except Exception as e:
            rows.append((elapsed, f.name, None, f"INVALID: {type(e).__name__}: {e}"))
            continue
        note = ""
        if lb is not None and score is not None:
            try:
                r = competition.grader.rank_score(score, lb)
                medal = ("gold" if r["gold_medal"] else "silver" if r["silver_medal"]
                         else "bronze" if r["bronze_medal"]
                         else "above-median" if r["above_median"] else "-")
                note = f"medal={medal}"
            except Exception:
                pass
        rows.append((elapsed, f.name, score, note))

    print(f"\n{'elapsed':>8}  {'score':>12}  file")
    print("-" * 88)
    best = None
    for elapsed, name, score, note in rows:
        s = "n/a" if score is None else f"{score:.6f}"
        print(f"{elapsed:7.2f}h  {s:>12}  {name}  {note}")
        if score is not None and (best is None or
                                  (score < best[0] if lower_better is not False else score > best[0])):
            best = (score, name)

    if best:
        direction = ("lower is better" if lower_better
                     else "higher is better" if lower_better is False
                     else "direction unknown — check the competition metric")
        print(f"\nbest: {best[0]:.6f}  ({best[1]})   [{direction}]")
    if lower_better is None:
        print("\nNote: no usable leaderboard, so metric direction and medals are unavailable.\n"
              "      spooky-author-identification is multi-class log loss: LOWER is better.")


if __name__ == "__main__":
    main()
