"""Grade several arms of one experiment and print a matched-K comparison table.

`grade_local.py` grades a pile of CSVs. This grades a set of RUN DIRECTORIES that are meant to
be compared with each other, and enforces the one comparison rule that is easy to get wrong:
only compare arms at the same ensemble size K.

    python utils/compare_arms.py \\
        -c learning-agency-lab-automated-essay-scoring-2 \\
        --data-dir /workspace/data/mlebench \\
        A=runs/20260817_193119_essay-base-s43 \\
        B=runs/20260817_193958_essay-kb-s43 \\
        C=runs/20260817_193958_essay-kbimp-s43

Each argument is `LABEL=path`. Labels are free-form; A/B/C is the convention used here for
baseline / KB-at-draft / KB-at-draft-and-improve.

── why matched K ───────────────────────────────────────────────────────────────────────
`submission_fusion_utils.py` writes `ensembles_csv/top{K}ens-total_run_time{H}h.csv`, adding
candidate solutions until the cumulative training time exceeds its budget. Arms that found
faster-training solutions get to stack more of them, so the arms often stop at different K:
comparing one arm's top-3 against another's top-1 measures how much fusion each arm could
afford, not whether the knowledge base helped. This script therefore reports each K separately
and computes deltas only where every arm has that K.

The cumulative runtime encoded in the filename is printed alongside, because equal K does not
imply equal compute — an arm reaching top-2 in 7.6 h and another in 8.5 h are matched on
ensemble size but not on cost, and that is worth seeing rather than hiding.

── the score printed here is the real one ──────────────────────────────────────────────
The metric the agent logs during the run is its own internal validation score. It is not
comparable across arms (different arms hold out different data) and is systematically
optimistic. Everything below is graded against mle-bench's private answers.
"""

import argparse
import re
import sys
from pathlib import Path

# ensembles_csv/top2ens-total_run_time8.48h.csv  ->  K=2, hours=8.48
ENS_RE = re.compile(r"top(\d+)ens-total_run_time([\d.]+)h\.csv$")


def _load_leaderboard(competition):
    """mle-bench's bundled leaderboards vary in schema; return one with a `score` column or None."""
    try:
        import pandas as pd

        from mlebench.data import get_leaderboard

        lb = get_leaderboard(competition)
        if "score" in lb.columns:
            return lb
        for col in ("score", "submissionScore", "displayScore", "publicScore", "_score"):
            if col in lb.columns:
                return lb.rename(columns={col: "score"})
        numeric = [c for c in lb.columns if pd.api.types.is_numeric_dtype(lb[c])]
        if len(numeric) == 1:
            return lb.rename(columns={numeric[0]: "score"})
    except Exception as e:
        print(f"(leaderboard unavailable: {e} — medals skipped)", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", metavar="LABEL=RUNDIR")
    ap.add_argument("-c", "--competition", required=True)
    ap.add_argument("--data-dir", required=True, help="mle-bench data root (has <comp>/prepared/)")
    ap.add_argument("--baseline", default="A", help="label to compute deltas against")
    args = ap.parse_args()

    from mlebench.registry import registry
    from mlebench.utils import load_answers, read_csv

    competition = registry.set_data_dir(Path(args.data_dir)).get_competition(args.competition)
    answers = load_answers(competition.answers)
    lb = _load_leaderboard(competition)
    lower_better = None
    if lb is not None:
        try:
            lower_better = competition.grader.is_lower_better(lb)
        except Exception:
            pass

    arms: dict[str, Path] = {}
    for spec in args.arms:
        if "=" not in spec:
            sys.exit(f"expected LABEL=RUNDIR, got {spec!r}")
        label, path = spec.split("=", 1)
        arms[label] = Path(path)

    # scores[label][K] = (score, hours, medal, filename)
    scores: dict[str, dict[int, tuple]] = {}
    for label, run in arms.items():
        ens_dir = run / "workspace" / "ensembles_csv"
        scores[label] = {}
        if not ens_dir.is_dir():
            print(f"[{label}] no ensembles_csv in {run} — arm produced no fused submission",
                  file=sys.stderr)
            continue
        for f in sorted(ens_dir.glob("*.csv")):
            m = ENS_RE.search(f.name)
            k = int(m.group(1)) if m else 0
            hours = float(m.group(2)) if m else float("nan")
            try:
                score = competition.grader(read_csv(f), answers)
            except Exception as e:
                print(f"[{label}] {f.name}: INVALID — {type(e).__name__}: {e}", file=sys.stderr)
                continue
            medal = ""
            if lb is not None and score is not None:
                try:
                    r = competition.grader.rank_score(score, lb)
                    medal = ("gold" if r["gold_medal"] else "silver" if r["silver_medal"]
                             else "bronze" if r["bronze_medal"]
                             else "above-med" if r["above_median"] else "-")
                except Exception:
                    pass
            scores[label][k] = (score, hours, medal, f.name)

    # ── per-arm detail ─────────────────────────────────────────────────────────────────
    print(f"\ncompetition: {args.competition}")
    direction = ("lower is better" if lower_better
                 else "higher is better" if lower_better is False
                 else "DIRECTION UNKNOWN — check the metric before reading these")
    print(f"metric direction: {direction}\n")

    print(f"{'arm':<6}{'K':>3}{'cum_hours':>11}{'score':>12}  medal   file")
    print("-" * 96)
    for label in arms:
        for k in sorted(scores[label]):
            s, h, medal, name = scores[label][k]
            print(f"{label:<6}{k:>3}{h:>11.2f}{s:>12.5f}  {medal:<9} {name}")
        if not scores[label]:
            print(f"{label:<6}  —  (nothing gradable)")

    # ── matched-K comparison ───────────────────────────────────────────────────────────
    common = set.intersection(*(set(v) for v in scores.values())) if all(scores.values()) else set()
    print(f"\nmatched-K comparison (K present in every arm: "
          f"{sorted(common) if common else 'NONE'})")
    if not common:
        print("  No K is shared by all arms, so no fair comparison can be made from the\n"
              "  ensembles alone. Grade the raw submission/ directories at a matched wall-clock\n"
              "  budget instead:  utils/grade_local.py --cutoff-hours <H> <run>/workspace/submission/")
        return

    base = args.baseline if args.baseline in scores else sorted(scores)[0]
    print(f"  deltas relative to arm {base}"
          f"{'  (negative = better)' if lower_better else '  (positive = better)'}\n")
    labels = list(arms)
    print("   K  " + "".join(f"{l:>14}" for l in labels))
    print("-" * (6 + 14 * len(labels)))
    for k in sorted(common):
        row = f"{k:>4}  "
        for l in labels:
            row += f"{scores[l][k][0]:>14.5f}"
        print(row)
        row = "      "
        for l in labels:
            d = scores[l][k][0] - scores[base][k][0]
            row += f"{('' if l == base else f'{d:+.5f}'):>14}"
        print(row)

    print("\nreminder: one seed is one draw. A difference here is a data point, not a result —\n"
          "the paired-difference sd on jigsaw was ~0.006, so single-seed gaps below that are\n"
          "indistinguishable from noise.")


if __name__ == "__main__":
    main()
