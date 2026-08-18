"""Grade every run's ensembles and emit one machine-readable scores.csv.

This is the bridge between the cluster and the local analysis. mle-bench's private answers only
exist here, so scores cannot be computed on a laptop from downloaded run directories; this
script produces the one file that carries them across.

    python utils/grade_all.py --runs /workspace/MLEvolve/runs -o /workspace/MLEvolve/runs/scores.csv

Output columns:

    run            run directory name — the join key for Agentic_Knowledge_Base/scripts/analyze_runs.py
    competition    mle-bench competition id, read from the run's own config.yaml
    k              ensemble size (top{K}ens...), 0 if the filename does not encode one
    cum_hours      cumulative training time of the fused solutions, from the filename
    score          graded against the private answers
    medal          gold | silver | bronze | above-med | - | "" if the leaderboard is unusable
    lower_better   1 / 0 / "" — metric direction, so the analysis never has to guess
    file           source CSV

Runs whose CSVs fail to grade are still written, with an empty score and the error in `note`.
Silence about a failure is worse than a row saying it failed: a missing arm looks identical to
an arm that was never launched.

Re-running is cheap and idempotent — grading is a metric computation, no model runs. Pass
--only to restrict to matching run names when iterating.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

ENS_RE = re.compile(r"top(\d+)ens-total_run_time([\d.]+)h\.csv$")

# Early runs encoded the arm into exp_id ("openadmet-kb"); the competition id is the stem.
ARM_SUFFIXES = ("-kbimp", "-kbfix", "-kb", "-base")


class _TagIgnoringLoader(yaml.SafeLoader):
    """config.yaml embeds !!python/object/apply:pathlib.PosixPath, which SafeLoader rejects and
    unsafe_load would execute. Map unknown tags to None."""


_TagIgnoringLoader.add_multi_constructor("", lambda loader, suffix, node: None)


def _competition_of(run_dir: Path) -> str:
    cfg_path = run_dir / "logs" / "config.yaml"
    if not cfg_path.exists():
        return ""
    try:
        cfg = yaml.load(cfg_path.read_text(errors="replace"), Loader=_TagIgnoringLoader) or {}
    except Exception:
        return ""
    exp = str(cfg.get("exp_id") or "")
    for suf in ARM_SUFFIXES:
        if exp.endswith(suf):
            return exp[: -len(suf)]
    return exp


def _load_leaderboard(competition):
    try:
        import pandas as pd

        from mlebench.data import get_leaderboard

        lb = get_leaderboard(competition)
        if "score" in lb.columns:
            return lb
        for col in ("submissionScore", "displayScore", "publicScore", "_score"):
            if col in lb.columns:
                return lb.rename(columns={col: "score"})
        numeric = [c for c in lb.columns if pd.api.types.is_numeric_dtype(lb[c])]
        if len(numeric) == 1:
            return lb.rename(columns={numeric[0]: "score"})
    except Exception:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="/workspace/MLEvolve/runs")
    ap.add_argument("--data-dir", default="/workspace/data/mlebench")
    ap.add_argument("-o", "--out", default=None,
                    help="output CSV (default: <runs>/scores.csv)")
    ap.add_argument("--only", default=None, help="substring filter on run directory names")
    args = ap.parse_args()

    root = Path(args.runs)
    out = Path(args.out) if args.out else root / "scores.csv"
    if not root.is_dir():
        sys.exit(f"no such runs directory: {root}")

    from mlebench.registry import registry
    from mlebench.utils import load_answers, read_csv

    reg = registry.set_data_dir(Path(args.data_dir))
    cache: dict[str, tuple] = {}          # competition -> (comp, answers, lb, lower_better)

    rows = []
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for run in run_dirs:
        if args.only and args.only not in run.name:
            continue
        ens_dir = run / "workspace" / "ensembles_csv"
        files = sorted(ens_dir.glob("*.csv")) if ens_dir.is_dir() else []
        if not files:
            continue

        comp_id = _competition_of(run)
        if not comp_id:
            print(f"[skip] {run.name}: cannot determine competition (no config.yaml)",
                  file=sys.stderr)
            continue

        if comp_id not in cache:
            try:
                comp = reg.get_competition(comp_id)
                answers = load_answers(comp.answers)
            except Exception as e:
                print(f"[skip] {run.name}: {comp_id} not gradable — {e}", file=sys.stderr)
                cache[comp_id] = None
                continue
            lb = _load_leaderboard(comp)
            lower = None
            if lb is not None:
                try:
                    lower = comp.grader.is_lower_better(lb)
                except Exception:
                    pass
            cache[comp_id] = (comp, answers, lb, lower)
        if cache[comp_id] is None:
            continue
        comp, answers, lb, lower = cache[comp_id]

        for f in files:
            m = ENS_RE.search(f.name)
            row = {"run": run.name, "competition": comp_id,
                   "k": int(m.group(1)) if m else 0,
                   "cum_hours": float(m.group(2)) if m else "",
                   "score": "", "medal": "",
                   "lower_better": "" if lower is None else int(bool(lower)),
                   "file": f.name, "note": ""}
            try:
                row["score"] = comp.grader(read_csv(f), answers)
            except Exception as e:
                row["note"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                print(f"[bad ] {run.name}/{f.name}: {row['note']}", file=sys.stderr)
                continue
            if lb is not None:
                try:
                    r = comp.grader.rank_score(row["score"], lb)
                    row["medal"] = ("gold" if r["gold_medal"] else "silver" if r["silver_medal"]
                                    else "bronze" if r["bronze_medal"]
                                    else "above-med" if r["above_median"] else "-")
                except Exception:
                    pass
            rows.append(row)
        print(f"[ok  ] {run.name}: {len(files)} ensemble(s), {comp_id}")

    if not rows:
        sys.exit("nothing graded — check --runs and --data-dir")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "competition", "k", "cum_hours", "score",
                                           "medal", "lower_better", "file", "note"])
        w.writeheader()
        w.writerows(rows)

    graded = sum(1 for r in rows if r["score"] != "")
    print(f"\nwrote {out}: {len(rows)} rows, {graded} graded, "
          f"{len(rows) - graded} failed, {len({r['run'] for r in rows})} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
