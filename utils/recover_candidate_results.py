"""Recover published candidate artifacts without resuming training or invoking an LLM."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.candidate_runtime.integration import export_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    result = export_results(args.run_dir / "workspace", args.run_dir / "logs", args.top_k)
    print(f"Recovered results: {result}" if result else "No eligible complete candidate snapshots")


if __name__ == "__main__":
    main()
