"""Download the ExpansionRx-OpenADMET challenge data and lay it out for MLEvolve.

Produces the layout run_single_task.sh expects:

    <dataset-dir>/<exp-id>/prepared/public/
        description.md
        train.csv              (Molecule Name, SMILES, 9 endpoints)
        test.csv               (Molecule Name, SMILES)   <- blinded, no labels
        sample_submission.csv  (Molecule Name + 9 endpoint columns)

Usage:
    python examples/openadmet-expansionrx/prepare_data.py --dataset-dir /workspace/mle-bench/data
    # then:
    bash run_single_task.sh openadmet-expansionrx /workspace/mle-bench/data

Needs network + pandas. If huggingface.co is blocked, set HF_ENDPOINT=https://hf-mirror.com.
"""
import argparse
import io
import os
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

TRAIN_REPO = "openadmet/openadmet-expansionrx-challenge-train-data"
TEST_REPO = "openadmet/openadmet-expansionrx-challenge-test-data-blinded"
TRAIN_FILE = "expansion_data_train.csv"          # ML-ready (in-range measurements only)
TEST_FILE = "expansion_data_test_blinded.csv"

ENDPOINTS = [
    "LogD",
    "KSOL",
    "MLM CLint",
    "HLM CLint",
    "Caco-2 Permeability Efflux",
    "Caco-2 Permeability Papp A>B",
    "MPPB",
    "MBPB",
    "MGMB",
]
ID_COL = "Molecule Name"
SMILES_COL = "SMILES"


def hf_url(repo: str, filename: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{repo}/resolve/main/{filename}"


def fetch(repo: str, filename: str) -> bytes:
    url = hf_url(repo, filename)
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True,
                    help="dataset root, e.g. /workspace/mle-bench/data")
    ap.add_argument("--exp-id", default="openadmet-expansionrx")
    ap.add_argument("--raw", action="store_true",
                    help="use expansion_data_train_raw.csv instead of the ML-ready train file")
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required: pip install pandas")

    out = Path(args.dataset_dir) / args.exp_id / "prepared" / "public"
    out.mkdir(parents=True, exist_ok=True)

    train_file = "expansion_data_train_raw.csv" if args.raw else TRAIN_FILE
    print("Downloading challenge data ...")
    train = pd.read_csv(io.BytesIO(fetch(TRAIN_REPO, train_file)))
    test = pd.read_csv(io.BytesIO(fetch(TEST_REPO, TEST_FILE)))

    # --- sanity checks: fail loudly if the upstream schema moved ---
    for name, df, required in (("train", train, [ID_COL, SMILES_COL]),
                               ("test", test, [ID_COL, SMILES_COL])):
        missing = [c] if (c := next((c for c in required if c not in df.columns), None)) else []
        if missing:
            sys.exit(f"FATAL: {name} is missing column(s) {missing}. Got: {list(df.columns)}")

    missing_ep = [e for e in ENDPOINTS if e not in train.columns]
    if missing_ep:
        sys.exit(f"FATAL: train is missing endpoint column(s): {missing_ep}\n"
                 f"Got: {list(train.columns)}")

    train = train[[ID_COL, SMILES_COL] + ENDPOINTS]
    test_out = test[[ID_COL, SMILES_COL]]

    # sample_submission: median of each endpoint (a valid, non-degenerate baseline in
    # ORIGINAL units — the official scorer applies the log transform itself).
    sample = test_out[[ID_COL]].copy()
    for ep in ENDPOINTS:
        med = train[ep].median()
        sample[ep] = 0.0 if pd.isna(med) else float(med)

    train.to_csv(out / "train.csv", index=False)
    test_out.to_csv(out / "test.csv", index=False)
    sample.to_csv(out / "sample_submission.csv", index=False)

    desc_src = HERE / "description.md"
    if desc_src.exists():
        shutil.copy(desc_src, out / "description.md")
    else:
        print(f"  WARN: {desc_src} not found — copy description.md into {out} manually")

    # --- report: label sparsity drives the modelling strategy, so surface it ---
    print(f"\nWrote -> {out}")
    print(f"  train.csv              {len(train):>6} molecules")
    print(f"  test.csv               {len(test_out):>6} molecules (blinded)")
    print(f"  sample_submission.csv  {len(sample):>6} rows")
    print("\nLabel coverage per endpoint (train):")
    for ep in ENDPOINTS:
        n = int(train[ep].notna().sum())
        print(f"  {ep:<32} {n:>5} / {len(train)}  ({100*n/len(train):4.1f}%)")

    print(f"\nNext:\n"
          f"  bash run_single_task.sh {args.exp_id} {args.dataset_dir}\n"
          f"  (this competition is not in mle-bench — add use_grading_server=False,\n"
          f"   e.g. EXTRA_RUN_ARGS='use_grading_server=False')")


if __name__ == "__main__":
    main()
