"""Public-data-only, versioned Jigsaw holdout and continuous-AUC evaluator."""

from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .io import atomic_json, check_hashes, digest, file_hashes, read_json, seal_directory

TASK_ID = "jigsaw-unintended-bias-in-toxicity-classification"
METRIC_VERSION = "jubias-continuous-auc-v1"
IDENTITIES = ["male", "female", "homosexual_gay_or_lesbian", "christian", "jewish",
              "muslim", "black", "white", "psychiatric_or_mental_illness"]


def predictions(values, rows):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {rows} finite scalar predictions; got {values.shape}")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Jigsaw predictions must be probabilities in [0, 1]")
    return values


def score(answers, values):
    values = predictions(values, len(answers))
    label = answers["target"].to_numpy() >= 0.5
    overall = float(roc_auc_score(label, values))
    parts = {"subgroup": [], "bpsn": [], "bnsp": []}
    for identity in IDENTITIES:
        group = answers[identity].fillna(0).to_numpy() >= 0.5
        masks = (group, (group & ~label) | (~group & label),
                 (group & label) | (~group & ~label))
        for kind, mask in zip(parts, masks):
            if np.unique(label[mask]).size != 2:
                raise ValueError(f"Undefined AUC for {identity}/{kind}; do not drop metric terms")
            parts[kind].append(float(roc_auc_score(label[mask], values[mask])))

    def power_mean(values):
        # The limit is zero if any AUC is zero. Avoid 0 ** -5 warnings.
        return 0.0 if min(values) == 0 else float(np.mean(np.power(values, -5.0)) ** (-0.2))

    return 0.25 * overall + 0.25 * sum(power_mean(v) for v in parts.values())


def prepare_contract(workspace, public_dir, seed, fraction, task_id=TASK_ID):
    if task_id != TASK_ID:
        raise ValueError(f"candidate_runtime has no validated task adapter for {task_id}; disable it")
    workspace, public_dir = Path(workspace), Path(public_dir)
    parent = workspace / "candidate_results"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / "contract"
    if destination.exists():
        contract = load_contract(destination)
        if contract["seed"] != seed or contract["validation_fraction"] != fraction:
            raise ValueError("Existing holdout contract does not match seed/fraction")
        return destination
    train_path, test_path = public_dir / "train.csv", public_dir / "test.csv"
    # This module never accesses prepared/private or MLE-bench's answers.
    train = pd.read_csv(train_path, usecols=["id", "target", *IDENTITIES], dtype={"id": str})
    test = pd.read_csv(test_path, usecols=["id"], dtype={"id": str})
    if train.id.isna().any() or test.id.isna().any() or not train.id.is_unique or not test.id.is_unique:
        raise ValueError("Jigsaw train/test IDs must be present and unique")
    if set(train.id) & set(test.id):
        raise ValueError("Public train/test IDs overlap")
    if not np.isfinite(train.target).all():
        raise ValueError("Invalid training labels")
    # Deterministic retries only ensure every metric component is defined; never optimize a score.
    for attempt in range(20):
        fit_idx, val_idx = train_test_split(np.arange(len(train)), test_size=fraction,
                                          random_state=seed + attempt, stratify=train.target >= 0.5)
        fit_idx, val_idx = np.sort(fit_idx), np.sort(val_idx)
        validation = train.iloc[val_idx]
        try:
            score(validation, np.full(len(validation), 0.5))
            break
        except ValueError:
            continue
    else:
        raise ValueError("Cannot build a holdout with all Jigsaw AUC components defined")
    with tempfile.TemporaryDirectory(prefix=".contract-", dir=parent) as tmp:
        staging = Path(tmp) / "sealed"
        staging.mkdir()
        np.savez_compressed(staging / "split.npz", train=fit_idx, validation=val_idx)
        train[["id"]].to_csv(staging / "train_ids.csv", index=False)
        validation.to_csv(staging / "validation.csv", index=False)
        test.to_csv(staging / "test_ids.csv", index=False)
        manifest = dict(version=1, task_id=task_id, metric_version=METRIC_VERSION,
                        maximize=True, seed=seed, split_attempt=attempt,
                        split_method="stratified-target-v1", validation_fraction=fraction,
                        train_rows=len(train), validation_rows=len(validation), test_rows=len(test),
                        public_train_sha256=digest(train_path), public_test_sha256=digest(test_path),
                        files=file_hashes(staging))
        manifest["contract_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        atomic_json(staging / "manifest.json", manifest)
        seal_directory(staging, destination)
    return destination


def load_contract(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    body = {k: v for k, v in manifest.items() if k != "contract_id"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != manifest["contract_id"]:
        raise ValueError("Changed holdout contract")
    if manifest["task_id"] != TASK_ID or manifest["metric_version"] != METRIC_VERSION:
        raise ValueError("Unsupported task/metric contract")
    check_hashes(directory, manifest["files"])
    return manifest


def check_csv(path, expected_ids):
    frame = pd.read_csv(path, dtype={"id": str})
    if list(frame.columns) != ["id", "prediction"] or frame.id.tolist() != list(expected_ids):
        raise ValueError("Prediction CSV schema, row count, IDs or order do not match the contract")
    return predictions(frame.prediction.to_numpy(), len(expected_ids))
