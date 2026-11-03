"""Opt-in Mac replay preparation; never imported by the legacy replay path."""

from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

PROFILE = "mac-mps"
TASK = "jigsaw-unintended-bias-in-toxicity-classification"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_mps():
    """Check before output replacement, model downloads or API calls."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("mac-mps requires native Apple Silicon macOS")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK or set it to 0; silent CPU fallback is not a paired MPS run")
    import torch
    import psutil
    from packaging.version import Version

    if Version(torch.__version__) < Version("2.14.1"):
        raise RuntimeError("mac-mps requires torch>=2.14.1 for dynamic-shape MPS memory fixes; "
                           "install requirements_macos_replay.txt in the existing .venv")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this Python environment")
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()
                if d.metadata["Name"]}
    return dict(profile=PROFILE, training_device="mps", memory_device="cpu",
                system=platform.system(), machine=platform.machine(), macos=platform.mac_ver()[0],
                python=platform.python_version(), memory_bytes=psutil.virtual_memory().total,
                packages=dict(sorted(packages.items())))


@contextmanager
def mac_process_environment():
    # Candidates are fresh subprocesses; don't change the user's shell or cluster env.
    updates = {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def apply_profile(cfg, dataset_dir=None):
    if cfg.exp_id != TASK or not cfg.candidate_runtime.enabled:
        raise ValueError("mac-mps currently supports only Jigsaw with candidate_runtime enabled")
    cfg.agent.memory_embedding_device = "cpu"
    cfg.exec.max_parallel_run = 1
    cfg.use_grading_server = False
    # Public validation uses the existing contract; private grading is explicitly local.
    cfg.dataset_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else ""
    if dataset_dir and not Path(cfg.dataset_dir).is_dir():
        raise FileNotFoundError("Local --dataset-dir is missing")
    # OmegaConf's Path field turns an empty string into Path('.'); retain opt-in separately.
    return str(cfg.dataset_dir) if dataset_dir else None


def ensure_separate(destination, *protected):
    destination = Path(destination).resolve()
    for path in protected:
        path = Path(path).resolve()
        if destination == path or destination.is_relative_to(path) or path.is_relative_to(destination):
            raise ValueError(f"Mac output overlaps protected input: {path}")


def source_identity(source, snapshot, budget):
    parent = snapshot["nodes"][-1]
    contract = source / "workspace/candidate_results/contract/manifest.json"
    memory = source / "workspace/global_memory/records.json"
    return dict(source_path=str(source.resolve()), original_parent_id=parent["id"],
                journal_sha256=digest(source / "logs/journal.json"),
                source_config_sha256=digest(source / "logs/config.yaml"),
                original_source_sha256=hashlib.sha256(parent["code"].encode()).hexdigest(),
                source_review_sha256=digest(source / "workspace/candidate_results/candidates" / parent["id"] / "review.json"),
                contract_sha256=digest(contract), memory_sha256=digest(memory) if memory.is_file() else None,
                replay_budget_seconds=budget)


def verify_public_inputs(inputs, contract_dir):
    from engine.candidate_runtime.jigsaw import load_contract

    contract = load_contract(contract_dir)
    for split in ("train", "test"):
        if digest(Path(inputs) / f"{split}.csv") != contract[f"public_{split}_sha256"]:
            raise ValueError(f"Public {split}.csv does not match the original validation contract")


def load_baseline(directory, expected, environment):
    """Verify a completed calibration before any existing arm can be removed."""
    from utils.replay_macos.adapters import get_adapter
    from engine.candidate_runtime.store import ResultStore

    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "baseline.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("profile") != PROFILE:
        raise ValueError("Mac baseline is not calibrated; manually run prepare_macos_parent.py --execute first")
    if manifest.get("source") != expected:
        raise ValueError("Mac baseline belongs to different source inputs or budget")
    adapter = get_adapter(expected["original_source_sha256"])
    if manifest.get("adapter_version") != adapter.ADAPTER_VERSION:
        raise ValueError("Mac baseline uses a different adapter version; prepare a new baseline")
    if manifest.get("environment") != environment:
        raise ValueError("Mac environment differs from the calibrated baseline; use the same environment")
    if not {"original_solution.py", "solution.py", "baseline_node.json"}.issubset(manifest.get("files", {})):
        raise ValueError("Mac baseline is missing required artifact hashes")
    for relative, wanted in manifest["files"].items():
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or digest(path) != wanted:
            raise ValueError(f"Mac baseline artifact changed: {relative}")
    original = (directory / "original_solution.py").read_text()
    code = (directory / "solution.py").read_text()
    if (hashlib.sha256(original.encode()).hexdigest() != expected["original_source_sha256"]
            or code != adapter.adapt(original)):
        raise ValueError("Mac baseline is not the locked deterministic parent adaptation")
    node = json.loads((directory / "baseline_node.json").read_text())
    if (node["code"] != code or node["id"] != manifest["effective_parent_id"]
            or not node["is_valid"] or node["is_buggy"]):
        raise ValueError("Mac baseline node is inconsistent")
    from replay_improve import component
    node_id, snapshot_id = component(node["id"]), component(node["best_snapshot_id"])
    calibration = directory / "calibration"
    store = ResultStore(calibration / "workspace", calibration / "logs")
    verified = store.verify(node_id, store.candidate_dir(node_id) / "snapshots" / snapshot_id)
    execution = json.loads((store.candidate_dir(node_id) / "execution.json").read_text())
    if (execution["status"] not in ("completed", "budget_exhausted")
            or verified["metric"] != node["metric"]["value"]
            or verified["metric"] != manifest["metric"]
            or verified["maximize"] != node["metric"]["maximize"]
            or verified["source_sha256"] != digest(directory / "solution.py")
            or digest(calibration / "workspace/candidate_results/contract/manifest.json") != expected["contract_sha256"]):
        raise ValueError("Mac calibration measurements do not match the baseline")
    return manifest, node


def attach_baseline(snapshot, directory, node, arm_dir):
    """Append a new measured node; never relabel old CUDA facts as MPS results."""
    from replay_improve import copy_runtime_history

    restored = copy.deepcopy(snapshot)
    node = copy.deepcopy(node)
    if node["id"] in {n["id"] for n in restored["nodes"]}:
        raise ValueError("Mac baseline node ID collides with the original journal")
    original_parent = restored["nodes"][-1]["id"]
    node.update(step=len(restored["nodes"]), parent=None, children=[], local_best_node=None,
                child_count_lock=None, lock=False, expected_child_count=0)
    restored["nodes"].append(node)
    restored["node2parent"][node["id"]] = original_parent
    restored["node2best_local_node"][node["id"]] = node["id"]
    copy_runtime_history(Path(directory) / "calibration", arm_dir, {"nodes": [node]})
    return restored
