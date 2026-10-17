"""Load the prepared task corpus and rank it once for this experiment."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time

from .technique_reranker import render_techniques, rerank_techniques

logger = logging.getLogger("MLEvolve")


def _read_limit(path):
    try:
        value = int(Path(path).read_text().strip())
        return value if 0 < value < 2 ** 60 else None
    except (OSError, ValueError):
        return None


def runtime_context(agent):
    import psutil

    now = time.time()
    remaining = max(0, agent.acfg.time_limit - (now - agent.start_time))
    deadline = os.environ.get("MLEVOLVE_RUN_DEADLINE")
    outer_remaining = max(0, float(deadline) - now) if deadline else None
    if outer_remaining is not None:
        remaining = min(remaining, outer_remaining)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    parallel = agent.cfg.agent.search.parallel_search_num
    configured_cpus = int(agent.cfg.cpu_number)
    memory = psutil.virtual_memory()
    memory_limit = (_read_limit("/sys/fs/cgroup/memory.max")
                    or _read_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    memory_used = (_read_limit("/sys/fs/cgroup/memory.current")
                   or _read_limit("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    cpu_quota = None
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpu_quota = int(quota) / int(period)
    except (OSError, ValueError, ZeroDivisionError):
        quota = _read_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period:
            cpu_quota = quota / period
    gpus = None
    try:
        import torch

        gpus = [{"device": index, "name": torch.cuda.get_device_properties(index).name,
                 "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                for index in range(torch.cuda.device_count())]
    except Exception as exc:
        logger.warning("[DraftTechniques] GPU properties unavailable: %s", exc)
    return {
        "remaining_seconds": int(remaining),
        "outer_deadline_remaining_seconds": outer_remaining,
        "single_execution_limit_seconds": min(agent.cfg.exec.timeout, int(remaining)),
        "steps_remaining": agent.acfg.steps - agent.current_step,
        "parallel_experiments": parallel,
        "configured_cpu_cores": configured_cpus,
        "cpu_affinity_count": len(affinity) if affinity is not None else None,
        "container_cpu_quota_cores": cpu_quota,
        "configured_cpu_slots_per_experiment": max(1, configured_cpus // parallel),
        "container_memory_limit_bytes": memory_limit,
        "observed_available_memory_bytes": min(memory.available, max(0, memory_limit - (memory_used or 0)))
            if memory_limit is not None else memory.available,
        "visible_gpus": gpus,
        "resource_sharing": "Concurrent experiments share visible GPUs and container RAM; "
                            "GPU memory is not reserved per experiment. CPU affinity does not reserve CPU time.",
        "pretrained_model_directory": str(agent.cfg.pretrain_model_dir or ""),
        "unknown_values": "null means unavailable, not zero; model weights and runtimes are not verified.",
    }


def prepare_draft_techniques(agent):
    cfg = agent.cfg
    if not re.fullmatch(r"[\w-][\w.-]*", str(cfg.exp_id)):
        raise ValueError("Draft techniques require a valid task exp_id")
    root = Path(cfg.draft_techniques.cache_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    corpus = root / cfg.exp_id
    logger.info("[DraftTechniques] loading prepared corpus from %s", corpus)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task_name") != cfg.exp_id or manifest.get("status") not in {"complete", "partial"}:
        raise ValueError("Draft technique corpus has a mismatched task or unpublished status")
    if manifest["status"] == "partial":
        logger.warning("[DraftTechniques] using published partial corpus; see its manifest for failed papers")
    raw = (corpus / "techniques.json").read_bytes()
    techniques = json.loads(raw)
    runtime = runtime_context(agent)
    if runtime["remaining_seconds"] <= 0:
        raise RuntimeError("No time remains for draft technique ranking")
    selected, audit = rerank_techniques(techniques, agent.task_desc, agent.data_preview, runtime, cfg)
    rendered = render_techniques(selected)
    audit.update({"task_name": cfg.exp_id, "corpus_path": str(corpus),
                  "corpus_status": manifest["status"], "corpus_sha256": hashlib.sha256(raw).hexdigest()})
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "draft_technique_rerank.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (log_dir / "draft_techniques.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (log_dir / "draft_techniques.md").write_text(rendered, encoding="utf-8")
    logger.info("[DraftTechniques] ready: selected=%d/%d; snapshot=%s; every draft reuses this ranking",
                len(selected), len(techniques), log_dir / "draft_technique_rerank.json")
    return rendered
