"""Manual retrieval orchestration; training never calls this module."""
import csv
import fcntl
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from threading import Lock, current_thread, local

from .jev_reranker import JEV_CRITERIA, JEV_PROMPT, JevReranker, split_paper
from .technique_agent import SYSTEM_PROMPT, CorpusTechniqueAgent
from .technique_deduplicator import SYSTEM_PROMPT as DEDUPLICATION_PROMPT, deduplicate_techniques

logger = logging.getLogger("MLEvolve")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "cache" / "techinique"


def describe_data(data_dir):
    """Bounded public-data metadata, without loading whole datasets into memory."""
    root = Path(data_dir)
    lines = ["Available files (bounded preview; omitted details are unknown):"]
    count = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            path = Path(directory) / name
            if name.startswith(".") or not path.is_file():
                continue
            # Do not follow dataset symlinks outside the explicitly supplied public directory.
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            lines.append(f"- {path.relative_to(root)} ({path.stat().st_size} bytes)")
            if path.suffix.lower() in {".csv", ".tsv"}:
                try:
                    with path.open(encoding="utf-8", errors="replace") as handle:
                        sample = handle.read(65536)
                    rows = csv.reader(io.StringIO(sample),
                                      delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
                    for i, row in enumerate(rows):
                        if i == 4:
                            break
                        label = "columns" if i == 0 else "sample row"
                        values = [cell[:160] for cell in row[:40]]
                        lines.append(f"  {label} (limited): {json.dumps(values, ensure_ascii=False)}")
                except (OSError, csv.Error) as exc:
                    lines.append(f"  preview unavailable: {type(exc).__name__}")
            count += 1
            if sum(len(line) + 1 for line in lines) > 12000:
                return "\n".join(lines)[:12000] + "\nData preview truncated; remaining details are unknown."
            if count == 40:
                lines.append("File preview limited to 40 files; remaining files were not inspected.")
                return "\n".join(lines)
    return "\n".join(lines)


def search_papers(task_description, cfg):
    """Reuse the existing centered hybrid retriever, without its extraction/filter pipeline."""
    from engine.coldstart.ondemand import _build_query, _load_abstract_index

    index_dir = Path(cfg.abstract_index_path)
    retriever = _load_abstract_index(index_dir, cfg)
    query = _build_query(task_description, cfg)
    hits = retriever.search(query, top_k=int(cfg.lazy_pool), alpha=float(cfg.retr_alpha))
    best = (hits[0][1] or 1.0) if hits else 1.0
    candidates = [(paper, score) for paper, score in hits
                  if score / best >= float(cfg.lazy_min_score)]
    embedding = retriever.embedding_model
    base = getattr(embedding, "base", embedding)
    tokenizer = base.model.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Corpus chunking requires the index model's fast tokenizer with offsets")
    return candidates, tokenizer, query


@contextmanager
def _stage(key, stage):
    worker = current_thread().name
    started = time.monotonic()
    logger.info("[Corpus] paper=%s worker=%s stage=%s starting", key, worker, stage)
    try:
        yield
    except BaseException:
        logger.warning("[Corpus] paper=%s worker=%s stage=%s failed after %.1fs",
                       key, worker, stage, time.monotonic() - started)
        raise
    else:
        logger.info("[Corpus] paper=%s worker=%s stage=%s done in %.1fs",
                    key, worker, stage, time.monotonic() - started)


def _parse_pdf(path):
    # PyMuPDF runs only inside spawned processes, never the paper threads.
    import pymupdf4llm
    return pymupdf4llm.to_markdown(path)


def load_full_text(paper, parser_pool):
    from engine.coldstart.ondemand import _download, _resolve_pdf

    key = paper_key(paper)
    with tempfile.TemporaryDirectory(prefix="mlevolve-paper-") as directory:
        path = Path(directory) / "paper.pdf"
        with _stage(key, "download"):
            url = _resolve_pdf(paper)
            if not url:
                raise ValueError("Paper has no resolvable PDF URL")
            if not _download(url, path):
                raise OSError("Could not download paper PDF")
        with _stage(key, "parse"):
            text = parser_pool.submit(_parse_pdf, str(path)).result()
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Paper PDF yielded no readable text")
    return text


def paper_key(paper):
    identity = str(getattr(paper, "id", "") or "")
    if not identity:
        identity = json.dumps([getattr(paper, key, "") or ""
                               for key in ("source", "pdf_url", "title", "abstract")])
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _render(techniques):
    lines = ["# Task-specific techniques", "",
             "Task adaptations are proposed experiments, not measured gains on this task.", ""]
    for number, technique in enumerate(techniques, 1):
        lines.extend([f"## {number}. {technique['name']}", "",
                      f"Paper: {technique['paper_title']}",
                      f"Source: {technique['source']}", ""])
        for key in ("paper_finding", "task_relevance", "task_adaptation", "proposed_experiment",
                    "requirements_and_limits", "paper_reported_result"):
            lines.extend([f"**{key.replace('_', ' ').capitalize()}**: "
                          f"{technique.get(key) or 'Not reported'}", ""])
        for evidence in technique["evidence"]:
            lines.extend([f"Evidence ({evidence['chunk_id']}):", f"> {evidence['quote']}", ""])
    return "\n".join(lines)


def _publish(staging, target):
    """Replace a whole task result so removed papers cannot survive a rerun."""
    previous = staging.parent / (staging.name + ".previous")
    if target.exists():
        target.rename(previous)
    try:
        staging.rename(target)
    except BaseException:
        if previous.exists():
            previous.rename(target)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def _screen_and_extract(record, chunks, paper_dir, task_description, data_description, cfg, api_key):
    settings = cfg.corpus_retriever
    jev = JevReranker(api_key, model=settings.jev_model, threshold=settings.jev_threshold)
    techniques = []
    key = record["paper_key"]
    try:
        logger.info("[Corpus] paper %s: screening %d chunks", key, len(chunks))
        with _stage(key, "jev"):
            selected, decisions = jev.filter(chunks, task_description, data_description)
        with _stage(key, "save_screening"):
            _write_json(paper_dir / "screening.json", {
                "chunks": [asdict(chunk) for chunk in chunks], "decisions": decisions,
                "selected_chunk_ids": [chunk.id for chunk in selected]})
        record["chunk_count"] = len(chunks)
        record["selected_chunk_ids"] = [chunk.id for chunk in selected]
        record["screening_complete"] = all(d["status"] == "ok" for d in decisions)
        if selected:
            logger.info("[Corpus] paper %s: extracting from %d selected chunks", key, len(selected))
            with _stage(key, "extract"):
                result = CorpusTechniqueAgent(cfg).extract(
                    chunks, selected, task_description, data_description)
            with _stage(key, "save_techniques"):
                _write_json(paper_dir / "techniques.json", result)
            record["status"] = result["status"]
            record["technique_count"] = len(result["techniques"])
            if result["status"] != "complete":
                logger.warning("[Corpus] paper %s: incomplete extraction: %s", key, result.get("errors", []))
            if result["status"] == "complete" and record["screening_complete"]:
                techniques = [{**technique, "technique_id": f"{key}:{index:04d}", "paper_key": key,
                    "paper_id": record["paper_id"], "paper_title": record["paper_title"],
                    "source": record["source"]} for index, technique in enumerate(result["techniques"])]
        else:
            record["status"] = "rejected"
        if not record["screening_complete"]:
            record["status"] = "incomplete"
    except (PermissionError, ImportError):
        raise
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        techniques = []
        logger.warning("[Corpus] paper %s failed: %s", key, record["error"])
    _write_json(paper_dir / "paper.json", record)
    return record, techniques


def run_retrieval(cfg, output_root=None):
    task_name = str(cfg.exp_id or "")
    if not re.fullmatch(r"[\w-][\w.-]*", task_name):
        raise ValueError("exp_id must be a task name, not a path")
    data_dir = Path(str(cfg.data_dir or ""))
    if not cfg.data_dir or not data_dir.is_dir():
        raise ValueError("data_dir must be the task's prepared public-data directory")
    if cfg.desc_file:
        task_description = Path(cfg.desc_file).read_text(encoding="utf-8")
    else:
        task_description = str(cfg.goal or "")
        if cfg.eval:
            task_description += "\nEvaluation: " + str(cfg.eval)
    if not task_description.strip():
        raise ValueError("Provide desc_file or a nonempty goal")
    index_dir = Path(str(cfg.abstract_index_path or ""))
    if not (index_dir / "manifest.json").is_file():
        raise ValueError("abstract_index_path must contain the existing abstract index")
    if not 1 <= int(cfg.lazy_pool) <= 100:
        raise ValueError("lazy_pool must be between 1 and 100")
    if not 0 <= float(cfg.lazy_min_score) <= 1 or not 0 <= float(cfg.retr_alpha) <= 1:
        raise ValueError("lazy_min_score and retr_alpha must be between 0 and 1")
    api_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not api_key:
        raise ValueError("Export TYPESAFE_API_KEY before running corpus retrieval")
    if not cfg.agent.code.api_key:
        raise ValueError("Configure agent.code credentials (normally LLM_API_KEY)")

    settings = cfg.corpus_retriever
    if not 0 <= settings.chunk_overlap < settings.chunk_tokens <= 10000:
        raise ValueError("Require 0 <= corpus_retriever.chunk_overlap < chunk_tokens <= 10000")
    root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    target = root / task_name
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("Task output must be a directory, not a file or symlink")

    with (root / f".{task_name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Retrieval for {task_name} is already running") from exc
        with tempfile.TemporaryDirectory(prefix=f".{task_name}.", dir=root) as directory:
            staging = Path(directory)
            data_description = describe_data(data_dir)
            context = {"task_name": task_name, "task_description": task_description,
                       "available_data": data_description}
            _write_json(staging / "task_context.json", context)
            _write_json(staging / "prompts.json", {"jev": JEV_PROMPT, "jev_criteria": JEV_CRITERIA,
                                                   "technique_agent": SYSTEM_PROMPT,
                                                   "technique_deduplication": DEDUPLICATION_PROMPT})
            candidates, tokenizer, query = search_papers(task_description, cfg)
            manifest = {
                "task_name": task_name, "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(target.resolve()), "query": query,
                "task_context_sha256": hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest(),
                "abstract_index": json.loads((index_dir / "manifest.json").read_text(encoding="utf-8")),
                "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                "token_count_note": "Chunk tokens use this local tokenizer, not Jev's private tokenizer.",
                "retrieval": {"pool": int(cfg.lazy_pool), "min_score": float(cfg.lazy_min_score),
                              "alpha": float(cfg.retr_alpha)},
                "jev_model": settings.jev_model, "jev_threshold": settings.jev_threshold,
                "chunk_tokens": settings.chunk_tokens, "chunk_overlap": settings.chunk_overlap,
                "technique_model": cfg.agent.code.model, "papers": [],
            }
            unique, seen = [], set()
            for paper, score in candidates:
                key = paper_key(paper)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((paper, score, key))
            results, pending = {}, {}
            fulltext_papers = {}
            hash_lock = Lock()
            worker_state = local()

            def init_worker():
                worker_state.tokenizer = deepcopy(tokenizer)

            def process_paper(paper, score, key):
                paper_dir = staging / "papers" / key
                paper_dir.mkdir(parents=True)
                record = {"paper_key": key, "paper_id": getattr(paper, "id", None),
                          "paper_title": getattr(paper, "title", ""),
                          "abstract": getattr(paper, "abstract", "") or "",
                          "source": getattr(paper, "source", "") or getattr(paper, "pdf_url", ""),
                          "retrieval_score": float(score), "status": "error"}
                logger.info("[Corpus] preparing paper %s worker=%s: %s",
                            key, current_thread().name, record["paper_title"])
                try:
                    text = load_full_text(paper, parser_pool)
                    record["fulltext_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                    with _stage(key, "chunk"):
                        chunks = split_paper(text, record["paper_id"], record["paper_title"],
                                             record["abstract"], worker_state.tokenizer,
                                             max_tokens=settings.chunk_tokens, overlap=settings.chunk_overlap)
                    with _stage(key, "save_fulltext"):
                        (paper_dir / "fulltext.md").write_text(text, encoding="utf-8")
                    # Claim only after preparation succeeds; never hold this lock during I/O.
                    with hash_lock:
                        duplicate_of = fulltext_papers.setdefault(record["fulltext_sha256"], key)
                    if duplicate_of != key:
                        (paper_dir / "fulltext.md").unlink()
                        record.update(status="duplicate", duplicate_of=duplicate_of)
                        _write_json(paper_dir / "paper.json", record)
                        logger.info("[Corpus] paper %s: duplicate full text of %s; skipping",
                                    key, duplicate_of)
                        return record, []
                except (PermissionError, ImportError, BrokenProcessPool):
                    raise
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    logger.warning("[Corpus] paper %s preparation failed: %s", key, record["error"])
                    _write_json(paper_dir / "paper.json", record)
                    return record, []
                return _screen_and_extract(record, chunks, paper_dir, task_description,
                                           data_description, cfg, api_key)

            def collect_ready(block=False):
                if not pending:
                    return
                done, _ = wait(pending, timeout=None if block else 0,
                               return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    record, techniques = future.result()
                    results[index] = (record, techniques)
                    logger.info("[Corpus] completed %d/%d: paper %s, status=%s, techniques=%d",
                                len(results), len(unique), record["paper_key"],
                                record["status"], len(techniques))

            logger.info("[Corpus] processing %d unique papers with up to 3 workers", len(unique))
            # Shut down paper threads before their parser pool and temporary files.
            with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as parser_pool, \
                 ThreadPoolExecutor(max_workers=3, thread_name_prefix="corpus",
                                    initializer=init_worker) as executor:
                try:
                    for index, (paper, score, key) in enumerate(unique):
                        collect_ready(block=len(pending) == 3)
                        pending[executor.submit(process_paper, paper, score, key)] = index
                    while pending:
                        collect_ready(block=True)
                except BaseException:
                    for future in pending:
                        future.cancel()
                    # Executor exit waits for running writers before staging is cleaned up.
                    raise

            all_techniques = []
            for index in range(len(unique)):
                record, techniques = results[index]
                manifest["papers"].append(record)
                all_techniques.extend(techniques)

            counts = Counter(paper["status"] for paper in manifest["papers"])
            manifest["counts"] = dict(counts)
            if manifest["papers"] and not (counts["complete"] or counts["rejected"]) and target.exists():
                raise RuntimeError("No paper completed successfully; previous task results were preserved")
            manifest["raw_technique_count"] = len(all_techniques)
            logger.info("[Corpus] deduplicating %d extracted techniques", len(all_techniques))
            try:
                all_techniques, deduplication = deduplicate_techniques(all_techniques, cfg)
            except (PermissionError, ImportError):
                raise
            except Exception as exc:
                deduplication = {"status": "incomplete", "input_count": len(all_techniques),
                                 "output_count": len(all_techniques),
                                 "error": f"{type(exc).__name__}: {exc}"}
                logger.warning("[Corpus] technique deduplication failed; keeping all %d techniques: %s",
                               len(all_techniques), deduplication["error"])
            _write_json(staging / "deduplication.json", deduplication)
            manifest["deduplication"] = {key: deduplication[key]
                                         for key in ("status", "input_count", "output_count")}
            manifest["technique_count"] = len(all_techniques)
            manifest["status"] = "partial" if (counts["error"] or counts["incomplete"] or
                                                deduplication["status"] != "complete") else "complete"
            logger.info("[Corpus] technique deduplication %s: %d -> %d",
                        deduplication["status"], deduplication["input_count"], len(all_techniques))
            _write_json(staging / "manifest.json", manifest)
            _write_json(staging / "techniques.json", all_techniques)
            (staging / "techniques.md").write_text(_render(all_techniques), encoding="utf-8")
            _publish(staging, target)
    return manifest
