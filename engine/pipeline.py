"""Overlap serial initial-draft generation with execution, deferring result feedback."""

import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict


logger = logging.getLogger("MLEvolve")


def run_search_pipeline(agent, interpreter, cfg, exec_callback, save_callback):
    total_steps = int(cfg.agent.steps)
    draft_count = min(int(cfg.agent.initial_drafts), total_steps)
    max_workers = interpreter.max_parallel_run
    if total_steps < 0 or draft_count < 0 or max_workers < 1:
        raise ValueError("steps/initial_drafts must be nonnegative and execution capacity positive")

    deadline = min(
        agent.start_time + cfg.agent.time_limit,
        float(os.environ.get("MLEVOLVE_RUN_DEADLINE", "inf")),
    )
    logger.info("Pipeline: %s execution slots; initial drafts stay serial and execute as ready",
                max_workers)
    execution_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="initial-execution")
    search_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search")
    interrupted = False

    def execute_initial(node):
        # Keep scores and memory out of later initial-draft prompts.
        logger.info("Starting initial draft execution: %s", node.id)
        result = exec_callback(node.code, node.id, True)
        try:
            output_dir = cfg.log_dir / "executions"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{node.id}.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Could not persist raw execution result for %s", node.id)
        logger.info("Initial draft %s finished execution (exception=%s); result awaits initial-generation barrier",
                    node.id, result.exc_type)
        return result

    def finish_initial(node, result_future):
        # Reuse parsing/validation without executing the candidate a second time.
        return agent.execute_deferred_node(node, lambda *args, **kwargs: result_future.result())

    def step_task(node=None):
        return agent.step(exec_callback=exec_callback, node=node)

    try:
        pending = []
        for draft_idx in range(draft_count):
            if time.time() >= deadline:
                break
            try:
                logger.info("Generating initial draft %s/%s", draft_idx + 1, draft_count)
                node = agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
                if not getattr(node, "pending_execution", False):
                    logger.warning("Initial draft %s produced no executable candidate", draft_idx + 1)
                    continue
                if time.time() >= deadline:
                    break
                pending.append((node, execution_pool.submit(execute_initial, node)))
                logger.info("Initial draft %s queued for execution immediately", node.id)
            except Exception:
                logger.exception("Initial draft %s generation failed", draft_idx + 1)

        logger.info("Initial generation complete; releasing %s results to the search", len(pending))
        futures = {search_pool.submit(finish_initial, node, result) for node, result in pending}
        while futures or len(agent.journal) - 1 < total_steps:
            # Completed-but-unconsumed futures may already be in the journal; count conservatively.
            while (len(futures) < max_workers and time.time() < deadline
                   and len(agent.journal) - 1 + len(futures) < total_steps):
                futures.add(search_pool.submit(step_task))
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)
            for future in done:
                futures.remove(future)
                try:
                    node = future.result()
                    if node:
                        logger.info("Task completed: node_id=%s, is_buggy=%s, metric=%s",
                                    node.id, node.is_buggy, node.metric.value if node.metric else None)
                    else:
                        logger.warning("Task returned no node")
                except Exception:
                    logger.exception("Search task failed")
                    node = None
                save_callback()
                completed = len(agent.journal) - 1
                if (time.time() < deadline and completed + len(futures) < total_steps
                        and len(futures) < max_workers):
                    futures.add(search_pool.submit(step_task, node))
                logger.info("Progress: %s/%s completed, %s search tasks outstanding",
                            completed, total_steps, len(futures))
        if time.time() >= deadline:
            logger.info("Runtime budget exhausted; no further search tasks submitted")
    except BaseException:
        interrupted = True
        logger.info("Stopping pipeline, including executions started during initial generation")
        # Stop new subprocess launches before cancelling queued work in either pool.
        interpreter.terminate_all_subprocesses()
        execution_pool.shutdown(wait=False, cancel_futures=True)
        search_pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            execution_pool.shutdown(wait=True)
            search_pool.shutdown(wait=True)
