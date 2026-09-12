"""Overlap serial initial-draft generation with candidate execution.

Only the interpreter runs during initial generation. Parsing, grading, global-memory
writes and search-tree updates wait until all initial prompts have been generated.
"""

import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict

logger = logging.getLogger("MLEvolve")


def run_search_pipeline(agent, interpreter, cfg, exec_callback, save_callback):
    agent.executor = interpreter  # resource observation only; execution scheduling is unchanged
    total_steps = int(cfg.agent.steps)
    draft_count = min(int(cfg.agent.initial_drafts), total_steps)
    search_workers = int(cfg.agent.search.parallel_search_num)
    if total_steps < 0 or draft_count < 0 or search_workers < 1:
        raise ValueError("steps/initial_drafts must be nonnegative and parallel_search_num positive")

    logger.info("Pipeline: %s search workers, %s execution slots; initial drafts stay serial",
                search_workers, interpreter.max_parallel_run)
    execution_pool = ThreadPoolExecutor(max_workers=interpreter.max_parallel_run,
                                        thread_name_prefix="initial-execution")
    search_pool = ThreadPoolExecutor(max_workers=search_workers, thread_name_prefix="search")
    interrupted = False

    def within_runtime_budget():
        return (not getattr(getattr(cfg, "candidate_runtime", None), "enabled", False)
                or time.time() < interpreter.run_deadline)

    def execute_initial(node):
        # No AgentSearch/node mutation here: later drafts must still see pending designs.
        result = exec_callback(node.code, node.id, True)
        try:
            output_dir = cfg.log_dir / "executions"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{node.id}.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Could not persist raw execution result for %s", node.id)
        logger.info("Initial draft %s finished execution; result awaits initial-generation barrier", node.id)
        return result

    def finish_initial(node, result_future):
        # Reuse the existing parse/validate/register path, without executing code twice.
        return agent.execute_deferred_node(node, lambda *args, **kwargs: result_future.result())

    def step_task(node=None):
        return agent.step(exec_callback=exec_callback, node=node)

    try:
        pending = []
        for draft_idx in range(draft_count):
            if not within_runtime_budget():
                break
            try:
                logger.info("Generating initial draft %s/%s", draft_idx + 1, draft_count)
                node = agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
                if not getattr(node, "pending_execution", False):
                    logger.warning("Initial draft %s produced no executable candidate", draft_idx + 1)
                    continue
                pending.append((node, execution_pool.submit(execute_initial, node)))
                logger.info("Initial draft %s queued for execution immediately", node.id)
            except Exception as exc:
                if getattr(exc, "transport_retry_exhausted", False):
                    # The LLM adapter already classified/retried this failure.
                    # Another draft would repeat it while retaining a GPU lease.
                    raise
                logger.exception("Initial draft %s generation failed", draft_idx + 1)

        logger.info("Initial generation complete; releasing %s results to the search", len(pending))
        futures = {search_pool.submit(finish_initial, node, result) for node, result in pending}
        while futures or len(agent.journal) - 1 < total_steps:
            if not futures and not within_runtime_budget():
                break
            # Journal may already contain nodes from finished, not-yet-consumed futures.
            # Counting those twice here is conservative and prevents budget oversubscription.
            while (len(futures) < search_workers
                   and within_runtime_budget()
                   and len(agent.journal) - 1 + len(futures) < total_steps):
                futures.add(search_pool.submit(step_task))
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)
            for future in done:
                futures.remove(future)
                try:
                    node = future.result()
                except Exception as exc:
                    if getattr(exc, "transport_retry_exhausted", False):
                        # Failed steps do not increment the journal: swallowing a
                        # terminal transport error would reschedule indefinitely.
                        raise
                    logger.exception("Search task failed")
                    node = None
                save_callback()
                completed = len(agent.journal) - 1
                if within_runtime_budget() and completed + len(futures) < total_steps and len(futures) < search_workers:
                    futures.add(search_pool.submit(step_task, node))
                logger.info("Progress: %s/%s completed, %s search tasks outstanding",
                            completed, total_steps, len(futures))
    except BaseException:
        interrupted = True
        # Stop slot waiters before cancelling queues, including interruptions during draft 1–3.
        interpreter.terminate_all_subprocesses()
        execution_pool.shutdown(wait=False, cancel_futures=True)
        search_pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            execution_pool.shutdown(wait=True)
            search_pool.shutdown(wait=True)
