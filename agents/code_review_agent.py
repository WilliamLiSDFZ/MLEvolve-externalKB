"""Code Review Agent: LLM-based code review and fix for node code."""

import logging
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import get_internet_clarification
from agents.prompts.impl_guideline import mac_replay_guidance

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                # An approved review may explicitly return null instead of a diff.
                "type": ["string", "null"],
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")
    material = getattr(agent, "replay_material", None)

    def finish(code):
        if material is not None:
            material.finish_review(code)
        return code

    prompt = get_code_review_prompt(
        task_desc=agent.task_desc,
        code=node.code,
    )
    mac_replay = getattr(agent, "replay_profile", None) == "mac-mps"
    internet_clarification = ([] if mac_replay else
        get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", "")))
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    if getattr(getattr(agent.cfg, "candidate_runtime", None), "enabled", False):
        from engine.candidate_runtime.prompt import instructions
        prompt["Instructions"]["Code review guidelines"] = [line for line in prompt["Instructions"]["Code review guidelines"]
            if "Execution time:" not in line and "Submission File Location" not in line]
        prompt["Instructions"]["Required candidate runtime protocol"] = instructions()
        prompt["Instructions"]["Runtime review"] = (
            "Check fixed split before preprocessing, no held-out rows in training, real backward/optimizer updates before "
            "session.step(), breaking all loops when requested, complete save/load callbacks and session.finish(). "
            "Repair missing runtime integration. Replace custom candidate deadlines derived from run.json, run started_at, "
            "parent timestamps or configured budgets with session.remaining(); the runtime already accounts for queueing "
            "and the whole-run deadline. Remove duplicate finalization-reserve subtraction and post-finish assertions "
            "on legacy submission paths; consume result['submission_path'] instead. "
            "Each predict callback must preserve row order and restore model mode."
        )
    if mac_replay:
        prompt["Instructions"]["Code review guidelines"] = [line for line in prompt["Instructions"]["Code review guidelines"]
            if not any(text in line for text in ("LATEST versions", "pip install", "ALL models",
                                                 "Unknown Models are Valid", "NO Compatibility Speculation",
                                                 "Execution time:"))]
        prompt["Instructions"]["Mac replay review"] = mac_replay_guidance(agent) + [
            "Treat incompatible MPS operations, CPU training fallback, float64 tensors on MPS and "
            "multiprocessing DataLoader workers as execution bugs. Repair them without changing the "
            "model, training objective, fixed split or seed. Never add packages during the run."
        ]
    elif "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification

    review_spec = CODE_REVIEW_SPEC
    if material is not None:
        prompt["Instructions"]["Replay material handoff"] = material.section("review")
        review_spec = FunctionSpec(name=CODE_REVIEW_SPEC.name, description=CODE_REVIEW_SPEC.description,
                                   json_schema=material.review_schema(CODE_REVIEW_SPEC.json_schema))

    use_diff_for_review = agent.acfg.use_diff_mode
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Code review retry attempt {attempt + 1}/{max_retries} for node {node.id}")
                time.sleep(5)

            kwargs = dict(system_message=prompt, user_message=None, func_spec=review_spec,
                          model=agent.acfg.code.model, role="code",
                          temperature=agent.acfg.code.temp, cfg=agent.cfg)
            review_response = cast(dict, material.call("review", query, **kwargs) if material is not None
                                   else query(**kwargs))
            if material is not None:
                if (not isinstance(review_response, dict) or
                        not isinstance(review_response.get("needs_revision"), bool) or
                        not isinstance(review_response.get("reasoning"), str) or
                        not review_response["reasoning"].strip() or
                        (review_response.get("revised_code") is not None and
                         not isinstance(review_response["revised_code"], str))):
                    raise ValueError("Invalid material protocol code-review response")
                material.record_review(review_response, node.code)

            needs_revision = review_response.get("needs_revision", False)
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            logger.info(f"Code review for node {node.id}: needs_revision={needs_revision}")
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})

            if needs_revision:
                if revised_code and revised_code.strip():
                    if use_diff_for_review and (
                        "<<<<<<< SEARCH" in revised_code or "< SEARCH" in revised_code
                        ):
                        try:
                            logger.info("Code review returned diff format, applying patch")
                            patcher = SearchReplacePatcher()
                            patched_code, count = patcher.apply_patch(
                                revised_code, node.code, strict=False
                            )
                            if material is not None:
                                search = "<<<<<<< SEARCH" if "<<<<<<< SEARCH" in revised_code else "< SEARCH"
                                end = ">>>>>>> REPLACE" if search == "<<<<<<< SEARCH" else "> REPLACE"
                                if count != revised_code.count(search) or count != revised_code.count(end):
                                    raise RuntimeError("Material protocol review diff did not apply completely")
                            if count > 0 and patched_code and patched_code != node.code:
                                logger.info(f"Successfully applied {count} review patch(es)")
                                return finish(patched_code.strip())
                            if material is not None:
                                raise RuntimeError("Material protocol review diff made no code change")
                            logger.warning(
                                f"Diff patch failed (count={count}), keeping original code to avoid writing raw diff to runfile"
                            )
                            return finish(node.code)
                        except Exception as e:
                            if material is not None or getattr(e, "transport_retry_exhausted", False):
                                raise
                            logger.warning(
                                f"Failed to apply diff patch in code review: {e}, keeping original code to avoid writing raw diff to runfile"
                            )
                            return finish(node.code)
                    else:
                        # Full code revision (original behavior)
                        if use_diff_for_review:
                            if material is not None:
                                raise RuntimeError("Material protocol review requires SEARCH/REPLACE fixes")
                            return finish(node.code)
                        else:
                            logger.info("Using revised code from reviewer")
                            return finish(revised_code.strip())

                if attempt < max_retries - 1:
                    logger.warning(f"Code review violation: needs_revision=True but revised_code is empty/None - Will retry ({attempt + 1}/{max_retries})")
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error(f"Code review violation: needs_revision=True but revised_code is empty/None - Max retries reached, returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                if material is not None:
                    raise RuntimeError("Material protocol review required fixes but supplied no diff")
                return finish(node.code)

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            return finish(node.code)

        except Exception as e:
            if getattr(e, "transport_retry_exhausted", False):
                raise
            error_msg = f"Code review failed with exception: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"{error_msg} - Will retry (attempt {attempt + 1}/{max_retries})")
                continue
            if material is not None:
                raise RuntimeError(f"Material protocol review failed after {max_retries} attempts: {e}") from e
            logger.error(f"{error_msg} - Max retries reached, returning original code")
            return finish(node.code)

    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    if material is not None:
        raise RuntimeError("Material protocol review exited without a validated response")
    return finish(node.code)
