"""Implementation guideline."""

import time

import humanize


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    tot_time_remaining = max(0, min(tot_time_remaining, getattr(agent, "runtime_deadline", float("inf")) - time.time()))
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    runtime = getattr(agent.cfg, "candidate_runtime", None)
    if getattr(runtime, "enabled", False):
        largest_budget = max(runtime.draft_budget_seconds or agent.cfg.exec.timeout,
                             runtime.candidate_budget_seconds or agent.cfg.exec.timeout)
        exec_timeout = int(min(exec_timeout, largest_budget))
    result = get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
    )
    if getattr(getattr(agent.cfg, "candidate_runtime", None), "enabled", False):
        from engine.candidate_runtime.prompt import instructions
        # Replace conflicting legacy output/logging rules; task semantics remain intact.
        result["Implementation guideline"] = [line for line in result["Implementation guideline"]
            if not any(text in line for text in ("MUST print:", "• Path:", "Print only 1 line per epoch",
                                                 "metric as the last line", "COMPLETE training dataset"))]
        result["Implementation guideline"].extend(instructions())
        result["Implementation guideline"].append(
            f"Stage budgets (including validation/export): draft/fusion_draft <= "
            f"{humanize.naturaldelta(min(runtime.draft_budget_seconds or agent.cfg.exec.timeout, exec_timeout))}; "
            f"debug/improve/evolution/fusion <= "
            f"{humanize.naturaldelta(min(runtime.candidate_budget_seconds or agent.cfg.exec.timeout, exec_timeout))}. "
            "The runtime's remaining() is authoritative after queueing and model loading."
        )
    return result


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
) -> dict:
    """Build implementation guideline from time and config."""
    impl_guideline = [
        f"**Resource Budget**: Time left ≈ {_format_time(tot_time_remaining)} | Steps left = {steps_remaining} | Max execution time per run = {humanize.naturaldelta(exec_timeout)}",
        "",
        f"**Note:** Training, validation and test inference together must fit within the stated {humanize.naturaldelta(exec_timeout)} execution budget.",
        "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "",
        "**1. Model Inference for ALL Predictions**",
        "• EVERY prediction (validation & test) MUST come from trained model's forward pass",
        "• Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions",
        "• ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "• ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "• Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        "**2. Generate submission.csv**",
        "• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
        "• Content: Model predictions on ALL test samples",
        "• Format: Follow task description exactly",
        "",
        "**3. Print Validation Metric**",
        "• MUST print: `print(f'Final Validation Score: {score}')`",
        "• Score MUST be computed on hold-out validation set using proper metric formula",
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "**4. Preserve Task Semantics in Local Validation**",
        "• Your local validation must evaluate the SAME core task as the final submission, not an easier proxy sub-problem.",
        "• ❌ FORBIDDEN: Silently redefining the task into a simpler objective (for example: multi-target -> single-target, detection -> presence-only classification, ranking -> plain classification, structured prediction -> one-field prediction).",
        "• If the submission predicts structured outputs, your validation metric must cover the key predicted structure rather than only a weak sub-component.",
        "• The validation target, prediction format, and post-processing logic must stay semantically aligned with the required submission format.",
        "",
        "**5. Make Validation Auditable**",
        "• In code and logs, make the validation setup easy to audit: split method, metric formula, predicted target, and any threshold/post-processing used.",
        "• The reported `Final Validation Score` must be computed with the official metric definition, or a task-faithful local implementation of that same metric.",
        "• ❌ FORBIDDEN: Using a proxy metric as the main validation score for model comparison, search ranking, or best-solution selection.",
        "• Do not report a validation score from a metric that ignores critical task dimensions required by the leaderboard.",
        "",
        "📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`",
        "",
        f"📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development."
        + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else ""),
        "",
        "⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`",
        "• AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`",
        "",
        "🚫 **Execution Guidelines**:",
        "• Each candidate sees at most one allocated CUDA GPU. Use cuda:0 (or CPU when CUDA is unavailable); preserve CUDA_VISIBLE_DEVICES and do not select host GPU IDs or launch multi-GPU training.",
        "• NO tqdm (not installed), NO verbose=1",
        "• Print only 1 line per epoch (minimize logging)",
        "• Use DataLoader with num_workers>=2 for speed",
        "",
        "⚠️  **Self-Check Before Finalizing**:",
        "□ Did predictions pass through model's learned weights during inference? (If NO → INVALID)",
        "□ Did I generate submission.csv in correct path with ALL test predictions?",
        "□ Did I print validation metric as the last line?",
        "□ Did I use the COMPLETE training dataset (not a tiny subset)?",
        "□ Did my local validation preserve the original task semantics instead of a simpler proxy?",
        "□ Is my reported `Final Validation Score` computed with the official metric definition rather than a proxy metric?",
    ]
    if expose_prediction:
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

    if k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}
