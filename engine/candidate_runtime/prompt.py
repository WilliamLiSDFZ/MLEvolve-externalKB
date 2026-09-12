"""One protocol description shared by every coder and the code reviewer."""


def instructions():
    return [
        "CANDIDATE RUNTIME PROTOCOL (required for this run): use the installed engine.candidate_runtime.CandidateSession. "
        "Validation/exports occur within this candidate, not in extra search nodes. The runtime owns budgets and the Jigsaw continuous-AUC metric.",
        "Create session = CandidateSession.from_env() early. Read input/train.csv and input/test.csv without changing original row order; "
        "then train_df, valid_df, test_df = session.split(train_df, test_df). Do this BEFORE fitting any transforms. "
        "Use the complete returned training partition; never train on valid_df. Do not make another validation split. "
        "A budget-limited partial epoch is allowed: log actual completed optimizer updates, never claim a full epoch.",
        "Use session.remaining() for CURRENT candidate seconds remaining and session.elapsed() for elapsed execution time. "
        "The candidate clock starts after acquiring its execution slot; the runtime already caps its deadline by the "
        "whole-run deadline. Preprocessing and model loading consume this same allowance. Do not derive a candidate "
        "deadline from candidate_results/run.json, run started_at, a parent candidate's timestamps, or configured stage "
        "budgets; do not cache an absolute deadline copied from a previous solution. If preprocessing needs a time check, "
        "query session.remaining() at that moment. During training, let session.step() decide when to stop and call "
        "session.finish(); do not subtract finalization reserves again or maintain a second training deadline.",
        "Bind FOUR callbacks using session.bind(predict_validation=..., predict_test=..., save_checkpoint=..., load_checkpoint=...). "
        "Each predict callback receives a NumPy array of POSITIONAL indices into valid_df or test_df and returns a 1D array "
        "of probabilities in precisely that order. Both callbacks use the same trained model, preprocessing and postprocessing. "
        "They must support arbitrary positional subsets for warmed timing samples as well as the full partition. "
        "Use inference mode, then restore the previous model train/eval mode before returning.",
        "save_checkpoint(directory) must save model weights, model configuration, tokenizer/feature transformations and inference state "
        "inside the supplied directory. load_checkpoint(directory) must restore them into the EXISTING model object "
        "(e.g. model.load_state_dict), preserving optimizer parameter references. No training, random reinitialization, or architecture "
        "replacement in these callbacks. The runtime reloads and compares validation predictions before publishing a submission.",
        "After binding callbacks call session.start_training(train_df['id'].astype(str).tolist()). "
        "After EACH real optimizer.step(), call stop = session.step(). If stop is True, exit ALL training loops immediately. "
        "The first few actual updates form the smoke test, including your backward/checkpointing path; do not use a smaller "
        "batch/sequence length solely to pass that test. For non-iterative estimators call step after actual fit, "
        "but internal callbacks/time limits are needed to stop a long fit cooperatively.",
        "Always call result = session.finish() at normal loop exit, including after step() returned True. "
        "finish() returns a DICT: result['best_validation_score'] is the full-validation score of the saved best checkpoint; "
        "result['submission_path'] and result['validation_path'] point to its complete prediction CSVs; "
        "result['checkpoint_id'], result['snapshot_id'], result['selected_optimizer_steps'], result['optimizer_steps'] "
        "and result['reason'] provide provenance. Repeated finish() calls return the same result without extra inference. "
        "Before finish, session.best_validation_score (also session.best_score) is None until formal validation, then a float. "
        "Use this documented API; do not probe guessed attributes or expect finish() to return a scalar. "
        "Do not retrain after finish. The session periodically validates, stores the best checkpoint, exports the first complete "
        "test submission early, then refreshes exports less frequently. It reserves validation/inference time from the same "
        "execution budget. Do not wait for an epoch boundary to call step; do not catch and suppress runtime/protocol errors.",
        "The session owns submission export and prints Final Validation Score itself. Do not run extra inference, recompute "
        "a score, or require another score attribute after finish(). Do not separately overwrite the submission or use another "
        "metric for search ranking. Use result['submission_path'] for any downstream file access; do not assert that a "
        "legacy ./submission/submission.csv or submission_<node_id>.csv exists after a successful runtime export. "
        "Runtime event logs replace the old epoch-only logging requirement. All weights and transforms "
        "must be reproducible from the saved checkpoint; small diagnostic predictions alone are never a scoreable submission.",
    ]
