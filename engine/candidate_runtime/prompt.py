"""One protocol description shared by every coder and the code reviewer."""


def instructions():
    return [
        "CANDIDATE RUNTIME PROTOCOL (required for this run): use the installed engine.candidate_runtime.CandidateSession. "
        "Validation/exports occur within this candidate, not in extra search nodes. The runtime owns budgets and the Jigsaw continuous-AUC metric.",
        "Create session = CandidateSession.from_env() early. Read input/train.csv and input/test.csv without changing original row order; "
        "then train_df, valid_df, test_df = session.split(train_df, test_df). Do this BEFORE fitting any transforms. "
        "Use the complete returned training partition; never train on valid_df. Do not make another validation split. "
        "A budget-limited partial epoch is allowed: log actual completed optimizer updates, never claim a full epoch.",
        "Bind FOUR callbacks using session.bind(predict_validation=..., predict_test=..., save_checkpoint=..., load_checkpoint=...). "
        "Each predict callback receives a NumPy array of POSITIONAL indices into valid_df or test_df and returns a 1D array "
        "of probabilities in precisely that order. Both callbacks use the same trained model, preprocessing and postprocessing. "
        "They must support both a small smoke slice and the full partition. Preserve and restore model train/eval mode.",
        "save_checkpoint(directory) must save model weights, model configuration, tokenizer/feature transformations and inference state "
        "inside the supplied directory. load_checkpoint(directory) must restore them into the EXISTING model object "
        "(e.g. model.load_state_dict), preserving optimizer parameter references. No training, random reinitialization, or architecture "
        "replacement in these callbacks. The runtime reloads and compares validation predictions before publishing a submission.",
        "After binding callbacks call session.start_training(train_df['id'].astype(str).tolist()). "
        "After EACH real optimizer.step(), call stop = session.step(). If stop is True, exit ALL training loops immediately. "
        "The first few actual updates form the smoke test, including your backward/checkpointing path; do not use a smaller "
        "batch/sequence length solely to pass that test. For non-iterative estimators call step after actual fit, "
        "but internal callbacks/time limits are needed to stop a long fit cooperatively.",
        "Always call session.finish() at normal loop exit. It is idempotent after a budget stop. "
        "Do not retrain after finish. The session periodically validates, stores the best checkpoint, exports the first complete "
        "test submission early, then refreshes exports less frequently. It reserves validation/inference time from the same "
        "execution budget. Do not wait for an epoch boundary to call step; do not catch and suppress runtime/protocol errors.",
        "The session owns submission export and Final Validation Score. Do not separately overwrite the submission or use another "
        "metric for search ranking. Runtime event logs replace the old epoch-only logging requirement. All weights and transforms "
        "must be reproducible from the saved checkpoint; small diagnostic predictions alone are never a scoreable submission.",
    ]
