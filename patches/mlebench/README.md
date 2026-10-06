# Jigsaw Unintended Bias: continuous-prediction AUC

`jubias-continuous-auc.patch` removes prediction thresholding from MLE-bench's
competition grader. Ground-truth toxicity and identity thresholds, ID alignment,
subgroup/BPSN/BNSP AUC, p=-5 aggregation, and component weights are unchanged.
All grading still goes through the original MLE-bench API.

The manifest pins upstream commit `507f92e1138bb6e40dac5c6ee7a6758e6424bf97` and
the original/patched file hashes. The cluster's `/workspace/mle-bench` checkout
was at this commit and its grader exactly matched the installed file. That
installation came from a local tree and has no VCS commit in package metadata;
do not infer an installation-wide commit solely from the one matching file.

Use the shared Linux environment from the dev pod after checking that no MLEvolve
training/grading process will switch metric definitions midway through a run:

```bash
/workspace/MLEvolve/.venv/bin/python /workspace/MLEvolve/utils/verify_mlebench_grading.py --candidate
/workspace/MLEvolve/.venv/bin/python /workspace/MLEvolve/utils/mlebench_patch.py --apply
/workspace/MLEvolve/.venv/bin/python /workspace/MLEvolve/utils/verify_mlebench_grading.py
/workspace/MLEvolve/.venv/bin/python /workspace/MLEvolve/utils/mlebench_patch.py --check
```

`--candidate` exercises a temporary patched module without modifying the installed
package. Tests include perfect ordering entirely below 0.5, monotone transforms,
ties, shuffled/mismatched IDs, preservation of ground-truth thresholding, and an
independent pairwise AUC oracle. Application is idempotent, rejects unknown source
hashes, keeps `grade.py.before-jubias-continuous-auc-v1`, and atomically replaces
only this grader. Restart any previously imported grading process after applying.
Restoring that backup rolls back the package edit; the new scoring entry points
then deliberately refuse to grade jubias until the corrected version is restored.

`k8s/setup-venv.sh` pins fresh installs and checks/applies the patch to existing
installs too. The three scoring utilities check provenance before grading, and
jubias Job startup checks that the fix remains installed. An upstream upgrade
requires a deliberate manifest/patch review rather than fuzzy patch application.

Write regraded results to a new file first, retaining the old scores:

```bash
python utils/grade_all.py --runs /workspace/MLEvolve/runs \
  -o /workspace/MLEvolve/runs/scores-continuous-auc-v1.csv
```

The result adds `metric_version`, `grader_sha256`, `mlebench_version`, and
`mlebench_commit` to the existing schema. An unknown install commit is empty; the
exact grader hash is always present. Other tasks retain their original graders.
Report jubias results as corrected MLE-bench scoring, not as results under the
legacy thresholded scorer. Agentic_Knowledge_Base's score loader rejects mixed
versions within a competition and rejects uncorrected/unversioned jubias rows.

The aggregate UPDATELOG is maintained in the sibling
`Agentic_Knowledge_Base/UPDATELOG.md`.
