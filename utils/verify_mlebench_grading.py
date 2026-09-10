"""CPU-only metric regressions. --candidate tests the patch without changing the install."""

import argparse
import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from mlebench.grade_helpers import InvalidSubmissionError

if __package__:
    from . import mlebench_patch as fix
else:
    import mlebench_patch as fix


def pairwise_auc(y, scores):
    """Independent small-data oracle: compare every positive with every negative."""
    positive, negative = scores[y], scores[~y]
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def metric_oracle(answers, prediction, identities):
    y = answers["toxicity"].to_numpy() >= 0.5
    bias = []
    for col in identities:
        group = answers[col].to_numpy() >= 0.5
        masks = (group, (group & ~y) | (~group & y), (group & y) | (~group & ~y))
        bias.append([pairwise_auc(y[m], prediction[m]) for m in masks])
    means = np.mean(np.asarray(bias) ** -5, axis=0) ** (-1 / 5)
    return float((pairwise_auc(y, prediction) + means.sum()) / 4)


class MetricTests(unittest.TestCase):
    grader = None

    def fixture(self):
        rng = np.random.default_rng(48)
        identities = self.grader["grade"].__defaults__[0]
        frame = pd.DataFrame({"id": np.arange(160),
                              "toxicity": np.tile([0.1, 0.9], 80)})
        for col in identities:
            frame[col] = rng.choice([0.0, 0.3, 0.8, np.nan], len(frame))
        prediction = 0.15 + 0.12 * (frame.toxicity.to_numpy() >= 0.5) + rng.uniform(-0.12, 0.12, len(frame))
        return frame, pd.DataFrame({"id": frame.id, "prediction": prediction}), identities

    def test_perfect_ranking_below_threshold(self):
        answers, submission, _ = self.fixture()
        submission["prediction"] = np.where(answers.toxicity >= 0.5, 0.4, 0.1)
        self.assertAlmostEqual(self.grader["grade"](submission, answers), 1.0)

    def test_monotone_transform_and_independent_oracle(self):
        answers, submission, identities = self.fixture()
        score = self.grader["grade"](submission, answers)
        self.assertAlmostEqual(score, metric_oracle(answers, submission.prediction.to_numpy(), identities), places=12)
        transformed = submission.copy()
        transformed["prediction"] = 0.6 + 0.3 * transformed.prediction
        self.assertAlmostEqual(score, self.grader["grade"](transformed, answers), places=12)

    def test_ties_and_id_alignment(self):
        answers, submission, _ = self.fixture()
        submission["prediction"] = 0.2
        self.assertAlmostEqual(self.grader["grade"](submission, answers), 0.5)
        _, submission, _ = self.fixture()
        expected = self.grader["grade"](submission, answers)
        self.assertAlmostEqual(self.grader["grade"](submission.iloc[::-1], answers.sample(frac=1, random_state=3)), expected)
        submission.loc[0, "id"] = -1
        with self.assertRaises(InvalidSubmissionError):
            self.grader["grade"](submission, answers)

    def test_only_ground_truth_is_thresholded(self):
        answers, submission, identities = self.fixture()
        before_answers, before_submission = answers.copy(), submission.copy()
        prepared, labels = self.grader["prepare_for_metric"](submission, answers, identities)
        np.testing.assert_array_equal(prepared.toxicity, before_submission.prediction)
        np.testing.assert_array_equal(labels.toxicity, before_answers.toxicity >= 0.5)
        for col in identities:
            np.testing.assert_array_equal(labels[col], before_answers[col].fillna(0) >= 0.5)
        pd.testing.assert_frame_equal(answers, before_answers)
        pd.testing.assert_frame_equal(submission, before_submission)

    def test_patch_is_idempotent_and_preserves_backup(self):
        # Recover exactly the pinned original from the hunk for an isolated lifecycle test.
        patched = fix.patched_source(fix.grader_path().read_bytes())
        lines = (fix.PATCH_DIR / "jubias-continuous-auc.patch").read_text().splitlines(True)
        body = lines[next(i for i, s in enumerate(lines) if s.startswith("@@")) + 1:]
        old = "".join(s[1:] for s in body if s.startswith((" ", "-"))).encode()
        new = "".join(s[1:] for s in body if s.startswith((" ", "+"))).encode()
        original = patched.replace(new, old, 1)
        self.assertEqual(fix.sha256(original), fix.MANIFEST["original_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grade.py"
            path.write_bytes(original)
            self.assertTrue(fix.apply_patch(path))
            self.assertFalse(fix.apply_patch(path))
            self.assertEqual(path.read_bytes(), patched)
            self.assertEqual(path.with_name(path.name + ".before-" + fix.MANIFEST["metric_version"]).read_bytes(), original)
            path.write_bytes(b"unrecognized source")
            with self.assertRaises(RuntimeError):
                fix.apply_patch(path)
            self.assertEqual(path.read_bytes(), b"unrecognized source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    if args.candidate:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grade.py"
            path.write_bytes(fix.patched_source(fix.grader_path().read_bytes()))
            MetricTests.grader = runpy.run_path(str(path))
    else:
        fix.grading_metadata(fix.COMPETITION)
        MetricTests.grader = runpy.run_path(str(fix.grader_path()))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
