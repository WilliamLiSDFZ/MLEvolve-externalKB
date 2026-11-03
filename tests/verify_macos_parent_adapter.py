"""Offline adapter checks; optional source fixture and tiny torch parity checks.

Set MLEVOLVE_MACOS_PARENT_SOURCE to the original saved solution.py when it is not
in the usual downloaded replay directory. No candidate/data/LLM is executed.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import typing
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.replay_macos.adapters import get_adapter, parent_36845c7c as adapter, parent_f327226c
from utils.replay_macos.adapters.common import replace


def original_source():
    default = Path.home() / (
        "nautilus/results/replay_jubias-noise-parent5-s67/A/source/candidates/"
        "36845c7c3f1c4beca8eebc7a37852e7c/solution.py")
    path = Path(os.environ.get("MLEVOLVE_MACOS_PARENT_SOURCE", default))
    if not path.is_file():
        raise unittest.SkipTest("Original parent fixture absent; set MLEVOLVE_MACOS_PARENT_SOURCE")
    return path.read_text()


def definitions(source, names):
    return ast.Module(body=[node for node in ast.parse(source).body
                           if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names],
                      type_ignores=[])


class AdapterGuardTests(unittest.TestCase):
    def test_unknown_parent_is_rejected(self):
        source = "print('unreviewed parent')\n"
        with self.assertRaisesRegex(ValueError, "Unsupported replay parent SHA256"):
            get_adapter(hashlib.sha256(source.encode()).hexdigest())
        for implementation in (adapter, parent_f327226c):
            with self.subTest(parent=implementation.__name__):
                with self.assertRaisesRegex(ValueError, "Unsupported replay parent SHA256"):
                    implementation.adapt(source)

    def test_registry_selects_the_exact_adapter(self):
        for implementation in (adapter, parent_f327226c):
            self.assertIs(get_adapter(implementation.ORIGINAL_SOURCE_SHA256), implementation)

    def test_replacement_requires_exact_occurrence_count(self):
        for source in ("missing", "target target"):
            with self.assertRaisesRegex(ValueError, "no longer matches its source template"):
                replace(source, "target", "replacement")
        self.assertEqual(replace("target target", "target", "replacement", count=2),
                         "replacement replacement")

    def test_mps_gate_rejects_cpu_and_fallback(self):
        torch = NS(backends=NS(mps=NS(is_available=lambda: False)))
        with self.assertRaisesRegex(RuntimeError, "available Apple MPS"):
            exec(adapter._MPS_SETUP, {"torch": torch, "os": os})
        torch.backends.mps.is_available = lambda: True
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(RuntimeError, "FALLBACK.*disabled"):
                exec(adapter._MPS_SETUP, {"torch": torch, "os": os})

    def test_converter_contains_no_torch_sparse_or_dense_materialization(self):
        tree = ast.parse(adapter._COMPONENT_CONVERTER)
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("torch.sparse_coo_tensor", calls)
        self.assertNotIn("feature_matrix.toarray", calls)
        self.assertEqual(calls.count("torch.as_tensor"), 3)


class SavedParentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = original_source()
        cls.adapted = adapter.adapt(cls.original)

    def test_exact_hash_and_determinism(self):
        self.assertEqual(hashlib.sha256(self.original.encode()).hexdigest(), adapter.ORIGINAL_SOURCE_SHA256)
        self.assertEqual(self.adapted, adapter.adapt(self.original))
        self.assertEqual(hashlib.sha256(self.adapted.encode()).hexdigest(),
                         "a886b017cbe3300004136b203219b7de2f09b2bbfe43dee0860b166b756d1120")
        self.assertEqual(get_adapter(adapter.ORIGINAL_SOURCE_SHA256).ADAPTER_VERSION,
                         "jigsaw-parent5-mps-v1")
        self.assertEqual(get_adapter(adapter.ORIGINAL_SOURCE_SHA256).adapt(self.original), self.adapted)
        for changed in (self.original + "\n", self.adapted):
            with self.assertRaisesRegex(ValueError, "Unsupported replay parent"):
                adapter.adapt(changed)

    def test_math_and_data_functions_remain_identical(self):
        names = {node.name for node in ast.parse(self.original).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        changed = {"DualViewSparseFairClassifier", "scipy_csr_to_torch_coo", "predict_partition",
                   "save_checkpoint", "load_checkpoint"}
        self.assertEqual(ast.dump(definitions(self.original, names - changed)),
                         ast.dump(definitions(self.adapted, names - changed)))
        self.assertIn("SEED = 20240517", self.adapted)
        self.assertIn("MODEL_BATCH_SIZE = 1024", self.adapted)
        self.assertIn("MODEL_EPOCHS = 2", self.adapted)
        split_calls = lambda source: [ast.dump(node) for node in ast.walk(ast.parse(source))
                                      if isinstance(node, ast.Call) and ast.unparse(node.func) == "session.split"]
        self.assertEqual(split_calls(self.original), split_calls(self.adapted))

    def test_training_and_prediction_share_mps_components(self):
        compile(self.adapted, "adapted_parent.py", "exec")
        self.assertEqual(self.adapted.count("scipy_csr_to_device_components(scipy_features, device)"), 2)
        for forbidden in ("torch.cuda", "amp_scaler", "scipy_csr_to_torch_coo", "prefetch_factor=",
                          "torch.sparse_coo", "to_sparse_coo()", "features.coalesce()"):
            self.assertNotIn(forbidden, self.adapted)
        self.assertIn("num_workers=0", self.adapted)
        self.assertIn('device = torch.device("mps")', self.adapted)
        self.assertIn('map_location="cpu"', self.adapted)


class TensorParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = original_source()
        cls.adapted = adapter.adapt(cls.original)
        try:
            import joblib
            import numpy as np
            from scipy import sparse
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"Optional tensor dependencies unavailable: {exc}")
        cls.torch, cls.np, cls.sparse, cls.joblib = torch, np, sparse, joblib

    def namespace(self, source):
        torch = self.torch
        namespace = dict(vars(typing), torch=torch, nn=torch.nn, F=torch.nn.functional,
                         np=self.np, sparse=self.sparse, math=math, Path=Path,
                         os=os, json=json, joblib=self.joblib)
        names = {"DualViewSparseFairClassifier", "BiasAwareAUCSurrogateLoss", "CounterfactualPairLoss",
                 "scipy_csr_to_torch_coo", "scipy_csr_to_device_components", "save_checkpoint", "load_checkpoint"}
        exec(compile(definitions(source, names), "parent_definitions.py", "exec"), namespace)
        return namespace

    def compare(self, target):
        torch, np = self.torch, self.np
        old, new = self.namespace(self.original), self.namespace(self.adapted)
        # Include duplicate entries, unsorted columns and an empty row.
        matrix = self.sparse.csr_matrix((np.array([.5, 1., .25, -.2, .8, .1], dtype=np.float32),
                                        np.array([4, 1, 4, 7, 0, 5]), np.array([0, 3, 3, 5, 6])),
                                       shape=(4, 8))
        original_matrix = matrix.copy()
        torch.manual_seed(7)
        cpu = old["DualViewSparseFairClassifier"](3, 3, 2, feature_dropout=0.0)
        with torch.no_grad():
            cpu.feature_weight.copy_(torch.linspace(-.3, .5, 8))
        migrated = new["DualViewSparseFairClassifier"](3, 3, 2, feature_dropout=0.0).to(target)
        migrated.load_state_dict(cpu.state_dict())
        original_batch = old["scipy_csr_to_torch_coo"](matrix)
        adapted_batch = new["scipy_csr_to_device_components"](matrix, target)
        self.assertEqual(adapted_batch[3], (4, 8))
        self.assertTrue(all(t.layout == torch.strided and t.device.type == target for t in adapted_batch[:3]))
        np.testing.assert_array_equal(matrix.data, original_matrix.data)
        predictions = cpu(original_batch)
        actual = migrated(adapted_batch)
        for key in predictions:
            torch.testing.assert_close(actual[key].detach().cpu(), predictions[key].detach(), rtol=2e-5, atol=2e-6)
        targets = torch.tensor([0.1, .8, 0., 1.], dtype=torch.float32)
        groups = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
        weights = torch.tensor([1., 2., .5, 1.])
        loss_fn = old["BiasAwareAUCSurrogateLoss"]()
        expected_loss = loss_fn(predictions, targets, subgroup_membership=groups, sample_weight=weights)
        actual_loss = loss_fn.to(target)(actual, targets.to(target), subgroup_membership=groups.to(target),
                                        sample_weight=weights.to(target))
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(actual_loss.detach().cpu(), expected_loss.detach(), rtol=2e-5, atol=2e-6)
        for expected, actual_parameter in zip(cpu.parameters(), migrated.parameters()):
            torch.testing.assert_close(actual_parameter.grad.cpu(), expected.grad, rtol=3e-5, atol=3e-6)
        original_optimizer = torch.optim.AdamW(cpu.parameters(), lr=.01)
        optimizer = torch.optim.AdamW(migrated.parameters(), lr=.01)
        original_optimizer.step()
        optimizer.step()
        for expected, actual_parameter in zip(cpu.parameters(), migrated.parameters()):
            torch.testing.assert_close(actual_parameter.detach().cpu(), expected.detach(), rtol=3e-5, atol=3e-6)

        # Exercise the actual adapted checkpoint callbacks, including optimizer state.
        state = {"sparse_feature_dimensions": {"raw_word": 3, "identity_neutral_char": 3, "dense": 2},
                 "text_column": "text", "identity_neutral_text_column": "neutral"}
        new.update(model=migrated, optimizer=optimizer,
                   lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
                   device=torch.device(target), EVALUATION_IDENTITY_COLUMNS=("male",),
                   dense_feature_names=("one", "two"), preprocessing_state=state,
                   dense_feature_scaler=None, raw_word_vectorizer=None, neutral_char_vectorizer=None,
                   IDENTITY_LEXICONS={}, INFERENCE_BATCH_SIZE=4)
        expected_after_update = migrated(adapted_batch)["logit"].detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            new["save_checkpoint"](directory)
            with torch.no_grad():
                migrated.feature_weight.zero_()
            new["load_checkpoint"](directory)
        torch.testing.assert_close(migrated(adapted_batch)["logit"].detach(), expected_after_update)
        for values in optimizer.state.values():
            self.assertEqual(values["step"].device.type, "cpu")
        original_optimizer.step()
        optimizer.step()
        for expected, actual_parameter in zip(cpu.parameters(), migrated.parameters()):
            torch.testing.assert_close(actual_parameter.detach().cpu(), expected.detach(), rtol=3e-5, atol=3e-6)

    def test_cpu_sparse_vs_component_predictions_gradients_and_checkpoint(self):
        self.compare("cpu")

    def test_mps_predictions_gradients_and_checkpoint(self):
        if not self.torch.backends.mps.is_available():
            self.skipTest("Apple MPS is unavailable in this process")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            self.skipTest("MPS fallback is enabled; native MPS parity requires it disabled")
        self.compare("mps")

    def test_mps_variable_shape_memory_remains_bounded(self):
        torch, np = self.torch, self.np
        if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            self.skipTest("Native Apple MPS is unavailable")
        namespace = self.namespace(self.adapted)
        torch.manual_seed(7)
        rng = np.random.default_rng(7)
        model = namespace["DualViewSparseFairClassifier"](32768, 32768, 10).to("mps")
        criterion = namespace["BiasAwareAUCSurrogateLoss"]().to("mps")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        torch.mps.synchronize()
        torch.mps.empty_cache()
        initial_memory = torch.mps.driver_allocated_memory()
        # Vary synthetic COO sizes: one-batch parity misses retained MPS graphs.
        for step in range(1, 129):
            rows, nnz = 128 + step % 23, 80000 + 137 * step
            matrix = self.sparse.csr_matrix(
                (rng.random(nnz, dtype=np.float32),
                 (rng.integers(rows, size=nnz), rng.integers(65546, size=nnz))),
                shape=(rows, 65546))
            batch = namespace["scipy_csr_to_device_components"](matrix, "mps")
            targets = torch.as_tensor(rng.random(rows, dtype=np.float32), device="mps")
            groups = torch.as_tensor(rng.integers(2, size=(rows, 9)).astype(np.float32), device="mps")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = criterion(outputs, targets, subgroup_membership=groups)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            self.assertTrue(np.isfinite(loss.item()))
            del batch, targets, groups, outputs, loss, matrix
            if step % 16 == 0:
                torch.mps.synchronize()
                self.assertLess(torch.mps.driver_allocated_memory() - initial_memory, 256 * 2**20,
                                "Unbounded MPS memory growth on variable-shape synthetic batches")


if __name__ == "__main__":
    unittest.main()
