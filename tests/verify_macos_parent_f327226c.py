"""Synthetic parity only; never executes the downloaded parent's data/training code.

Set MLEVOLVE_MACOS_F327226C_SOURCE to the original downloaded solution.py.
Native MPS checks are bounded to tiny tensors and require fallback disabled.
"""

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Dict, Optional, Sequence
import unittest

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.replay_macos.adapters import get_adapter
from utils.replay_macos.adapters.parent_f327226c import ADAPTER_VERSION, ORIGINAL_SOURCE_SHA256, adapt


SOURCE = Path(os.environ.get("MLEVOLVE_MACOS_F327226C_SOURCE", Path.home() / (
    "nautilus/replay-inputs/jubias-f327226c/runs/20260912_081234_jubias-base-gpt56sol-s57/"
    "workspace/candidate_results/candidates/f327226c666446d58718f3801048032c/solution.py")))
DEFINITIONS = {
    "_MacEmbeddingBagFunction", "MacSparseEmbeddingBag", "MacSparseAdam",
    "SparseWideDeepToxicityModel", "JigsawFairnessCriterion", "CompositeOptimizer",
    "clip_sparse_gradients", "atomic_torch_save", "atomic_joblib_save",
    "save_checkpoint", "load_checkpoint",
}


def definitions(source):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS]
    namespace = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<synthetic-parent>", "exec"), namespace)
    return namespace


def close(left, right, tolerance=2e-6):
    torch.testing.assert_close(left.detach().cpu(), right.detach().cpu(),
                               atol=tolerance, rtol=tolerance)


@unittest.skipUnless(SOURCE.is_file(), "Original parent fixture not downloaded")
class ParentPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.port = adapt(cls.source)
        cls.original = definitions(cls.source)
        cls.adapted = definitions(cls.port)

    def test_hash_lock_and_preserved_experiment(self):
        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(), ORIGINAL_SOURCE_SHA256)
        self.assertEqual(ADAPTER_VERSION, "jigsaw-f327226c-mps-v1")
        self.assertEqual(self.port, adapt(self.source))
        self.assertEqual(hashlib.sha256(self.port.encode()).hexdigest(),
                         "13178cd664223bd5d240a051d37501e9589bbce7079d1be0a43c2908c03a3499")
        self.assertEqual(self.port, get_adapter(ORIGINAL_SOURCE_SHA256).adapt(self.source))
        self.assertEqual(get_adapter(ORIGINAL_SOURCE_SHA256).ADAPTER_VERSION, ADAPTER_VERSION)
        with self.assertRaises(ValueError):
            adapt(self.source + "\n")
        original = {node.name: ast.dump(node) for node in ast.parse(self.source).body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        adapted = {node.name: ast.dump(node) for node in ast.parse(self.port).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in original.keys() - {"SparseWideDeepToxicityModel", "clip_sparse_gradients"}:
            self.assertEqual(original[name], adapted[name], name)
        for setting in ("SEED = 2025", "TRAIN_BASE_BATCH_SIZE = 1024",
                        "INFERENCE_BATCH_SIZE = 2048", "MAX_EPOCHS = 3",
                        "WARMUP_UPDATES = 100", "GRADIENT_CLIP_NORM = float("):
            self.assertTrue(setting in self.port, setting)
        self.assertIn('device = torch.device("mps")', self.port)
        self.assertIn("num_workers=0", self.port)
        self.assertNotIn("torch.cuda", self.port)

    def check_sparse_updates(self, device):
        torch.manual_seed(17)
        original = nn.EmbeddingBag(11, 3, mode="sum", sparse=True, include_last_offset=True)
        adapted = self.adapted["MacSparseEmbeddingBag"](
            11, 3, mode="sum", sparse=True, include_last_offset=True).to(device)
        adapted.load_state_dict(original.state_dict())
        settings = dict(lr=0.015, betas=(0.9, 0.999), eps=1e-8)
        old_optimizer = torch.optim.SparseAdam(original.parameters(), **settings)
        new_optimizer = self.adapted["MacSparseAdam"](adapted.parameters(), **settings)
        batches = ([2, 2, 4, 2], [1, 1, 7, 1], [2, 4, 2, 4], [])
        for step, indices in enumerate(batches):
            inputs = (torch.tensor(indices, dtype=torch.long),
                      torch.tensor([0, 2, 2, 4] if indices else [0, 0, 0, 0]),
                      torch.tensor([1., -2., 3., 4.] if indices else []))
            output_gradient = torch.tensor([[10., -2., 0.1], [0., 4., 0.], [20., 3., -9.]])
            old_optimizer.zero_grad()
            new_optimizer.zero_grad()
            old_output = original(inputs[0], inputs[1], per_sample_weights=inputs[2])
            new_output = adapted(inputs[0].to(device), inputs[1].to(device),
                                 per_sample_weights=inputs[2].to(device))
            close(old_output, new_output)
            old_output.backward(output_gradient)
            new_output.backward(output_gradient.to(device))
            old_grad = original.weight.grad
            new_indices, new_values = adapted.weight._mac_sparse_gradient
            self.assertEqual(new_values.device.type, device)
            close(old_grad._indices()[0], new_indices)
            close(old_grad._values(), new_values)
            self.original["clip_sparse_gradients"]([original.weight])
            self.adapted["clip_sparse_gradients"]([adapted.weight])
            close(old_grad._values(), new_values)
            before = adapted.weight.detach().clone()
            before_state = copy.deepcopy(new_optimizer.state_dict())
            old_optimizer.step()
            new_optimizer.step()
            close(original.weight, adapted.weight)
            untouched = sorted(set(range(11)) - set(indices))
            close(before[untouched], adapted.weight[untouched], tolerance=0)
            for key in ("exp_avg", "exp_avg_sq"):
                close(old_optimizer.state[original.weight][key], new_optimizer.state[adapted.weight][key])
                if step:
                    close(before_state["state"][0][key][untouched],
                          new_optimizer.state[adapted.weight][key][untouched], tolerance=0)
            self.assertEqual(new_optimizer.state[adapted.weight]["step"], step + 1)
            if step == 1:
                # Ordinary optimizer checkpoints also preserve sparse Adam state.
                state = copy.deepcopy(new_optimizer.state_dict())
                new_optimizer = self.adapted["MacSparseAdam"](adapted.parameters(), **settings)
                new_optimizer.load_state_dict(state)

    def test_cpu_sparse_clipping_and_optimizer_parity(self):
        self.check_sparse_updates("cpu")

    def check_full_model(self, device):
        torch.manual_seed(23)
        settings = dict(feature_dimension=13, number_numeric_features=3,
                        number_auxiliary_targets=2, deep_embedding_dimension=4,
                        numeric_hidden_dimension=3, hidden_dimension=8, dropout=0.0)
        old_model = self.original["SparseWideDeepToxicityModel"](**settings)
        new_model = self.adapted["SparseWideDeepToxicityModel"](**settings).to(device)
        new_model.load_state_dict(old_model.state_dict())
        criteria = [namespace["JigsawFairnessCriterion"]([0, 1], minimum_group_examples=2).to(where)
                    for namespace, where in ((self.original, "cpu"), (self.adapted, device))]
        inputs = (torch.tensor([2, 2, 1, 4, 2, 4, 8, 1, 4, 6, 4, 8]),
                  torch.tensor([0, 2, 3, 5, 7, 7, 9, 10, 12]),
                  torch.tensor([1., 0.2, -0.4, 0.7, 1.1, 0.1, 0.8, 0.9, 1., -0.1, 1., 0.5]),
                  torch.randn(8, 3))
        targets = dict(soft_targets=torch.tensor([.8, .1, .7, .2, .9, .3, .6, .4]),
                       auxiliary_targets=torch.rand(8, 2), auxiliary_mask=torch.ones(8, 2),
                       identity_targets=torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]] * 2),
                       identity_mask=torch.ones(8, 2), sample_weight=torch.arange(1., 9.))
        optimizers = []
        parameter_sets = []
        for model, namespace, kind in ((old_model, self.original, torch.optim.SparseAdam),
                                        (new_model, self.adapted, self.adapted["MacSparseAdam"])):
            sparse = [model.wide_embedding.weight, model.deep_embedding.weight]
            dense = [p for p in model.parameters() if all(p is not s for s in sparse)]
            children = [kind(sparse, lr=.015, betas=(.9, .999), eps=1e-8),
                        torch.optim.AdamW(dense, lr=8e-4, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)]
            optimizers.append(namespace["CompositeOptimizer"](children))
            parameter_sets.append((sparse, dense))
        for step in range(3):
            outputs = [old_model(*inputs), new_model(*(value.to(device) for value in inputs))]
            for name in outputs[0]:
                close(outputs[0][name], outputs[1][name], tolerance=2e-5)
            losses = []
            for criterion, output, where in zip(criteria, outputs, ("cpu", device)):
                criterion.set_training_progress((step + 1) / 3)
                losses.append(criterion(output, **{k: v.to(where) for k, v in targets.items()}))
                self.assertGreater(criterion.last_components["fairness"], 0)
            close(*losses, tolerance=2e-5)
            for optimizer in optimizers:
                optimizer.zero_grad()
            for loss in losses:
                loss.backward()
            for old, new in zip(parameter_sets[0][0], parameter_sets[1][0]):
                close(old.grad._values(), new._mac_sparse_gradient[1], tolerance=2e-5)
            for old, new in zip(parameter_sets[0][1], parameter_sets[1][1]):
                close(old.grad, new.grad, tolerance=2e-5)
            for namespace, optimizer, (sparse, dense) in zip(
                    (self.original, self.adapted), optimizers, parameter_sets):
                namespace["clip_sparse_gradients"](sparse)
                torch.nn.utils.clip_grad_norm_(dense, 2.)
                optimizer.step()
            for old, new in zip(old_model.parameters(), new_model.parameters()):
                close(old, new, tolerance=3e-5)
        # Exercise unchanged production checkpoint callbacks and cross-loadability.
        with tempfile.TemporaryDirectory() as temporary:
            before = {k: v.detach().cpu().clone() for k, v in new_model.state_dict().items()}
            self.adapted.update(model=new_model, device=torch.device(device), optimizer_steps=3,
                                model_config={**settings, "official_identity_columns": ["male", "female"]},
                                feature_state={"fixture": [1, 2]}, prepared_data={})
            self.adapted["save_checkpoint"](temporary)
            with torch.no_grad():
                for parameter in new_model.parameters():
                    parameter.zero_()
            self.adapted["load_checkpoint"](temporary)
            for name, tensor in new_model.state_dict().items():
                close(before[name], tensor, tolerance=0)
            saved = torch.load(Path(temporary) / "model.pt", weights_only=False)
            old_model.load_state_dict(saved["model_state_dict"], strict=True)
            self.assertEqual(saved["optimizer_steps"], 3)

    def test_cpu_model_loss_gradient_update_checkpoint_parity(self):
        self.check_full_model("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available() and
                         os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1",
                         "Native MPS required with fallback disabled")
    def test_native_mps_bounded_parity(self):
        self.check_sparse_updates("mps")
        self.check_full_model("mps")
        torch.mps.synchronize()


if __name__ == "__main__":
    unittest.main()
