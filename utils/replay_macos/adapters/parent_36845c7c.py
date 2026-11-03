"""Hash-locked MPS port of Jigsaw parent 36845c7c (S60)."""

import ast

from .common import replace as _replace, require_source


ADAPTER_VERSION = "jigsaw-parent5-mps-v1"
ORIGINAL_SOURCE_SHA256 = "e5f0ab92a8932140f472ab3d13e4139d14dde3dfa5d9b721570470022078fcec"

_MPS_SETUP = '''# mac-mps: require Apple GPU execution; never silently fall back to CPU.
if not torch.backends.mps.is_available():
    raise RuntimeError("mac-mps requires an available Apple MPS device")
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
    raise RuntimeError("mac-mps requires PYTORCH_ENABLE_MPS_FALLBACK to be disabled")
torch.set_default_dtype(torch.float32)
'''

_COMPONENT_CONVERTER = '''def scipy_csr_to_device_components(feature_matrix, target_device):
    """Keep sparse storage on CPU; transfer only ordinary COO component tensors."""
    # Match torch COO coalesce ordering and duplicate summation before dropout.
    canonical = feature_matrix.tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    coo = canonical.tocoo(copy=False)
    return (
        torch.as_tensor(coo.row, dtype=torch.int64, device=target_device),
        torch.as_tensor(coo.col, dtype=torch.int64, device=target_device),
        torch.as_tensor(coo.data, dtype=torch.float32, device=target_device),
        tuple(coo.shape),
    )
'''


def adapt(source: str) -> str:
    """Port only the reviewed parent; callers must retain the original provenance."""
    require_source(source, ORIGINAL_SOURCE_SHA256)

    source = _replace(source, "import gc\n", "import gc\nfrom contextlib import nullcontext\n")
    source = _replace(source, "from engine.candidate_runtime import CandidateSession\n",
                      "from engine.candidate_runtime import CandidateSession\n\n" + _MPS_SETUP)
    source = _replace(source, 'if torch.cuda.is_available():\n    torch.cuda.manual_seed_all(SEED)',
                      'torch.mps.manual_seed(SEED)')
    source = _replace(source, 'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
                      'device = torch.device("mps")')

    # Preserve the segmented linear math; only change its sparse input container.
    source = _replace(source, '''        if features.layout != torch.sparse_coo:
            features = features.to_sparse_coo()
        features = features.coalesce()

        if features.ndim != 2:
            raise ValueError("Sparse features must be two-dimensional.")
        if features.shape[1] != self.total_feature_dim:
            raise ValueError(
                f"Expected {self.total_feature_dim} features, "
                f"received {features.shape[1]}."
            )

        indices = features.indices()
        row_indices = indices[0]
        column_indices = indices[1]
        feature_values = features.values()
''', '''        row_indices, column_indices, feature_values, shape = features
        if len(shape) != 2 or shape[1] != self.total_feature_dim:
            raise ValueError("Sparse component shape does not match the model.")
''')
    source = _replace(source, '            features.shape[0] * 3,', '            shape[0] * 3,')
    source = _replace(source, '            features.shape[0],\n            3,', '            shape[0],\n            3,')
    source = _replace(source, '''        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch Tensor.")

        if features.layout == torch.strided:
            components = self._dense_segmented_linear(features)
        else:
            components = self._sparse_segmented_linear(features)
''', '''        if isinstance(features, tuple) and len(features) == 4:
            components = self._sparse_segmented_linear(features)
        elif isinstance(features, torch.Tensor) and features.layout == torch.strided:
            components = self._dense_segmented_linear(features)
        else:
            raise TypeError("features must be dense or (rows, columns, values, shape).")
''')

    # Replace the whole CPU-COO constructor without carrying the parent into the repo.
    converter = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == "scipy_csr_to_torch_coo")
    lines = source.splitlines(keepends=True)
    lines[converter.lineno - 1:converter.end_lineno] = [_COMPONENT_CONVERTER]
    source = "".join(lines)
    for indent in ("                ", "        "):
        source = _replace(source, indent + '''torch_features = scipy_csr_to_torch_coo(scipy_features).to(
''' + indent + '''    device=device,
''' + indent + '''    non_blocking=False,
''' + indent + ''')''', indent + 'torch_features = scipy_csr_to_device_components(scipy_features, device)')

    source = _replace(source, "    num_workers=2,", "    num_workers=0,")
    source = _replace(source, "    prefetch_factor=2,\n", "")
    source = _replace(source, 'with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):',
                      "with nullcontext():", count=2)
    source = _replace(source, 'amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))\n', "")
    source = _replace(source, '        "amp_scaler_state_dict": (amp_scaler.state_dict()),\n', "")
    source = _replace(source, '''    if (
        "amp_scaler_state_dict" in checkpoint_payload
        and checkpoint_payload["amp_scaler_state_dict"]
    ):
        amp_scaler.load_state_dict(checkpoint_payload["amp_scaler_state_dict"])

''', "")
    source = _replace(source, '''        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)''', "        loss.backward()")
    source = _replace(source, '''        scale_before_step = float(amp_scaler.get_scale())
        amp_scaler.step(optimizer)
        amp_scaler.update()
        scale_after_step = float(amp_scaler.get_scale())

        optimizer_step_completed = (
            not amp_scaler.is_enabled() or scale_after_step >= scale_before_step
        )''', '''        optimizer.step()
        optimizer_step_completed = True''')
    # load_state_dict moves parameter states to MPS, keeping AdamW step counters on CPU.
    source = _replace(source, "        map_location=device,", '        map_location="cpu",')
    source = _replace(source, 'if device.type == "cuda":\n    torch.cuda.empty_cache()',
                      "torch.mps.empty_cache()")
    ast.parse(source)
    return source
