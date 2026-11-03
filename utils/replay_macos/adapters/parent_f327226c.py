"""Hash-locked MPS port of Jigsaw parent f327226c (S57)."""

from .common import replace as _replace, require_source


ADAPTER_VERSION = "jigsaw-f327226c-mps-v1"
ORIGINAL_SOURCE_SHA256 = "626b762e35a3aa5e023c56002a1198a1a0d73c2bbdd8b689dcf7739f9541556e"

_MPS_SETUP = '''# mac-mps: require native Apple GPU kernels, including embedding updates.
if not torch.backends.mps.is_available():
    raise RuntimeError("mac-mps requires an available Apple MPS device")
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
    raise RuntimeError("mac-mps requires PYTORCH_ENABLE_MPS_FALLBACK to be disabled")
torch.set_default_dtype(torch.float32)
'''

# Sparse MPS tensors are unsupported. Keep the original uncoalesced gradient as
# ordinary tensors: clipping must precede duplicate-index summation, and unused
# rows must retain their SparseAdam moments without any decay or parameter update.
_SPARSE_COMPATIBILITY = '''class _MacEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices, offsets, values):
        ctx.weight = weight
        ctx.save_for_backward(indices, offsets, values)
        return F.embedding_bag(
            indices, weight, offsets, mode="sum", per_sample_weights=values,
            sparse=False, include_last_offset=True,
        )

    @staticmethod
    def backward(ctx, output_gradient):
        indices, offsets, values = ctx.saved_tensors
        bag_indices = torch.repeat_interleave(
            torch.arange(offsets.numel() - 1, device=indices.device),
            offsets[1:] - offsets[:-1], output_size=indices.numel(),
        )
        gradient_values = output_gradient.index_select(0, bag_indices) * values[:, None]
        if getattr(ctx.weight, "_mac_sparse_gradient", None) is not None:
            raise RuntimeError("This fixed replay parent requires zero_grad between backwards")
        ctx.weight._mac_sparse_gradient = (indices, gradient_values)
        return None, None, None, None


class MacSparseEmbeddingBag(nn.EmbeddingBag):
    def forward(self, indices, offsets, per_sample_weights):
        if per_sample_weights.requires_grad:
            raise RuntimeError("This fixed replay parent requires constant feature weights")
        return _MacEmbeddingBagFunction.apply(self.weight, indices, offsets, per_sample_weights)


class MacSparseAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, maximize=False))

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter._mac_sparse_gradient = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = getattr(parameter, "_mac_sparse_gradient", None)
                if gradient is None:
                    continue
                indices, values = gradient
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                if values.numel() == 0:
                    continue
                rows, inverse = torch.unique(indices, sorted=True, return_inverse=True)
                coalesced = values.new_zeros((rows.numel(), parameter.shape[1]))
                coalesced.index_add_(0, inverse, values)
                # Match torch.optim._functional.sparse_adam's operation order.
                old_mean = state["exp_avg"].index_select(0, rows)
                mean_delta = coalesced.sub(old_mean).mul_(1 - beta1)
                state["exp_avg"].index_add_(0, rows, mean_delta)
                old_variance = state["exp_avg_sq"].index_select(0, rows)
                variance_delta = coalesced.pow(2).sub_(old_variance).mul_(1 - beta2)
                state["exp_avg_sq"].index_add_(0, rows, variance_delta)
                numerator = mean_delta.add_(old_mean)
                denominator = variance_delta.add_(old_variance).sqrt_().add_(group["eps"])
                step_size = group["lr"] * math.sqrt(1 - beta2 ** state["step"])
                step_size /= 1 - beta1 ** state["step"]
                parameter.index_add_(0, rows, numerator.div_(denominator).mul_(-step_size))
        return loss


'''


def adapt(source: str) -> str:
    """Only change execution placement and the unsupported sparse representation."""
    require_source(source, ORIGINAL_SOURCE_SHA256)
    source = _replace(source, "from torch.optim import AdamW, SparseAdam\n",
                      "from torch.optim import AdamW\n")
    source = _replace(source, "from engine.candidate_runtime import CandidateSession\n",
                      "from engine.candidate_runtime import CandidateSession\n\n" + _MPS_SETUP)
    source = _replace(source, "if torch.cuda.is_available():\n    torch.cuda.manual_seed_all(SEED)",
                      "torch.mps.manual_seed(SEED)")
    source = _replace(source, 'device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")',
                      'device = torch.device("mps")')
    source = _replace(source, "class SparseWideDeepToxicityModel(nn.Module):",
                      _SPARSE_COMPATIBILITY + "class SparseWideDeepToxicityModel(nn.Module):")
    source = _replace(source, " = nn.EmbeddingBag(", " = MacSparseEmbeddingBag(", count=2)
    source = _replace(source, "sparse_optimizer = SparseAdam(", "sparse_optimizer = MacSparseAdam(")
    source = _replace(source, '''        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            gradient._values().clamp_(
                min=-maximum_absolute_value,
                max=maximum_absolute_value,
            )
''', '''        gradient = getattr(parameter, "_mac_sparse_gradient", None)
        if gradient is None:
            continue
        gradient[1].clamp_(
            min=-maximum_absolute_value,
            max=maximum_absolute_value,
        )
''')
    source = _replace(source, "    num_workers=2,", "    num_workers=0,")
    source = _replace(source, "    persistent_workers=True,", "    persistent_workers=False,")
    source = _replace(source, "    prefetch_factor=2,\n", "")
    source = _replace(source, '''if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
''', "")
    source = _replace(source, "if torch.cuda.is_available():\n    torch.cuda.empty_cache()",
                      "torch.mps.empty_cache()")
    compile(source, "<macos-parent-f327226c>", "exec")
    return source
