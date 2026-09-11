"""Resolve CUDA visibility without GPU model or memory-size allowlists."""

import os


def visible_gpu_devices() -> list[str]:
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        return []
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        # Preserve the parent's CUDA_DEVICE_ORDER and CUDA ordinal namespace.
        return [str(index) for index in range(count)]
    devices = [token.strip() for token in mask.split(",")][:count]
    if len(devices) != count or any(not token or token == "-1" for token in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES disagrees with PyTorch's visible GPU count")
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU identifiers")
    # Preserve numeric IDs, GPU UUIDs and MIG IDs; child cuda:0 maps to this exact device.
    return devices
