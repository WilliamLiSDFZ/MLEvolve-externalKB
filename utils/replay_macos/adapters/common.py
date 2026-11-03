"""Strict source and replacement checks shared by the fixed parent ports."""

import hashlib


def require_source(source: str, expected_sha256: str) -> None:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Unsupported replay parent SHA256: {digest}")


def replace(source: str, old: str, new: str, count: int = 1) -> str:
    if source.count(old) != count:
        raise ValueError("The fixed parent adapter no longer matches its source template")
    return source.replace(old, new)
