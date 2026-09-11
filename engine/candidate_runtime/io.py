"""Small crash-safe files; all publication renames stay on the same filesystem."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def seal_directory(staging, destination):
    staging, destination = Path(staging), Path(destination)
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Artifact must contain regular files, not symlinks: {path}")
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        sync_directory(path)
    sync_directory(staging)
    os.replace(staging, destination)
    sync_directory(destination.parent)


def file_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def check_hashes(directory, hashes):
    directory = Path(directory).resolve()
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Incomplete or changed artifact file: {name}")
