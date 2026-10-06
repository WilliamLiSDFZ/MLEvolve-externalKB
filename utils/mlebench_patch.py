"""Apply/check the versioned MLE-bench metric fix; never replace the grading API."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches" / "mlebench"
MANIFEST = json.loads((PATCH_DIR / "manifest.json").read_text())
COMPETITION = MANIFEST["competition"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def grader_path(competition: str = COMPETITION) -> Path:
    dist = importlib.metadata.distribution("mlebench")
    return Path(dist.locate_file(f"mlebench/competitions/{competition}/grade.py"))


def patched_source(source: bytes) -> bytes:
    """Apply our single-hunk patch in memory, guarded by both complete-file hashes."""
    digest = sha256(source)
    if digest == MANIFEST["patched_sha256"]:
        return source
    if digest != MANIFEST["original_sha256"]:
        raise RuntimeError(f"Unrecognized jubias grader {digest}; review the upstream change "
                           "before updating patches/mlebench/manifest.json.")
    lines = (PATCH_DIR / "jubias-continuous-auc.patch").read_text().splitlines(keepends=True)
    hunks = [i for i, line in enumerate(lines) if line.startswith("@@")]
    if len(hunks) != 1:
        raise RuntimeError("Expected exactly one patch hunk")
    body = lines[hunks[0] + 1:]
    old = "".join(line[1:] for line in body if line.startswith((" ", "-"))).encode()
    new = "".join(line[1:] for line in body if line.startswith((" ", "+"))).encode()
    if not old or source.count(old) != 1:
        raise RuntimeError("Patch context must occur exactly once")
    result = source.replace(old, new, 1)
    if sha256(result) != MANIFEST["patched_sha256"]:
        raise RuntimeError("Patched grader hash does not match the manifest")
    return result


def apply_patch(path: Path) -> bool:
    original = path.read_bytes()
    patched = patched_source(original)
    if patched == original:
        return False
    backup = path.with_name(path.name + ".before-" + MANIFEST["metric_version"])
    if backup.exists():
        if backup.read_bytes() != original:
            raise RuntimeError(f"Existing backup differs: {backup}")
    else:
        with backup.open("xb") as fh:
            fh.write(original)
    # Write beside the target and rename, so readers never see a partially written module.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".grade-", delete=False) as fh:
            temporary = Path(fh.name)
            fh.write(patched)
            fh.flush()
            os.fsync(fh.fileno())
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def grading_metadata(competition: str) -> dict[str, str]:
    """Fail closed for an unpatched jubias grader; describe other graders unchanged."""
    digest = sha256(grader_path(competition).read_bytes())
    if competition == COMPETITION and digest != MANIFEST["patched_sha256"]:
        raise RuntimeError("Jubias continuous-AUC fix is missing or changed. Run "
                           "python utils/mlebench_patch.py --apply, then restart this process. "
                           f"Found grader sha256={digest}")
    dist = importlib.metadata.distribution("mlebench")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    return {
        "metric_version": MANIFEST["metric_version"] if competition == COMPETITION else "mlebench-unmodified",
        "grader_sha256": digest,
        "mlebench_version": dist.version,
        # An install from a local source tree may have no VCS metadata. Do not invent it.
        "mlebench_commit": direct.get("vcs_info", {}).get("commit_id", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--upstream-revision", action="store_true")
    args = parser.parse_args()
    if args.upstream_revision:
        print(MANIFEST["upstream_revision"])
        return
    if args.apply:
        changed = apply_patch(grader_path())
        print("Applied jubias continuous-AUC patch" if changed else "Jubias patch already applied")
    print(json.dumps(grading_metadata(COMPETITION), sort_keys=True))


if __name__ == "__main__":
    main()
