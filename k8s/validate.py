"""Validate the Job manifests in this directory before `kubectl apply`.

Written after a manifest that parsed as valid YAML was rejected by the API server:

    strict decoding error: unknown field "spec.template.repeat"

A generator had inserted a label one indent level too high, so it landed on the PodTemplateSpec
instead of on its metadata.labels. YAML parsing cannot catch that — the document is well
formed, the key is just in the wrong place. This checks the *shape* against the fields the Job
and Pod schemas actually accept, plus the project-specific invariants that make a paired A/B
comparison valid.

    python k8s/validate.py
"""
import sys
from pathlib import Path

import yaml

JOB_SPEC_FIELDS = {
    "parallelism", "completions", "activeDeadlineSeconds", "podFailurePolicy",
    "backoffLimit", "backoffLimitPerIndex", "maxFailedIndexes", "selector",
    "manualSelector", "template", "ttlSecondsAfterFinished", "completionMode",
    "suspend", "podReplacementPolicy", "managedBy", "successPolicy",
}
POD_TEMPLATE_FIELDS = {"metadata", "spec"}
METADATA_FIELDS = {"name", "generateName", "namespace", "labels", "annotations",
                   "ownerReferences", "finalizers"}


def check(ok: bool, msg: str, errors: list) -> None:
    if not ok:
        errors.append(msg)


def validate_doc(doc: dict, where: str, errors: list) -> None:
    check(doc.get("apiVersion") == "batch/v1", f"{where}: apiVersion is not batch/v1", errors)
    check(doc.get("kind") == "Job", f"{where}: kind is not Job", errors)

    spec = doc.get("spec", {})
    for k in spec:
        check(k in JOB_SPEC_FIELDS, f"{where}: unknown field spec.{k}", errors)

    tmpl = spec.get("template", {})
    for k in tmpl:
        # This is the check that the earlier failure needed.
        check(k in POD_TEMPLATE_FIELDS,
              f'{where}: unknown field spec.template.{k} '
              f'(labels belong under spec.template.metadata.labels)', errors)

    for md_path, md in (("metadata", doc.get("metadata", {})),
                        ("spec.template.metadata", tmpl.get("metadata", {}))):
        for k in md:
            check(k in METADATA_FIELDS, f"{where}: unknown field {md_path}.{k}", errors)
        for k, v in (md.get("labels") or {}).items():
            check(isinstance(v, str),
                  f"{where}: label {k} is {type(v).__name__}, must be a quoted string", errors)

    pod = tmpl.get("spec", {})
    check(bool(pod.get("containers")), f"{where}: no containers", errors)
    for c in pod.get("containers", []):
        res = c.get("resources", {})
        check(res.get("requests") == res.get("limits"),
              f"{where}/{c.get('name')}: Nautilus requires requests == limits", errors)
        env = {e["name"]: e.get("value") for e in c.get("env", [])}
        if "CPUS_PER_TASK" in env:
            check(env["CPUS_PER_TASK"] == str(res.get("limits", {}).get("cpu")),
                  f"{where}/{c.get('name')}: CPUS_PER_TASK != cpu limit", errors)


def main() -> int:
    here = Path(__file__).parent
    errors: list = []
    jobs = []

    for f in sorted(here.glob("job-*.yaml")):
        try:
            docs = [d for d in yaml.safe_load_all(f.read_text()) if d]
        except yaml.YAMLError as e:
            errors.append(f"{f.name}: not valid YAML: {e}")
            continue
        for i, d in enumerate(docs):
            where = f"{f.name}[{i}]"
            validate_doc(d, where, errors)
            c = d["spec"]["template"]["spec"]["containers"][0]
            jobs.append({
                "file": f.name, "name": d["metadata"]["name"],
                "labels": d["metadata"].get("labels", {}),
                "env": {e["name"]: e.get("value") for e in c.get("env", [])},
            })

    # Cross-file invariants: a collision here silently corrupts a run rather than failing it.
    for key, get in (("job name", lambda j: j["name"]),
                     ("EXP_NAME", lambda j: j["env"].get("EXP_NAME")),
                     ("grading port", lambda j: j["env"].get("SERVER_ID"))):
        seen = {}
        for j in jobs:
            v = get(j)
            if v is None:
                continue
            if v in seen:
                errors.append(f"duplicate {key} {v!r}: {seen[v]} and {j['file']}")
            seen[v] = j["file"]

    print(f"checked {len(jobs)} Job document(s) across "
          f"{len({j['file'] for j in jobs})} file(s)")
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("all manifests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
