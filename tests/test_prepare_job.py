import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@unittest.skipIf(yaml is None, "PyYAML is required to read the Job manifest")
class PrepareJobTests(unittest.TestCase):
    def setUp(self):
        manifest = Path(__file__).resolve().parents[1] / "k8s/job-prepare-jigsaw-unintended.yaml"
        container = yaml.safe_load(manifest.read_text())["spec"]["template"]["spec"]["containers"][0]
        self.script = container["args"][0]
        self.tmp = tempfile.TemporaryDirectory(prefix="prepare job ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.env = {**os.environ, **{e["name"]: e["value"] for e in container["env"] if "value" in e}}
        self.env.update(LLM_API_KEY="test-llm", TYPESAFE_API_KEY="test-jev",
                        CALL_LOG=str(self.root / "calls"), ARGS_LOG=str(self.root / "args"))
        for name in ("DATASET_DIR", "ABSTRACT_INDEX_PATH", "KAGGLE_CONFIG_DIR", "HF_HOME", "TORCH_HOME"):
            path = self.root / name.lower()
            path.mkdir()
            self.env[name] = str(path)
        self.env["TRANSFORMERS_CACHE"] = self.env["HF_HOME"]
        for name in ("manifest.json", "records.jsonl", "embeddings.npy"):
            (Path(self.env["ABSTRACT_INDEX_PATH"]) / name).touch()
        (Path(self.env["KAGGLE_CONFIG_DIR"]) / "kaggle.json").write_text("{}")
        self.desc = Path(self.env["DATASET_DIR"]) / self.env["EXP_ID"] / "prepared/public/description.md"
        bin_dir = self.root / ".venv/bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "activate").write_text(
            f"export VIRTUAL_ENV={shlex.quote(str(bin_dir.parent))}\n"
            f"export PATH={shlex.quote(str(bin_dir))}:\"$PATH\"\n"
        )
        self._executable(bin_dir / "python", '''
if [ "$1" = "-u" ] && [ "$2" = "-" ]; then
    echo dependencies >> "$CALL_LOG"
    cat >/dev/null
else
    echo retrieval >> "$CALL_LOG"
    printf '%s\\n' "$@" > "$ARGS_LOG"
    exit "${RETRIEVAL_RC:-0}"
fi
''')
        self._executable(bin_dir / "mlebench", '''
echo prepare >> "$CALL_LOG"
test "$*" = "prepare -c $EXP_ID --data-dir $DATASET_DIR" || exit 9
mkdir -p "$DATASET_DIR/$EXP_ID/prepared/public"
echo task > "$DATASET_DIR/$EXP_ID/prepared/public/description.md"
''')

    def _executable(self, path, body):
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    def _prepared(self):
        self.desc.parent.mkdir(parents=True)
        self.desc.write_text("task")

    def _run(self):
        return subprocess.run(["bash", "-c", self.script], cwd=self.root, env=self.env,
                              capture_output=True, text=True, timeout=10)

    def _calls(self):
        path = Path(self.env["CALL_LOG"])
        return path.read_text().splitlines() if path.exists() else []

    def test_existing_dataset_skips_preparation_and_uses_cpu_retrieval(self):
        self._prepared()
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._calls(), ["dependencies", "retrieval"])
        self.assertIn("[prepare] starting task", result.stdout)
        self.assertIn("venv activated; checking retrieval dependencies", result.stdout)
        self.assertIn("using existing prepared dataset", result.stdout)
        self.assertIn("starting corpus retrieval", result.stdout)
        self.assertEqual(Path(self.env["ARGS_LOG"]).read_text().splitlines(), [
            "-u", "-m", "corpus_retriever", f"exp_id={self.env['EXP_ID']}",
            f"desc_file={self.desc}", f"data_dir={self.desc.parent}",
            f"abstract_index_path={self.env['ABSTRACT_INDEX_PATH']}",
            "retr_embedding_device=cpu", "lazy_pool=100",
        ])

    def test_missing_dataset_is_prepared_before_retrieval(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.desc.exists())
        self.assertEqual(self._calls(), ["dependencies", "prepare", "retrieval"])

    def test_partial_retrieval_exit_code_is_preserved(self):
        self._prepared()
        self.env["RETRIEVAL_RC"] = "2"
        self.assertEqual(self._run().returncode, 2)
        self.assertEqual(self._calls(), ["dependencies", "retrieval"])

    def test_missing_credentials_fail_before_any_calls(self):
        for key in ("LLM_API_KEY", "TYPESAFE_API_KEY"):
            with self.subTest(key=key):
                value = self.env.pop(key)
                result = self._run()
                self.env[key] = value
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)
                self.assertEqual(self._calls(), [])

    def test_embedded_shell_syntax(self):
        result = subprocess.run(["bash", "-n"], input=self.script, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_failure_reports_elapsed_time_and_stops(self):
        preflight = self.script.split("python -u - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        self.assertNotIn("faulthandler", preflight)
        fake_imports = '''
import importlib
from types import SimpleNamespace
def fake_import(name):
    if name == "config":
        return SimpleNamespace(Config=object)
    raise ImportError("test dependency failure: " + name)
importlib.import_module = fake_import
'''
        result = subprocess.run([sys.executable, "-u", "-"], input=fake_imports + preflight,
                                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(": starting", result.stdout)
        self.assertRegex(result.stdout, r"import config: done in \d+\.\d+s")
        self.assertRegex(result.stdout, r"import llm: FAILED after \d+\.\d+s")
        self.assertIn("ImportError: test dependency failure: llm", result.stderr)
        self.assertNotIn("import agents.memory.embedding_models", result.stdout)
        self.assertNotIn("retrieval dependencies available", result.stdout)


if __name__ == "__main__":
    unittest.main()
