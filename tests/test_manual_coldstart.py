import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engine.coldstart import knowledge
from utils import dump_injected


class ManualColdstartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        tasks = self.root / "tasks.json"
        models = self.root / "models.json"
        tasks.write_text(json.dumps({"demo": "images"}))
        models.write_text(json.dumps({"images": {"model-a": {
            "Description": "A model recommendation",
            "Code_template": "load('{TORCH_HUB_DIR}/model-a')",
        }}}))
        self.cfg = SimpleNamespace(
            exp_id="demo", torch_hub_dir="/models/", log_dir=self.root / "logs",
            methodology_kb_path="/old/kb", methodology_retrieval="lazy",
            coldstart=SimpleNamespace(
                task_json_path=str(tasks), model_json_path=str(models),
                methodology_text="stale literature", inject_into_improve=True,
            ),
        )

    def test_legacy_modes_never_retrieve_or_write_injection_artifacts(self):
        with patch("engine.coldstart.methodology_agent.build_methodology_guidance") as retrieve, \
                patch.object(knowledge, "_build_methodology_text") as static, \
                patch("engine.coldstart.kb_snapshot.write_kb_snapshot") as snapshot:
            for mode in ("lazy", "vector", "llm", "static"):
                for task_desc in ("classify images", ""):
                    with self.subTest(mode=mode, task_desc=task_desc):
                        self.cfg.methodology_retrieval = mode
                        self.cfg.coldstart.methodology_text = "stale literature"
                        with self.assertLogs("MLEvolve", level="WARNING") as logs:
                            text = knowledge.build_guidance_description(self.cfg, task_desc)
                        self.assertIn("python -m corpus_retriever", " ".join(logs.output))
                        self.assertNotIn("returned empty", " ".join(logs.output))
                        self.assertEqual(self.cfg.coldstart.methodology_text, "")
                        self.assertIn("Model1: model-a", text)
                        self.assertIn("load('/models/model-a')", text)
                        self.assertNotIn("stale literature", text)
            retrieve.assert_not_called()
            static.assert_not_called()
            snapshot.assert_not_called()
        self.assertFalse(self.cfg.log_dir.exists())

    def test_no_model_sentinel_and_manual_mode_are_preserved(self):
        self.cfg.exp_id = "unknown"
        self.cfg.methodology_kb_path = ""
        self.cfg.methodology_retrieval = "manual"
        with self.assertNoLogs("MLEvolve", level="WARNING"):
            text = knowledge.build_guidance_description(self.cfg, "task")
        self.assertEqual(text, "None model")
        self.assertEqual(self.cfg.coldstart.methodology_text, "")

    def test_historical_verifier_never_rebuilds_or_changes_saved_files(self):
        for name in ("missing", "matched", "mismatched"):
            logs = self.root / name / "logs"
            logs.mkdir(parents=True)
            original = "historical techniques"
            (logs / "MLEvolve.log").write_text(
                f"Knowledge injected at draft: {len(original)} chars, "
                f"digest {knowledge.text_digest(original)}\n"
            )
            if name != "missing":
                (logs / "injected_knowledge.md").write_text(
                    original if name == "matched" else "different content"
                )
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        output = io.StringIO()
        with patch("sys.argv", ["dump_injected.py", "--runs", str(self.root), "--write"]), \
                patch.object(knowledge, "build_guidance_description") as guidance, \
                contextlib.redirect_stdout(output):
            self.assertEqual(dump_injected.main(), 0)
        guidance.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertIn("1 verified, 1 mismatched, 1 unavailable", output.getvalue())
        self.assertIn("historical replay is no longer supported", output.getvalue())


if __name__ == "__main__":
    unittest.main()
