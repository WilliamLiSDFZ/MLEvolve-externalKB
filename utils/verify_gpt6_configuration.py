"""Offline configuration/preflight and rendered GPT-6 Job template regressions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import Config
from utils.llm_preflight import record_configuration


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"LLM_MODEL": "gpt-6-astra", "LLM_REASONING_EFFORT": "high",
            "LLM_BASE_URL": "https://proxy.invalid/v1", "LLM_API_KEY": "test-secret", "MLEVOLVE_REQUIRE_GPT6": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        raw = OmegaConf.load(ROOT/"config/config.yaml")
        # prep_cfg supplies these required runtime paths/name before schema merge.
        raw.data_dir = self.root / "public"
        raw.exp_name = "synthetic-gpt6-config-test"
        self.cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
        self.cfg.log_dir = self.root

    def test_preflight_effective_slots_and_sanitized_record(self):
        self.cfg.agent.code.base_url = "https://private-user:private-password@proxy.invalid/v1?secret=hidden"
        info = record_configuration(self.cfg)
        self.assertEqual(info["effective_analogy_input_cap"], 196608)
        self.assertFalse(info["live_compatibility_checked_here"])
        for role in ("code", "feedback", "analogy"):
            self.assertEqual(info["slots"][role]["model"], "gpt-6-astra")
            self.assertEqual(info["slots"][role]["reasoning_effort"], "high")
            self.assertEqual(info["slots"][role]["endpoint_type"], "responses")
        saved = (self.root/"llm_preflight.json").read_text()
        self.assertEqual(json.loads(saved), info)
        for private in ("private-user", "private-password", "hidden", "test-secret"):
            self.assertNotIn(private, saved)

    def test_stale_model_effort_and_context_fail_gpt6_preflight(self):
        for path, value in [("agent.feedback.model", "gpt-5"), ("agent.code.reasoning_effort", "none"),
                            ("analogy.context.version", 1), ("agent.code.max_output_tokens", 0),
                            ("analogy.max_output_tokens", 0), ("analogy.max_output_tokens", -1)]:
            with self.subTest(path=path):
                cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
                OmegaConf.update(cfg, path, value)
                with self.assertRaises(ValueError):
                    record_configuration(cfg)

    def test_old_saved_config_without_context_uses_v1(self):
        old = OmegaConf.to_container(self.cfg, resolve=True)
        old["analogy"].pop("context")
        old["agent"]["code"]["model"] = old["agent"]["feedback"]["model"] = "gpt-5"
        old_cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.create(old))
        with patch.dict(os.environ, {"MLEVOLVE_REQUIRE_GPT6": "0"}):
            info = record_configuration(old_cfg)
        self.assertEqual(info["context"]["version"], 1)
        self.assertEqual(info["slots"]["code"]["endpoint_type"], "legacy")

    def test_rendered_af_template_has_string_labels_and_explicit_gpt6(self):
        template = (ROOT/"k8s/job-jigsaw-unintended-af-gpt6.template.yaml").read_text()
        docs = list(yaml.safe_load_all(template.replace("__SEED__", "57")))
        self.assertEqual(len(docs), 2)
        configs = []
        for doc in docs:
            self.assertIn("gpt6-s57", doc["metadata"]["name"])
            for labels in (doc["metadata"]["labels"], doc["spec"]["template"]["metadata"]["labels"]):
                self.assertTrue(all(isinstance(value, str) for value in labels.values()), labels)
                self.assertEqual(labels["repeat"], "57")
            container = doc["spec"]["template"]["spec"]["containers"][0]
            env = {item["name"]: item["value"] for item in container["env"]}
            self.assertEqual(env["LLM_MODEL"], "gpt-6-astra")
            self.assertEqual(env["LLM_REASONING_EFFORT"], "high")
            self.assertEqual(env["MLEVOLVE_REQUIRE_GPT6"], "1")
            self.assertEqual(container["resources"]["requests"], container["resources"]["limits"])
            cfg = OmegaConf.merge(self.cfg, OmegaConf.from_dotlist(shlex.split(env["EXTRA_RUN_ARGS"])))
            record_configuration(cfg)
            configs.append(cfg)
        self.assertFalse(configs[0].analogy.enabled)
        self.assertTrue(configs[1].analogy.enabled)
        self.assertTrue(configs[1].analogy.draft and configs[1].analogy.improve and configs[1].analogy.fulltext.enabled)
        for key in ("draft_budget_seconds", "candidate_budget_seconds", "validation_fraction", "keep_snapshots"):
            self.assertEqual(configs[0].candidate_runtime[key], configs[1].candidate_runtime[key])


if __name__ == "__main__":
    unittest.main()
