import copy
import shlex
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DraftRerankJobTests(unittest.TestCase):
    def _pair(self, seed):
        path = ROOT / f"k8s/job-jigsaw-unintended-draft-rerank-s{seed}.yaml"
        docs = list(yaml.safe_load_all(path.read_text()))
        self.assertEqual(len(docs), 2)
        return docs

    def test_pairs_differ_only_in_injection_and_run_identity(self):
        for seed in (42, 43, 44):
            with self.subTest(seed=seed):
                normalized = []
                for index, doc in enumerate(self._pair(seed)):
                    doc = copy.deepcopy(doc)
                    arm = ("a-baseline", "b-draft-rerank")[index]
                    for metadata in (doc["metadata"], doc["spec"]["template"]["metadata"]):
                        self.assertEqual(metadata["labels"].pop("arm"), arm)
                    doc["metadata"].pop("name")
                    container = doc["spec"]["template"]["spec"]["containers"][0]
                    env = {item["name"]: item["value"] for item in container.pop("env")}
                    env.pop("EXP_NAME")
                    env.pop("SERVER_ID")
                    overrides = dict(item.split("=", 1) for item in shlex.split(env.pop("EXTRA_RUN_ARGS")))
                    self.assertEqual(overrides.pop("draft_techniques.enabled"), ("False", "True")[index])
                    self.assertEqual(overrides["agent.seed"], str(seed))
                    normalized.append((doc, env, overrides))
                self.assertEqual(normalized[0], normalized[1])

    def test_jobs_use_current_corpus_and_matched_runtime_budget(self):
        identities = {key: set() for key in ("name", "EXP_NAME", "SERVER_ID")}
        for seed in (42, 43, 44):
            for doc in self._pair(seed):
                container = doc["spec"]["template"]["spec"]["containers"][0]
                env = {item["name"]: item["value"] for item in container["env"]}
                overrides = dict(item.split("=", 1) for item in shlex.split(env["EXTRA_RUN_ARGS"]))
                self.assertEqual(overrides["methodology_retrieval"], "none")
                self.assertEqual(overrides["coldstart.use_coldstart"], "True")
                self.assertEqual(overrides["coldstart.inject_into_improve"], "False")
                self.assertEqual(overrides["draft_techniques.cache_dir"], "/workspace/MLEvolve/cache/techinique")
                self.assertEqual(overrides["draft_techniques.context_window"], "258400")
                self.assertEqual(overrides["agent.time_limit"], env["TIME_LIMIT_SECS"])
                self.assertEqual(env["TIME_LIMIT_SECS"], "43200")
                self.assertEqual(env["LLM_MODEL"], "gpt-5.6-terra")
                self.assertEqual(env["LLM_BASE_URL"], "http://cliproxy:8317/v1")
                resources = container["resources"]
                self.assertEqual(resources["requests"], resources["limits"])
                self.assertEqual(resources["limits"], {"cpu": "8", "memory": "48Gi", "nvidia.com/gpu": 1})
                self.assertEqual(env["CPUS_PER_TASK"], resources["limits"]["cpu"])
                for key, value in (("name", doc["metadata"]["name"]),
                                   ("EXP_NAME", env["EXP_NAME"]), ("SERVER_ID", env["SERVER_ID"])):
                    self.assertNotIn(value, identities[key])
                    identities[key].add(value)
        self.assertEqual(identities["SERVER_ID"], {str(number) for number in range(78, 84)})


if __name__ == "__main__":
    unittest.main()
