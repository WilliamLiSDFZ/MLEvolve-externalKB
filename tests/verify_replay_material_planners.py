"""Material v2 reaches both planner routes; legacy prompts stay byte-identical."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.planner import base_planner as direct, planner_with_memory as memory
from agents.replay_material import MaterialProtocol


class MaterialPlannerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.agent = NS(acfg=NS(code=NS(model="gpt-5.6-sol", temp=0)), cfg=NS(),
                        global_memory=NS(retrieve_similar_records=Mock(return_value=[])))
        self.prompt = {
            "Introduction": "Improve the model.", "Task description": "Predict toxicity.",
            "Memory": "Prior result.",
            "Instructions": {"Implementation guideline": [
                "Use MPS float32.", "Keep split, seed, metric and budget fixed."]},
            "Previous solution": {"Code": "x = 1\n"},
        }
        self.context = {"execution_output": "score .5"}
        self.plan = {"reason": "Sampling needs improvement.", "module": ["training_evaluation"],
                     "plan": {"training_evaluation": "Change training sampling."}}
        self.components = {"training_evaluation": "Training and evaluation."}

    def enable(self, arm):
        text = "" if arm == "A" else "MATERIAL_START\n" + arm * 11972 + "\nMATERIAL_END"
        self.agent.replay_material = MaterialProtocol(self.root / arm, text, arm, "x = 1\n")
        return text

    def decision_plan(self, decision):
        result = deepcopy(self.plan)
        result["material_decision"] = {
            "decision": decision,
            "source_quotes": ["MATERIAL_START"] if decision == "adopt" else [],
            "reason": "This addresses the observed sampling bottleneck." if decision == "adopt"
                      else "No actionable task evidence in the material." if decision == "reject"
                      else "No supplementary material was supplied.",
            "adaptation": "Change the sampling logic." if decision == "adopt" else "",
            "target_locations": ["train: sampler"] if decision == "adopt" else [],
        }
        return result

    def run_stage(self, stage):
        if stage == "direct":
            return direct.run_planner(self.agent, deepcopy(self.prompt), "tiny data", self.context,
                                      "Plan the improvement.", "Planning suffix.")
        if stage == "initial":
            return memory.generate_initial_plan(self.agent, deepcopy(self.prompt), "tiny data", self.context)
        if stage == "refine_history":
            self.agent.global_memory.retrieve_similar_records.side_effect = [
                [(NS(description="Previous plan", method="Previous method"), .7)], []]
        return memory.refine_plan_to_json(self.agent, "Initial proposal", deepcopy(self.prompt),
                                          "tiny data", self.context)

    def test_legacy_prompt_and_schema_fingerprints_are_unchanged(self):
        expected = {
            "direct": "4617fec63d53d22430e8b3f63fd663827f590fd5b1be23ded9aa2e7b48a7e1c9",
            "initial": "5f8a639454a6f99253bf256ba18440b608b377688488db215fc4328d3f663b05",
            "refine": "6ec7cc02062993f4c9646bad8548154abc00af76fd8efa760b2c3b610804c618",
            "refine_history": "71c02361eb7119d02ae7ff990288e25dbd1e1be0e48379a072618f324a546507",
        }
        for stage, wanted in expected.items():
            with self.subTest(stage=stage):
                module = direct if stage == "direct" else memory
                response = "Initial proposal" if stage == "initial" else self.plan
                with patch.object(module, "get_component_descriptions", return_value=self.components), \
                        patch.object(module, "generate", return_value=response) as generate:
                    self.run_stage(stage)
                captured = {key: generate.call_args.kwargs[key] for key in ("prompt", "json_schema")}
                self.assertEqual(hashlib.sha256(json.dumps(captured, sort_keys=True).encode()).hexdigest(), wanted)

    def test_four_arms_share_templates_and_schemas_with_complete_material(self):
        for stage in ("direct", "initial", "refine", "refine_history"):
            templates, wrappers, schemas = [], [], []
            module = direct if stage == "direct" else memory
            for arm in "ABCD":
                with self.subTest(stage=stage, arm=arm):
                    text = self.enable(arm)
                    response = ("Initial proposal" if stage == "initial" else
                                self.decision_plan("no_material" if arm == "A" else "reject"))
                    with patch.object(module, "get_component_descriptions", return_value=self.components), \
                            patch.object(module, "generate", return_value=response) as generate:
                        result = self.run_stage(stage)
                    generate.assert_called_once()
                    prompt = generate.call_args.kwargs["prompt"]
                    if text:
                        self.assertEqual(prompt["user"].count(text), 1)
                        self.assertIn("MATERIAL_END", prompt["user"])
                    self.assertIn("Use MPS float32.", prompt["user"])
                    self.assertIn("Keep split, seed, metric and budget fixed.", prompt["user"])
                    section = self.agent.replay_material.section("planner_" + stage.replace("_history", ""))
                    self.assertTrue(prompt["user"].endswith(section))
                    templates.append({**prompt, "user": prompt["user"][:-len(section)]})
                    wrappers.append(section.replace(f"<replay_material>\n{text}\n</replay_material>",
                                                    "<replay_material>\n[FROZEN BODY]\n</replay_material>"))
                    schemas.append(generate.call_args.kwargs["json_schema"])
                    material = self.agent.replay_material
                    recorded = json.loads((material.directory / material.audit["calls"][-1]).read_text())
                    self.assertEqual(recorded["prompt"], prompt)
                    self.assertEqual(recorded["schema"], schemas[-1])
                    self.assertEqual(recorded["response"], response)
                    if stage == "initial":
                        self.assertEqual(result, "Initial proposal")
                        self.assertIsNone(schemas[-1])
                    else:
                        self.assertEqual(result["material_decision"], response["material_decision"])
                        self.assertIn("material_decision", schemas[-1]["required"])
                    if stage.startswith("refine"):
                        self.assertNotIn("DO NOT deviate", str(prompt))
                        self.assertNotIn("direction closely", str(prompt))
            self.assertTrue(all(value == templates[0] for value in templates))
            self.assertTrue(all(value == wrappers[0] for value in wrappers))
            self.assertTrue(all(value == schemas[0] for value in schemas))

    def test_adoption_retains_exact_quote_and_implementation_intent(self):
        self.enable("D")
        for stage in ("direct", "refine"):
            module = direct if stage == "direct" else memory
            response = self.decision_plan("adopt")
            with patch.object(module, "get_component_descriptions", return_value=self.components), \
                    patch.object(module, "generate", return_value=response):
                result = self.run_stage(stage)
            self.assertEqual(result["material_decision"], response["material_decision"])

    def test_invalid_decision_retries_and_exhaustion_never_returns_fallback(self):
        self.enable("C")
        for stage in ("direct", "refine"):
            module = direct if stage == "direct" else memory
            invalid = self.decision_plan("adopt")
            invalid["material_decision"]["source_quotes"] = ["not present in the material"]
            valid = self.decision_plan("reject")
            with patch.object(module, "get_component_descriptions", return_value=self.components), \
                    patch.object(module, "generate", side_effect=[invalid, valid]) as generate:
                self.assertEqual(self.run_stage(stage)["material_decision"], valid["material_decision"])
                self.assertEqual(generate.call_count, 2)
            for response in (invalid, "not JSON", self.plan):
                with self.subTest(stage=stage, response=response), \
                        patch.object(module, "get_component_descriptions", return_value=self.components), \
                        patch.object(module, "generate", return_value=response) as generate:
                    with self.assertRaisesRegex(RuntimeError, "Material planner failed"):
                        self.run_stage(stage)
                    self.assertEqual(generate.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
