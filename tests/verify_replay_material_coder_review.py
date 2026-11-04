"""Offline v2 coder/reviewer handoff: exact material, bounded retries and provenance."""

import copy
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import code_review_agent
from agents.coder.diff_coder import diff_generate
from agents.replay_material import MaterialProtocol, source_hash
from engine.search_node import SearchNode
from llm import compile_prompt_to_md


PARENT = "x = 1\n"
DIFF = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"


class MaterialCoderReviewTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def setup_arm(self, arm):
        body = "" if arm == "A" else "Use the sampled rows.\n" + (arm * 11950) + "\nEND OF MATERIAL"
        if arm in "BC":
            body += "." * (12000 - len(body))
        if arm == "D":
            body = "Use the sampled rows.\n" + ("Mechanism details.\n" * 12) + "END OF MATERIAL"
        material = MaterialProtocol(self.root / arm, body, arm, PARENT)
        decision = dict(decision="no_material" if arm == "A" else "adopt" if arm == "D" else "reject",
                        source_quotes=["Use the sampled rows."] if arm == "D" else [],
                        reason="The sampled rows address the diagnosed problem." if arm == "D" else
                               "No material was provided." if arm == "A" else "The material does not address this task.",
                        adaptation="Change the training update." if arm == "D" else "",
                        target_locations=["training update"] if arm == "D" else [])
        plan = dict(reason="Improve the training update.", module=["training_evaluation"],
                    plan={"training_evaluation": "Change x to 2."}, parse_success=True,
                    material_decision=decision)
        material.accept_plan(plan)
        stage = NS(model="gpt-5.6-sol", temp=0)
        agent = NS(acfg=NS(code=stage, use_diff_mode=True), task_desc="Synthetic task",
                   cfg=NS(candidate_runtime=NS(enabled=False)), replay_profile="mac-mps",
                   replay_material=material)
        return agent, plan, material

    def code(self, agent, plan):
        return diff_generate.diff_generate_and_apply(agent, plan, PARENT, "Synthetic data", "Output", "Implement plan")

    def calls(self, material):
        audit = json.loads((material.directory / "audit.json").read_text())
        return [json.loads((material.directory / path).read_text()) for path in audit["calls"]]

    def review(self, **kwargs):
        result = dict(needs_revision=False, reasoning="The code has no execution issue.", revised_code=None,
                      material_review=dict(verdict="consistent", reason="The code follows the frozen decision.",
                                           evidence=["x = 1"]))
        result.update(kwargs)
        return result

    def test_all_arms_deliver_complete_material_and_frozen_decision_on_regeneration(self):
        for arm in "ABCD":
            with self.subTest(arm=arm):
                agent, plan, material = self.setup_arm(arm)
                frozen = copy.deepcopy(plan)
                with patch.object(diff_generate, "generate", side_effect=["invalid diff", DIFF]) as generate:
                    saved_plan, code = self.code(agent, plan)
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(code.strip(), "x = 2")
                self.assertEqual(json.loads(saved_plan), frozen)
                self.assertEqual(material.audit["plan"], frozen)
                calls = self.calls(material)
                for call in calls:
                    user = call["prompt"]["user"]
                    self.assertEqual(user.count(material.section("coder")), 1)
                    if material.text:
                        self.assertEqual(user.count(material.text), 1)
                    self.assertEqual(call["material_sha256"], source_hash(material.text))
                self.assertEqual([c["response"] for c in calls], ["invalid diff", DIFF])
                self.assertEqual(material.audit["sources"]["generated"], source_hash(code))

    def test_explicit_conflicts_abort_initial_or_regenerated_calls_without_extra_generation(self):
        for responses in (["MATERIAL_CONFLICT: unavailable operation"],
                          ["invalid diff", "MATERIAL_CONFLICT: unavailable operation"]):
            with self.subTest(responses=responses):
                agent, plan, material = self.setup_arm("D")
                with patch.object(diff_generate, "generate", side_effect=responses) as generate:
                    with self.assertRaisesRegex(RuntimeError, "MATERIAL_CONFLICT: unavailable operation"):
                        self.code(agent, plan)
                self.assertEqual(generate.call_count, len(responses))
                self.assertNotIn("generated", material.audit["sources"])
                self.assertEqual(self.calls(material)[-1]["response"], responses[-1])

    def test_noop_and_exhausted_partial_diff_fail_instead_of_training_parent(self):
        agent, plan, material = self.setup_arm("A")
        with patch.object(diff_generate, "generate", return_value="no diff") as generate:
            with self.assertRaisesRegex(RuntimeError, "did not complete its diff"):
                self.code(agent, plan)
        self.assertEqual(generate.call_count, 3)
        self.assertNotIn("generated", material.audit["sources"])
        with patch.object(diff_generate, "generate", return_value=DIFF), \
                patch.object(diff_generate, "apply_diff_with_retry", return_value=("x = 2\n", 1, "incomplete block")):
            with self.assertRaisesRegex(RuntimeError, "incomplete block"):
                self.code(agent, plan)

    def test_legacy_noop_still_returns_parent(self):
        agent, plan, _ = self.setup_arm("A")
        del agent.replay_material
        del plan["material_decision"]
        with patch.object(diff_generate, "generate", return_value="no diff") as generate:
            _, code = self.code(agent, plan)
        self.assertEqual(code, PARENT)
        self.assertEqual(generate.call_count, 3)
        self.assertNotIn("Supplementary material", generate.call_args.kwargs["prompt"]["user"])

    def test_unmatched_complete_block_cannot_silently_accept_partial_plan(self):
        agent, plan, material = self.setup_arm("A")
        missing = "<<<<<<< SEARCH\nmissing = 1\n=======\nmissing = 2\n>>>>>>> REPLACE"
        with patch.object(diff_generate, "generate", side_effect=[DIFF + "\n" + missing, missing, missing]) as generate:
            with self.assertRaisesRegex(RuntimeError, "remaining patches"):
                self.code(agent, plan)
        self.assertEqual(generate.call_count, 3)
        self.assertNotIn("generated", material.audit["sources"])
        retry = generate.call_args_list[1].kwargs["prompt"]
        self.assertIn("1 of 2 requested blocks", retry["user"])
        self.assertIn("x = 2", retry["assistant"])

    def test_remaining_patch_can_complete_partial_diff_within_same_retry_limit(self):
        agent, plan, material = self.setup_arm("A")
        missing = "<<<<<<< SEARCH\nmissing = 1\n=======\nmissing = 2\n>>>>>>> REPLACE"
        remaining = "<<<<<<< SEARCH\nx = 2\n=======\nx = 3\n>>>>>>> REPLACE"
        with patch.object(diff_generate, "generate", side_effect=[DIFF + "\n" + missing, remaining]) as generate:
            _, code = self.code(agent, plan)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(code.strip(), "x = 3")
        self.assertEqual(material.audit["sources"]["generated"], source_hash(code))

    def test_legacy_partial_diff_keeps_original_acceptance_policy(self):
        agent, plan, _ = self.setup_arm("A")
        del agent.replay_material
        del plan["material_decision"]
        missing = "<<<<<<< SEARCH\nmissing = 1\n=======\nmissing = 2\n>>>>>>> REPLACE"
        with patch.object(diff_generate, "generate", return_value=DIFF + "\n" + missing) as generate:
            _, code = self.code(agent, plan)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(code.strip(), "x = 2")

    def test_all_arms_share_review_schema_and_receive_full_material(self):
        original_schema = copy.deepcopy(code_review_agent.CODE_REVIEW_SPEC.json_schema)
        schemas = []
        for arm in "ABCD":
            with self.subTest(arm=arm):
                agent, _, material = self.setup_arm(arm)
                node = SearchNode(code=PARENT, stage="improve")
                with patch.object(code_review_agent, "query", return_value=self.review()) as query:
                    self.assertEqual(code_review_agent.run(agent, node), PARENT)
                self.assertEqual(query.call_count, 1)
                call = self.calls(material)[0]
                prompt = compile_prompt_to_md(call["prompt"]["system"])
                self.assertEqual(prompt.count(material.section("review")), 1)
                if material.text:
                    self.assertEqual(prompt.count(material.text), 1)
                schemas.append(call["schema"])
                final = material.audit["material_review"]
                self.assertEqual(final["verdict"], "consistent")
                self.assertEqual(final["source_sha256"], source_hash(PARENT))
        self.assertTrue(all(schema == schemas[0] for schema in schemas))
        self.assertEqual(code_review_agent.CODE_REVIEW_SPEC.json_schema, original_schema)

    def test_review_patch_keeps_input_evidence_and_marks_final_source_uncertain(self):
        agent, _, material = self.setup_arm("D")
        node = SearchNode(code=PARENT, stage="improve")
        with patch.object(code_review_agent, "query", return_value=self.review(needs_revision=True, revised_code=DIFF)):
            code = code_review_agent.run(agent, node)
        self.assertEqual(code, "x = 2")
        self.assertEqual(material.audit["input_code_review"]["source_sha256"], source_hash(PARENT))
        self.assertEqual(material.audit["input_code_review"]["evidence"], ["x = 1"])
        self.assertEqual(material.audit["material_review"]["verdict"], "uncertain")
        self.assertEqual(material.audit["material_review"]["source_sha256"], source_hash(code))
        self.assertEqual((material.directory / "reviewed.py").read_text(), code)

    def test_material_mismatch_is_recorded_without_forcing_a_repair(self):
        agent, _, material = self.setup_arm("D")
        response = self.review(material_review=dict(verdict="inconsistent", reason="The promised adaptation is missing.",
                                                    evidence=["x = 1"]))
        with patch.object(code_review_agent, "query", return_value=response) as query:
            code = code_review_agent.run(agent, SearchNode(code=PARENT, stage="improve"))
        self.assertEqual(query.call_count, 1)
        self.assertEqual(code, PARENT)
        self.assertEqual(material.audit["material_review"]["verdict"], "inconsistent")

    def test_invalid_review_or_unapplied_fix_fails_after_same_three_attempts(self):
        invalid = [self.review(material_review=None), self.review(needs_revision="false"),
                   self.review(reasoning=""), self.review(revised_code=5),
                   self.review(material_review=dict(verdict="consistent", reason="Claimed implementation.",
                                                     evidence=["not in the code"])),
                   self.review(needs_revision=True, revised_code=None),
                   self.review(needs_revision=True, revised_code=DIFF.replace("x = 1", "missing = 1")),
                   self.review(needs_revision=True, revised_code="x = 2")]
        for response in invalid:
            with self.subTest(response=response):
                agent, _, material = self.setup_arm("D")
                with patch.object(code_review_agent, "query", return_value=response) as query, \
                        patch.object(code_review_agent.time, "sleep"):
                    with self.assertRaisesRegex(RuntimeError, "review failed after 3 attempts"):
                        code_review_agent.run(agent, SearchNode(code=PARENT, stage="improve"))
                self.assertEqual(query.call_count, 3)
                self.assertNotIn("reviewed", material.audit["sources"])


if __name__ == "__main__":
    logging.getLogger("MLEvolve").setLevel(logging.CRITICAL)
    unittest.main(verbosity=2)
