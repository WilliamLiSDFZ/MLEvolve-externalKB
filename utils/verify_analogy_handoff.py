"""CPU regressions for explicit analogy selection, coder handoff and provenance.

Run: python utils/verify_analogy_handoff.py
All model calls are mocked. No cluster or live API calls are made.
"""

from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.analogy_handoff import (
    CONTEXT_KEY, adoption_from_plan, attach_report, planning_instruction, planning_schema,
    record_child_handoff, selected_mechanism_brief,
)


def report():
    return dict(context_version=2, observed_facts=["read training objective"], unknowns=["loss curve unavailable"],
        mechanisms=[dict(mechanism_id="m1", name="sampled pair ranking", intervention="add current-positive ranking loss",
                         assumptions=["cached negatives are detached"], constraints=["preserve fixed split and sigmoid export"],
                         target_fit="ranking metric", validation_plan="compare public AUC at equal steps",
                         rejection_criterion="reject if gradients rank positive below negative",
                         code_refs=[dict(node_id="parent", source_sha256="a" * 64, start_line=20, end_line=40)],
                         evidence_refs=[dict(paper_id="paper1", source="fulltext", quote="sampled negative comparison")],
                         evidence_limitations="No matched compute comparison yet"),
                    dict(mechanism_id="m2", name="other method", constraints=["do not combine with m1"])])


def plan(**updates):
    result = dict(reason="Use pair ranking", module=["training_evaluation"],
                  plan={"training_evaluation": "Add detached negative ranking with correct sign"},
                  selected_mechanism_id="m1", analogy_decision_reason="Matches observed pairwise objective gap",
                  analogy_adaptation="Keep BCE and add current-positive minus cached-negative term", parse_success=True)
    result.update(updates)
    return result


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {"Task description": "Task", "Introduction": "Improve", "Instructions": {}, "Memory": "",
                       "Previous solution": {"Code": "x = 1\n"}}
        attach_report(self.prompt, report())
        self.agent = SimpleNamespace(acfg=SimpleNamespace(code=SimpleNamespace(model="gpt-6-astra", temp=0.3),
                                                         use_global_memory=False), cfg=SimpleNamespace(),
                                     global_memory=Mock())

    def test_schema_is_optional_v2_only_and_does_not_mutate_base(self):
        from agents.planner.base_planner import PLANNING_JSON_SCHEMA
        original = deepcopy(PLANNING_JSON_SCHEMA)
        self.assertIs(planning_schema(PLANNING_JSON_SCHEMA, {}), PLANNING_JSON_SCHEMA)
        changed = planning_schema(PLANNING_JSON_SCHEMA, self.prompt)
        self.assertEqual(changed["properties"]["selected_mechanism_id"]["enum"], [None, "m1", "m2"])
        self.assertNotIn("selected_mechanism_id", changed["required"])
        self.assertEqual(PLANNING_JSON_SCHEMA, original)

    def test_declared_selection_rejection_and_unknown_are_distinct(self):
        selected = adoption_from_plan(self.prompt, plan())
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(selected["selected_mechanism_id"], "m1")
        rejected = adoption_from_plan(self.prompt, plan(selected_mechanism_id=None))
        self.assertEqual(rejected["status"], "rejected")
        for value in (plan(selected_mechanism_id=["m1", "m2"]), plan(selected_mechanism_id="unknown"),
                      plan(parse_success=False), plan(analogy_adaptation=""), "I adopt sampled pair ranking"):
            self.assertEqual(adoption_from_plan(self.prompt, value)["status"], "unknown")
        self.assertIsNone(adoption_from_plan({}, plan()))

    def test_full_rewrite_requires_explicit_json_not_name_matching(self):
        text = "ANALOGY_ADOPTION: " + json.dumps(plan()) + "\nNow update the training loop."
        self.assertEqual(adoption_from_plan(self.prompt, text)["status"], "selected")
        self.assertEqual(adoption_from_plan(self.prompt, "Use m1 sampled ranking loss")["status"], "unknown")

    def test_coder_brief_preserves_full_selected_constraints_and_evidence(self):
        selected = adoption_from_plan(self.prompt, plan())
        brief = selected_mechanism_brief(selected)
        for field in ("constraints", "assumptions", "code_refs", "evidence_refs", "evidence_limitations", "validation_plan", "rejection_criterion"):
            self.assertIn(field, brief)
        self.assertIn("current-positive minus cached-negative", brief)
        self.assertNotIn("other method", brief)
        self.assertEqual(selected_mechanism_brief(adoption_from_plan(self.prompt, plan(selected_mechanism_id=None))), "")

    def test_prompt_reports_are_isolated_for_parallel_children(self):
        original = report()
        first, second = {}, {}
        attach_report(first, original)
        attach_report(second, original)
        first[CONTEXT_KEY]["mechanisms"][0]["constraints"].append("first child only")
        self.assertNotIn("first child only", str(second))
        self.assertNotIn("first child only", str(original))

    def test_direct_planner_receives_optional_schema_and_explicit_id_request(self):
        from agents.planner import base_planner
        with patch.object(base_planner, "get_component_descriptions", return_value={"training_evaluation": "Training"}), \
             patch.object(base_planner, "generate", return_value=json.dumps(plan())) as generate:
            result = base_planner.run_planner(self.agent, self.prompt, "data", {}, "Plan", "Suffix")
        kwargs = generate.call_args.kwargs
        self.assertIn("selected_mechanism_id", kwargs["json_schema"]["properties"])
        self.assertIn("Offered mechanism IDs: m1, m2", kwargs["prompt"]["user"])
        self.assertEqual(adoption_from_plan(self.prompt, result)["status"], "selected")

    def test_memory_refinement_retains_original_mechanism_not_only_initial_plan(self):
        from agents.planner import planner_with_memory
        self.agent.global_memory.retrieve_similar_records.return_value = []
        with patch.object(planner_with_memory, "get_component_descriptions", return_value={"training_evaluation": "Training"}), \
             patch.object(planner_with_memory, "generate", return_value=json.dumps(plan())) as generate:
            result = planner_with_memory.refine_plan_to_json(self.agent, "Short initial plan", self.prompt, "data", {})
        kwargs = generate.call_args.kwargs
        self.assertIn("selected_mechanism_id", kwargs["json_schema"]["properties"])
        self.assertIn("reject if gradients rank positive below negative", kwargs["prompt"]["user"])
        self.assertIn("sampled negative comparison", kwargs["prompt"]["user"])
        self.assertEqual(adoption_from_plan(self.prompt, result)["status"], "selected")

    def test_diff_regeneration_keeps_mechanism_brief(self):
        from agents.coder.diff_coder import diff_generate
        brief = selected_mechanism_brief(adoption_from_plan(self.prompt, plan()))
        final_diff = "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n"
        with patch.object(diff_generate, "generate", side_effect=["No valid patch", final_diff]) as generate:
            _, code = diff_generate.diff_generate_and_apply(self.agent, plan(), "x = 1\n", "data", "output",
                                                            "Improve", analogy_mechanism_brief=brief)
        self.assertIn("x = 2", code)
        self.assertEqual(generate.call_count, 2)
        for call in generate.call_args_list:
            self.assertIn(brief, call.kwargs["prompt"]["user"])

    def test_improve_diff_forwards_selected_mechanism_separately(self):
        from agents import improve_agent
        parent = SimpleNamespace(code="x = 1\n", term_out="output", code_summary="summary")
        with patch.object(improve_agent, "run_planner", return_value=plan()), \
             patch.object(improve_agent, "diff_generate_and_apply", return_value=("plan", "code")) as coder:
            improve_agent._diff_improve(self.agent, self.prompt, "data", parent)
        self.assertIn("preserve fixed split", coder.call_args.kwargs["analogy_mechanism_brief"])
        self.assertEqual(coder.call_args.kwargs["planning_result"]["analogy_adoption"]["status"], "selected")

    def test_v2_injection_keeps_string_contract_and_no_parent_or_agent_state(self):
        from agents import improve_agent
        from engine.analogy import agent as retrieval
        from config import AnalogyConfig
        from engine.analogy.context import ContextOptions
        parent = SimpleNamespace(id="parent")
        # Use the real nested config contract, not an invented top-level field.
        self.agent.cfg.analogy = AnalogyConfig(enabled=True, improve=True, context=ContextOptions(version=2))
        injected = {"Instructions": {}}
        returned = SimpleNamespace(report_md="MECHANISM REPORT", report=report())
        with patch.object(retrieval, "retrieve_for_node", return_value=returned) as retrieve:
            text = improve_agent._inject_analogy(self.agent, injected, parent)
        self.assertEqual(text, "MECHANISM REPORT")
        self.assertEqual(retrieve.call_args.kwargs, {"with_result": True})
        self.assertIn(CONTEXT_KEY, injected)
        self.assertEqual(adoption_from_plan(injected, plan())["status"], "selected")
        self.assertIn("selected_mechanism_id", planning_schema({"properties": {}}, injected)["properties"])
        self.assertNotIn(CONTEXT_KEY, vars(self.agent))
        self.assertEqual(vars(parent), {"id": "parent"})
        with patch.object(retrieval, "retrieve_for_node", return_value=SimpleNamespace(report_md="", report=report())):
            empty = {"Instructions": {}}
            self.assertEqual(improve_agent._inject_analogy(self.agent, empty, parent), "")
            self.assertNotIn(CONTEXT_KEY, empty)

    def test_legacy_config_without_context_keeps_v1_retrieval_contract(self):
        from agents import improve_agent
        from engine.analogy import agent as retrieval
        self.agent.cfg.analogy = SimpleNamespace(enabled=True, improve=True)
        prompt = {"Instructions": {}}
        parent = SimpleNamespace(id="parent")
        with patch.object(retrieval, "retrieve_for_node", return_value="LEGACY REPORT") as retrieve:
            self.assertEqual(improve_agent._inject_analogy(self.agent, prompt, parent), "LEGACY REPORT")
        self.assertEqual(retrieve.call_args.kwargs, {})
        self.assertNotIn(CONTEXT_KEY, prompt)

    def test_child_provenance_refreshes_after_review_and_execution(self):
        from engine.search_node import SearchNode
        from engine.executor import ExecutionResult
        with tempfile.TemporaryDirectory() as tmp:
            self.agent.cfg.log_dir = Path(tmp)
            parent = SearchNode(code="x = 1\n", stage="draft")
            child = SearchNode(code="x = 2\n", plan=json.dumps(plan()), parent=parent, stage="improve")
            record_child_handoff(self.agent, child, self.prompt, generation_mode="diff")
            self.assertEqual(child.analogy_adoption["code_provenance"]["phase"], "generated_before_code_review")
            child.code = "x = 3\n"  # A later code-review fix must be reflected in provenance.
            child.absorb_exec_result(ExecutionResult(["result"], 123, "TimeoutError", execution_status="timeout"))
            provenance = child.analogy_adoption["code_provenance"]
            self.assertEqual(provenance["phase"], "executed_source")
            self.assertEqual(provenance["child_source_sha256"], hashlib.sha256(child.code.encode()).hexdigest())
            self.assertEqual(child.analogy_adoption["execution_outcome"]["status"], "timeout")
            artifact = json.loads(Path(child.analogy_adoption["artifact_path"]).read_text())
            self.assertIn("+x = 3", artifact["actual_diff"])
            self.assertEqual(artifact["outcome_reference"]["node_id"], child.id)
            self.assertIn("analogy_adoption", {f.name for f in fields(SearchNode)})
            from engine.search_node import Journal
            from utils import serialize
            journal = Journal()
            journal.append(SearchNode(code="x = 3\n", stage="draft", analogy_adoption=child.analogy_adoption))
            saved = json.loads(serialize.dumps_json(journal))
            self.assertEqual(saved["nodes"][0]["analogy_adoption"]["status"], "selected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
