"""Offline material-protocol boundaries, opt-in isolation and paired replay wiring."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from agents import improve_agent
from agents.replay_material import MaterialProtocol
from engine.analogy.agent import AnalogyResult
from engine.search_node import SearchNode
from tests import verify_replay_macos as mac_tests
from utils.metric import MetricValue
from utils.replay_macos import runtime as mac
import replay_improve as replay


def material_text(arm):
    if arm == "A":
        return ""
    text = f"START_{arm}: consider balancing training groups.\n"
    if arm in "BC":
        text += "." * (12000 - len(text) - len(f"END_{arm}"))
    return text + f"END_{arm}"


def plan_for(arm, decision=None):
    decision = decision or ("no_material" if arm == "A" else "reject")
    return {
        "reason": "Improve generalization while preserving the fixed validation contract.",
        "module": ["training_evaluation"],
        "plan": {"training_evaluation": "Change training group sampling with a fixed seed."},
        "parse_success": True,
        "material_decision": {
            "decision": decision,
            "source_quotes": ["consider balancing training groups."] if decision == "adopt" else [],
            "reason": "The evidence supports this sampling change." if decision == "adopt" else
                      "The supplied material does not justify a change to this model." if decision == "reject" else
                      "There is no supplementary material to assess.",
            "adaptation": "Balance the training groups in the sampler." if decision == "adopt" else "",
            "target_locations": ["training sampler"] if decision == "adopt" else [],
        },
    }


class MaterialBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent_code = "x = 1\n"

    def material(self, arm="D", text=None):
        return MaterialProtocol(self.root / arm, material_text(arm) if text is None else text,
                                arm, self.parent_code)

    def test_input_budget_and_missing_material_fail_closed(self):
        for arm, text in (("A", "unexpected"), ("B", ""), ("C", ""), ("D", ""),
                          ("D", " " * 20), ("D", "x" * 12001), ("B", "x" * 11999),
                          ("C", "x" * 12001), ("X", "unknown")):
            with self.subTest(arm=arm, length=len(text)), self.assertRaises(ValueError):
                self.material(arm, text)

    def test_four_arms_share_schema_and_carry_complete_body_once(self):
        base = {"type": "object", "properties": {}, "required": []}
        schemas = []
        for arm in "ABCD":
            protocol = self.material(arm)
            schemas.append(protocol.plan_schema(base))
            protocol.accept_plan(plan_for(arm))
            for stage in ("planner_initial", "planner_direct", "planner_refine", "coder", "review"):
                with self.subTest(arm=arm, stage=stage):
                    section = protocol.section(stage)
                    if arm != "A":
                        self.assertEqual(section.count(material_text(arm)), 1)
                        self.assertIn(f"START_{arm}", section)
                        self.assertIn(f"END_{arm}", section)
        self.assertTrue(all(schema == schemas[0] for schema in schemas))
        self.assertEqual(base, {"type": "object", "properties": {}, "required": []})

    def test_decision_requires_reason_and_exact_source_evidence(self):
        protocol = self.material()
        valid = plan_for("D", "adopt")
        protocol.validate_plan(valid)
        invalid = []
        for field, value in (("reason", ""), ("source_quotes", ["not present in material"]),
                             ("source_quotes", []), ("adaptation", ""), ("target_locations", []),
                             ("decision", "no_material"), ("decision", "maybe")):
            case = copy.deepcopy(valid)
            case["material_decision"][field] = value
            invalid.append(case)
        case = copy.deepcopy(valid)
        del case["material_decision"]
        invalid.append(case)
        case = copy.deepcopy(valid)
        case["plan"] = {}
        invalid.append(case)
        for case in invalid:
            with self.subTest(decision=case.get("material_decision")), self.assertRaises(ValueError):
                protocol.validate_plan(case)
        for arm in "BCD":
            self.material(arm).validate_plan(plan_for(arm))

    def test_empty_control_only_allows_no_material(self):
        protocol = self.material("A")
        protocol.validate_plan(plan_for("A"))
        for decision in ("adopt", "reject"):
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                protocol.validate_plan(plan_for("A", decision))

    def test_accepted_decision_is_a_snapshot_of_the_plan(self):
        protocol = self.material()
        plan = plan_for("D", "adopt")
        protocol.accept_plan(plan)
        before = protocol.section("coder")
        plan["material_decision"]["reason"] = "silently changed after acceptance"
        plan["material_decision"]["source_quotes"].clear()
        self.assertEqual(protocol.section("coder"), before)
        self.assertNotIn("silently changed", before)

    def test_actual_prompt_schema_response_and_failure_are_persisted(self):
        protocol = self.material()
        prompt = {"system": "Plan the change", "user": protocol.section("planner_direct")}
        schema = protocol.plan_schema({"type": "object", "properties": {}, "required": []})
        response = plan_for("D", "reject")

        def generate(**kwargs):
            in_flight = json.loads((protocol.directory / "calls/01_planner_direct.json").read_text())
            self.assertEqual(in_flight["status"], "started")
            self.assertEqual(in_flight["prompt"], prompt)
            return response

        self.assertEqual(protocol.call("planner_direct", generate, prompt=prompt, json_schema=schema), response)
        first = json.loads((protocol.directory / "calls/01_planner_direct.json").read_text())
        self.assertEqual(first["schema"], schema)
        self.assertEqual(first["response"], response)
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["material_sha256"], hashlib.sha256(material_text("D").encode()).hexdigest())
        error = RuntimeError("synthetic transport failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic transport failure"):
            protocol.call("planner_direct", Mock(side_effect=error), prompt=prompt, json_schema=schema)
        protocol.fail(error)
        failed = json.loads((protocol.directory / "calls/02_planner_direct.json").read_text())
        self.assertEqual(failed["prompt"], prompt)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("synthetic transport failure", failed["error"])
        audit = json.loads((protocol.directory / "audit.json").read_text())
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(len(audit["calls"]), 2)

    def test_review_evidence_and_final_assessment_are_bound_to_source(self):
        protocol = self.material()
        protocol.accept_plan(plan_for("D", "adopt"))
        code = "x = 2\n"
        response = {"material_review": {
            "verdict": "consistent", "reason": "The requested change is present.", "evidence": ["x = 2"],
        }}
        protocol.record_code("generated", code)
        protocol.record_review(response, code)
        protocol.finish_review(code)
        self.assertEqual(protocol.audit["material_review"]["verdict"], "consistent")
        self.assertEqual(protocol.audit["material_review"]["source_sha256"], hashlib.sha256(code.encode()).hexdigest())
        changed = "x = 3\n"
        protocol.finish_review(changed)
        self.assertEqual(protocol.audit["input_code_review"]["verdict"], "consistent")
        self.assertEqual(protocol.audit["material_review"]["verdict"], "uncertain")
        self.assertEqual(protocol.audit["material_review"]["source_sha256"], hashlib.sha256(changed.encode()).hexdigest())
        self.assertEqual((protocol.directory / "reviewed.py").read_text(), changed)
        self.assertIn("+x = 3", (protocol.directory / "reviewed.diff").read_text())

    def test_invented_code_evidence_or_unsupported_consistency_is_rejected(self):
        protocol = self.material()
        protocol.accept_plan(plan_for("D", "adopt"))
        for evidence in ([], ["x = 2"]):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                protocol.record_review({"material_review": {
                    "verdict": "consistent", "reason": "Claimed adoption.", "evidence": evidence,
                }}, "x = 1\n")


class ImproveMaterialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = SearchNode(code="x = 1\n", stage="improve", branch_id=1,
                                 metric=MetricValue(.5, maximize=True), _term_out=["score: 0.5"])
        self.material = MaterialProtocol(Path(self.temp.name), material_text("D"), "D", self.parent.code)
        self.agent = SimpleNamespace(
            acfg=SimpleNamespace(time_limit=1000, steps=10, use_diff_mode=True,
                                 use_global_memory=False, code=SimpleNamespace(model="gpt-5.6-sol")),
            cfg=SimpleNamespace(exec=SimpleNamespace(timeout=600)),
            current_step=3, start_time=0, replay_budget_seconds=300,
            replay_profile="mac-mps", replay_material=self.material,
            task_desc="Predict the target", data_preview="Two numeric features",
        )
        self.enterContext(patch.object(improve_agent, "get_impl_guideline_from_agent", return_value={}))
        self.enterContext(patch.object(improve_agent, "register_node"))
        self.enterContext(patch.object(improve_agent, "record_child_handoff"))
        self.retrieve = self.enterContext(patch.object(improve_agent, "_inject_analogy"))
        self.rewrite = self.enterContext(patch.object(improve_agent, "plan_and_code_query"))

    def test_v2_does_not_fallback_to_untraced_full_rewrite(self):
        with patch.object(improve_agent, "_diff_improve", side_effect=RuntimeError("planner failed")):
            with self.assertRaisesRegex(RuntimeError, "planner failed"):
                improve_agent.run(self.agent, self.parent, injection_text=material_text("D"))
        self.rewrite.assert_not_called()
        self.retrieve.assert_not_called()

    def test_protocol_failure_never_reaches_review_or_candidate_execution(self):
        self.agent.executor = Mock()
        with patch.object(improve_agent, "_diff_improve", side_effect=RuntimeError("protocol failed")), \
                patch("agents.code_review_agent.run") as review:
            with self.assertRaisesRegex(RuntimeError, "protocol failed"):
                replay.execute_one(self.agent, self.parent, material_text("D"))
            review.assert_not_called()
        self.agent.executor.run.assert_not_called()
        self.rewrite.assert_not_called()

    def test_changed_injection_cannot_bypass_frozen_body(self):
        with patch.object(improve_agent, "_diff_improve") as generate:
            with self.assertRaises((ValueError, RuntimeError)):
                improve_agent.run(self.agent, self.parent, injection_text="replacement material")
            generate.assert_not_called()
        self.rewrite.assert_not_called()

    def test_final_plan_is_accepted_before_coder_without_hidden_report(self):
        plan = plan_for("D", "adopt")

        def coder(**kwargs):
            self.assertEqual(kwargs["planning_result"]["material_decision"], plan["material_decision"])
            self.assertEqual(kwargs["analogy_mechanism_brief"], "")
            self.assertIn(plan["material_decision"]["reason"], self.material.section("coder"))
            return json.dumps(plan), "x = 2\n"

        with patch.object(improve_agent, "run_planner", return_value=plan) as planner, \
                patch.object(improve_agent, "diff_generate_and_apply", side_effect=coder):
            child = improve_agent.run(self.agent, self.parent, injection_text=material_text("D"))
        self.assertEqual(child.code, "x = 2\n")
        self.assertNotIn("Supplementary material", planner.call_args.kwargs["prompt_base"]["Instructions"])
        self.retrieve.assert_not_called()
        self.rewrite.assert_not_called()


class ReplayMaterialPairingTests(unittest.TestCase):
    setUp = mac_tests.MacReplayTests.setUp
    save_manifest = mac_tests.MacReplayTests.save_manifest

    def args(self, arm="A"):
        args = mac_tests.MacReplayTests.args(self, arm)
        args.material_protocol = "v2"
        return args

    def test_v2_requires_explicit_mac_before_runtime_or_outputs(self):
        args = self.args()
        args.profile = None
        with patch.object(replay, "_run") as run, patch.object(mac, "require_mps") as probe:
            with self.assertRaises(ValueError):
                replay.run(args)
            run.assert_not_called()
            probe.assert_not_called()

    def test_v2_rejects_full_rewrite_before_arm_replacement(self):
        cfg = replay.load_config(self.cfg_path)
        cfg.agent.use_diff_mode = False
        OmegaConf.save(cfg, self.cfg_path)
        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", return_value=(self.manifest, self.node)), \
                patch.object(replay, "prepare_arm") as prepare:
            with self.assertRaisesRegex(ValueError, "diff"):
                replay.run(self.args())
            prepare.assert_not_called()

    def test_protocol_failure_is_saved_with_decision_and_never_trains(self):
        def execute(agent, parent, text):
            agent.replay_material.accept_plan(plan_for("D", "adopt"))
            raise RuntimeError("MATERIAL_CONFLICT: sampler cannot preserve the fixed split")

        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", return_value=(self.manifest, self.node)), \
                patch("engine.candidate_runtime.jigsaw.load_contract", return_value=self.contract), \
                patch("engine.candidate_runtime.integration.prepare"), \
                patch("engine.executor.Interpreter") as interpreter, \
                patch("engine.analogy.agent.retrieve_for_node", return_value=AnalogyResult(report_md=material_text("D"))), \
                patch.object(replay, "execute_one", side_effect=execute):
            with self.assertRaisesRegex(RuntimeError, "MATERIAL_CONFLICT"):
                replay.run(self.args("D"))
            interpreter.return_value.run.assert_not_called()
        arm_dir = self.args().output_root / "replay_paired/D"
        result = json.loads((arm_dir / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["material_decision"], plan_for("D", "adopt")["material_decision"])
        audit = json.loads((arm_dir / result["material_handoff"]).read_text())
        self.assertEqual(audit["status"], "failed")
        self.assertIn("MATERIAL_CONFLICT", audit["error"])

    def test_four_arms_share_common_identity_with_individual_material_hashes(self):
        observed = []

        def prepare_input(arm, agent, parent, output_dir, seed):
            text = material_text(arm)
            (output_dir / "injection.txt").write_text(text)
            return text, None, {}

        def execute(agent, parent, text):
            self.assertIsInstance(agent.replay_material, MaterialProtocol)
            observed.append((parent.id, agent.replay_budget_seconds, hashlib.sha256(text.encode()).hexdigest()))
            agent.replay_material.accept_plan(plan_for(self.current_arm))
            agent.replay_material.record_code("generated", "x = 2\n")
            agent.replay_material.record_review({"material_review": {
                "verdict": "consistent", "reason": "The code follows the declared independent change.",
                "evidence": ["x = 2"],
            }}, "x = 2\n")
            agent.replay_material.finish_review("x = 2\n")
            return SearchNode(code="x = 2\n", id="child" + self.current_arm,
                              stage="improve", parent=parent, metric=MetricValue(.75, maximize=True),
                              is_valid=True, is_buggy=False)

        with patch.object(mac, "require_mps", return_value=self.environment), \
                patch.object(mac, "load_baseline", return_value=(self.manifest, self.node)), \
                patch("engine.candidate_runtime.jigsaw.load_contract", return_value=self.contract), \
                patch("engine.candidate_runtime.integration.prepare"), \
                patch("engine.executor.Interpreter"), \
                patch("utils.replay_inputs.prepare_input", side_effect=prepare_input), \
                patch("engine.analogy.agent.retrieve_for_node", return_value=AnalogyResult(report_md=material_text("D"))), \
                patch.object(replay, "execute_one", side_effect=execute):
            for arm in "ABCD":
                self.current_arm = arm
                result = replay.run(self.args(arm))
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["injection_sha256"], hashlib.sha256(material_text(arm).encode()).hexdigest())
                self.assertEqual(result["material_decision"], plan_for(arm)["material_decision"])
                self.assertEqual(result["material_review"]["verdict"], "consistent")
                self.assertEqual(result["material_protocol"], "v2")
        identity = json.loads((self.args().output_root / "replay_paired/experiment.json").read_text())
        self.assertEqual(identity["material_protocol"], "v2")
        self.assertNotIn("material_sha256", identity)
        self.assertNotIn("injection_sha256", identity)
        self.assertEqual({(parent, budget) for parent, budget, _ in observed}, {("macparent", self.budget)})
        self.assertEqual(len({digest for _, _, digest in observed}), 4)
        self.assertEqual((self.source / "logs/journal.json").read_bytes(), self.before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
