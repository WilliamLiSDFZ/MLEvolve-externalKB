"""Offline regressions for paired replay inputs and frozen prompt budgets.

Run: python tests/verify_replay_injection.py
Generation is mocked; no API, corpus, training or GPU is needed.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import improve_agent
from agents.analogy_handoff import CONTEXT_KEY
from agents.prompts import impl_guideline
from engine.search_node import SearchNode


class ReplayInjectionTests(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(
            acfg=SimpleNamespace(time_limit=1000, steps=10, use_diff_mode=False,
                                 use_global_memory=False,
                                 code=SimpleNamespace(model="gpt-5.6-sol")),
            cfg=SimpleNamespace(exec=SimpleNamespace(timeout=600)),
            current_step=3, start_time=0, replay_budget_seconds=300,
            task_desc="Predict the target", data_preview="Two numeric features",
        )
        self.retrieval = self._patch("_inject_analogy", return_value="legacy report")
        self._patch("get_patience_counter", return_value=(0, 0, 0.5))
        self.register = self._patch("register_node")
        self.rewrite = self._patch("plan_and_code_query", return_value=("plan", "x = 2"))
        self.planner = self._patch("run_planner", return_value={
            "module": ["training"], "plan": {"training": "Improve training"},
            "parse_success": True,
        })
        self.coder = self._patch("diff_generate_and_apply", return_value=("plan", "x = 2"))

    def _patch(self, name, **kwargs):
        patcher = patch.object(improve_agent, name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def parent(self):
        return SearchNode(code="x = 1", stage="draft", _term_out=["score: 0.5"])

    def test_arm_a_empty_text_skips_retrieval_and_injection(self):
        node = improve_agent.run(self.agent, self.parent(), injection_text="")
        self.retrieval.assert_not_called()
        self.assertIsNone(node.analogy_report)
        self.assertNotIn("Supplementary material", str(self.rewrite.call_args.args[1]))

    def test_default_none_keeps_existing_analogy_path(self):
        parent = self.parent()
        node = improve_agent.run(self.agent, parent)
        self.retrieval.assert_called_once()
        self.assertIs(self.retrieval.call_args.args[0], self.agent)
        self.assertIs(self.retrieval.call_args.args[2], parent)
        self.assertEqual(node.analogy_report, "legacy report")

    def test_each_text_reaches_full_rewrite_and_saved_prompt(self):
        for arm in "BCD":
            with self.subTest(arm=arm):
                text = f"Unique material for arm {arm}"
                node = improve_agent.run(self.agent, self.parent(), injection_text=text)
                prompt = self.rewrite.call_args.args[1]
                self.assertIn(text, str(prompt))
                self.assertIn("Supplementary material", str(prompt))
                self.assertEqual(self.register.call_args.args[2], prompt)
                self.assertEqual(node.analogy_report, text)
                self.assertIsNone(node.analogy_adoption)
        self.retrieval.assert_not_called()

    def test_each_text_reaches_diff_planner_with_identical_wrapper(self):
        self.agent.acfg.use_diff_mode = True
        wrappers = []
        for arm in "BCD":
            with self.subTest(arm=arm):
                text = f"Unique material for arm {arm}"
                node = improve_agent.run(self.agent, self.parent(), injection_text=text)
                prompt = self.planner.call_args.kwargs["prompt_base"]
                section = prompt["Instructions"]["Supplementary material"]
                self.assertEqual(section[-1], text)
                wrappers.append(section[:-1])
                self.assertNotIn(CONTEXT_KEY, prompt)
                self.assertNotIn("Explicit analogy selection", prompt["Instructions"])
                self.assertEqual(self.coder.call_args.kwargs["analogy_mechanism_brief"], "")
                self.assertEqual(node.analogy_report, text)
                self.assertIsNone(node.analogy_adoption)
        self.assertEqual(wrappers, [wrappers[0]] * 3)
        self.retrieval.assert_not_called()
        self.rewrite.assert_not_called()

    def test_diff_failure_preserves_text_in_full_rewrite_fallback(self):
        self.agent.acfg.use_diff_mode = True
        self.planner.side_effect = RuntimeError("planner failed")
        improve_agent.run(self.agent, self.parent(), injection_text="fallback material")
        self.assertIn("fallback material", str(self.rewrite.call_args.args[1]))
        self.retrieval.assert_not_called()

    def test_replay_prompt_budget_does_not_depend_on_clock_or_deadline(self):
        self.agent.runtime_deadline = 500
        with patch.object(impl_guideline.time, "time", return_value=100):
            first = impl_guideline.get_impl_guideline_from_agent(self.agent)
        with patch.object(impl_guideline.time, "time", return_value=10000):
            later = impl_guideline.get_impl_guideline_from_agent(self.agent)
        self.assertEqual(first, later)
        self.assertIn("Time left ≈ 0h 5m 0s", first["Implementation guideline"][0])

    def test_normal_prompt_budget_still_observes_runtime_deadline(self):
        del self.agent.replay_budget_seconds
        self.agent.runtime_deadline = 700
        with patch.object(impl_guideline.time, "time", return_value=500):
            result = impl_guideline.get_impl_guideline_from_agent(self.agent)
        self.assertIn("Time left ≈ 0h 3m 20s", result["Implementation guideline"][0])


if __name__ == "__main__":
    unittest.main()
