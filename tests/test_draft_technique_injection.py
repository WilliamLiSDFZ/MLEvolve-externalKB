"""Exercise real draft prompt construction without SDKs, execution, or network."""

import importlib
import importlib.util
import json
import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
COMPLETION = "Use a held-out validation split.\n```python\nscore = 0.5\nprint(score)\n```"


def package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    return module


@contextmanager
def draft_modules():
    """Stub external interfaces while importing the actual compiler and coders."""
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name.split(".")[0] in {"agents", "engine", "llm", "utils"}:
                del sys.modules[name]
        llm = package("llm", ROOT / "llm")
        llm.generate = Mock(return_value=COMPLETION)
        genai = SimpleNamespace(Client=object, types=SimpleNamespace(Tool=object))
        sys.modules.update({
            "agents": package("agents", ROOT / "agents"),
            "engine": package("engine", ROOT / "engine"),
            "utils": package("utils", ROOT / "utils"), "llm": llm,
            "backoff": SimpleNamespace(on_predicate=lambda **kw: lambda fn: fn, constant=object()),
            "jsonschema": SimpleNamespace(),
            "dataclasses_json": SimpleNamespace(DataClassJsonMixin=object),
            "funcy": SimpleNamespace(once=lambda fn: fn, notnone=lambda v: v is not None,
                                     select_values=lambda fn, values: values),
            "google": SimpleNamespace(genai=genai), "google.genai": genai,
            "config": SimpleNamespace(Config=object),
            "black": SimpleNamespace(format_str=lambda code, mode: code, FileMode=lambda: None,
                                     parsing=SimpleNamespace(InvalidInput=ValueError)),
            "humanize": SimpleNamespace(naturaldelta=lambda seconds: f"{seconds} seconds"),
            "engine.search_node": SimpleNamespace(
                SearchNode=lambda **kw: SimpleNamespace(**kw, id="test-node", branch_id=0)),
            "agents.triggers": SimpleNamespace(register_node=Mock(), get_patience_counter=Mock()),
        })
        spec = importlib.util.spec_from_file_location("_test_prompt_compiler", ROOT / "llm/gemini.py")
        compiler = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = compiler
        spec.loader.exec_module(compiler)
        llm.compile_prompt_to_md = compiler.compile_prompt_to_md
        yield SimpleNamespace(
            draft=importlib.import_module("agents.draft_agent"),
            improve=importlib.import_module("agents.improve_agent"),
            stepwise=importlib.import_module("agents.coder.stepwise_coder"),
            reranker=importlib.import_module("engine.coldstart.technique_reranker"),
            generate=llm.generate,
        )


def agent_fixture(modules, stepwise=False, enabled=True):
    records = [{
        "technique_id": f"selected-{i:02}", "name": f"Technique {i}", "paper_title": "Source paper",
        "paper_finding": "A source finding.", "task_relevance": "Relevant to toxicity AUC.",
        "task_adaptation": "Use the existing training labels.",
        "proposed_experiment": f"Compare experimental intervention {i} to the baseline.",
        "requirements_and_limits": "Requires training labels.",
    } for i in range(25)]
    acfg = SimpleNamespace(code=SimpleNamespace(model="gpt-5.6-terra", temp=0),
                           time_limit=43200, steps=100)
    lock = threading.Lock()
    root = SimpleNamespace(expected_child_count=0, fetch_child_memory=lambda: "Earlier draft used a baseline.")

    def add_child():
        with lock:
            root.expected_child_count += 1

    root.add_expected_child_count = add_child
    return SimpleNamespace(
        task_desc="Predict toxicity and maximize AUC.", data_preview="train.csv contains subtype labels.",
        use_coldstart=False, coldstart_description="None model", methodology_text="",
        draft_techniques=modules.reranker.render_techniques(records) if enabled else "",
        use_stepwise_generation=stepwise, virtual_root=root, acfg=acfg,
        start_time=time.time(), current_step=0,
        cfg=SimpleNamespace(agent=acfg, exec=SimpleNamespace(timeout=3600),
                            draft_techniques=SimpleNamespace(context_window=258400),
                            coldstart=SimpleNamespace(inject_into_improve=True)),
    )


def prompt_text(prompt):
    return "\n".join(prompt.values()) if isinstance(prompt, dict) else prompt


class DraftTechniqueInjectionTests(unittest.TestCase):
    def test_draft_guideline_obeys_the_outer_deadline(self):
        with draft_modules() as modules:
            agent = agent_fixture(modules)
            agent.start_time = 100
            guideline = importlib.import_module("agents.prompts.impl_guideline")
            with patch.object(guideline.time, "time", return_value=500), \
                    patch.dict(os.environ, {"MLEVOLVE_RUN_DEADLINE": "600"}), \
                    patch.object(guideline, "get_impl_guideline", return_value={}) as build:
                guideline.get_impl_guideline_from_agent(agent)
            self.assertEqual(build.call_args.kwargs["tot_time_remaining"], 100)
            self.assertEqual(build.call_args.kwargs["exec_timeout"], 100)

    def test_all_selected_techniques_reach_three_initial_and_concurrent_later_drafts(self):
        for stepwise in (False, True):
            with self.subTest(stepwise=stepwise), draft_modules() as modules:
                agent = agent_fixture(modules, stepwise=stepwise)
                selected_text = agent.draft_techniques
                with patch.object(modules.reranker, "rerank_techniques",
                                  side_effect=AssertionError("Draft must reuse startup ranking")) as rerank:
                    nodes = [modules.draft.run(agent) for _ in range(3)]
                    barrier = threading.Barrier(3)

                    def concurrent_completion(**kwargs):
                        barrier.wait(timeout=5)
                        return COMPLETION

                    modules.generate.side_effect = concurrent_completion
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        nodes.extend(pool.map(lambda _: modules.draft.run(agent), range(3)))
                    rerank.assert_not_called()
                self.assertEqual(len(nodes), 6)
                self.assertEqual(agent.virtual_root.expected_child_count, 6)
                self.assertEqual(modules.generate.call_count, 24 if stepwise else 6)
                self.assertEqual(agent.draft_techniques, selected_text)
                self.assertTrue(all(node.stage == "draft" and node.code for node in nodes))
                for call in modules.generate.call_args_list:
                    text = prompt_text(call.kwargs["prompt"])
                    self.assertIn(selected_text, text)
                    self.assertIn(agent.task_desc, text)
                    self.assertIn(agent.data_preview, text)
                    for i in range(25):
                        self.assertEqual(text.count(f'"technique_id": "selected-{i:02}"'), 1)
                if stepwise:
                    first_draft = [prompt_text(call.kwargs["prompt"])
                                   for call in modules.generate.call_args_list[:4]]
                    self.assertIn("# Current step: data_processing_and_feature_engineering", first_draft[0])
                    self.assertIn("# Current step: model_design", first_draft[1])
                    self.assertIn("# Current step: training_evaluation", first_draft[2])
                    self.assertIn("# Step results", first_draft[3])

    def test_disabled_injection_leaves_both_generation_paths_without_candidates(self):
        for stepwise in (False, True):
            with self.subTest(stepwise=stepwise), draft_modules() as modules:
                agent = agent_fixture(modules, stepwise=stepwise, enabled=False)
                modules.draft.run(agent)
                self.assertEqual(modules.generate.call_count, 4 if stepwise else 1)
                for call in modules.generate.call_args_list:
                    text = prompt_text(call.kwargs["prompt"])
                    self.assertNotIn("Ranked task-specific techniques", text)
                    self.assertNotIn("selected-00", text)

    def test_improve_ignores_draft_candidates_even_when_old_injection_is_enabled(self):
        with draft_modules() as modules:
            agent = agent_fixture(modules)
            agent.use_coldstart = True
            prompt = {"Instructions": {"Original instruction": "Keep this instruction."}}
            before = json.dumps(prompt)
            modules.improve._inject_methodology(agent, prompt, SimpleNamespace(id="parent"))
            self.assertEqual(json.dumps(prompt), before)
            self.assertNotIn("selected-00", json.dumps(prompt))

    def test_single_shot_rejects_oversize_prompt_before_model_call(self):
        with draft_modules() as modules:
            agent = agent_fixture(modules)
            agent.cfg.draft_techniques.context_window = 40000
            agent.data_preview = "Large dataset preview " * 3000
            with self.assertRaisesRegex(ValueError, "nothing was truncated"):
                modules.draft.run(agent)
            modules.generate.assert_not_called()

    def test_stepwise_checks_added_previous_code_before_next_model_call(self):
        with draft_modules() as modules:
            agent = agent_fixture(modules, stepwise=True)
            agent.cfg.draft_techniques.context_window = 100000
            modules.generate.return_value = "Plan.\n```python\n# " + "x" * 100000 + "\npass\n```"
            with self.assertRaisesRegex(ValueError, "nothing was truncated"):
                modules.draft.run(agent)
            self.assertEqual(modules.generate.call_count, 1)

    def test_merge_checks_all_step_code_before_model_call(self):
        with draft_modules() as modules:
            agent = agent_fixture(modules, stepwise=True)
            agent.cfg.draft_techniques.context_window = 100000
            base = {"Task description": agent.task_desc, "Instructions": {
                "Ranked task-specific techniques": agent.draft_techniques}}
            results = [{"name": "model_design", "plan": "Plan", "code": "# " + "x" * 100000}]
            with self.assertRaisesRegex(ValueError, "nothing was truncated"):
                modules.stepwise.MetaAgent().merge(
                    agent.task_desc, agent.data_preview, results, base, agent,
                    modules.stepwise.StepwiseContext(stage="draft"))
            modules.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
