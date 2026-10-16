import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from corpus_retriever.technique_agent import CorpusTechniqueAgent


def chunk(chunk_id, content):
    return SimpleNamespace(id=chunk_id, content=content, paper_id="paper",
                           paper_title="Paper", abstract="Abstract",
                           start_char=0, end_char=len(content))


def action(kind="finish", chunk_id=None, techniques=None, notes=""):
    return {"action": kind, "chunk_id": chunk_id, "notes": notes,
            "techniques": techniques or []}


def technique(chunk_id="a", quote="Use validation."):
    return {
        "name": "Grouped validation", "paper_finding": "Use grouped validation.",
        "task_relevance": "Training rows share subjects.",
        "task_adaptation": "Split by subject in this task.",
        "proposed_experiment": "Compare grouped and random validation.",
        "requirements_and_limits": "Requires subject IDs.",
        "evidence": [{"chunk_id": chunk_id, "quote": quote}],
        "paper_reported_result": None,
    }


class FakeAgent(CorpusTechniqueAgent):
    def __init__(self, actions, **kwargs):
        super().__init__(None, **kwargs)
        self.actions = iter(actions)
        self.prompts = []

    def _query(self, prompt):
        self.prompts.append(json.loads(prompt))
        return next(self.actions)


class TechniqueAgentTests(unittest.TestCase):
    def test_all_selected_are_read_even_after_early_finish(self):
        chunks = [chunk("a", "Use validation."), chunk("b", "More methods.")]
        found = technique()
        agent = FakeAgent([action(techniques=[found]), action(techniques=[found])])
        result = agent.extract(chunks, chunks, "Predict labels", {"columns": ["subject"]})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["read_chunk_ids"], ["a", "b"])
        self.assertEqual(agent.prompts[1]["techniques"], [found])
        self.assertEqual(agent.prompts[0]["task_description"], "Predict labels")
        self.assertEqual(agent.prompts[0]["available_data"], {"columns": ["subject"]})
        self.assertNotIn("Use validation.", agent.prompts[1]["current_chunk"]["content"])

    def test_reader_can_reach_unselected_evidence(self):
        chunks = [chunk("a", "See the appendix."), chunk("b", "Use  validation.\n")]
        agent = FakeAgent([action("read_full_text", "b"),
                           action(techniques=[technique("b")])])
        result = agent.extract(chunks, chunks[:1], "Task", "Data")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["read_chunk_ids"], ["a", "b"])
        self.assertEqual(result["trace"][0]["requested_chunk_id"], "b")

    def test_reader_cannot_access_another_paper(self):
        chunks = [chunk("a", "Text")]
        agent = FakeAgent([action("read_full_text", "other-paper")])
        result = agent.extract(chunks, chunks, "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("outside the current paper", result["errors"][0])

    def test_reader_can_choose_selected_order_without_losing_coverage(self):
        chunks = [chunk("a", "Text A"), chunk("b", "Text B"), chunk("c", "Text C")]
        agent = FakeAgent([action("read_full_text", "c"), action(), action()],
                          max_extra_reads=0)
        result = agent.extract(chunks, chunks, "Task", "Data")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["read_chunk_ids"], ["a", "c", "b"])

    def test_unread_or_fabricated_quotes_are_not_successful(self):
        chunks = [chunk("a", "Use validation."), chunk("b", "Use validation.")]
        for candidate in [technique("b"), technique("a", "Invented result")]:
            with self.subTest(candidate=candidate):
                result = FakeAgent([action(techniques=[candidate])]).extract(
                    chunks, chunks[:1], "Task", "Data")
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["techniques"], [])
                self.assertTrue(result["errors"])

    def test_valid_empty_result_is_distinct_from_malformed_output(self):
        chunks = [chunk("a", "No applicable findings.")]
        good = FakeAgent([action()]).extract(chunks, chunks, "Task", "Data")
        bad = FakeAgent([{}]).extract(chunks, chunks, "Task", "Data")
        self.assertEqual(good["status"], "complete")
        self.assertEqual(good["techniques"], [])
        self.assertEqual(bad["status"], "incomplete")

    def test_read_budget_never_silently_truncates(self):
        chunks = [chunk("a", "Text"), chunk("b", "Other text")]
        agent = FakeAgent([action("read_full_text", "b")], max_extra_reads=0)
        result = agent.extract(chunks, chunks[:1], "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("call budget exhausted", result["errors"][0])
        self.assertEqual(result["read_chunk_ids"], ["a"])

    def test_input_and_notes_budgets_are_explicit_failures(self):
        chunks = [chunk("a", "Text")]
        agent = FakeAgent([], max_input_chars=10)
        result = agent.extract(chunks, chunks, "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(agent.prompts, [])
        result = FakeAgent([action(notes="long")], max_notes_chars=1).extract(
            chunks, chunks, "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        result = FakeAgent([action(techniques=[technique()])], max_result_chars=1).extract(
            chunks, chunks, "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("results exceed", result["errors"][0])

    def test_preserves_previous_valid_results_on_later_error(self):
        chunks = [chunk("a", "Use validation."), chunk("b", "Other text")]
        found = technique()
        agent = FakeAgent([action("next_selected", techniques=[found]), {}])
        result = agent.extract(chunks, chunks, "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["techniques"], [found])

    def test_auth_and_dependency_errors_abort_the_run(self):
        chunks = [chunk("a", "Text")]
        denied = RuntimeError("Invalid model credentials")
        denied.status_code = 401
        gemini_denied = RuntimeError("Gemini permission denied")
        gemini_denied.code = 403
        for failure, expected in [(denied, PermissionError),
                                  (gemini_denied, PermissionError),
                                  (PermissionError("Denied"), PermissionError),
                                  (ImportError("Missing provider SDK"), ImportError)]:
            with self.subTest(failure=failure):
                agent = FakeAgent([])
                with patch.object(agent, "_query", side_effect=failure):
                    with self.assertRaises(expected):
                        agent.extract(chunks, chunks, "Task", "Data")

    def test_mixed_papers_and_mismatched_selection_are_rejected(self):
        a, b = chunk("a", "Text"), chunk("b", "Other text")
        b.paper_id = "another paper"
        result = FakeAgent([]).extract([a, b], [a], "Task", "Data")
        self.assertEqual(result["status"], "incomplete")
        result = FakeAgent([]).extract([a], [chunk("a", "Changed")], "Task", "Data")
        self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
