"""Offline checks for control-arm input provenance, isolation and exact lengths."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import replay_inputs as inputs


class ReplayInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.records = [{"id": f"p{i}", "title": f"Paper {i}", "abstract": "ABSTRACT ONLY"}
                        for i in range(3)]
        (self.corpus / "records.jsonl").write_text("\n".join(map(json.dumps, self.records)))
        (self.corpus / "manifest.json").write_text('{"records_sha1": "corpus-hash"}')
        self.agent = NS(cfg=NS(analogy=NS(report_char_budget=12000, max_output_tokens=16384,
                                        corpus_path=str(self.corpus)),
                               agent=NS(code=NS(model="configured-model", temp=0.4))),
                        task_desc="TASK_MARKER")
        self.parent = NS(code="SECRET_PARENT_CODE", term_out="SECRET_PARENT_LOG")
        self.document = {"pdf_sha256": "pdf-hash", "text_sha256": "text-hash", "warnings": [],
                         "chunks": [{"text": "FULLTEXT_FIRST"}, {"text": "FULLTEXT_LAST"}]}

    def run_arm(self, arm):
        return inputs.prepare_input(arm, self.agent, self.parent, self.root / arm, 42)

    def test_b_raw_novel_exact_and_repeatable(self):
        raw = "CHAPTER I\nA story, without stripping headings.\n" * 500
        nltk = NS(data=NS(path=[]), download=Mock())
        reader = NS(raw=Mock(return_value=raw))
        with patch.dict(sys.modules, {"nltk": nltk, "nltk.corpus": NS(gutenberg=reader)}):
            text, report, metadata = self.run_arm("B")
            repeat = self.run_arm("B")
        self.assertEqual(len(text), 12000)
        self.assertEqual((text, report, metadata), repeat)
        self.assertEqual(text, raw[metadata["offset_chars"]:metadata["offset_chars"] + 12000])
        self.assertIn("CHAPTER I", text)
        self.assertIn(metadata["file_id"], inputs.NOVELS)
        nltk.download.assert_not_called()

    def test_b_missing_corpus_downloads_into_arm(self):
        nltk = NS(data=NS(path=[]), download=Mock())
        reader = NS(raw=Mock(side_effect=[LookupError(), "novel " * 3000]))
        with patch.dict(sys.modules, {"nltk": nltk, "nltk.corpus": NS(gutenberg=reader)}):
            self.run_arm("B")
        self.assertEqual(nltk.download.call_args.kwargs["download_dir"],
                         str(self.root / "B" / "nltk_data"))
        self.assertEqual(nltk.data.path, [str(self.root / "B" / "nltk_data")])

    def test_c_rejects_related_and_uses_whole_paper_without_parent(self):
        judgements = iter([True, False])
        suggestions = iter(["### Mechanism\n" + "a" * 7000, "### Implementation\n" + "b" * 6000])
        prompts = []

        def ask(agent, system, user, schema=None):
            prompts.append((system, user, schema))
            self.assertIn("FULLTEXT_FIRST\n\nFULLTEXT_LAST", user)
            self.assertNotIn("ABSTRACT ONLY", user)
            self.assertNotIn("SECRET_PARENT", user)
            if schema:
                self.assertIn("TASK_MARKER", user)
                return {"related": next(judgements), "reason": "judgement evidence"}
            self.assertNotIn("TASK_MARKER", user)
            self.assertIn("Do not diagnose", system)
            return next(suggestions)

        with patch("engine.analogy.fulltext.FullTextStore") as store, patch.object(inputs, "_ask", ask):
            store.return_value.get.return_value = self.document, False
            text, report, metadata = self.run_arm("C")
        self.assertEqual(len(text), 12000)
        self.assertIsNone(report)
        self.assertEqual(metadata["suggestion_calls"], 2)
        self.assertGreater(metadata["truncated_chars"], 0)
        draws = [c.args[0]["id"] for c in store.return_value.get.call_args_list]
        self.assertEqual(len(set(draws)), 2)
        attempts = json.loads((self.root / "C" / "paper_attempts.json").read_text())
        self.assertEqual([a["status"] for a in attempts], ["related", "unrelated"])
        self.assertIn("FULLTEXT_LAST", (self.root / "C" / "paper.txt").read_text())
        self.assertTrue((self.root / "C" / "suggestions_2.md").exists())
        self.assertEqual(len(prompts), 4)

    def test_c_short_suggestions_fail_without_filler(self):
        with patch("engine.analogy.fulltext.FullTextStore") as store, patch.object(inputs, "_ask") as ask:
            store.return_value.get.return_value = self.document, False
            ask.side_effect = [{"related": False, "reason": "different subject"}, "first", "second", "third"]
            with self.assertRaisesRegex(RuntimeError, "refusing to pad C"):
                self.run_arm("C")
        self.assertFalse((self.root / "C" / "injection.txt").exists())
        self.assertTrue((self.root / "C" / "suggestions_3.md").exists())

    def test_c_all_related_fails_and_records_each_draw(self):
        with patch("engine.analogy.fulltext.FullTextStore") as store, patch.object(inputs, "_ask") as ask:
            store.return_value.get.return_value = self.document, False
            ask.return_value = {"related": True, "reason": "relevant"}
            with self.assertRaisesRegex(RuntimeError, "3 random draws"):
                self.run_arm("C")
        attempts = json.loads((self.root / "C" / "paper_attempts.json").read_text())
        self.assertEqual(len({a["paper_id"] for a in attempts}), 3)

    def test_c_repeated_continuation_is_not_used_as_padding(self):
        with patch("engine.analogy.fulltext.FullTextStore") as store, patch.object(inputs, "_ask") as ask:
            store.return_value.get.return_value = self.document, False
            ask.side_effect = [{"related": False, "reason": "different subject"}, "suggestion", "suggestion"]
            with self.assertRaisesRegex(ValueError, "repeated existing text"):
                self.run_arm("C")
        self.assertFalse((self.root / "C" / "injection.txt").exists())

    def test_c_missing_pages_are_skipped_before_llm(self):
        self.document["warnings"] = ["Pages without extracted text: [2]"]
        with patch("engine.analogy.fulltext.FullTextStore") as store, patch.object(inputs, "_ask") as ask:
            store.return_value.get.return_value = self.document, False
            with self.assertRaisesRegex(RuntimeError, "No readable unrelated paper"):
                self.run_arm("C")
        ask.assert_not_called()
        attempts = json.loads((self.root / "C" / "paper_attempts.json").read_text())
        self.assertTrue(all(a["status"] == "unavailable" for a in attempts))

    def test_oversized_request_is_never_sent_or_truncated(self):
        llm = NS(FunctionSpec=Mock(), query=Mock())
        with patch.dict(sys.modules, {"llm": llm}), \
                patch("engine.analogy.observed_loop.ContextBudget") as budget:
            budget.return_value.limit = 100
            budget.return_value.estimate.return_value = 101
            with self.assertRaisesRegex(ValueError, "complete paper.*exceed"):
                inputs._ask(self.agent, "system", "whole paper")
        llm.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
