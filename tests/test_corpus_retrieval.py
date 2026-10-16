import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from corpus_retriever import retrieval as rt
from corpus_retriever.__main__ import main


class Tokenizer:
    is_fast = True
    name_or_path = "test-character-tokenizer"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def paper(number):
    return SimpleNamespace(id=f"p{number}", title=f"Paper {number}", abstract="abstract",
                           source=f"https://example.org/{number}", pdf_url="")


def screen(chunks, task, data):
    return chunks, [{"chunk_id": chunk.id, "status": "ok", "score": 0.9, "selected": True}
                    for chunk in chunks]


def extract(chunks, selected, task, data):
    return {"status": "complete", "read_chunk_ids": [chunk.id for chunk in chunks],
            "techniques": [{"name": "A concrete experiment", "paper_finding": "finding",
                "task_relevance": task, "task_adaptation": "adaptation",
                "proposed_experiment": "comparison", "requirements_and_limits": "constraints",
                "evidence": [{"chunk_id": selected[0].id, "quote": selected[0].content}],
                "paper_reported_result": None}], "errors": [], "trace": []}


class CorpusPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        public = self.root / "public"
        public.mkdir()
        (public / "train.csv").write_text("text,target\nexample,1\n", encoding="utf-8")
        index = self.root / "index"
        index.mkdir()
        (index / "manifest.json").write_text('{"count": 30}', encoding="utf-8")
        self.cfg = SimpleNamespace(exp_id="task-a", data_dir=public, desc_file=None,
            goal="classify task-specific text", eval="F1", abstract_index_path=index,
            lazy_pool=100, lazy_min_score=0.05, retr_alpha=0.5,
            agent=SimpleNamespace(code=SimpleNamespace(api_key="test", model="fake")),
            corpus_retriever=SimpleNamespace(jev_model="jev-1.13.0", jev_threshold=0.5,
                                             chunk_tokens=10000, chunk_overlap=200))
        self.output = self.root / "cache"
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"TYPESAFE_API_KEY": "test"}).start()
        self.search = patch.object(rt, "search_papers", return_value=(
            [(paper(1), 1.0)], Tokenizer(), "search query")).start()
        self.load = patch.object(rt, "load_full_text", return_value="A useful technique.").start()
        self.jev = patch.object(rt, "JevReranker").start().return_value
        self.jev.filter.side_effect = screen
        self.agent = patch.object(rt, "CorpusTechniqueAgent").start().return_value
        self.agent.extract.side_effect = extract

    def run_pipeline(self):
        return rt.run_retrieval(self.cfg, self.output)

    def test_rerun_replaces_the_same_task_without_stale_papers(self):
        self.run_pipeline()
        target = self.output / "task-a"
        stale = target / "papers" / rt.paper_key(paper(1))
        self.assertTrue(stale.exists())
        self.search.return_value = ([(paper(2), 0.8)], Tokenizer(), "new query")
        result = self.run_pipeline()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(stale.exists())
        self.assertEqual([p.name for p in self.output.iterdir() if p.is_dir()], ["task-a"])
        techniques = json.loads((target / "techniques.json").read_text())
        self.assertEqual([item["paper_id"] for item in techniques], ["p2"])
        self.assertEqual(self.load.call_count, 2)
        self.assertEqual(self.agent.extract.call_args.args[2], "classify task-specific text\nEvaluation: F1")
        self.assertIn("target", self.agent.extract.call_args.args[3])

    def test_all_selected_papers_extracted_without_old_twenty_paper_cap(self):
        self.search.return_value = ([(paper(i), 1.0) for i in range(23)], Tokenizer(), "query")
        result = self.run_pipeline()
        self.assertEqual(result["technique_count"], 23)
        self.assertEqual(self.agent.extract.call_count, 23)

    def test_late_fulltext_is_preserved_and_all_selected_chunks_reach_agent(self):
        self.load.return_value = "x" * 65000 + "LATE METHOD"
        self.run_pipeline()
        chunks, selected, _, _ = self.agent.extract.call_args.args
        self.assertIn("LATE METHOD", chunks[-1].content)
        self.assertEqual([c.id for c in chunks], [c.id for c in selected])
        saved = self.output / "task-a" / "papers" / rt.paper_key(paper(1)) / "fulltext.md"
        self.assertEqual(saved.read_text(), self.load.return_value)
        self.assertEqual(self.load.call_count, 1)

    def test_duplicate_paper_is_only_processed_once(self):
        self.search.return_value = ([(paper(1), 1.0), (paper(1), 0.8)], Tokenizer(), "query")
        result = self.run_pipeline()
        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(self.load.call_count, 1)

    def test_normal_rejection_does_not_fall_back_to_unfiltered_papers(self):
        self.jev.filter.side_effect = lambda chunks, *_: ([], [
            {"chunk_id": c.id, "status": "ok", "score": 0.1, "selected": False} for c in chunks])
        result = self.run_pipeline()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"], {"rejected": 1})
        self.agent.extract.assert_not_called()

    def test_partial_screening_does_not_publish_techniques_as_complete(self):
        self.load.return_value = "x" * 11000
        self.jev.filter.side_effect = lambda chunks, *_: (chunks[:1], [
            {"chunk_id": c.id, "status": "ok" if i == 0 else "error"}
            for i, c in enumerate(chunks)])
        result = self.run_pipeline()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["technique_count"], 0)
        self.assertEqual(result["counts"], {"incomplete": 1})
        self.assertTrue((self.output / "task-a" / "papers" / rt.paper_key(paper(1)) /
                         "techniques.json").exists())

    def test_auth_failure_preserves_previous_task_result(self):
        self.run_pipeline()
        path = self.output / "task-a" / "manifest.json"
        previous = path.read_bytes()
        self.jev.filter.side_effect = PermissionError("invalid credentials")
        with self.assertRaises(PermissionError):
            self.run_pipeline()
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual([p.name for p in self.output.iterdir() if p.is_dir()], ["task-a"])

    def test_failed_extraction_is_not_a_successful_empty_result(self):
        self.agent.extract.side_effect = lambda *_: {
            "status": "incomplete", "techniques": [], "errors": ["bad response"], "trace": []}
        result = self.run_pipeline()
        self.assertEqual(result["counts"], {"incomplete": 1})
        self.assertEqual(result["status"], "partial")

    def test_all_failed_rerun_preserves_previous_task(self):
        self.run_pipeline()
        path = self.output / "task-a" / "manifest.json"
        previous = path.read_bytes()
        self.load.side_effect = OSError("PDF server unavailable")
        with self.assertRaisesRegex(RuntimeError, "previous task results were preserved"):
            self.run_pipeline()
        self.assertEqual(path.read_bytes(), previous)

    def test_missing_dependency_is_a_fatal_error_not_a_rejected_paper(self):
        self.load.side_effect = ImportError("missing PDF reader")
        with self.assertRaises(ImportError):
            self.run_pipeline()
        self.assertFalse((self.output / "task-a").exists())

    def test_task_name_cannot_escape_output_root(self):
        self.cfg.exp_id = "../other-task"
        with self.assertRaises(ValueError):
            self.run_pipeline()
        self.search.assert_not_called()

    def test_fulltext_reader_does_not_truncate_at_old_character_limit(self):
        from engine.coldstart import ondemand
        fake_pdf = SimpleNamespace(to_markdown=Mock(return_value="a" * 65000 + "END"))
        with patch.dict("sys.modules", {"pymupdf4llm": fake_pdf}), \
             patch.object(ondemand, "_download", return_value=True):
            source = paper(1)
            source.pdf_url = "https://example.org/paper.pdf"
            self.assertTrue(self.real_loader(source).endswith("END"))

    real_loader = staticmethod(rt.load_full_text)


class RetrievalHelperTests(unittest.TestCase):
    def test_search_uses_one_hundred_and_the_existing_embedding_tokenizer(self):
        from engine.coldstart import ondemand
        token = Tokenizer()
        retriever = SimpleNamespace(
            embedding_model=SimpleNamespace(base=SimpleNamespace(model=SimpleNamespace(tokenizer=token))),
            search=Mock(return_value=[(paper(1), 1.0), (paper(2), 0.01)]))
        cfg = SimpleNamespace(abstract_index_path="unused", lazy_pool=100, retr_alpha=0.5,
                              lazy_min_score=0.05)
        with patch.object(ondemand, "_load_abstract_index", return_value=retriever), \
             patch.object(ondemand, "_build_query", return_value="query"):
            hits, actual, query = rt.search_papers("full task", cfg)
        retriever.search.assert_called_once_with("query", top_k=100, alpha=0.5)
        self.assertEqual(len(hits), 1)
        self.assertIs(actual, token)
        self.assertEqual(query, "query")

    def test_missing_ids_do_not_collide(self):
        self.assertNotEqual(rt.paper_key(SimpleNamespace(title="A")),
                            rt.paper_key(SimpleNamespace(title="B")))

    def test_help_requires_no_model_dependencies(self):
        with patch("builtins.print"):
            self.assertEqual(main(["--help"]), 0)


if __name__ == "__main__":
    unittest.main()
