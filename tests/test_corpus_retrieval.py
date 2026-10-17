import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, patch

from corpus_retriever import retrieval as rt
from corpus_retriever import technique_deduplicator
from corpus_retriever.__main__ import main
from corpus_retriever.technique_agent import CorpusTechniqueAgent


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


def keep_all(techniques, cfg):
    return techniques, {"status": "complete", "input_count": len(techniques),
                        "output_count": len(techniques), "groups": [
                            {"keep_id": item["technique_id"], "member_ids": [item["technique_id"]],
                             "reason": "Distinct experiment"} for item in techniques]}


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
        self.load = patch.object(rt, "load_full_text",
                                 side_effect=lambda paper, parser_pool: f"A useful technique for {paper.id}.").start()
        self.parser_factory = patch.object(rt, "ProcessPoolExecutor").start()
        self.parser_pool = self.parser_factory.return_value.__enter__.return_value
        self.jev = patch.object(rt, "JevReranker").start().return_value
        self.jev.filter.side_effect = screen
        self.agent = patch.object(rt, "CorpusTechniqueAgent").start().return_value
        self.agent.extract.side_effect = extract
        self.deduplicate = patch.object(rt, "deduplicate_techniques", side_effect=keep_all).start()

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
        self.load.side_effect = None
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

    def test_simultaneous_duplicate_fulltexts_only_screen_and_extract_once(self):
        self.search.return_value = ([(paper(i), 1.0) for i in range(1, 4)], Tokenizer(), "query")
        self.load.side_effect = None
        self.load.return_value = "The same paper under different corpus identifiers."
        prepared = Barrier(3)
        split = rt.split_paper

        def prepare_together(*args, **kwargs):
            chunks = split(*args, **kwargs)
            prepared.wait(timeout=5)
            return chunks

        with patch.object(rt, "split_paper", side_effect=prepare_together):
            result = self.run_pipeline()
        self.assertEqual(self.load.call_count, 3)
        self.jev.filter.assert_called_once()
        self.agent.extract.assert_called_once()
        self.assertEqual(result["counts"], {"complete": 1, "duplicate": 2})
        representative = next(record for record in result["papers"] if record["status"] == "complete")
        for duplicate in (record for record in result["papers"] if record["status"] == "duplicate"):
            self.assertEqual(duplicate["duplicate_of"], representative["paper_key"])
            self.assertEqual(duplicate["fulltext_sha256"], representative["fulltext_sha256"])
            saved = self.output / "task-a" / "papers" / duplicate["paper_key"]
            self.assertEqual(json.loads((saved / "paper.json").read_text()), duplicate)
            self.assertFalse((saved / "fulltext.md").exists())
            self.assertFalse((saved / "screening.json").exists())
            self.assertFalse((saved / "techniques.json").exists())

    def test_failed_chunk_preparation_does_not_mark_fulltext_as_processed(self):
        self.search.return_value = ([(paper(1), 1.0), (paper(2), 0.8)], Tokenizer(), "query")
        failed = Event()
        split = rt.split_paper

        def load(source, parser_pool):
            if source.id == "p2":
                self.assertTrue(failed.wait(5))
            return "The same full text."

        def fail_first(text, paper_id, *args, **kwargs):
            if paper_id == "p1":
                failed.set()
                raise ValueError("bad chunk preparation")
            return split(text, paper_id, *args, **kwargs)

        self.load.side_effect = load
        with patch.object(rt, "split_paper", side_effect=fail_first):
            result = self.run_pipeline()
        self.assertEqual(result["counts"], {"error": 1, "complete": 1})
        self.jev.filter.assert_called_once()
        self.assertEqual(self.agent.extract.call_args.args[0][0].paper_id, "p2")

    def test_llm_dedup_retains_original_record_and_all_per_paper_outputs(self):
        self.search.return_value = ([(paper(1), 1.0), (paper(2), 0.8)], Tokenizer(), "query")
        self.deduplicate.side_effect = technique_deduplicator.deduplicate_techniques
        first_id, second_id = (f"{rt.paper_key(paper(i))}:0000" for i in (1, 2))

        def answer(prompt, cfg):
            items = json.loads(prompt)["techniques"]
            self.assertEqual([item["technique_id"] for item in items], [first_id, second_id])
            return {"groups": [{"keep_id": second_id, "member_ids": [first_id, second_id],
                                "reason": "Both propose the same comparison"}]}

        with patch.object(technique_deduplicator, "_query", side_effect=answer):
            result = self.run_pipeline()
        self.assertEqual(result["raw_technique_count"], 2)
        self.assertEqual(result["technique_count"], 1)
        self.assertEqual(result["status"], "complete")
        target = self.output / "task-a"
        saved = json.loads((target / "techniques.json").read_text())
        originals = self.deduplicate.call_args.args[0]
        self.assertEqual(saved, [originals[1]])
        self.assertEqual(saved[0]["technique_id"], second_id)
        for i in (1, 2):
            raw = json.loads((target / "papers" / rt.paper_key(paper(i)) / "techniques.json").read_text())
            self.assertEqual(len(raw["techniques"]), 1)
            self.assertNotIn("technique_id", raw["techniques"][0])
        audit = json.loads((target / "deduplication.json").read_text())
        self.assertEqual(audit["groups"][0]["member_ids"], [first_id, second_id])
        self.assertEqual(result["deduplication"],
                         {"status": "complete", "input_count": 2, "output_count": 1})
        prompts = json.loads((target / "prompts.json").read_text())
        self.assertEqual(prompts["technique_deduplication"], technique_deduplicator.SYSTEM_PROMPT)

    def test_dedup_failure_keeps_all_techniques_and_marks_partial(self):
        self.search.return_value = ([(paper(1), 1.0), (paper(2), 0.8)], Tokenizer(), "query")
        for error in (ValueError("invalid groups"), RuntimeError("service unavailable")):
            with self.subTest(error=error):
                self.deduplicate.side_effect = error
                result = self.run_pipeline()
                self.assertEqual(result["status"], "partial")
                self.assertEqual(result["raw_technique_count"], 2)
                self.assertEqual(result["technique_count"], 2)
                target = self.output / "task-a"
                saved = json.loads((target / "techniques.json").read_text())
                self.assertEqual([item["paper_id"] for item in saved], ["p1", "p2"])
                audit = json.loads((target / "deduplication.json").read_text())
                self.assertEqual(audit["status"], "incomplete")
                self.assertIn(str(error), audit["error"])

    def test_fatal_dedup_failure_preserves_previous_task_result(self):
        self.run_pipeline()
        target = self.output / "task-a"
        previous = (target / "manifest.json").read_bytes()
        for error in (PermissionError("invalid credentials"), ImportError("missing client")):
            with self.subTest(error=error):
                self.deduplicate.side_effect = error
                with self.assertRaises(type(error)):
                    self.run_pipeline()
                self.assertEqual((target / "manifest.json").read_bytes(), previous)
                self.assertEqual([p.name for p in self.output.iterdir() if p.is_dir()], ["task-a"])

    def test_three_workers_prepare_concurrently_with_private_reused_tokenizers(self):
        entered = {f"p{i}": Event() for i in range(1, 5)}
        extracting = {key: Event() for key in entered}
        release = {key: Event() for key in entered}
        finished = {key: Event() for key in entered}
        lock = Lock()
        active, peak = 0, 0
        completed, preparation_threads, tokenizers = [], {}, {}
        caller_threads, jev_clients, agents = [], [], []

        class RecordingTokenizer(Tokenizer):
            def __call__(self, text, **kwargs):
                with lock:
                    tokenizers.setdefault(get_ident(), set()).add(id(self))
                return super().__call__(text, **kwargs)

        original_tokenizer = RecordingTokenizer()
        self.search.return_value = (
            [(paper(i), 1.0) for i in range(1, 5)], original_tokenizer, "query")

        def load(source, parser_pool):
            nonlocal active, peak
            self.assertIs(parser_pool, self.parser_pool)
            with lock:
                active += 1
                peak = max(peak, active)
                preparation_threads[source.id] = get_ident()
            entered[source.id].set()
            if source.id == "p1":
                self.assertTrue(release["p1"].wait(5))
            return f"A useful technique for {source.id}."

        def extract_in_worker(chunks, selected, task, data):
            nonlocal active
            identity = chunks[0].paper_id
            self.assertEqual(get_ident(), preparation_threads[identity])
            extracting[identity].set()
            try:
                self.assertTrue(release[identity].wait(5), identity)
                result = extract(chunks, selected, task, data)
                with lock:
                    completed.append(identity)
                return result
            finally:
                with lock:
                    active -= 1
                finished[identity].set()

        def new_jev(*args, **kwargs):
            owner = get_ident()
            client = Mock()

            def filter_in_worker(*args):
                self.assertEqual(get_ident(), owner)
                return screen(*args)

            client.filter.side_effect = filter_in_worker
            jev_clients.append(client)
            return client

        def new_agent(*args, **kwargs):
            owner = get_ident()
            agent = Mock()

            def extract_with_owner(*args):
                self.assertEqual(get_ident(), owner)
                return extract_in_worker(*args)

            agent.extract.side_effect = extract_with_owner
            agents.append(agent)
            return agent

        def run():
            caller_threads.append(get_ident())
            return self.run_pipeline()

        self.load.side_effect = load
        with patch.object(rt, "JevReranker", side_effect=new_jev), \
             patch.object(rt, "CorpusTechniqueAgent", side_effect=new_agent), \
             ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(run)
            try:
                for identity in ("p1", "p2", "p3"):
                    self.assertTrue(entered[identity].wait(5), identity)
                for identity in ("p2", "p3"):
                    self.assertTrue(extracting[identity].wait(5), identity)
                self.assertFalse(extracting["p1"].is_set())
                self.assertFalse(entered["p4"].is_set())
                self.assertEqual(peak, 3)
                for identity in ("p3", "p4", "p2", "p1"):
                    self.assertTrue(entered[identity].wait(5), identity)
                    release[identity].set()
                    self.assertTrue(finished[identity].wait(5), identity)
            finally:
                for event in release.values():
                    event.set()
            result = future.result(timeout=5)

        self.assertEqual(completed, ["p3", "p4", "p2", "p1"])
        self.assertEqual(peak, 3)
        worker_threads = set(preparation_threads.values())
        self.assertEqual(len(worker_threads), 3)
        self.assertTrue(worker_threads.isdisjoint(caller_threads))
        self.assertEqual(set(tokenizers), worker_threads)
        self.assertTrue(all(len(copies) == 1 for copies in tokenizers.values()))
        copy_ids = set().union(*tokenizers.values())
        self.assertEqual(len(copy_ids), 3)
        self.assertNotIn(id(original_tokenizer), copy_ids)
        self.assertEqual(preparation_threads["p3"], preparation_threads["p4"])
        self.assertEqual(sum(client.filter.call_count for client in jev_clients), 4)
        self.assertEqual(sum(agent.extract.call_count for agent in agents), 4)
        self.assertEqual([record["paper_id"] for record in result["papers"]],
                         ["p1", "p2", "p3", "p4"])
        saved = json.loads((self.output / "task-a" / "techniques.json").read_text())
        self.assertEqual([item["paper_id"] for item in saved], ["p1", "p2", "p3", "p4"])
        options = self.parser_factory.call_args.kwargs
        self.assertEqual(options["max_workers"], 3)
        self.assertEqual(options["mp_context"].get_start_method(), "spawn")

    def test_one_worker_failure_does_not_discard_other_papers(self):
        self.search.return_value = ([(paper(i), 1.0) for i in range(1, 5)], Tokenizer(), "query")

        def fail_one(chunks, *args):
            if chunks[0].paper_id == "p2":
                raise RuntimeError("bad paper response")
            return extract(chunks, *args)

        self.agent.extract.side_effect = fail_one
        result = self.run_pipeline()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["counts"], {"complete": 3, "error": 1})
        self.assertEqual(result["papers"][1]["error"], "RuntimeError: bad paper response")
        saved = json.loads((self.output / "task-a" / "techniques.json").read_text())
        self.assertEqual([item["paper_id"] for item in saved], ["p1", "p3", "p4"])

    def test_fatal_worker_waits_for_running_papers_before_removing_staging(self):
        self.run_pipeline()
        target = self.output / "task-a"
        previous = (target / "manifest.json").read_bytes()
        self.search.return_value = ([(paper(i), 1.0) for i in range(1, 5)], Tokenizer(), "query")
        entered = {f"p{i}": Event() for i in range(1, 5)}
        fail, shutting_down, release = Event(), Event(), Event()
        completed = []

        class RecordingExecutor(ThreadPoolExecutor):
            def shutdown(self, *args, **kwargs):
                shutting_down.set()
                return super().shutdown(*args, **kwargs)

        def extract_in_worker(chunks, *args):
            identity = chunks[0].paper_id
            entered[identity].set()
            if identity == "p1":
                self.assertTrue(fail.wait(5))
                raise PermissionError("invalid credentials")
            self.assertTrue(release.wait(5))
            staging = [path for path in self.output.iterdir() if path.is_dir() and path != target]
            self.assertEqual(len(staging), 1)
            self.assertTrue((staging[0] / "papers" / rt.paper_key(paper(int(identity[1:]))) /
                             "fulltext.md").exists())
            completed.append(identity)
            return extract(chunks, *args)

        self.agent.extract.side_effect = extract_in_worker
        with patch.object(rt, "ThreadPoolExecutor", RecordingExecutor), \
             ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(self.run_pipeline)
            try:
                for identity in ("p1", "p2", "p3"):
                    self.assertTrue(entered[identity].wait(5), identity)
                fail.set()
                self.assertTrue(shutting_down.wait(5))
                self.assertFalse(future.done())
                self.assertFalse(entered["p4"].is_set())
            finally:
                fail.set()
                release.set()
            with self.assertRaises(PermissionError):
                future.result(timeout=5)

        self.assertEqual(set(completed), {"p2", "p3"})
        self.assertFalse(entered["p4"].is_set())
        self.assertEqual((target / "manifest.json").read_bytes(), previous)
        self.assertEqual([path.name for path in self.output.iterdir() if path.is_dir()], ["task-a"])

    def test_normal_rejection_does_not_fall_back_to_unfiltered_papers(self):
        self.jev.filter.side_effect = lambda chunks, *_: ([], [
            {"chunk_id": c.id, "status": "ok", "score": 0.1, "selected": False} for c in chunks])
        result = self.run_pipeline()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"], {"rejected": 1})
        self.agent.extract.assert_not_called()

    def test_partial_screening_does_not_publish_techniques_as_complete(self):
        self.load.side_effect = None
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

    def test_quote_warning_still_publishes_the_papers_techniques(self):
        agent = CorpusTechniqueAgent(self.cfg)
        self.agent.extract.side_effect = agent.extract

        def answer(prompt):
            current = json.loads(prompt)["current_chunk"]
            found = extract([], [SimpleNamespace(id=current["id"], content=current["content"])],
                            "Task", "Data")["techniques"]
            found[0]["evidence"][0]["quote"] = '"A useful technique."'
            return {"action": "finish", "chunk_id": None, "notes": "", "techniques": found}

        with patch.object(agent, "_query", side_effect=answer), \
             self.assertLogs("MLEvolve", level="WARNING"):
            result = self.run_pipeline()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["counts"], {"complete": 1})
        self.assertEqual(result["technique_count"], 1)
        saved = json.loads((self.output / "task-a" / "techniques.json").read_text())
        self.assertEqual(saved[0]["evidence"][0]["quote"], '"A useful technique."')

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

    def test_broken_parser_pool_preserves_previous_task_result(self):
        self.run_pipeline()
        path = self.output / "task-a" / "manifest.json"
        previous = path.read_bytes()
        self.load.side_effect = BrokenProcessPool("PDF parser process exited")
        with self.assertRaises(BrokenProcessPool):
            self.run_pipeline()
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual([p.name for p in self.output.iterdir() if p.is_dir()], ["task-a"])

    def test_fulltext_reader_waits_for_parser_before_removing_pdf(self):
        from engine.coldstart import ondemand
        source = paper(1)
        source.pdf_url = "https://example.org/paper.pdf"
        paths = []

        def parse_result():
            path = Path(self.parser_pool.submit.call_args.args[1])
            self.assertTrue(path.exists())
            paths.append(path)
            return "a" * 65000 + "END"

        def download(url, path):
            path.write_bytes(b"PDF fixture")
            return True

        self.parser_pool.submit.return_value.result.side_effect = parse_result
        with patch.object(ondemand, "_download", side_effect=download):
            self.assertTrue(self.real_loader(source, self.parser_pool).endswith("END"))
        self.assertIs(self.parser_pool.submit.call_args.args[0], rt._parse_pdf)
        self.assertFalse(paths[0].exists())

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
