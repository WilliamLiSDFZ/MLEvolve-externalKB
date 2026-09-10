"""CPU-only reading regressions; optional live publisher/cache smoke test (no LLM calls).

python utils/verify_analogy_fulltext.py
python utils/verify_analogy_fulltext.py --corpus /path/to/paper_corpus \
    --paper-id venue/id --cache /shared/paper_fulltext --out smoke.json [--offline]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.analogy import agent
from engine.analogy.fulltext import (FullTextConfig, FullTextError, FullTextStore,
                                    PaperReadingSession, digest, resolve_urls)
from engine.analogy.fulltext_worker import split_pages, title_matches, download_pdf, check_url

QUOTE = "Minority group performance depends on the sampling distribution."
RECORD = {"id": "venue/test", "title": "Sampling Distribution and Minority Group Performance",
          "venue": "venue", "abstract": QUOTE, "source": "https://arxiv.org/abs/2407.13957v1",
          "pdf_url": ""}


class Corpus:
    digest = "test-corpus"
    def __init__(self, records=None):
        self.by_id = {r["id"]: r for r in (records or [RECORD])}
    def __contains__(self, pid):
        return pid in self.by_id
    def __len__(self):
        return len(self.by_id)
    def search(self, q, k=10):
        return [{"id": r["id"], "title": r["title"], "venue": r["venue"], "score": 1, "tldr": ""}
                for r in list(self.by_id.values())[:k]]
    def get(self, ids):
        return [self.by_id[pid] for pid in ids if pid in self]


def document(text=QUOTE):
    chunks = split_pages([{"text": "# Method\n\n" + text}, {"text": "# Appendix\n\n" + QUOTE}])
    return {"paper_id": RECORD["id"], "title": RECORD["title"], "chunks": chunks,
            "pdf_sha256": "pdf-test-hash", "text_sha256": digest(chunks), "parser": {},
            "page_count": 2, "source_url": "https://arxiv.org/pdf/2407.13957v1", "warnings": []}


def report(source="full_text", quote=QUOTE, cid="p001-c001"):
    return {"bottlenecks": [{"statement": "Subgroup sampling mismatch"}], "mechanisms": [{
        "title": "Sampling correction", "paper_ids": [RECORD["id"]], "mechanism": "Use corrected sampling",
        "intervention": "Change only the sampler", "evidence_refs": [{"paper_id": RECORD["id"],
            "source": source, "quote": quote, "chunk_id": cid}],
        "assumptions": "Group labels are known", "target_fit": "Labels are available",
        "limitations": "Ranking transfer remains unverified", "validation_plan": "Hold the model fixed; reject if validation AUC falls"}]}


def response(name, args):
    return NS(usage=NS(prompt_tokens=10, completion_tokens=5), choices=[NS(message=NS(
        content="", tool_calls=[NS(id=name, function=NS(name=name, arguments=json.dumps(args)))]))])


class ReadingTests(unittest.TestCase):
    def session(self, **kwargs):
        return PaperReadingSession(Corpus(), FullTextConfig(enabled=True, **kwargs))

    def test_resolvers_and_doi(self):
        self.assertEqual(resolve_urls(RECORD), ["https://arxiv.org/pdf/2407.13957v1"])
        for source, expected in [("https://aclanthology.org/2024.acl-long.1/", "https://aclanthology.org/2024.acl-long.1.pdf"),
                                 ("https://openreview.net/forum?id=abc", "https://openreview.net/pdf?id=abc")]:
            self.assertEqual(resolve_urls({"source": source}), [expected])
        pages = [gzip.compress(b'<meta name="citation_pdf_url" content="https://ojs.aaai.org/index.php/AAAI/article/download/1/2">'), b"%PDF-test"]
        class Page:
            def __init__(self, body, url): self.body, self.url = body, url
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def geturl(self): return self.url
            def read(self, n): return self.body[:n]
        opener = NS(open=lambda req, timeout: Page(pages.pop(0), req.full_url))
        with patch("urllib.request.build_opener", return_value=opener):
            data, url = download_pdf(["https://doi.org/10.1609/example"])
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertIn("/download/", url)
        for url in ["file:///etc/passwd", "http://arxiv.org/pdf/1", "https://127.0.0.1/x",
                    "https://raw.githubusercontent.com/random/repo/paper.pdf"]:
            with self.assertRaises(FullTextError): check_url(url)

    def test_page_chunking_and_appendix(self):
        docs = split_pages([{"text": "# Method\n\n" + "A" * 6001}, {"text": "# Appendix\n\n" + QUOTE}])
        self.assertTrue(all(c["chars"] <= 2000 for c in docs))
        self.assertEqual(docs[-1]["page"], 2)
        self.assertEqual(docs[-1]["section"], "Appendix")
        self.assertEqual(sum(c["text"].count("A") for c in docs if c["page"] == 1), 6001)
        self.assertTrue(title_matches(RECORD["title"], RECORD["title"].upper()))
        self.assertFalse(title_matches(RECORD["title"], "An unrelated paper about chemistry"))

    def test_unseen_open_and_unopened_read_are_rejected(self):
        s = self.session()
        with patch.object(s.store, "get") as get:
            self.assertEqual(s.call("open_paper", {"paper_id": RECORD["id"]}, set())["status"], "rejected")
            get.assert_not_called()
        self.assertEqual(s.call("read_paper", {"paper_id": RECORD["id"], "chunk_ids": ["p001-c001"]}, {RECORD["id"]})["status"], "not_open")

    def test_opening_is_not_reading_and_quotes_are_grounded(self):
        s = self.session()
        with patch.object(s.store, "get", return_value=(document(), False)):
            opened = s.call("open_paper", {"paper_id": RECORD["id"]}, {RECORD["id"]})
        self.assertNotIn("text", opened["outline"][0])
        clean, _ = agent.validate_report(report(), {RECORD["id"]}, s.corpus, 3, reading=s)
        self.assertFalse(clean["mechanisms"])
        s.call("read_paper", {"paper_id": RECORD["id"], "chunk_ids": ["p001-c001"]}, {RECORD["id"]})
        clean, _ = agent.validate_report(report(), {RECORD["id"]}, s.corpus, 3, reading=s)
        self.assertEqual(clean["mechanisms"][0]["evidence_refs"][0]["page"], 1)
        for invalid in [report(quote="The invented treatment always improves every metric."), report(cid="p002-c001")]:
            clean, _ = agent.validate_report(invalid, {RECORD["id"]}, s.corpus, 3, reading=s)
            self.assertFalse(clean["mechanisms"])

    def test_failed_fulltext_allows_honest_abstract_report(self):
        s = self.session()
        with patch.object(s.store, "get", side_effect=FullTextError("download_error", "offline")):
            self.assertEqual(s.call("open_paper", {"paper_id": RECORD["id"]}, {RECORD["id"]})["status"], "download_error")
        clean, _ = agent.validate_report(report(source="abstract"), {RECORD["id"]}, s.corpus, 3,
                                         reading=s, abstracts={RECORD["id"]: QUOTE})
        self.assertEqual(clean["mechanisms"][0]["evidence_level"], "abstract_only")
        self.assertIn("abstract only", agent.render_report(clean, s.corpus, 8000))
        clean, _ = agent.validate_report(report(source="abstract"), {RECORD["id"]}, s.corpus, 3, reading=s)
        self.assertFalse(clean["mechanisms"])

    def test_read_budgets_charge_repeats_and_preserve_whole_chunks(self):
        doc = document("a" * 6000)
        s = self.session(read_chars=2200, total_chars=4100, max_read_calls=2)
        s.documents[RECORD["id"]] = doc
        args = {"paper_id": RECORD["id"], "chunk_ids": [c["chunk_id"] for c in doc["chunks"][:4]]}
        one = s.call("read_paper", args, {RECORD["id"]})
        self.assertLessEqual(sum(c["chars"] for c in one["chunks"]), 2200)
        two = s.call("read_paper", args, {RECORD["id"]})
        self.assertLessEqual(s.chars, 4100)
        self.assertEqual(s.chars, sum(c["chars"] for r in [one, two] for c in r["chunks"]))
        self.assertEqual(s.call("read_paper", args, {RECORD["id"]})["status"], "read_budget_exhausted")

    def test_failures_count_against_paper_budget(self):
        s = self.session(max_papers=1)
        second = {**RECORD, "id": "venue/other"}
        s.corpus.by_id[second["id"]] = second
        seen = set(s.corpus.by_id)
        with patch.object(s.store, "get", side_effect=FullTextError("not_pdf", "not PDF")) as get:
            for pid in [RECORD["id"], RECORD["id"], second["id"]]:
                result = s.call("open_paper", {"paper_id": pid}, seen)
            self.assertEqual(result["status"], "paper_budget_exhausted")
            self.assertEqual(get.call_count, 1)

    def test_cached_document_hash_and_offline_versions(self):
        with tempfile.TemporaryDirectory() as d:
            store = FullTextStore(FullTextConfig(cache_dir=d, offline=True))
            with self.assertRaises(FullTextError) as exc: store.get(RECORD, 5)
            self.assertEqual(exc.exception.status, "offline_cache_miss")
            self.assertEqual(list(Path(d).iterdir()), [])
            key = digest({"paper_id": RECORD["id"], "title": RECORD["title"],
                          "urls": resolve_urls(RECORD), "parser": store.versions})
            p = Path(d) / key; p.mkdir()
            pdf = b"%PDF-fixture"; (p / "paper.pdf").write_bytes(pdf)
            doc = document(); doc["parser"] = store.versions
            doc["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
            (p / "document.json").write_text(json.dumps(doc))
            self.assertTrue(store.get(RECORD, 5)[1])
            (p / "paper.pdf").write_bytes(b"modified")
            with self.assertRaises(FullTextError) as exc: store.get(RECORD, 5)
            self.assertEqual(exc.exception.status, "cache_error")

    def test_concurrent_cache_publication(self):
        with tempfile.TemporaryDirectory() as d:
            store = FullTextStore(FullTextConfig(cache_dir=d))
            def worker(cmd, **kwargs):
                directory = Path(cmd[-1]); pdf = b"%PDF-fixture"
                doc = document(); doc["parser"] = store.versions
                doc["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
                (directory / "paper.pdf").write_bytes(pdf)
                (directory / "document.json").write_text(json.dumps(doc))
                return NS(returncode=0)
            with patch("engine.analogy.fulltext.subprocess.run", side_effect=worker) as run:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _: store.get(RECORD, 5), range(4)))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(sum(not hit for _, hit in results), 1)

    def test_worker_timeout_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as d:
            s = self.session(cache_dir=d)
            with patch("engine.analogy.fulltext.subprocess.run", side_effect=subprocess.TimeoutExpired("reader", 1)):
                result = s.call("open_paper", {"paper_id": RECORD["id"]}, {RECORD["id"]})
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(s.snapshot()["attempts"][RECORD["id"]]["status"], "timeout")

    def test_real_agent_loop_modes_artifacts_and_feature_off(self):
        import openai
        for mode in ["draft", "improve"]:
            calls = [response("search_papers", {"query": "sampling"}),
                     response("read_abstract", {"ids": [RECORD["id"]]}),
                     response("open_paper", {"paper_id": RECORD["id"]}),
                     response("read_paper", {"paper_id": RECORD["id"], "chunk_ids": ["p001-c001"]}),
                     response("submit_report", report())]
            create = unittest.mock.Mock(side_effect=calls)
            with patch.object(openai, "OpenAI", return_value=NS(chat=NS(completions=NS(create=create)))), \
                 patch.object(FullTextStore, "get", return_value=(document(), True)):
                res = agent.run_analogy_agent("packet", Corpus(), NS(model="test", api_key="test", base_url=""),
                        mode=mode, fulltext=FullTextConfig(enabled=True))
            self.assertTrue(res.report_md)
            self.assertEqual(res.report["mechanisms"][0]["evidence_level"], "full_text")
            self.assertEqual(res.fulltext["read_calls"], 1)
            self.assertLessEqual(len(res.report_md), 8000)
            for message in create.call_args.kwargs["messages"]:
                if message["role"] == "tool" and message["content"].startswith("{"):
                    json.loads(message["content"])
            with tempfile.TemporaryDirectory() as d:
                agent._write_artifacts(Path(d), mode, "packet", res, Corpus(), {})
                manifest = next((Path(d) / "analogy").glob("*.fulltext.json"))
                self.assertEqual(json.loads(manifest.read_text())["events"][-1]["result"]["chunks"][0]["text"],
                                 document()["chunks"][0]["text"])
        plain_tools = copy.deepcopy(agent.TOOLS)
        agent.reading_tools()
        self.assertEqual(agent.TOOLS, plain_tools)
        create = unittest.mock.Mock(side_effect=[response("search_papers", {"query": "sampling"}),
                                                response("submit_report", report())])
        with patch.object(openai, "OpenAI", return_value=NS(chat=NS(completions=NS(create=create)))):
            res = agent.run_analogy_agent("packet", Corpus(), NS(model="test", api_key="test", base_url=""))
        self.assertTrue(res.report_md)
        self.assertIsNone(res.fulltext)
        self.assertEqual(create.call_args.kwargs["tools"], plain_tools)

    def test_llm_failure_keeps_reading_manifest(self):
        import openai
        create = unittest.mock.Mock(side_effect=[response("search_papers", {"query": "sampling"}),
            response("open_paper", {"paper_id": RECORD["id"]}), RuntimeError("provider unavailable")])
        with patch.object(openai, "OpenAI", return_value=NS(chat=NS(completions=NS(create=create)))), \
             patch.object(FullTextStore, "get", return_value=(document(), True)):
            res = agent.run_analogy_agent("packet", Corpus(), NS(model="test", api_key="test", base_url=""),
                                         fulltext=FullTextConfig(enabled=True))
        self.assertFalse(res.report_md)
        self.assertIn(RECORD["id"], res.fulltext["documents"])

    def test_render_budget_is_hard_for_reading_reports(self):
        s = self.session()
        clean, _ = agent.validate_report(report(source="abstract"), {RECORD["id"]}, s.corpus, 3,
                                         reading=s, abstracts={RECORD["id"]: QUOTE})
        self.assertEqual(agent.render_report(clean, s.corpus, 20), "")

    def test_yaml_defaults_match_fulltext_dataclass(self):
        import yaml
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/config.yaml").read_text())
        self.assertEqual(config["analogy"]["fulltext"], dataclasses.asdict(FullTextConfig()))

    def test_draft_and_improve_forward_reading_config(self):
        cfg = NS(analogy=NS(enabled=True, draft=True, corpus_path="test", fulltext=FullTextConfig(enabled=True)),
                 agent=NS(code=NS()), log_dir=".")
        owner = NS(cfg=cfg)
        parent = NS(id="node", branch_id="branch", metric=None)
        with patch.object(agent, "load_corpus", return_value=Corpus()), \
             patch.object(agent, "packet_from_search", return_value="packet"), \
             patch.object(agent, "_resources", return_value={}), \
             patch.object(agent, "_write_artifacts"), \
             patch.object(agent, "run_analogy_agent", return_value=agent.AnalogyResult()) as run:
            agent.retrieve_for_node(owner, parent)
            agent.retrieve_for_draft(owner)
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertTrue(call.kwargs["fulltext"].enabled)
                self.assertEqual(call.kwargs["fulltext"].total_chars, 40000)


def smoke(args) -> int:
    records = [json.loads(line) for line in (Path(args.corpus) / "records.jsonl").read_text().splitlines() if line]
    corpus = Corpus(records)
    cfg = FullTextConfig(enabled=True, cache_dir=args.cache, offline=args.offline,
                         max_papers=max(3, len(args.paper_id)))
    session = PaperReadingSession(corpus, cfg)
    seen = set(args.paper_id)
    failures = []
    for pid in args.paper_id:
        opened = session.call("open_paper", {"paper_id": pid}, seen)
        if opened["status"] != "ok":
            failures.append(pid)
            print(json.dumps(opened, ensure_ascii=False))
            continue
        doc = session.documents[pid]
        method = next((c for c in doc["chunks"] if re_search_method(c["section"])), doc["chunks"][0])
        selected = list(dict.fromkeys([doc["chunks"][0]["chunk_id"], method["chunk_id"], doc["chunks"][-1]["chunk_id"]]))
        read = session.call("read_paper", {"paper_id": pid, "chunk_ids": selected}, seen)
        if read["status"] != "ok" or len(read["chunks"]) != len(selected): failures.append(pid)
        print(json.dumps({"paper_id": pid, "status": read["status"], "pages": doc["page_count"],
            "chunks": len(doc["chunks"]), "cache_hit": opened["cache_hit"], "pdf_sha256": doc["pdf_sha256"],
            "read_pages": [c["page"] for c in read.get("chunks", [])]}, ensure_ascii=False))
    snap = session.snapshot()
    snap["smoke_failures"] = failures
    Path(args.out).write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return bool(failures)


def re_search_method(section):
    return any(word in section.lower() for word in ["method", "approach", "algorithm", "framework"])


if __name__ == "__main__":
    if "--corpus" in sys.argv:
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--corpus", required=True)
        ap.add_argument("--paper-id", action="append", required=True)
        ap.add_argument("--cache", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--offline", action="store_true")
        raise SystemExit(smoke(ap.parse_args()))
    unittest.main(verbosity=2)
