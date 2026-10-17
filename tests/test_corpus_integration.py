"""Offline integration checks using the project's real optional dependencies."""

import importlib.util
import json
import multiprocessing
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from corpus_retriever.jev_reranker import JevReranker, PaperChunk, split_paper


def installed(*modules):
    return all(importlib.util.find_spec(module) is not None for module in modules)


@unittest.skipUnless(installed("omegaconf", "coolname", "rich", "shutup", "dataclasses_json"),
                     "Config integration requires the project's config dependencies")
class CorpusCLIIntegrationTests(unittest.TestCase):
    def test_real_yaml_and_cli_merge_without_creating_training_workspace(self):
        from config import Config, CorpusRetrieverConfig
        from omegaconf import OmegaConf
        from corpus_retriever.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = root / "public"
            public.mkdir()
            description = public / "description.md"
            description.write_text("Predict a class using the provided training data.")
            manifest = {"status": "complete", "output_dir": str(root / "techinique" / "demo")}
            arguments = [
                "exp_id=demo", f"data_dir={public}", f"desc_file={description}",
                f"abstract_index_path={root / 'index'}", f"log_dir={root / 'runs'}",
                f"workspace_dir={root / 'workspace'}", "corpus_retriever.jev_threshold=0.25",
                "agent.feedback.model=offline-model",
            ]
            with patch("corpus_retriever.retrieval.run_retrieval", return_value=manifest) as run, \
                    patch("config.load_cfg", side_effect=AssertionError("Training setup must not run")), \
                    patch.object(Path, "mkdir", side_effect=AssertionError("CLI must not create workspaces")), \
                    patch.dict(os.environ, {"LLM_API_KEY": "offline-test-key"}), \
                    self.assertLogs("MLEvolve", level="INFO"):
                self.assertEqual(main(arguments), 0)
                run.assert_called_once()
                cfg = run.call_args.args[0]
                self.assertIs(OmegaConf.get_type(cfg), Config)
                self.assertIs(OmegaConf.get_type(cfg.corpus_retriever), CorpusRetrieverConfig)
                self.assertEqual(cfg.exp_id, "demo")
                self.assertEqual(cfg.exp_name, "demo")
                self.assertEqual(cfg.data_dir, public)
                self.assertEqual(cfg.desc_file, description)
                self.assertEqual(cfg.lazy_pool, 100)
                self.assertEqual(cfg.corpus_retriever.jev_threshold, 0.25)
                self.assertEqual(cfg.corpus_retriever.chunk_tokens, 10000)
                self.assertEqual(cfg.corpus_retriever.chunk_overlap, 200)
                self.assertEqual(cfg.agent.feedback.model, "offline-model")
                self.assertEqual(cfg.agent.feedback.api_key, "offline-test-key")
            self.assertFalse((root / "runs").exists())
            self.assertFalse((root / "workspace").exists())
            self.assertFalse((root / "techinique").exists())


@unittest.skipUnless(installed("httpx"), "HTTP integration requires httpx")
class JevHTTPIntegrationTests(unittest.TestCase):
    def test_real_httpx_serializes_typed_request_and_reads_each_answer(self):
        import httpx

        chunks = [PaperChunk(f"论文 evidence {i}", f"paper:{i}", "paper", "Title", "Abstract", i, i + 1)
                  for i in range(3)]
        requests = []
        scores = [0.8, 0.1, 0.5]

        def handler(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(request.headers["Authorization"], "Bearer offline-key")
            self.assertIn("application/json", request.headers["Content-Type"])
            payload = json.loads(request.content)
            requests.append(payload)
            question = payload["questions"]["useful"]
            self.assertEqual(question["type"], "noul")
            self.assertIsInstance(question["instructions"], str)
            self.assertEqual(set(question["criteria"]), {"true", "false"})
            self.assertEqual(payload["model"], "jev-1.13.0")
            self.assertEqual(payload["state"]["task_description"], "Classify documents")
            self.assertEqual(payload["state"]["available_data"], {"files": ["train.csv"]})
            self.assertEqual(payload["state"]["paper_title"], "Title")
            self.assertEqual(payload["state"]["abstract"], "Abstract")
            return httpx.Response(200, json={
                "model": "jev-1.13.0",
                "answers": {"useful": {"type": "noul", "noul": scores[len(requests) - 1]}},
                "usage": {"input_tokens": 123, "output_tokens": 5},
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            selected, decisions = JevReranker("offline-key", client=client).filter(
                chunks, "Classify documents", {"files": ["train.csv"]})
        self.assertEqual(len(requests), 3)
        self.assertEqual([p["state"]["chunk"]["content"] for p in requests], [c.content for c in chunks])
        self.assertEqual(selected, [chunks[0], chunks[2]])
        self.assertEqual([d["score"] for d in decisions], scores)
        self.assertTrue(all(d["status"] == "ok" for d in decisions))
        self.assertTrue(all(d["usage"] == {"input_tokens": 123, "output_tokens": 5} for d in decisions))


@unittest.skipUnless(installed("tokenizers"), "Tokenizer integration requires tokenizers")
class RustTokenizerIntegrationTests(unittest.TestCase):
    def test_locally_trained_bpe_preserves_unicode_and_bounds_every_slice(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        rust = Tokenizer(models.BPE(unk_token="[UNK]"))
        rust.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=300, min_frequency=2, special_tokens=["[UNK]"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
        paragraph = "\t论文 α🧪: an experiment tests café and e\u0301 features.\n\n"
        rust.train_from_iterator([paragraph] * 20, trainer=trainer)

        class LocalFastTokenizer:
            is_fast = True

            def __call__(self, text, *, add_special_tokens, truncation, return_offsets_mapping):
                self_test.assertFalse(add_special_tokens)
                self_test.assertFalse(truncation)
                self_test.assertTrue(return_offsets_mapping)
                encoded = rust.encode(text, add_special_tokens=add_special_tokens)
                return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

        self_test = self
        text = paragraph * 250 + "终点: 🧪\n  "
        tokenizer = LocalFastTokenizer()
        chunks = split_paper(text, "unicode-paper", "Title", "Abstract", tokenizer,
                             max_tokens=32, overlap=7)
        self.assertGreater(len(chunks), 10)
        covered = 0
        reconstructed = []
        for chunk in chunks:
            self.assertEqual(chunk.content, text[chunk.start_char:chunk.end_char])
            self.assertLessEqual(len(rust.encode(chunk.content, add_special_tokens=False).ids), 32)
            self.assertLessEqual(chunk.start_char, covered)
            self.assertGreater(chunk.end_char, covered)
            reconstructed.append(chunk.content[covered - chunk.start_char:])
            covered = chunk.end_char
        self.assertEqual(covered, len(text))
        self.assertEqual("".join(reconstructed), text)
        self.assertTrue(any(a.end_char > b.start_char for a, b in zip(chunks, chunks[1:])))


@unittest.skipUnless(installed("pymupdf", "pymupdf4llm"), "PDF integration requires the PDF dependencies")
class PaperPDFIntegrationTests(unittest.TestCase):
    def test_real_pdf_conversion_reads_through_the_final_page(self):
        import pymupdf
        from corpus_retriever.retrieval import load_full_text

        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "fixture.pdf"
            with pymupdf.open() as document:
                document.new_page().insert_text((72, 72), "Main paper methodology.")
                document.new_page().insert_text((72, 72), "Appendix experiment: use grouped validation.")
                document.save(pdf)
            with patch("engine.coldstart.ondemand._resolve_pdf", return_value="offline://fixture"), \
                    patch("engine.coldstart.ondemand._download", side_effect=lambda url, path: shutil.copyfile(pdf, path)), \
                    ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
                text = load_full_text(SimpleNamespace(), pool)
            self.assertIn("Main paper methodology.", text)
            self.assertIn("Appendix experiment: use grouped validation.", text)


if __name__ == "__main__":
    unittest.main()
