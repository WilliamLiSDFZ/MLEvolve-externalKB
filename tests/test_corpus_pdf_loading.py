"""PDF loading keeps temporary files alive until isolated parsing completes."""

import importlib.util
import multiprocessing
import os
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from corpus_retriever import retrieval as rt


_parser_barrier = None


def _set_parser_barrier(barrier):
    global _parser_barrier
    _parser_barrier = barrier


def _parse_in_parallel(path):
    _parser_barrier.wait(timeout=20)
    text = rt._parse_pdf(path)
    return text, os.getpid(), [name for name in (
        "torch", "transformers", "sentence_transformers", "engine.coldstart.ondemand")
        if name in sys.modules]


class PDFLoadingTests(unittest.TestCase):
    def setUp(self):
        self.paths = []
        self.resolve = Mock(return_value="https://example.org/paper.pdf")
        self.download = Mock(side_effect=self._download)
        ondemand = SimpleNamespace(_resolve_pdf=self.resolve, _download=self.download)
        patcher = patch.dict(sys.modules, {"engine.coldstart.ondemand": ondemand})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.pool = Mock()
        self.pool.submit.return_value.result.return_value = "Readable paper text."
        self.paper = SimpleNamespace(id="paper-1")

    def _download(self, url, path):
        self.paths.append(path)
        path.write_bytes(b"PDF bytes")
        return True

    def test_file_lives_until_parser_result_and_is_then_removed(self):
        def submit(parser, filename):
            self.assertIs(parser, rt._parse_pdf)
            self.assertIsInstance(filename, str)
            self.assertEqual(Path(filename).read_bytes(), b"PDF bytes")
            future = Mock()

            def result():
                self.assertTrue(Path(filename).is_file())
                return "Readable paper text."

            future.result.side_effect = result
            return future

        self.pool.submit.side_effect = submit
        self.assertEqual(rt.load_full_text(self.paper, self.pool), "Readable paper text.")
        self.resolve.assert_called_once_with(self.paper)
        self.download.assert_called_once_with(self.resolve.return_value, self.paths[0])
        self.pool.submit.assert_called_once()
        self.assertFalse(self.paths[0].parent.exists())

    def test_missing_url_never_downloads_or_parses(self):
        self.resolve.return_value = None
        with self.assertRaisesRegex(ValueError, "resolvable PDF"):
            rt.load_full_text(self.paper, self.pool)
        self.download.assert_not_called()
        self.pool.submit.assert_not_called()

    def test_failed_download_is_cleaned_up_without_parsing(self):
        def failed_download(url, path):
            self._download(url, path)
            return False

        self.download.side_effect = failed_download
        with self.assertRaisesRegex(OSError, "download"):
            rt.load_full_text(self.paper, self.pool)
        self.pool.submit.assert_not_called()
        self.assertFalse(self.paths[0].parent.exists())

    def test_parser_error_propagates_after_cleanup(self):
        self.pool.submit.return_value.result.side_effect = ValueError("Malformed PDF")
        with self.assertRaisesRegex(ValueError, "Malformed PDF"):
            rt.load_full_text(self.paper, self.pool)
        self.assertFalse(self.paths[0].parent.exists())

    def test_unreadable_parser_output_is_rejected_after_cleanup(self):
        for text in ("", " \n\t", None, ["page text"]):
            with self.subTest(text=text):
                self.pool.submit.return_value.result.return_value = text
                with self.assertRaisesRegex(ValueError, "readable text"):
                    rt.load_full_text(self.paper, self.pool)
                self.assertFalse(self.paths[-1].parent.exists())

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("fitz", "pymupdf4llm")),
        "Real PDF process integration requires PyMuPDF and pymupdf4llm")
    def test_three_spawn_parsers_read_real_pdf_reuse_processes_and_avoid_model_imports(self):
        import fitz

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.pdf"
            with fitz.open() as document:
                page = document.new_page()
                page.insert_text((72, 72), "A concrete experiment for toxicity classification.")
                document.save(path)
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=3, mp_context=context,
                                     initializer=_set_parser_barrier,
                                     initargs=(context.Barrier(3),)) as pool:
                process_waves = []
                for _ in range(2):
                    futures = [pool.submit(_parse_in_parallel, str(path)) for _ in range(3)]
                    process_ids = set()
                    for future in futures:
                        text, process_id, model_imports = future.result(timeout=30)
                        self.assertIn("toxicity classification", text)
                        self.assertNotEqual(process_id, os.getpid())
                        self.assertEqual(model_imports, [])
                        process_ids.add(process_id)
                    self.assertEqual(len(process_ids), 3)
                    process_waves.append(process_ids)
                self.assertEqual(process_waves[0], process_waves[1])


if __name__ == "__main__":
    unittest.main()
