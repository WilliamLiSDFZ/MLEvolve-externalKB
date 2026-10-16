import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from corpus_retriever.jev_reranker import JevReranker, PaperChunk, split_paper


class OffsetTokenizer:
    """Offline byte-like tokenizer with repeated offsets for Unicode characters."""

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False,
                          "return_offsets_mapping": True}
        offsets = [(i, i + 1) for i, char in enumerate(text)
                   for _ in char.encode("utf-8")]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class BoundaryTokenizer(OffsetTokenizer):
    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        # A substring starting at b has one additional boundary token.
        if text.startswith("b"):
            encoded["input_ids"].insert(0, -1)
            encoded["offset_mapping"].insert(0, (0, 1))
        return encoded


class HTTPFailure(Exception):
    def __init__(self, response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def response(score=0.8, status=200, headers=None):
    result = Mock(status_code=status, headers=headers or {})
    result.json.return_value = {
        "answers": {"useful": {"type": "noul", "noul": score}},
        "model": "jev-test", "usage": {"input_tokens": 42},
    }
    if status >= 400:
        result.raise_for_status.side_effect = HTTPFailure(result)
    return result


class SplitPaperTests(unittest.TestCase):
    def assert_covered(self, text, chunks, tokenizer, cap):
        covered = 0
        for chunk in chunks:
            self.assertLessEqual(chunk.start_char, covered)
            self.assertGreater(chunk.end_char, covered)
            self.assertEqual(chunk.content, text[chunk.start_char:chunk.end_char])
            self.assertLessEqual(len(tokenizer(chunk.content, add_special_tokens=False,
                                               truncation=False, return_offsets_mapping=True)["input_ids"]), cap)
            covered = chunk.end_char
        self.assertEqual(covered, len(text))

    def test_long_unicode_and_whitespace_are_preserved(self):
        text = " 论文α🧪 ab\n\n" * 3000 + "\n  "
        tokenizer = OffsetTokenizer()
        chunks = split_paper(text, "paper", "Title", "Abstract", tokenizer)
        self.assertGreater(len(chunks), 1)
        self.assert_covered(text, chunks, tokenizer, 10000)
        self.assertTrue(all(a.end_char > b.start_char for a, b in zip(chunks, chunks[1:])))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))
        self.assertTrue(all(c.paper_id == "paper" and c.abstract == "Abstract" for c in chunks))

    def test_retokenizes_boundary_and_never_decodes(self):
        text = "ababababab" * 30
        tokenizer = BoundaryTokenizer()
        chunks = split_paper(text, "p", "t", "a", tokenizer, max_tokens=6, overlap=1)
        self.assert_covered(text, chunks, tokenizer, 6)

    def test_prefers_paragraph_boundary(self):
        chunks = split_paper("abcde\n\nfghijklmnop", "p", "t", "a", OffsetTokenizer(),
                             max_tokens=10, overlap=2)
        self.assertEqual(chunks[0].content, "abcde\n\n")

    def test_missing_id_is_stable_and_papers_remain_distinct(self):
        args = ("some text", None, "title", "abstract", OffsetTokenizer())
        first = split_paper(*args)
        self.assertEqual(first, split_paper(*args))
        self.assertIsNone(first[0].paper_id)
        other = split_paper("different text", None, "title", "abstract", OffsetTokenizer())
        self.assertNotEqual(first[0].id, other[0].id)

    def test_empty_and_invalid_budgets(self):
        self.assertEqual(split_paper("", None, "", "", OffsetTokenizer()), [])
        for cap, overlap in ((10001, 0), (0, 0), (10, 10), (10, -1)):
            with self.subTest(cap=cap, overlap=overlap), self.assertRaises(ValueError):
                split_paper("text", None, "", "", OffsetTokenizer(), cap, overlap)

    def test_character_that_exceeds_budget_fails_without_losing_text(self):
        with self.assertRaises(ValueError):
            split_paper("🧪", None, "", "", OffsetTokenizer(), max_tokens=2, overlap=0)


class JevRerankerTests(unittest.TestCase):
    def setUp(self):
        httpx = patch.dict("sys.modules", {"httpx": SimpleNamespace(TransportError=OSError)})
        httpx.start()
        self.addCleanup(httpx.stop)
        self.chunks = [PaperChunk(f"evidence {i}", f"paper:{i}", "paper", "title", "abstract")
                       for i in range(3)]
        self.client = Mock()
        self.reranker = JevReranker("test-key", client=self.client)

    def test_reads_all_chunks_after_first_positive_and_preserves_order(self):
        self.client.post.side_effect = [response(.8), response(.1), response(.5)]
        selected, decisions = self.reranker.filter(self.chunks, "full task", {"files": ["train.csv"]})
        self.assertEqual(selected, [self.chunks[0], self.chunks[2]])
        self.assertEqual(self.client.post.call_count, 3)
        self.assertEqual([d["selected"] for d in decisions], [True, False, True])
        self.assertTrue(all(d["status"] == "ok" for d in decisions))
        payload = self.client.post.call_args_list[0].kwargs["json"]
        self.assertEqual(payload["state"]["task_description"], "full task")
        self.assertEqual(payload["state"]["available_data"], {"files": ["train.csv"]})
        self.assertEqual(payload["state"]["chunk"]["content"], "evidence 0")
        self.assertEqual(payload["questions"]["useful"]["type"], "noul")
        self.assertEqual(decisions[0]["model"], "jev-test")
        self.assertEqual(decisions[0]["usage"], {"input_tokens": 42})

    def test_all_negative_remains_empty(self):
        self.client.post.return_value = response(0)
        selected, decisions = self.reranker.filter(self.chunks, "task", "data")
        self.assertEqual(selected, [])
        self.assertTrue(all(d["selected"] is False for d in decisions))

    def test_invalid_scores_are_errors_not_negative_decisions(self):
        for score in (None, True, "0.8", float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(score=score):
                self.client.post.return_value = response(score)
                selected, decisions = self.reranker.filter(self.chunks[:1], "task", "data")
                self.assertEqual(selected, [])
                self.assertEqual(decisions[0]["status"], "error")
                self.assertIsNone(decisions[0]["selected"])
                self.assertIsNone(decisions[0]["score"])

    def test_missing_answer_does_not_prevent_next_chunk(self):
        malformed = response()
        malformed.json.return_value = {"answers": {}}
        self.client.post.side_effect = [malformed, response(.8)]
        selected, decisions = self.reranker.filter(self.chunks[:2], "task", "data")
        self.assertEqual(selected, self.chunks[1:2])
        self.assertEqual([d["status"] for d in decisions], ["error", "ok"])

    @patch("corpus_retriever.jev_reranker.time.sleep")
    def test_rate_limit_respects_retry_after(self, sleep):
        self.client.post.side_effect = [response(status=429, headers={"Retry-After": "3"}), response(.8)]
        selected, _ = self.reranker.filter(self.chunks[:1], "task", "data")
        self.assertEqual(selected, self.chunks[:1])
        sleep.assert_called_once_with(3)

    @patch("corpus_retriever.jev_reranker.time.sleep")
    def test_timeout_and_server_error_retry_then_succeed(self, sleep):
        self.client.post.side_effect = [TimeoutError("timeout"), response(status=529), response(.8)]
        selected, _ = self.reranker.filter(self.chunks[:1], "task", "data")
        self.assertEqual(selected, self.chunks[:1])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    @patch("corpus_retriever.jev_reranker.time.sleep")
    def test_exhausted_retry_is_unresolved_and_next_chunk_is_checked(self, sleep):
        self.client.post.side_effect = [response(status=503)] * 3 + [response(.8)]
        selected, decisions = self.reranker.filter(self.chunks[:2], "task", "data")
        self.assertEqual(selected, self.chunks[1:2])
        self.assertEqual(decisions[0]["http_status"], 503)
        self.assertIsNone(decisions[0]["selected"])

    def test_bad_request_is_not_retried(self):
        self.client.post.side_effect = [response(status=422), response(.8)]
        selected, decisions = self.reranker.filter(self.chunks[:2], "task", "data")
        self.assertEqual(self.client.post.call_count, 2)
        self.assertEqual(decisions[0]["http_status"], 422)
        self.assertEqual(selected, self.chunks[1:2])

    def test_invalid_request_configuration_is_not_retried(self):
        self.client.post.side_effect = ValueError("Invalid request URL")
        _, decisions = self.reranker.filter(self.chunks[:1], "task", "data")
        self.assertEqual(self.client.post.call_count, 1)
        self.assertEqual(decisions[0]["status"], "error")

    def test_auth_failure_aborts_without_retries(self):
        for status in (401, 403):
            with self.subTest(status=status):
                client = Mock()
                client.post.return_value = response(status=status)
                with self.assertRaises(PermissionError):
                    JevReranker("key", client=client).filter(self.chunks, "task", "data")
                self.assertEqual(client.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
