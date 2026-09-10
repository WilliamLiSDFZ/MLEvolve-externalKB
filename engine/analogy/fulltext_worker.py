"""Bounded CPU PDF worker, launched by fulltext.py, never imported by the agent loop."""
from __future__ import annotations

import hashlib
import gzip
import io
import json
import re
import sys
import time
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.analogy.fulltext import CHUNK_CHARS, FullTextError, digest  # noqa: E402

MAX_BYTES = 30 * 1024 * 1024
MAX_PAGES = 100
ALLOWED_HOSTS = {"aclanthology.org", "openreview.net", "arxiv.org", "export.arxiv.org",
                 "doi.org", "ojs.aaai.org", "proceedings.mlr.press", "papers.nips.cc",
                 "proceedings.neurips.cc", "openaccess.thecvf.com", "raw.githubusercontent.com"}


def check_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in ALLOWED_HOSTS or p.username or p.password or p.port not in (None, 443):
        raise FullTextError("unsupported_source", "PDF URL is outside the supported scholarly sources")
    if p.hostname == "raw.githubusercontent.com" and not p.path.startswith("/mlresearch/"):
        raise FullTextError("unsupported_source", "Only the proceedings' mlresearch PDF repository is supported")


class ScholarlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PDFLink(HTMLParser):
    def __init__(self):
        super().__init__()
        self.url = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "meta" and attrs.get("name", "").lower() == "citation_pdf_url":
            self.url = attrs.get("content", "")


def download_pdf(urls: list[str]) -> tuple[bytes, str]:
    last = FullTextError("unavailable", "No usable PDF URL")
    opener = urllib.request.build_opener(ScholarlyRedirect())
    for url in urls:
        try:
            for _ in range(3):  # DOI -> publisher HTML -> PDF, never follow arbitrary page links
                check_url(url)
                req = urllib.request.Request(url, headers={"User-Agent": "MLEvolve-PaperReader/1.0"})
                with opener.open(req, timeout=20) as response:
                    final = response.geturl()
                    check_url(final)
                    data = response.read(MAX_BYTES + 1)
                # Some publisher/CDN responses use gzip even without Accept-Encoding.
                # Bound decompressed bytes too; decoding compressed HTML hides the PDF meta tag.
                if data.startswith(b"\x1f\x8b"):
                    with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
                        data = compressed.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise FullTextError("size_limit", "PDF exceeds the 30 MiB download limit")
                if data.lstrip().startswith(b"%PDF-"):
                    return data, final
                page = PDFLink()
                page.feed(data.decode("utf-8", errors="replace"))
                if not page.url:
                    raise FullTextError("not_pdf", "Source returned neither a PDF nor citation_pdf_url")
                url = urljoin(final, page.url)
        except FullTextError as exc:
            last = exc
        except Exception as exc:
            last = FullTextError("download_error", f"{type(exc).__name__}: {exc}"[:400])
    raise last


def split_pages(pages: list[dict]) -> list[dict]:
    chunks, section = [], "(page text; heading unavailable)"
    for page_num, page in enumerate(pages, 1):
        text = page.get("text", "")
        # Heading spans and ordinary paragraphs stay in source order. Section labels are
        # navigational hints, not LLM-generated interpretations of the paper.
        paragraphs = re.split(r"\n\s*\n", text)
        serial, buffer = 0, ""
        def emit():
            nonlocal serial, buffer
            if buffer:
                serial += 1
                chunks.append({"chunk_id": f"p{page_num:03d}-c{serial:03d}", "page": page_num,
                               "section": section, "chars": len(buffer), "text": buffer})
                buffer = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            heading = re.match(r"^(?:#{1,6}\s+|\*\*)([^\n]{3,180})", paragraph)
            if heading:
                emit()
                section = heading.group(1).strip("* ")[:160]
            for offset in range(0, len(paragraph), CHUNK_CHARS):
                body = paragraph[offset:offset + CHUNK_CHARS]
                if buffer and len(buffer) + len(body) + 2 > CHUNK_CHARS:
                    emit()
                buffer += ("\n\n" if buffer else "") + body
        emit()
    return chunks


def title_matches(title: str, text: str) -> bool:
    def tokens(s):
        return set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", s).lower()))
    words = {w for w in tokens(title) if len(w) > 2}
    return bool(words) and len(words & tokens(text)) / len(words) >= 0.8


def parse_pdf(pdf: Path, request: dict, source_url: str) -> dict:
    import pymupdf
    import pymupdf4llm

    # The layout-model backend starts ONNX thread pools and was too slow under the pod's
    # CPU quota. The deterministic text/table extractor needs no layout inference or OCR.
    pymupdf4llm.use_layout(False)
    data = pdf.read_bytes()  # avoid many random reads against a network PVC
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise FullTextError("encrypted", "The paper requires a PDF password")
        if len(doc) > MAX_PAGES:
            raise FullTextError("page_limit", f"PDF exceeds the {MAX_PAGES}-page parsing limit")
        front = "\n".join(doc[i].get_text() for i in range(min(2, len(doc))))[:16000]
        if not front.strip():
            raise FullTextError("no_text", "No extractable title/body text; image-only PDFs are unsupported")
        if not title_matches(request["record"]["title"], front):
            raise FullTextError("title_mismatch", "Downloaded PDF could not be matched to the corpus title")
        pages = pymupdf4llm.to_markdown(doc, page_chunks=True, ignore_images=True,
                                       ignore_graphics=True, show_progress=False)
        page_count = len(doc)
    chunks = split_pages(pages)
    if sum(c["chars"] for c in chunks) < 200:
        raise FullTextError("no_text", "Too little extractable body text")
    empty = [i + 1 for i, p in enumerate(pages) if not p.get("text", "").strip()]
    return {"paper_id": request["record"]["id"], "title": request["record"]["title"],
            "corpus_source": request["record"].get("source", ""), "source_url": source_url,
            "pdf_sha256": hashlib.sha256(data).hexdigest(),
            "text_sha256": digest(chunks), "parser": request["parser"],
            "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extraction_mode": "pymupdf4llm-legacy-text",
            "page_count": page_count, "title_check": "token_overlap_passed",
            "warnings": ["Text extraction only: figures and complex equations may be incomplete; OCR is disabled"]
                        + ([f"Pages without extracted text: {empty}"] if empty else []),
            "chunks": chunks}


def main() -> int:
    directory = Path(sys.argv[1])
    try:
        request = json.load(sys.stdin)
        pdf, receipt = directory / "paper.pdf", directory / "download.json"
        previous = json.loads(receipt.read_text()) if receipt.exists() and pdf.exists() else {}
        if (previous.get("urls") == request["urls"] and
                previous.get("pdf_sha256") == hashlib.sha256(pdf.read_bytes()).hexdigest()):
            source_url = previous["source_url"]
        else:
            data, source_url = download_pdf(request["urls"])
            temp_pdf = directory / "paper.pdf.tmp"
            temp_pdf.write_bytes(data)
            temp_receipt = directory / "download.json.tmp"
            temp_receipt.write_text(json.dumps({"urls": request["urls"], "source_url": source_url,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pdf_sha256": hashlib.sha256(data).hexdigest()}), encoding="utf-8")
            temp_pdf.replace(pdf)
            temp_receipt.replace(receipt)
        document = parse_pdf(pdf, request, source_url)
        document["fetched_at"] = json.loads(receipt.read_text()).get("fetched_at")
        temp = directory / "document.json.tmp"
        temp.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        temp.replace(directory / "document.json")  # publish success last, under the parent lock
        return 0
    except Exception as exc:
        error = {"status": exc.status if isinstance(exc, FullTextError) else "parse_error",
                 "message": str(exc)[:400], "retry_after": time.time() + 300}
        temp = directory / "failure.json.tmp"
        temp.write_text(json.dumps(error), encoding="utf-8")
        temp.replace(directory / "failure.json")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
