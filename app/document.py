#!/usr/bin/env python3
"""Document text extraction: a .docx or .pdf upload -> the text inside it.

The translator only knows text, so a document is read down to its text here
and then goes through the same lilly.translate/reply path as anything typed.
The two readers are lazy imports, loaded on first use like every other model
part: python-docx for .docx, pypdf for .pdf. Neither is a heavyweight or GPU
dependency, and neither is needed to start the server.

Everything is bounded before it reaches a reader, as with every other upload:
the server refuses oversized files on their declared size, and the text kept
out of a document is capped here, because a 400-page PDF could otherwise pour
a novel's worth of text into the translation engine that truncates it anyway.

Usage:
    python3 app/document.py document.docx     # prints the extracted text
"""
import sys
from pathlib import Path

if __name__ == "__main__":
    # Run as a script, sys.path[0] is app/ and `from app.lilly import ...`
    # cannot resolve -- the usage above never worked that way. The repository
    # goes first, as app/lilly.py does.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.lilly import BadInput  # noqa: E402

# The suffixes a document may arrive under. The suffix travels with the temp
# file the server writes, and is checked here rather than against the upload's
# declared name, so a misnamed file is refused for what it is.
SUFFIXES = (".docx", ".pdf")

# How much text one document may ask for. Generous on purpose — a full page of
# prose is a few thousand characters — but finite: the extraction below stops
# reading once this much is out, and the translation engine applies its own
# token budget on top (truncate=True, so the beginning is translated rather
# than the request refused).
MAX_DOC_CHARS = 20_000
# A PDF with hundreds of pages should cost hundreds of page reads at most,
# not an unbounded walk. Even a full page of dense text stays under
# MAX_DOC_CHARS, so this only ever bites on a pathological file.
MAX_PDF_PAGES = 100


class UnreadableDocument(BadInput):
    """Raised when the upload is not a document we can open, or holds no text."""


def _bounded_text() -> tuple:
    """A piecewise text collector that stops accepting once the cap is met."""
    parts = []
    size = [0]

    def add(chunk: str) -> None:
        if size[0] >= MAX_DOC_CHARS or not chunk:
            return
        parts.append(chunk)
        size[0] += len(chunk)

    return parts, size, add


def _finish(parts: list, size: list) -> str:
    text = "\n".join(parts).strip()
    if size[0] > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]
    if not text:
        raise UnreadableDocument("that document has no text we could read")
    return text


def _extract_docx(path: str) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        # A missing library is not the caller's document and not a crash.
        raise FileNotFoundError(
            "document reading is not installed -- pip install python-docx") from exc
    try:
        doc = Document(path)
        parts, size, add = _bounded_text()
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                add(paragraph.text.strip())
        # Tables are their own tree in a docx; skipping them would quietly
        # drop half of some documents. Cell text is added after the body, in
        # document order, and the cap still governs.
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        add(cell.text.strip())
        return _finish(parts, size)
    except BadInput:
        raise
    except Exception as exc:
        raise UnreadableDocument(
            "that file is not a document we can read") from exc


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FileNotFoundError(
            "document reading is not installed -- pip install pypdf") from exc
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise UnreadableDocument(
                "that PDF is password-protected and we cannot open it")
        parts, size, add = _bounded_text()
        for page in reader.pages[:MAX_PDF_PAGES]:
            add(page.extract_text() or "")
        return _finish(parts, size)
    except BadInput:
        raise
    except Exception as exc:
        raise UnreadableDocument(
            "that file is not a document we can read") from exc


def extract_text(path: str) -> str:
    """The text of a .docx or .pdf upload, bounded and readable."""
    suffix = Path(path).suffix.lower()
    if suffix not in SUFFIXES:
        raise BadInput("Lilly reads .docx and .pdf documents")
    return _extract_docx(path) if suffix == ".docx" else _extract_pdf(path)


def main() -> int:
    if len(sys.argv) < 2 or not Path(sys.argv[1]).is_file():
        print("usage: python3 app/document.py <document.docx|document.pdf>",
              file=sys.stderr)
        return 1
    print(extract_text(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
