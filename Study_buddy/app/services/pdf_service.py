"""PDF text extraction and page-aware chunking."""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from app.core.config import get_settings


@dataclass(frozen=True)
class PdfChunk:
    """A text chunk tied to its source page."""

    text: str
    page: int


def extract_pdf_chunks(pdf_bytes: bytes) -> tuple[int, list[PdfChunk]]:
    """Extract page text and split it into overlapping, page-aware chunks."""
    reader = PdfReader(BytesIO(pdf_bytes))
    chunks: list[PdfChunk] = []
    settings = get_settings()

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + settings.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(PdfChunk(text=chunk_text, page=page_number))
            if end == len(text):
                break
            start = end - settings.chunk_overlap

    if not chunks:
        raise ValueError("This PDF contains no extractable text. Scanned PDFs are not supported.")
    return len(reader.pages), chunks
