from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz


@dataclass
class PageText:
    page: int
    text: str


@dataclass
class TextChunk:
    page: int
    text: str


def extract_pages(path: Path) -> List[PageText]:
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ValueError("The PDF is damaged, encrypted, or unreadable.") from exc

    try:
        pages = [PageText(page=i + 1, text=page.get_text("text").strip()) for i, page in enumerate(document)]
    finally:
        document.close()
    if not any(page.text for page in pages):
        raise ValueError("The PDF contains no extractable text. Scanned PDFs need OCR.")
    return pages


def chunk_pages(pages: List[PageText], chunk_size: int = 1200, overlap: int = 200) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    step = chunk_size - overlap
    for page in pages:
        normalized = " ".join(page.text.split())
        for start in range(0, len(normalized), step):
            chunk = normalized[start : start + chunk_size].strip()
            if chunk:
                chunks.append(TextChunk(page=page.page, text=chunk))
            if start + chunk_size >= len(normalized):
                break
    return chunks

