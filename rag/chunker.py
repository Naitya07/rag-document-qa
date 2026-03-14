"""PDF parsing and text chunking."""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract text from a PDF file, returning a list of page dicts."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split page texts into overlapping chunks for embedding.

    Each chunk dict has: text, page, chunk_index.
    """
    chunks = []
    for page_info in pages:
        words = page_info["text"].split()
        start = 0
        idx = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "text": chunk_text,
                "page": page_info["page"],
                "chunk_index": idx,
            })
            start += chunk_size - overlap
            idx += 1
    return chunks
