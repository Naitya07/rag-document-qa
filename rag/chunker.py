"""Document parsing and text chunking. Supports PDF, DOCX, PPTX, TXT, and CSV."""

import csv
import io

import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text from a PDF file."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


def extract_text_from_docx(file_bytes: bytes) -> list[dict]:
    """Extract text from a DOCX file."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Group every 10 paragraphs as a "section"
    sections = []
    for i in range(0, len(paragraphs), 10):
        section_text = "\n".join(paragraphs[i:i + 10])
        if section_text.strip():
            sections.append({"page": i // 10 + 1, "text": section_text.strip()})
    return sections


def extract_text_from_pptx(file_bytes: bytes) -> list[dict]:
    """Extract text from a PPTX file (one page per slide)."""
    from pptx import Presentation

    prs = Presentation(io.BytesIO(file_bytes))
    slides = []
    for i, slide in enumerate(prs.slides):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                texts.append(shape.text_frame.text)
        slide_text = "\n".join(texts)
        if slide_text.strip():
            slides.append({"page": i + 1, "text": slide_text.strip()})
    return slides


def extract_text_from_txt(file_bytes: bytes) -> list[dict]:
    """Extract text from a plain text file."""
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    # Group every 30 lines as a "section"
    sections = []
    for i in range(0, len(lines), 30):
        section_text = "\n".join(lines[i:i + 30])
        if section_text.strip():
            sections.append({"page": i // 30 + 1, "text": section_text.strip()})
    return sections


def extract_text_from_csv(file_bytes: bytes) -> list[dict]:
    """Extract text from a CSV file (rows grouped into sections)."""
    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = [", ".join(row) for row in reader if any(cell.strip() for cell in row)]
    # Group every 20 rows as a "section"
    sections = []
    for i in range(0, len(rows), 20):
        section_text = "\n".join(rows[i:i + 20])
        if section_text.strip():
            sections.append({"page": i // 20 + 1, "text": section_text.strip()})
    return sections


def extract_text(file_bytes: bytes, filename: str) -> list[dict]:
    """Route to the correct extractor based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    extractors = {
        "pdf": extract_text_from_pdf,
        "docx": extract_text_from_docx,
        "pptx": extract_text_from_pptx,
        "txt": extract_text_from_txt,
        "csv": extract_text_from_csv,
    }
    extractor = extractors.get(ext)
    if not extractor:
        raise ValueError(f"Unsupported file type: .{ext}")
    return extractor(file_bytes)


def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split page texts into overlapping chunks for embedding."""
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
