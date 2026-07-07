"""Tests for PDF extraction behavior."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from parser.pdf_extractor import PDFExtractionError, extract_text_from_pdf


def _create_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for page_text in pages:
        page = document.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)
    document.save(path)
    document.close()


def test_extract_text_from_multi_page_pdf(tmp_path: Path) -> None:
    """Extract text from all pages in a valid PDF."""
    pdf_path = tmp_path / "resume.pdf"
    _create_pdf(pdf_path, ["Jane Doe\nPython Developer", "Skills\nPython\nSQL"])

    text = extract_text_from_pdf(pdf_path)

    assert "Jane Doe" in text
    assert "Python Developer" in text
    assert "Skills" in text
    assert "SQL" in text


def test_extract_text_from_corrupted_pdf_raises_error(tmp_path: Path) -> None:
    """Raise a parser-specific error for corrupted PDFs."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"this is not a real pdf")

    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf(pdf_path)

