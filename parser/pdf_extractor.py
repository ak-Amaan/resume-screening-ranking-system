"""PDF text extraction module using PyMuPDF."""

from __future__ import annotations

import logging
import re
from pathlib import Path

try:
    import fitz
except ImportError as exc:  # pragma: no cover - exercised only without dependency.
    fitz = None
    _FITZ_IMPORT_ERROR = exc
else:
    _FITZ_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when text cannot be extracted from a PDF resume."""


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving readable line breaks.

    Args:
        text: Raw text extracted from one or more PDF pages.

    Returns:
        Cleaned text with normalized whitespace and collapsed blank lines.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract clean text from every page in a PDF resume.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        Clean text extracted from all PDF pages.

    Raises:
        FileNotFoundError: If the PDF path does not exist.
        ValueError: If the path is not a PDF file.
        PDFExtractionError: If the PDF cannot be opened or text extraction fails.
    """
    if fitz is None:
        raise PDFExtractionError(
            "PyMuPDF is required for PDF extraction. Install the 'pymupdf' package."
        ) from _FITZ_IMPORT_ERROR

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF resumes are supported: {path}")

    logger.info("Extracting text from PDF: %s", path)
    try:
        with fitz.open(path) as document:
            if document.page_count == 0:
                raise PDFExtractionError(f"PDF has no pages: {path}")

            page_texts: list[str] = []
            for page_number in range(document.page_count):
                page = document.load_page(page_number)
                page_text = page.get_text("text")
                logger.debug(
                    "Extracted %d characters from page %d of %s",
                    len(page_text),
                    page_number + 1,
                    path.name,
                )
                page_texts.append(page_text)
    except PDFExtractionError:
        raise
    except Exception as exc:
        logger.exception("Failed to extract text from PDF: %s", path)
        raise PDFExtractionError(f"Unable to extract text from PDF: {path}") from exc

    cleaned_text = clean_text("\n".join(page_texts))
    if not cleaned_text:
        raise PDFExtractionError(f"No readable text found in PDF: {path}")

    logger.info("Extracted %d clean characters from %s", len(cleaned_text), path.name)
    return cleaned_text
