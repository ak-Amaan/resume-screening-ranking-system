"""File input and output helpers for local project artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from parser.pdf_extractor import extract_text_from_pdf

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str | Path) -> str:
    """Load a PDF resume and return extracted clean text.

    Args:
        pdf_path: Path to a PDF resume.

    Returns:
        Clean text extracted from the PDF.
    """
    return extract_text_from_pdf(pdf_path)


def save_json(data: Any, output_path: str | Path) -> None:
    """Save JSON-serializable data to disk.

    Args:
        data: Data to serialize. Dataclasses are converted with ``asdict``.
        output_path: Destination JSON file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = asdict(data) if is_dataclass(data) else data
    logger.info("Saving JSON file: %s", path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable_data, file, indent=2, ensure_ascii=False)


def load_json(input_path: str | Path) -> Any:
    """Load JSON data from disk.

    Args:
        input_path: Source JSON file path.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(input_path)
    logger.info("Loading JSON file: %s", path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
