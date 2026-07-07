"""Logging configuration helpers for terminal execution."""

from __future__ import annotations

import logging
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent terminal logging for project entry points.

    Args:
        level: Logging level to apply to the root logger.
    """
    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
