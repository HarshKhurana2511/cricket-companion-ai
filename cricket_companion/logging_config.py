from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Baseline logging setup for API/UI/MCP servers/pipelines.

    Keep it simple: log to stdout with a consistent, readable format.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)

