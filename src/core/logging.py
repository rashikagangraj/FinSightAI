from __future__ import annotations

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=_console,
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
