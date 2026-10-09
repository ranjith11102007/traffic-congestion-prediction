"""Central logging configuration."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Third-party loggers that are too chatty at DEBUG level.
NOISY_LOGGERS = (
    "uvicorn.access",
    "sqlalchemy.engine.Engine",
)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, at application start-up."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )
    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)