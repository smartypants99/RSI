"""Logging configuration for the Time Dilation Runtime."""
import json
import logging
import sys


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record):
        log_entry = {
            "ts": self.formatTime(record, "%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(log_entry)


def setup_logging(verbose: bool = False, structured: bool = False) -> None:
    """Configure logging for the timedilate package."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    root = logging.getLogger("timedilate")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
