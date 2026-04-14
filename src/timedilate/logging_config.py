import logging
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the timedilate package."""
    level = logging.DEBUG if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("timedilate")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
