import logging
from timedilate.logging_config import setup_logging


def test_setup_logging_verbose():
    setup_logging(verbose=True)
    logger = logging.getLogger("timedilate")
    assert logger.level == logging.DEBUG


def test_setup_logging_quiet():
    setup_logging(verbose=False)
    logger = logging.getLogger("timedilate")
    assert logger.level == logging.WARNING
