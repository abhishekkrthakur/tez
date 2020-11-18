import logging
import sys


def logger(name=None):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger_fn = logging.getLogger(name=name)
    return logger_fn
