import sys


try:
    from loguru import logger

    logger.configure(handlers=[dict(sink=sys.stderr, format="> <level>{level:<7} {message}</level>")])
except ImportError:

    import logging

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("tez")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
