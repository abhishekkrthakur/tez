try:
    from loguru import logger
except ImportError:

    import logging

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("tez")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
