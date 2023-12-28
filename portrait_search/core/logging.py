import logging


def init_logging() -> None:
    logging.getLogger("backoff").addHandler(logging.StreamHandler())
