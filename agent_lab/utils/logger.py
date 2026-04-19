import logging
import os


def get_logger(name: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        fh = logging.FileHandler("logs/agent.log")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger