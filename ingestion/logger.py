"""
Centralised logger for the ingestion pipeline.
Logs to both console (UTF-8 safe) and a rotating log file.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from core.config import Config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — force UTF-8 on Windows (avoids cp1252 UnicodeEncodeError)
    console_stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
    console = logging.StreamHandler(console_stream)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — always UTF-8
    os.makedirs(os.path.dirname(Config.LOG_PATH), exist_ok=True)
    file_handler = RotatingFileHandler(
        Config.LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger