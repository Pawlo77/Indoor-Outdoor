"""Utility functions for logging and timing."""

import logging
import time
import warnings
from typing import Any

warnings.simplefilter("ignore")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger with the specified name.

    Args:
        name: The name of the logger.
        level: The logging level.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=level)
    handler = logging.StreamHandler()
    handler.setLevel(level=level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class TimedLog:
    """Context manager for logging the time taken by a block of code."""

    def __init__(
        self, logger: logging.Logger, message: str, level: int = logging.INFO
    ) -> None:
        """
        Initializes the TimedLog context manager.

        Args:
            logger: The logger to use.
            message: The message to log.
            level: The logging level.
        """
        self.logger = logger
        self.message = message
        self.level = level
        self.start_time: float | None = None

    def __enter__(self) -> None:
        """
        Enters the context manager and logs the start time.
        """
        self.start_time = time.time()
        self.logger.debug(f"{self.message} - Start")

    def __exit__(self, *_: Any) -> None:
        """
        Exits the context manager and logs the end time.
        """
        if self.start_time is None:
            raise RuntimeError("Context manager was not entered properly.")

        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.logger.log(
            self.level, f"{self.message}  (Elapsed time: {elapsed_time:.2f} s)"
        )
