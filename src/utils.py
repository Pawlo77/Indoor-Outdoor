"""Utility functions for logging and timing."""

import io
import logging
import time
import warnings
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def plot_images(images: List[Any], ncols: int = 5, labels: List[str] = None) -> None:
    """
    Plots a grid of images.

    Args:
        images: List of image data (e.g., bytes or PIL images).
        ncols: Number of columns in the grid.
        labels: Optional list of labels for each image.
    """
    nrows = len(images) // ncols + (len(images) % ncols > 0)
    _, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))

    for i, (ax, img) in enumerate(zip(axes.ravel(), images)):
        if isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        ax.imshow(img)
        ax.axis("off")

        if labels is not None:
            ax.set_title(labels[i])

    plt.tight_layout()
    plt.show()
