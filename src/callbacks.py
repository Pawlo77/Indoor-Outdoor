"""Callbacks for training a model."""

import os
from typing import Optional

import numpy as np
import torch
from torch.nn import Module

from .utils import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint_model.pth")


# pylint: disable=too-few-public-methods
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve sufficiently after a given patience.

    Args:
        patience: How long to wait after last time validation loss improved.
        verbose: If True, prints a message for each validation loss improvement.
        path: Path for the checkpoint to be saved to.
        min_improvement: Minimum decrease in validation loss to qualify as an improvement.
    """

    def __init__(
        self,
        patience: int = 20,
        verbose: bool = False,
        path: str = CHECKPOINT_PATH,
        min_improvement: float = 1e-8,
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path
        self.min_improvement = min_improvement

    def __call__(self, val_loss: float, model: Module) -> None:
        """
        Call this method after each epoch to check if early stopping should be performed.

        Args:
            val_loss: Current validation loss.
            model: Model to save if validation loss decreases sufficiently.
        """
        # Check if validation loss improved by at least min_improvement
        if self.val_loss_min == np.inf or self.val_loss_min - val_loss >= self.min_improvement:
            if self.verbose and self.val_loss_min != np.inf:
                logger.info(
                    "Validation loss improved (%.6f --> %.6f). Saving model ...",
                    self.val_loss_min,
                    val_loss,
                )
            self._save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping counter: %s out of %s", self.counter, self.patience
                )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")

    def _save_checkpoint(self, val_loss: float, model: Module) -> None:
        """
        Saves model when validation loss decreases sufficiently.

        Args:
            val_loss: Current validation loss.
            model: Model to save.
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
