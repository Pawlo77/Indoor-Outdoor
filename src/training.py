"""Training loop for the model."""

import os

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from .callbacks import EarlyStopping
from .dataset_loader import ActiveLearningSampler
from .utils import get_logger

logger = get_logger(__name__)


# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals, too-many-statements
def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str = "cuda",
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    uncertainty_threshold: float = 0.9,
    sampler: Sampler = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    criterion: nn.Module = None,
    optimizer: optim.Optimizer = None,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    save_path: str = "best_model.pth",
) -> nn.Module:
    """
    Train the model using training and validation datasets.

    Parameters:
        model: The model to be trained.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        device: Device to use ('cuda' or 'cpu'). Default is 'cuda'.
        epochs: Number of epochs. Default is 25.
        batch_size: Batch size. Default is 64.
        learning_rate: Learning rate for optimizer. Default is 1e-3.
        weight_decay: Weight decay for optimizer. Default is 1e-4.
        patience: Patience for early stopping and LR scheduler. Default is 5.
        uncertainty_threshold: Threshold for uncertainty in active learning. Default is 0.9.
        sampler: Custom sampler for training dataset. If None, uses ActiveLearningSampler.
        train_loader: Custom DataLoader for training data. If None, a new one is created.
        val_loader: Custom DataLoader for validation data.
            If None, a new one is created with shuffle=False.
        criterion: Loss function. If None, uses CrossEntropyLoss.
        optimizer: Optimizer. If None, uses AdamW.
        scheduler: Learning rate scheduler. If None, uses ReduceLROnPlateau scheduler.
        save_path: Path to save the best model weights. Default is 'best_model.pth'.

    Returns:
        The trained model with the best weights.
    """
    writer = SummaryWriter(log_dir=os.path.splitext(save_path)[0])
    main_bar = tqdm(
        range(epochs), desc="Training", unit="epoch", leave=True
    )

    model = model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )

    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)

    # Setup sampler if not provided.
    # if sampler is None:
    #     sampler = ActiveLearningSampler(
    #         train_dataset,
    #         model,
    #         batch_size=batch_size,
    #         device=device,
    #         uncertainty_threshold=uncertainty_threshold,
    #     )

    # Setup train DataLoader if not provided.
    if train_loader is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )

    # Setup validation DataLoader if not provided.
    if val_loader is None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    activities = [ProfilerActivity.CPU]
    if device == "cuda":  # pylint: disable=magic-value-comparison
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("runs/profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ):
        for epoch in main_bar:
            model.train()
            running_loss = 0.0
            train_steps = 0

            train_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"
            )
            for images, labels in train_bar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                with autocast(device_type=device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                train_steps += 1
                train_bar.set_postfix(loss=f"{loss.item():.4e}")
                train_bar.refresh()

            avg_train_loss = running_loss / train_steps
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)

            # Validation phase.
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for images, labels in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Validation]",
                ):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item() * images.size(0)

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

            logger.info(
                "Epoch %d/%d - Training Loss: %.4f, Validation Loss: %.4f",
                epoch + 1,
                epochs,
                avg_train_loss,
                avg_val_loss,
            )

            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            main_bar.set_postfix(
                loss=f"{avg_val_loss:.4e}",
                lr=f"{optimizer.param_groups[0]['lr']:.4e}",
            )
            if early_stopping.early_stop:
                break

    # Load the best saved model weights.
    writer.close()
    model.load_state_dict(
        torch.load(early_stopping.path, map_location=device)  # nosec
    )
    return model
