"""Dataset loader for images and labels stored in .npz files."""

import os
import re
from itertools import chain
from typing import Any, Callable, List, Optional, Tuple, cast

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .data_preparation_utils import RESULTS_DIR
from .utils import get_logger
from tqdm.notebook import tqdm

logger = get_logger(__name__)


# Was too slow for low res images to give training
# performance boost, good for high res images.
class ActiveLearningSampler(Sampler):
    """
    Sampler that selects samples based on model confidence for active learning.
    A sample is selected if the model's maximum predicted probability is less
    than the uncertainty_threshold, meaning the model is not confident
    about its prediction.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        dataset: Any,
        model: torch.nn.Module,
        *args: Any,
        batch_size: int = 32,
        uncertainty_threshold: float = 0.9,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dataset: The dataset to sample from.
            model: The model used to compute prediction confidence.
            batch_size: Batch size for model inference.
            uncertainty_threshold: Confidence threshold; samples with
                max probability below this are selected.
            device: Device to run model inference on.
        """
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.model = model.to(device)
        self.batch_size = batch_size
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device
        self.selected_indices = self._select_uncertain_samples()

    def _select_uncertain_samples(self) -> List[int]:
        self.model.eval()
        selected = []
        with torch.no_grad():
            n = len(self.dataset)
            pbar = tqdm(range(0, n, self.batch_size), desc="Selecting uncertain samples", 
                              unit="batch")
            for start in pbar:
                batch_indices = list(range(start, min(start + self.batch_size, n)))
                # Use list comprehension for faster image collection.
                images = [self.dataset[i][0] for i in batch_indices]
                batch = torch.stack(images).to(self.device)
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                max_confidences, _ = torch.max(probs, dim=1)
                # Vectorized selection of uncertain samples.
                mask = max_confidences < self.uncertainty_threshold
                uncertain_indices = np.array(batch_indices)[mask.cpu().numpy()]
                selected.extend(uncertain_indices.tolist())
                pbar.set_postfix({"selected": len(selected)})
                pbar.refresh()

        return selected

    def update(
        self,
        model: Optional[torch.nn.Module] = None,
        uncertainty_threshold: Optional[float] = None,
    ) -> None:
        """
        Dynamically update the sampler based on the current model state
        or a new uncertainty threshold.
        This method re-computes the uncertain sample indices.

        Args:
            model: (Optional) A new model; if provided, will update the current model.
            uncertainty_threshold: (Optional) New confidence threshold.
        """
        if model is not None:
            self.model = model.to(self.device)
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold
        self.selected_indices = self._select_uncertain_samples()
        logger.info(
            "Updated sampler with %d uncertain samples.", len(self.selected_indices)
        )

    def __iter__(self):
        # Optionally, you can shuffle the selected indices each iteration.
        return iter(self.selected_indices)

    def __len__(self) -> int:
        return len(self.selected_indices)


# Was too slow to use a generator for the dataset.
# pylint: disable=too-many-instance-attributes
class NPZImageDataset(Dataset):
    """
    PyTorch Dataset for loading images and labels from multiple .npz files.
    Each .npz file is expected to contain 'images' and 'labels' arrays.
    Efficiently caches the last opened file to minimize disk I/O.
    """

    def __init__(
        self,
        *args: Any,
        npz_dir: str = RESULTS_DIR,
        transform: Optional[Callable] = None,
        file_filter: Optional[str] = r"train",
        max_cache_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            npz_dir: Directory containing .npz files.
            transform: Optional transform to be applied on a sample.
            file_filter: Optional regex pattern to filter files by name (default: 'train').
            max_cache_size: Maximum number of .npz files to cache.
        """
        super().__init__(*args, **kwargs)
        if not os.path.isdir(npz_dir):
            raise ValueError(f"Directory {npz_dir} does not exist.")

        # Use os.scandir for faster directory listing
        with os.scandir(npz_dir) as it:
            all_files = [
                entry.name
                for entry in it
                if entry.is_file() and entry.name.endswith(".npz")
            ]

        if file_filter is not None:
            pattern = re.compile(file_filter)
            filtered_files = [f for f in all_files if pattern.search(f)]
        else:
            filtered_files = all_files

        self.npz_files: List[str] = [os.path.join(npz_dir, f) for f in filtered_files]
        self.transform: Optional[Callable] = transform
        self.data_index: List[Tuple[str, int]] = []
        self.file_sample_counts: List[int] = []
        self._build_index()

        # Cache only the last opened file.
        self._last_file_path: Optional[str] = None
        self._last_data: Optional[Any] = None

        self._max_cache_size = max_cache_size
        if max_cache_size <= 0:
            raise ValueError("max_cache_size must be greater than 0.")

    def _build_index(self) -> None:
        """
        Builds an index mapping each sample to its file and index within that file.
        Uses list comprehensions to reduce overhead when concatenating indices.
        """
        indices = []
        for file_path in self.npz_files:
            # We use mmap_mode to avoid loading the whole file into memory.
            with np.load(file_path, mmap_mode="r") as data:
                num_samples = cast(np.array, data["images"]).shape[0]  # pylint: disable=no-member
                indices.append([(file_path, i) for i in range(num_samples)])
                self.file_sample_counts.append(num_samples)
        self.data_index = list(chain.from_iterable(indices))

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of samples across all .npz files.
        """
        return len(self.data_index)

    # pylint: disable=attribute-defined-outside-init
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (image, label).
        """
        file_path, sample_idx = self.data_index[idx]
        # Initialize a cache for multiple NPZ files if not already created.
        if not hasattr(self, "_file_cache"):
            self._file_cache = {}
            self._file_cache_order = []

        if file_path in self._file_cache:
            data = self._file_cache[file_path]
            # Refresh cache order.
            self._file_cache_order.remove(file_path)
            self._file_cache_order.append(file_path)
        else:
            data = np.load(file_path, mmap_mode="r")
            self._file_cache[file_path] = data
            self._file_cache_order.append(file_path)
            # If cache exceeds max size, remove the least recently used file.
            if len(self._file_cache_order) > self._max_cache_size:
                oldest = self._file_cache_order.pop(0)
                self._file_cache.pop(oldest).close()

        image = data["images"][sample_idx]
        label = data["labels"][sample_idx]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label

    def __del__(self) -> None:
        """
        Ensures that the last opened .npz file is properly closed.
        """
        if hasattr(self, "_last_data") and self._last_data is not None:
            self._last_data.close()
