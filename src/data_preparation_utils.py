"""Data preparation utilities for handling file operations and data loading."""

import gc
import os
from typing import Dict, List, Tuple

import humanize
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from .data_extraction_pipeline import RESULTS_DIR as DATA_PREPARATION_RESULTS_DIR
from .utils import TimedLog, get_logger

logger = get_logger(__name__)

RESULTS_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "datasets")
)


def get_files(
    part: str, results_dir: str = DATA_PREPARATION_RESULTS_DIR
) -> Tuple[List[str], pd.DataFrame]:
    """
    Get the list of files in the results directory that match the given part.
    Args:
        part: The part of the file name to match.
        results_dir: The directory to search for files.
    Returns:
        A tuple containing:
            - A list of file paths that match the part.
            - A DataFrame with file names and their sizes.
    """
    results_files = os.listdir(results_dir)
    data = sorted(
        [
            os.path.join(results_dir, file_name)
            for file_name in results_files
            if file_name.endswith(".npz") and part in file_name
        ]
    )

    stats = [
        {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "size": humanize.naturalsize(os.path.getsize(file_path)),
            "part": part,
            "rows": 0,
            "cols": 0,
        }
        for file_path in data
    ]

    return data, pd.DataFrame(stats)


def load_parts(
    files: List[str], part: str, part_df: pd.DataFrame | None = None
) -> np.ndarray:
    """
    Load the parts from the given files.
    Args:
        files: The list of file paths to load.
        part: The part of the file to load.
        part_df: The DataFrame to update with file statistics.
    Returns:
        A numpy array containing the loaded parts.
    """
    parts = []
    for file in files:
        with np.load(file) as data:
            parts.append(data[part])

            if part_df is not None:
                part_df.loc[part_df["file_path"] == file, "rows"] = data[part].shape[0]
                part_df.loc[part_df["file_path"] == file, "cols"] = (
                    data[part].shape[1] if len(data[part].shape) > 1 else 1
                )

    return np.concatenate(parts, axis=0)


def create_df(indices_to_remove: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Create a DataFrame from the indices to remove.
    Args:
        indices_to_remove: The dictionary with file paths and indices to remove.
    Returns:
        A DataFrame with file paths and indices to remove.
    """
    df = pd.DataFrame(
        [
            {"file_path": file_path, "total_removed": len(indices)}
            for file_path, indices in sorted(
                indices_to_remove.items(), key=lambda x: x[0]
            )
        ]
    )
    return df


def save_batch(
    i: int,
    images: np.ndarray,
    labels: np.ndarray,
    option: str,
    results_dir: str = RESULTS_DIR,
):
    """
    Saves a batch of data to a parquet file.

    Args:
        i: The index of the batch.
        images: The images to save.
        labels: The labels to save.
        option: The option to save (e.g., "train", "test").
        results_dir: The directory where the batch will be saved.
    """
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(
        results_dir,
        f"{option}_{i}.npz",
    )

    np.savez_compressed(
        file_path,
        images=images,
        labels=labels,
    )

    logger.debug("Saved batch %d for %s.", i, option)


# pylint: disable=too-many-locals,too-many-positional-arguments,too-many-arguments
def process_and_save_batches(
    training_indices_to_remove_per_file: Dict[str, np.ndarray],
    validation_indices_to_remove_per_file: Dict[str, np.ndarray],
    test_indices_to_remove_per_file: Dict[str, np.ndarray],
    labels_files: List[str],
    images_files: List[str],
    target_file_size: int = 10000,
) -> None:
    """
    Process and save batches of images and labels in a memory efficient way.

    Args:
        training_indices_to_remove_per_file: Indices to remove for training.
        validation_indices_to_remove_per_file: Indices to remove for validation.
        test_indices_to_remove_per_file: Indices to remove for testing.
        labels_files: List of label files.
        images_files: List of image files.
        target_file_size: Target size for each batch file.
    """
    datasets = {
        "train": {
            "images": None,
            "labels": None,
            "counter": 0,
            "remove_map": training_indices_to_remove_per_file,
            "max_size": 80000,
            "size": 0,
        },
        "val": {
            "images": None,
            "labels": None,
            "counter": 0,
            "remove_map": validation_indices_to_remove_per_file,
            "max_size": 10000,
            "size": 0,
        },
        "test": {
            "images": None,
            "labels": None,
            "counter": 0,
            "remove_map": test_indices_to_remove_per_file,
            "max_size": 10000,
            "size": 0,
        },
    }

    # pylint: disable=unsubscriptable-object
    stop = False
    with TimedLog(logger, "Processing and saving batches of images and labels"):
        for i in tqdm(range(len(labels_files)), desc="Processing files", unit="file"):
            labels_file = labels_files[i]
            images_file = images_files[i]

            # Load data and promptly release memory when possible.
            labels = load_parts([labels_file], "labels")
            images = load_parts([images_file], "images")

            for key, data in datasets.items():
                indices_to_remove = data["remove_map"][labels_file]
                mask = np.ones(len(labels), dtype=bool)
                if len(indices_to_remove) > 0:
                    mask[indices_to_remove] = False

                # Append processed data
                if data["labels"] is None:
                    data["labels"] = labels[mask]
                    data["images"] = images[mask]
                else:
                    data["labels"] = np.concatenate(
                        (data["labels"], labels[mask]), axis=0
                    )
                    data["images"] = np.concatenate(
                        (data["images"], images[mask]), axis=0
                    )

                # Save batch if current data exceeds target_file_size
                if len(data["labels"]) > target_file_size:
                    save_batch(
                        data["counter"],
                        data["images"][:target_file_size],
                        data["labels"][:target_file_size],
                        key,
                    )
                    data["images"] = data["images"][target_file_size:]
                    data["labels"] = data["labels"][target_file_size:]
                    data["counter"] += 1
                    data["size"] += target_file_size

            if (
                datasets["test"]["size"] >= datasets["test"]["max_size"]
                and datasets["val"]["size"] >= datasets["val"]["max_size"]
                and datasets["train"]["size"] >= datasets["train"]["max_size"]
            ):
                stop = True

            # Free file-specific data and trigger garbage collection
            del labels, images, mask, indices_to_remove
            gc.collect()

            if stop:
                break

        # Save any remaining samples and clean up
        for key, data in datasets.items():
            if not stop and data["labels"] is not None and len(data["labels"]) > 0:
                save_batch(data["counter"], data["images"], data["labels"], key)
            del data["images"], data["labels"], data["counter"]
            gc.collect()


def get_remove_indices_per_file(
    mask: np.ndarray, files_stats_df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Translate the mask to the file level.
    Args:
        mask: The mask to translate.
        files_stats_df: The DataFrame with file names and their sizes.
    Returns:
        A dictionary where the keys are file paths and the values are arrays of indices to remove.
    """
    indices_to_remove = np.where(~mask)[0]

    i = 0
    cumulative_rows = 0
    indices_to_remove_per_file = {}
    for _, row in files_stats_df.sort_values(by="file_path").iterrows():
        file_path = row["file_path"]
        rows = row["rows"]
        translated = []

        while (
            i < len(indices_to_remove) - 1
            and indices_to_remove[i] < cumulative_rows + rows - 1
        ):
            translated.append(indices_to_remove[i] - cumulative_rows)
            i += 1

        cumulative_rows += rows
        indices_to_remove_per_file[file_path] = np.array(translated)

    return indices_to_remove_per_file
