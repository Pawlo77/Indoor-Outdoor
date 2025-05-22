"""Utility functions for data loading and preparation."""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image
from PIL.ImageFile import ImageFile

from .llm_utils import get_outdoor_model_prediction
from .utils import TimedLog, get_logger

logger = get_logger(__name__)

URL_TEMPLATE: str = (
    "https://huggingface.co/datasets/wikimedia/wit_base/resolve"
    "/main/data/train-{id}-of-00330.parquet?download=true"
)

DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


# pylint: disable=magic-value-comparison
def get_file_path(dataset_id: int = 0, target: str = DATA_DIR) -> str:
    """
    Returns the file path for a specific part of the dataset.

    Args:
        dataset_id: The part number to get the file path for.
        target: The target directory where the part is saved
    Returns:
        str: The file path for the specified part.
    Raises:
        ValueError: If the id is not between 0 and 330.
    """
    if dataset_id < 0 or dataset_id > 330:
        raise ValueError("id must be between 0 and 330")
    os.makedirs(target, exist_ok=True)
    dataset_id = str(dataset_id).zfill(5)
    return os.path.join(target, f"train-{dataset_id}-of-00330.parquet")


def download_dataset_part(dataset_id: int = 0, target: str = DATA_DIR) -> None:
    """
    Downloads a part of the dataset.

    Args:
        dataset_id: The part number to download.
        target: The target directory to save the downloaded part.
    """
    file_path = get_file_path(dataset_id=dataset_id, target=target)
    dataset_id = str(dataset_id).zfill(5)

    if os.path.exists(file_path):
        logger.debug("File %s already exists, skipping download.", file_path)
        return

    response = requests.get(URL_TEMPLATE.format(id=dataset_id), stream=True, timeout=10)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    logger.debug("Downloaded file to %s", file_path)


def load_dataset_part(dataset_id: int = 0, target: str = DATA_DIR) -> pd.DataFrame:
    """
    Loads a part of the dataset.

    Args:
        dataset_id: The part number to load.
        target: The target directory where the part is saved.

    Returns:
        DatasetDict: The loaded dataset part.
    """
    download_dataset_part(dataset_id=dataset_id, target=target)

    file_path = get_file_path(dataset_id=dataset_id, target=target)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    return pd.read_parquet(file_path)


# pylint: disable=magic-value-comparison
def load_cleaned_dataset_part(
    dataset_id: int = 0,
    include_embedding: bool = True,
    target: str = DATA_DIR,
    show_example: bool = True,
) -> List[Dict[str, Any]]:
    """
    Loads a cleaned part of the dataset.

    Args:
        dataset_id: The part number to load.
        include_embedding: Whether to include the embedding in the loaded dataset.
        target: The target directory where the part is saved.
        show_example: Whether to show an example entry from the dataset.
    Returns:
        The loaded dataset part.
    """
    # Load the dataset part
    dataset_part = load_dataset_part(dataset_id=dataset_id, target=target)

    # Print the dataset part
    if show_example:
        logger.info("Sample entry from dataset part %d:", dataset_id)
        example = dataset_part.head(1).to_dict()
        for k, v in example.items():
            v = str(v)
            formatted_v = v[:100] + "..." if len(v) > 100 else v
            logger.info("\t%s: %s", k, formatted_v)

        language_idx = list(example["wit_features"]["language"]).index("en")
        for k, v in example["wit_features"].items():
            v = str(v[language_idx])
            if len(v) > 100:
                formatted_v = v[:100] + "..."
            else:
                formatted_v = v
            logger.info("\t\t%s: %s", k, formatted_v)

    dt = []
    for i, entry in dataset_part.iterrows():
        try:
            language_idx = list(entry["wit_features"]["language"]).index("en")
        except ValueError:
            logger.debug(
                "Language 'en' not found in entry %d - %s, skipping...",
                i,
                entry["wit_features"]["language"],
            )
            continue

        new_entry = {
            "image": entry["image"]["bytes"],
            "title": entry["wit_features"]["hierarchical_section_title"][language_idx],
            "description": entry["wit_features"]["context_section_description"][
                language_idx
            ]
            or entry["wit_features"]["context_page_description"][language_idx],
        }

        if include_embedding:
            new_entry["embedding"] = entry["embedding"]

        dt.append(new_entry)

    logger.debug("Loaded %d entries from dataset part %d.", len(dt), dataset_id)
    return dt


# pylint: disable=too-many-positional-arguments,too-many-arguments
def perform_experiment(
    model: str,
    experiment_id: int = 0,
    target_dir: str = os.path.join(os.path.dirname(__file__), "experiments"),
    dataset_part: List[Dict[str, Any]] = None,
    return_dataset_part: bool = False,
    overwrite: bool = False,
) -> str | List[Dict[str, Any]]:
    """
    Prepares a part of the dataset for testing.

    Args:
        model: The model to use for predictions.
        experiment_id: The part number to prepare.
        target_dir: The target directory to save the prepared part.
        dataset_part: The dataset part to prepare. If None, it will be loaded.
        return_dataset_part: Whether to return the prepared dataset part.
        overwrite: Whether to overwrite the existing prepared part. If False, it will skip
            if it already exists.
    Returns:
        str: The path to the prepared dataset part.
    """
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, f"{experiment_id}.parquet")
    if not overwrite and os.path.exists(file_path):
        logger.debug("File %s already exists, skipping preparation.", file_path)
        if return_dataset_part:
            return pd.read_parquet(file_path)
        return file_path

    with TimedLog(logger, f"Prepared dataset part {experiment_id} with model {model}"):
        dataset_part = dataset_part or load_cleaned_dataset_part(
            dataset_id=experiment_id
        )
        prepared_dataset_part = get_outdoor_model_prediction(
            dt=dataset_part, model=model
        )

        with open(file_path, "wb") as f:
            df = pd.DataFrame(prepared_dataset_part)
            df.to_parquet(f)

        if return_dataset_part:
            return prepared_dataset_part
        return file_path


def process_image(img: Any, image_shape: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Processes the image bytes into a numpy array.
    Args:
        img: The image bytes to process.
        image_shape: The shape of the image to resize to.
    Returns:
        The processed image as a numpy array.
    """
    if isinstance(img, bytes):
        img = io.BytesIO(img)
    if not isinstance(img, ImageFile):
        img = Image.open(img)
    return np.array(img.convert("RGB").resize(image_shape))


# pylint: disable=magic-value-comparison
def load_cleaned_dataset_part_minimal(
    dataset_id: int = 0,
    target: str = DATA_DIR,
    image_shape: Tuple[int, int] = (256, 256),
    executor: ThreadPoolExecutor | None = None,
) -> pd.DataFrame:
    """
    Loads a cleaned part of the dataset.

    Args:
        dataset_id: The part number to load.
        image_shape: The shape of the image to resize to.
        target: The target directory where the part is saved.
        executor: The executor to use for parallel processing.
            If not provided, one will be created with the number of available CPU cores.
    Returns:
        The loaded dataset part.
    """
    # Load the dataset part
    dataset_part = load_dataset_part(dataset_id=dataset_id, target=target)

    df = dataset_part[["image", "embedding"]]
    df["image"] = df["image"].apply(lambda x: x["bytes"])

    local_executor = False
    if executor is None:
        executor = ThreadPoolExecutor()
        local_executor = True

    df["image"] = list(
        executor.map(
            partial(
                process_image,
                image_shape=image_shape,
            ),
            df["image"],
        )
    )

    df = df.convert_dtypes()

    if local_executor:
        executor.shutdown(wait=True)

    logger.debug("Loaded %d entries from dataset part %d.", len(df), dataset_id)
    return df
