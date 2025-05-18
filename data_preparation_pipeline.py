"""Pipelines for preparing data for further model training."""

import io
import os
from typing import List, Tuple, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipForImageClassification

from data_utils import get_file_path, load_cleaned_dataset_part
from utils import TimedLog, get_logger

logger = get_logger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _get_indoor_outdoor_net() -> Tuple[
    SiglipForImageClassification, AutoImageProcessor
]:
    """
    Loads the IndoorOutdoorNet model and processor.
    Returns:
        model: The loaded SiglipForImageClassification model.
        processor: The loaded AutoImageProcessor.
    """
    model_name = "prithivMLmods/IndoorOutdoorNet"
    model = SiglipForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model.to(device)
    return model, processor


def _classify_environment_images_batch(
    images: List[ImageFile],
    processor: AutoImageProcessor,
    model: SiglipForImageClassification,
) -> List[float]:
    """
    Classifies a batch of images as indoor or outdoor.
    Args:
        images: A list of PIL Image objects to classify.
        processor: The AutoImageProcessor to preprocess the images.
        model: The SiglipForImageClassification model to use for classification.
    Returns:
        A list of tuples containing the probabilities for outdoor class.
    """

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    return [p[1] for p in probs]


def _save_batch(
    i: int,
    df: pd.DataFrame,
    results_dir: str = os.path.join(os.path.dirname(__file__), "data_prepared"),
):
    """
    Saves a batch of data to a parquet file.

    Args:
        i: The index of the batch.
        df: The DataFrame containing the data to save.
        results_dir: The directory to save the parquet file.
    Returns:
        The file path of the saved parquet file.
    """
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(
        results_dir,
        f"initial_set_part_{i}.parquet",
    )

    df.to_parquet(
        file_path,
        index=False,
    )
    logger.debug("Saved batch %d to parquet file.", i)
    return file_path


def deduplicate_embeddings(
    embeddings: np.ndarray, threshold: float = 0.95, return_pairs: bool = False
) -> Tuple[List[int], List[Tuple[int, int]]] | List[int]:
    """
    Identifies duplicates in the embeddings based on cosine similarity.

    Args:
        embeddings: A 2D numpy array of shape (num_embeddings, embedding_dim).
        threshold: Cosine similarity threshold for considering two embeddings as duplicates.
        return_pairs: Whether to return the pairs of duplicate indices.
    Returns:
        unique_indices: List of indices of unique embeddings.
        duplicate_pairs: List of tuples containing indices of duplicate pairs.
    """
    similarity_matrix = cosine_similarity(embeddings)
    num_embeddings = embeddings.shape[0]
    to_remove = set()
    duplicate_pairs = []

    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            if similarity_matrix[i, j] >= threshold:
                to_remove.add(j)
                duplicate_pairs.append((i, j))

    unique_indices = [i for i in range(num_embeddings) if i not in to_remove]
    if not return_pairs:
        return unique_indices
    return unique_indices, duplicate_pairs


# pylint: disable=too-many-statements,too-many-branches
# pylint: disable=magic-value-comparison,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments
def prepare_initial_set(
    start_id: int = 0,
    end_id: int | None = None,
    stop_after_k_rows: int = 2 * 10**6,
    batch_size: int = 128,
    rows_per_file: int = 10**4,
    min_unique_pixels_num: int = 10,
    certainty_cutoff: float = 0.1,
    deduplication_threshold: float = 0.95,
    results_dir: str = os.path.join(os.path.dirname(__file__), "data_prepared"),
    data_dir: str = os.path.join(os.path.dirname(__file__), "data"),
    remove_original_files: bool = True,
) -> List[str]:
    """
    Prepares the initial set of data for processing.

    Args:
        start_id: The starting part number to load.
        end_id: The ending part number to load.
        stop_after_k_rows: The maximum number of rows to process.
        batch_size: The size of the batch for processing.
        rows_per_file: The number of rows per parquet file.
        min_unique_pixels_num: The minimum number of unique pixels to consider an image valid.
        certainty_cutoff: The cutoff for classifying images as indoor or outdoor.
        deduplication_threshold: The threshold for deduplicating embeddings.
        results_dir: The directory to save the processed data.
        data_dir: The directory to load and save the original data from.
        remove_original_files: Whether to remove the original files after processing.
    Returns:
        A list of file paths to the saved parquet files.
    """
    if end_id is None:
        end_id = 330
    else:
        stop_after_k_rows = None

    with TimedLog(logger, f"Prepared dataset parts {start_id} to {end_id}"):
        dt = []

        saved_files = []
        saved_rows = 0
        df_to_save = None

        processed_parts = 0
        processed_rows = 0

        sig_stop = False
        model, processor = _get_indoor_outdoor_net()
        for i in range(start_id, end_id):
            with TimedLog(
                logger, f"Processed dataset ({i - start_id} / {end_id - start_id})"
            ):
                with TimedLog(
                    logger, f"Loaded images from dataset part {i - start_id}."
                ):
                    df = pd.DataFrame(
                        load_cleaned_dataset_part(
                            dataset_id=i, target=data_dir, show_example=False
                        )
                    )
                    initial_rows = len(df)

                    df["image"] = df["image"].apply(
                        lambda x: np.array(Image.open(io.BytesIO(x)).convert("RGB"))
                    )

                # remove corrupted images
                with TimedLog(
                    logger,
                    f"Removed corrupted images from dataset part {i - start_id}.",
                ):
                    unique_pixels_num = df["image"].apply(lambda x: len(np.unique(x)))
                    corrupted_mask = unique_pixels_num < min_unique_pixels_num
                    df = df[~corrupted_mask]
                    df = df.reset_index(drop=True)

                # remove duplicates based on embeddings
                with TimedLog(
                    logger,
                    f"Removed duplicated images from dataset part {i - start_id}.",
                ):
                    embeddings = np.array(df["embedding"].tolist())
                    unique_indices = deduplicate_embeddings(
                        embeddings,
                        threshold=deduplication_threshold,
                        return_pairs=False,
                    )
                    df = cast(pd.DataFrame, df.iloc[unique_indices, :])
                    df = df.reset_index(drop=True)

                logger.info(
                    "Unique images from dataset part %d: %d", i - start_id, len(df)
                )

                # classify images in batches
                with TimedLog(
                    logger, f"Classified images from dataset part {i - start_id}."
                ):
                    images = df["image"].tolist()
                    batches = [
                        images[i : i + batch_size]  # noqa: E203
                        for i in range(0, len(images), batch_size)
                    ]
                    probs = []
                    for batch in tqdm(batches, desc="Classifying images", unit="batch"):
                        probs.extend(
                            _classify_environment_images_batch(batch, processor, model)
                        )
                    df["outdoor_prob"] = probs

                # filtering based on certainty
                with TimedLog(
                    logger,
                    f"Removed uncertain images from dataset part {i - start_id}.",
                ):
                    certain_mask = (df["outdoor_prob"] < certainty_cutoff) | (
                        df["outdoor_prob"] > 1 - certainty_cutoff
                    )
                    df = df[certain_mask]
                    df = df.reset_index(drop=True)

                    df["is_outdoor"] = df["outdoor_prob"] > 0.5
                    df.drop(columns=["outdoor_prob"], inplace=True)

                processed_rows += len(df)
                processed_parts += 1

                # saving and removing original files
                with TimedLog(
                    logger, f"Performed IO operations from dataset part {i - start_id}."
                ):
                    df = df[["image", "is_outdoor", "embedding"]]

                    while True:
                        if df_to_save is None:
                            df_to_save = df.head(rows_per_file)
                            df = df.iloc[rows_per_file:]
                        else:
                            to_be_added = rows_per_file - len(df_to_save)
                            df_to_save = pd.concat(
                                [df_to_save, df.head(to_be_added)], ignore_index=True
                            )
                            df = df.iloc[to_be_added:]

                        if len(df_to_save) >= rows_per_file:
                            saved_files.append(
                                _save_batch(
                                    len(saved_files),
                                    df_to_save,
                                    results_dir=results_dir,
                                )
                            )
                            df_to_save = None
                            saved_rows += rows_per_file
                        if len(df) < rows_per_file:
                            break

                    if saved_rows >= stop_after_k_rows or (
                        end_id is not None and i >= end_id - 1
                    ):
                        if df_to_save is not None:
                            saved_files.append(
                                _save_batch(
                                    len(saved_files),
                                    df_to_save,
                                    results_dir=results_dir,
                                )
                            )
                            df_to_save = None
                        sig_stop = True

                    if remove_original_files:
                        file_path = get_file_path(dataset_id=i, target=data_dir)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.debug("Removed original file %s", file_path)

                # logging using lazy formatting
                if stop_after_k_rows is not None:
                    logger.info(
                        "Processed part %d (%d / %d rows accepted). Total saved rows: %d / %d.",
                        i - start_id,
                        processed_rows,
                        initial_rows,
                        saved_rows,
                        stop_after_k_rows,
                    )
                else:
                    logger.info(
                        "Processed part %d of %d (%d / %d rows accepted). Total saved rows: %d.",
                        i - start_id,
                        end_id - start_id,
                        processed_rows,
                        initial_rows,
                        saved_rows,
                    )

            if sig_stop:
                break

    logger.info(
        "Processed %d parts. Processed %d rows. Saved %d rows.",
        processed_parts,
        processed_rows,
        saved_rows,
    )
    return dt


if __name__ == "__main__":
    prepare_initial_set()
