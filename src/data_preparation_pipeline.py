"""Pipelines for preparing data for further model training."""

import argparse
import concurrent.futures
import gc
import os
from functools import partial
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL.ImageFile import ImageFile
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipForImageClassification

from .data_utils import (
    download_dataset_part,
    get_file_path,
    load_cleaned_dataset_part_minimal,
)
from .utils import TimedLog, get_logger

logger = get_logger(__name__)

RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data_prepared")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
    model.to(DEVICE)
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
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    return [p[1] for p in probs]


def save_batch(
    i: int,
    df: pd.DataFrame,
    results_dir: str = RESULTS_DIR,
    name_format: str = "initial_set_part_{i}_",
):
    """
    Saves a batch of data to a parquet file.

    Args:
        i: The index of the batch.
        df: The DataFrame containing the data to save.
        results_dir: The directory to save the parquet file.
        name_format: The format for the file name.
    """
    os.makedirs(results_dir, exist_ok=True)
    file_path_raw = os.path.join(
        results_dir,
        name_format.format(i=i),
    )

    images = np.array(df["image"].tolist(), dtype=np.uint8)
    labels = np.array(df["is_outdoor"].tolist(), dtype=np.bool_)
    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

    np.savez_compressed(
        file_path_raw + "images.npz",
        images=images,
    )
    np.savez_compressed(
        file_path_raw + "embeddings.npz",
        embeddings=embeddings,
    )
    np.savez_compressed(
        file_path_raw + "labels.npz",
        labels=labels,
    )

    logger.debug("Saved batch %d.", i)


def _normalize_batch(batch: np.ndarray) -> np.ndarray:
    """
    Normalizes a batch of embeddings.

    Args:
        batch: A 2D numpy array of shape (num_embeddings, embedding_dim).

    Returns:
        A 2D numpy array of normalized embeddings.
    """
    batch_norm = np.linalg.norm(batch, axis=1)
    batch_mask = batch_norm > 0
    normalized_batch = batch.copy()
    normalized_batch[batch_mask] /= batch_norm[batch_mask][:, np.newaxis]
    return normalized_batch


def deduplicate_embeddings_efficient(
    embeddings: np.ndarray, threshold: float = 0.95, file_size: int = 14000
) -> np.ndarray:
    """
    Efficiently deduplicates embeddings using cosine similarity with a dynamic nearest
    neighbor search. This function uses sklearn's NearestNeighbors (with cosine metric)
    to compare incoming batches of embeddings against the current set of unique embeddings.
    It is optimized for large datasets (e.g. around 1 million embeddings)
    and assumes an average batch size defined by file_size (default 14,000).

    Args:
        embeddings: A 2D numpy array of shape (num_embeddings, embedding_dim)
            containing the embeddings.
        threshold: A float representing the cosine similarity threshold above which
            two embeddings are considered duplicates. An embedding is accepted as unique
            only if its cosine similarity with each unique embedding is below this threshold.
        file_size: An integer representing the number of embeddings to process per batch.
            Also used as the initial count of embeddings to consider when
            building the starting unique set.

    Returns:
        A numpy array containing the indices of embeddings deemed unique.
    """
    unique_indices = np.arange(file_size)
    normalized_embeddings = _normalize_batch(embeddings[:file_size])

    # For unit vectors in cosine similarity, cosine distance = 1 - cosine similarity.
    distance_threshold = 1 - threshold
    nn_model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(
        normalized_embeddings
    )

    batches = [
        embeddings[i : i + file_size]  # noqa
        for i in range(file_size, len(embeddings), file_size)
    ]
    j = file_size
    for batch in tqdm(batches, desc="Processing batches", unit="batch"):
        normalized_batch = _normalize_batch(batch)

        distances, _ = nn_model.kneighbors(normalized_batch)

        unique = np.where(distances > distance_threshold)[0]
        unique_indices = np.concatenate([unique_indices, unique + j])
        normalized_embeddings = np.concatenate(
            [normalized_embeddings, normalized_batch[unique]]
        )
        j += len(unique)

        nn_model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(
            normalized_embeddings
        )

    return unique_indices


def deduplicate_embeddings(
    embeddings: np.ndarray, threshold: float = 0.95, return_pairs: bool = False
) -> Tuple[List[int], List[Tuple[int, int]]] | List[int]:
    """
    Identifies duplicates in the embeddings based on cosine similarity in a
    space efficient way. This version avoids constructing the full cosine
    similarity matrix at the cost of increased computation.

    Args:
        embeddings: A 2D numpy array of shape (num_embeddings, embedding_dim).
        threshold: Cosine similarity threshold for considering two embeddings as duplicates.
        return_pairs: Whether to return the pairs of duplicate indices.
    Returns:
        unique_indices: List of indices of unique embeddings.
        duplicate_pairs: List of tuples containing indices of
            duplicate pairs (if return_pairs is True).
    """
    num_embeddings = embeddings.shape[0]
    unique_indices = []
    duplicate_pairs = []
    # Store normalized embeddings for only the unique ones to save space.
    normalized_uniques = []

    for i in range(num_embeddings):
        vector = embeddings[i]
        norm = np.linalg.norm(vector)
        if norm == 0:
            # Skip zero vectors as they cannot be meaningfully compared.
            continue
        norm_vector = vector / norm

        is_duplicate = False
        # Compare with each unique embedding computed so far.
        for idx, uniq_vec in zip(unique_indices, normalized_uniques):
            # Cosine similarity for normalized vectors is just the dot product.
            cosine_sim = np.dot(norm_vector, uniq_vec)
            if cosine_sim >= threshold:
                is_duplicate = True
                duplicate_pairs.append((idx, i))
                break
        if not is_duplicate:
            unique_indices.append(i)
            normalized_uniques.append(norm_vector)

    if return_pairs:
        return unique_indices, duplicate_pairs
    return unique_indices


# pylint: disable=too-many-statements,too-many-branches
# pylint: disable=magic-value-comparison,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments
def prepare_initial_set(
    start_id: int = 0,
    end_id: int | None = None,
    stop_after_k_rows: int = 10**7,
    batch_size: int = 64,
    min_unique_pixels_num: int = 10,
    certainty_cutoff: float = 0.1,
    deduplication_threshold: float = 0.95,
    results_dir: str = RESULTS_DIR,
    data_dir: str = os.path.join(os.path.dirname(__file__), "data"),
    remove_original_files: bool = True,
    image_shape: Tuple[int, int] = (256, 256),
) -> None:
    """
    Prepares the initial set of data for processing.

    Args:
        start_id: The starting part number to load.
        end_id: The ending part number to load.
        stop_after_k_rows: The maximum number of rows to process.
        batch_size: The size of the batch for processing.
        min_unique_pixels_num: The minimum number of unique pixels to consider an image valid.
        certainty_cutoff: The cutoff for classifying images as indoor or outdoor.
        deduplication_threshold: The threshold for deduplicating embeddings.
        results_dir: The directory to save the processed data.
        data_dir: The directory to load and save the original data from.
        remove_original_files: Whether to remove the original files after processing.
        image_shape: The shape to resize the images to.
    """
    if end_id is None:
        end_id = 330
    else:
        stop_after_k_rows = None

    saved_rows = 0
    sig_stop = False
    model, processor = _get_indoor_outdoor_net()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Preload the first dataframe.
        future = executor.submit(
            partial(download_dataset_part, target=data_dir), start_id
        )

        for i in range(start_id, end_id):
            if future is not None:
                future.result()
                future = None

            # Preload the next df (if any) concurrently.
            if i < end_id - 1:
                next_future = executor.submit(
                    partial(download_dataset_part, target=data_dir), i + 1
                )
            else:
                next_future = None

            with TimedLog(logger, f"Loaded dataset part {i - start_id}"):
                df = load_cleaned_dataset_part_minimal(
                    dataset_id=i,
                    target=data_dir,
                    image_shape=image_shape,
                    executor=executor,
                )
                initial_rows = len(df)

            # Process the current dataset part.
            with TimedLog(logger, f"Processed dataset part {i - start_id}"):
                # Remove corrupted images.
                with TimedLog(
                    logger, f"Removed corrupted images from part {i - start_id}"
                ):
                    unique_pixels_num = np.array(
                        list(executor.map(lambda img: np.unique(img).size, df["image"]))
                    )
                    corrupted_mask = unique_pixels_num < min_unique_pixels_num
                    df = df[~corrupted_mask].reset_index(drop=True)

                    logger.debug(
                        "Removed %d corrupted images from dataset part %d",
                        np.sum(corrupted_mask),
                        i - start_id,
                    )

                # Remove duplicates based on embeddings.
                with TimedLog(logger, f"Removed duplicates from part {i - start_id}"):
                    embeddings = np.array(df["embedding"].tolist())
                    unique_indices = deduplicate_embeddings(
                        embeddings,
                        threshold=deduplication_threshold,
                        return_pairs=False,
                    )
                    removed_num = len(embeddings) - len(unique_indices)
                    df = df.iloc[unique_indices, :].reset_index(drop=True)

                    logger.debug(
                        "Removed %d duplicates from dataset part %d",
                        removed_num,
                        i - start_id,
                    )

                # Classify images in batches.
                with TimedLog(logger, f"Classified images from part {i - start_id}"):
                    images = df["image"].tolist()
                    batches = [
                        images[j : j + batch_size]  # noqa: E203
                        for j in range(0, len(images), batch_size)
                    ]
                    probs = []
                    for batch in tqdm(batches, desc="Classifying images", unit="batch"):
                        probs.extend(
                            _classify_environment_images_batch(batch, processor, model)
                        )
                    df["outdoor_prob"] = probs

                # Filtering based on certainty.
                with TimedLog(
                    logger, f"Filtered uncertain images from part {i - start_id}"
                ):
                    certain_mask = (df["outdoor_prob"] < certainty_cutoff) | (
                        df["outdoor_prob"] > 1 - certainty_cutoff
                    )
                    df = df[certain_mask].reset_index(drop=True)
                    df["is_outdoor"] = df["outdoor_prob"] > 0.5
                    df.drop(columns=["outdoor_prob"], inplace=True)

                    logger.debug(
                        "Filtered %d uncertain images from dataset part %d",
                        np.sum(~certain_mask),
                        i - start_id,
                    )

                # Save and remove original files.
                with TimedLog(logger, f"IO operations for part {i - start_id}"):
                    df = df[["image", "is_outdoor", "embedding"]]
                    save_batch(i, df, results_dir=results_dir)
                    saved_rows += len(df)

                    if (
                        stop_after_k_rows is not None
                        and saved_rows >= stop_after_k_rows
                    ) or (end_id is not None and i >= end_id - 1):
                        sig_stop = True

                    if remove_original_files:
                        file_path = get_file_path(dataset_id=i, target=data_dir)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.debug("Removed original file %s", file_path)

                    logger.info(
                        "Processed part %d (%d / %d rows accepted). Total saved rows: %d / %s.",
                        i - start_id,
                        len(df),
                        initial_rows,
                        saved_rows,
                        str(stop_after_k_rows)
                        if stop_after_k_rows is not None
                        else "?",
                    )

                # Cleanup to free memory.
                del (
                    df,
                    images,
                    batches,
                    probs,
                    unique_pixels_num,
                    corrupted_mask,
                    embeddings,
                )
                gc.collect()

            # Set future for next iteration.
            if next_future is not None:
                future = next_future
            if sig_stop:
                break


def main():
    """Main function to parse arguments and prepare the initial dataset."""

    parser = argparse.ArgumentParser(description="Prepare the initial dataset.")
    parser.add_argument(
        "--start", type=int, default=0, help="Starting part id (default: 0)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Ending part id (default: 330)"
    )
    parser.add_argument(
        "--remove-original-files",
        action="store_true",
        help="Flag to remove original files after processing",
        default=False,
    )
    args = parser.parse_args()

    prepare_initial_set(
        start_id=args.start,
        end_id=args.end,
        remove_original_files=args.remove_original_files,
    )


if __name__ == "__main__":
    main()
