"""Logic for querying the LLM Studio API."""

from typing import Any, Dict, List, Tuple

import lmstudio as lms
from tqdm.notebook import tqdm

from utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant. You will be given an image and its description.
Your task is to return one digit indicating if image shows outdoor or indoor scene.

For outdoor scene return 1, for indoor scene return 0. If image depicts neither, return N.

For example, if the image shows a person in a park, return 1.
If an image shows a person in a room, return 0.
If the image is not clear, return N. If image is a photo of an map, return N.
Try your best not to return N.

Think twice before answering, I really need this for my research. I believe in you.
Remember to respond with a single digit, without any additional text or explanation.
"""

DESCRIPTION_PROMPT = """
# Title: {title}.

# Description
{description}
"""

CHAT_KWARGS: Dict[str, Any] = {
    "temperature": 0.0,
    "topP": 0.95,
    "maxTokens": 1,
    "cpuThreads": 6,
}
LLM_KWARGS: Dict[str, Any] = {
    "contextLength": 1024,
    "gpuOffload": "max",
    "flashAttention": True,
    "keepModelInMemory": True,
    "seed": 42,
    "evalBatchSize": 1024,
    "tryMmap": True,
}


# pylint: disable=magic-value-comparison
def query_llm(model: str, dt: List[Dict[str, Any]]) -> List[Tuple[bool | None, str]]:
    """
    Queries the LM Studio API with the given prompts.

    Args:
        model: The model to use for generating responses.
        dt: The dataset to process.
    Returns:
        A list of tuples containing the validated response and the original response.
    """
    client = lms.get_default_client()
    llm = client.llm.model(model, config=LLM_KWARGS)

    try:
        responses = []
        for i, entry in enumerate(tqdm(dt, desc="Querying LLM", unit="request")):
            try:
                chat = lms.Chat()

                chat.add_system_prompt(SYSTEM_PROMPT)
                chat.add_user_message(
                    DESCRIPTION_PROMPT.format(
                        title=entry["title"], description=entry["description"]
                    ),
                    images=[lms.prepare_image(entry["image"])],
                )

                result = llm.respond(chat, config=CHAT_KWARGS)
                prediction = result.content.strip()

                if result.stats.stop_reason != "eosFound" and prediction not in [
                    "0",
                    "1",
                    "N",
                ]:
                    logger.warning(
                        "Unexpected stop reason: %s for entry %s: %s",
                        result.stats.stop_reason,
                        i,
                        prediction,
                    )
                elif prediction not in ["0", "1", "N"]:
                    logger.warning(
                        "Unexpected prediction: %s for entry %s: %s",
                        prediction,
                        i,
                        entry,
                    )

                parsed = (
                    True if prediction == "1" else False if prediction == "0" else None
                )
                responses.append((parsed, prediction))

            except Exception as e:
                logger.error("Error querying LLM with entry %s", entry)
                raise e
    finally:
        llm.unload()

    return responses


def get_outdoor_model_prediction(
    dt: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Retrieves the outdoor model prediction for a specific part of the dataset.

    Args:
        dt: The dataset to process.
        model: The model to use for predictions.
    Returns:
        The dataset with the outdoor model predictions added.
    """
    answers = query_llm(model=model, dt=dt)
    for i, entry in enumerate(answers):
        dt[i]["is_outdoor"] = entry[0]
        dt[i]["prediction"] = entry[1]
    return dt
