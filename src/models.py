import os
import time
from functools import lru_cache, wraps
from typing import Dict, Optional

from langchain_core.language_models import BaseLLM
from langchain_huggingface import HuggingFaceEndpoint
from loguru import logger


@lru_cache(maxsize=1)
def get_llm_model(
    model_id: str,
    hf_token: Optional[str] = None,
    max_new_tokens=512,
    **models_kwargs: Dict,
) -> BaseLLM:
    """
    Load a CausalLM model (local or cloud).

    Args:
        model_id (str): Model ID to load.
        hf_token (Optional[str]): Hugging Face API access token.

        max_new_tokens (int): Maximum number of tokens to generate.
        models_kwargs (Optional[dict]): Arguments to pass to the model.

    Returns:
        BaseLLM: Language model.
    """
    if hf_token is None:
        hf_token = os.environ["HF_TOKEN"]

    llm_model = _get_llm_model_hf_cloud(
        model_id=model_id,
        hf_token=hf_token,
        max_new_tokens=max_new_tokens,
        models_kwargs=models_kwargs,
    )

    logger.debug(f'Model "{model_id}" loaded successfully.')
    llm_model.name = model_id

    return llm_model


def _get_llm_model_hf_cloud(
    model_id: str,
    hf_token: str,
    max_new_tokens=512,
    models_kwargs: Optional[dict] = None,
):
    """Load a model from Hugging Face Cloud."""
    logger.debug(f"Loading model '{model_id}' from Hugging Face Cloud.")

    llm_model = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=max_new_tokens,
        **models_kwargs,
    )

    return llm_model


def log_inference(model_id: str):
    """Decorator to log the inference time of a model."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Inference of model '{model_id}' started. (function: {func.__name__})")
            time_start = time.time()

            result = func(*args, **kwargs)

            time_end = time.time()
            logger.debug(f"Inference takes {time_end - time_start:.2f} seconds.")
            return result

        return wrapper

    return decorator
