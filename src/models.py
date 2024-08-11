import os
import time
from functools import lru_cache, wraps
from typing import Dict, Optional, Union

from gradio import ChatMessage
from langchain import hub
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import SystemMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src._typing import History


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
    """Décorateur pour logger l'exécution d'un modèle."""

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



if __name__ == "__main__":
    history = [
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing great. How can I help you today?"),
        ChatMessage(role="user", content="I'd like to show off how chat templating works!"),
        ChatMessage(role="assistant", content="I'm doing great. How can I help you today?"),
    ]

    chat_template = hub.pull("athroniaeth/rag-chat-template")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    history_to_template(history, chat_template, tokenizer)
