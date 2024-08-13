"""Module for simplifying complex type hints in the project."""

from typing import Callable, List

from gradio import ChatMessage

""" History of chat messages for a chatbot. """
History = List[ChatMessage]

""" List of examples for a Gradio Examples component. """
Examples = List[List[str]]

""" Inference callable for a chatbot. """
InferenceCallable = Callable[[str, History], str]
