from typing import List

from gradio import ChatMessage

""" History of chat messages for a chatbot. """
History = List[ChatMessage]

""" List of examples for a Gradio Examples component. """
Examples = List[List[str]]
