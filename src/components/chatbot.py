import random
import time
from typing import Tuple

import gradio

from src._typing import History


class Chatbot:
    def __init__(self):
        gradio.ChatInterface(
            fn=respond,
            examples=["Quel est le sujet de ce document ?", "Résume moi ce document"],
        )


def respond(message: str, history: History) -> Tuple[str, History]:
    responses = ["Hello World!", "Hi there!", "Greetings!"]
    time.sleep(random.uniform(0, 0.333))
    response = random.choice(responses)

    if message == "42":
        response = "The answer to the ultimate question of life, the universe and everything is 42."

    return response
