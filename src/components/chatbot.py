import random
import time
from typing import List, Tuple

import gradio
from gradio import ChatMessage

History = List[ChatMessage]


class Chatbot:
    def __init__(self):
        gradio.ChatInterface(
            fn=respond,
            examples=["Quel est le sujet de ce document ?", "Résume moi ce document"],
            title="Echo Bot"
        )


def respond(message: str, history: History) -> Tuple[str, History]:
    responses = ["Hello World!", "Hi there!", "Greetings!"]
    time.sleep(random.uniform(0, 0.333))
    response = random.choice(responses)

    if message == "42":
        response = "The answer to the ultimate question of life, the universe and everything is 42."

    print(history, len(history))
    return response
