import logging
from typing import Optional

import gradio as gr

from src.components import PDFReader
from src.components.chatbot import Chatbot


def app(
        host: str = "127.0.0.1",
        port: int = 7860,
        debug: bool = False,

        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
):
    """
    Main function to run Gradio application.

    Args:
        host (str): Host IP address of the server
        port (int): Port number of host server
        debug (bool): Debug mode for development

        ssl_keyfile (Optional[str]): The SSL key file path.
        ssl_certfile (Optional[str]): The SSL certificate file path.

    Returns:
        None
    """
    logging.debug("Starting the Gradio application")

    with gr.Blocks() as application:
        with gr.Row():
            with gr.Column(scale=1):  # Prend 2/3 de la largeur
                PDFReader()

            with gr.Column(scale=2):  # Prend 2/3 de la largeur
                Chatbot()

    application.launch(
        server_name=host,
        server_port=port,
        debug=debug,

        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


if __name__ == "__main__":
    app()
