import logging
from functools import partial
from typing import Optional

import gradio
from gradio import ChatInterface

from src._typing import History
from src.client import RagClient
from src.components import PDFReader


def app(
        model_id: str,
        hf_token: str,

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

    def echo(
            message: str,
            history: History,
            pdf_reader: PDFReader,
            rag_client: RagClient,
    ) -> str:
        return message

    rag_client = RagClient(model_id=model_id, hf_token=hf_token)

    with gradio.Blocks() as application:
        with gradio.Row():
            with gradio.Column(scale=1):  # Prend 2/3 de la largeur
                pdf_reader = PDFReader()

            with gradio.Column(scale=2):  # Prend 2/3 de la largeur
                interface = ChatInterface(
                    fn=partial(echo, pdf_reader=pdf_reader, rag_client=rag_client),
                    examples=["Quel est le sujet de ce document ?", "Résume moi ce document"],
                )

    application.launch(
        server_name=host,
        server_port=port,
        debug=debug,

        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


if __name__ == "__main__":
    app()
