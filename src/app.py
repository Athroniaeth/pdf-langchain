import logging
from typing import Optional

import gradio

from src.components.rag_interface import RagInterface


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
        model_id (str): The model ID of Hugging Face LLM.
        hf_token (str): The Hugging Face token.
        host (str): Host IP address of the server
        port (int): Port number of host server
        debug (bool): Debug mode for development

        ssl_keyfile (Optional[str]): The SSL key file path.
        ssl_certfile (Optional[str]): The SSL certificate file path.

    Returns:
        None
    """
    logging.debug("Starting the Gradio application")

    css = """
        #counter input { text-align: center; }
        .lg { font-size: calc(var(--button-large-text-size) - 1px); }
        """

    with gradio.Blocks(css=css) as application:
        rag_interface = RagInterface(model_id=model_id, hf_token=hf_token)

        rag_interface.bind_events()

    application.launch(
        ssl_verify=False,
        debug=debug,
        server_name=host,
        server_port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        max_file_size="1mb",
    )


if __name__ == "__main__":
    app()
