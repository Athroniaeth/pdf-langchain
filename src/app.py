import logging
from typing import Optional

import gradio as gr


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
    logging.info("Starting the Gradio application")
    def greet(name):
        return "Hello " + name + "!"

    demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

    demo.launch(
        server_name=host,
        server_port=port,
        debug=debug,
    )


if __name__ == "__main__":
    app()
