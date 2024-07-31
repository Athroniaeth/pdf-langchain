import gradio as gr


def app(
        host: str = "127.0.0.1",
        port: int = 7860,
        debug: bool = False,
):
    """
    Main function to run Gradio application.

    Args:
        host (str): Host IP address of the server
        port (int): Port number of host server
        debug (bool): Debug mode for development

    Returns:
        None
    """

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
