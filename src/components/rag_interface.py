import fitz
import gradio
import gradio as gr
from dotenv import load_dotenv
from gradio import ChatMessage
from overrides import overrides

from src import ENV_PATH
from src._pymupdf import highlight_text
from src._typing import History
from src.client import RagClient
from src.components import PDFReader
from src.components.chat_interface import ChatInterface


class RagInterface(ChatInterface):
    """
    Rag interface for the Gradio application.
    
    gr.ChatInterface don't allow to return gr.update, because 'echo'
    function must just return the response in string format. This class
    return the response in string format and update the PDF display.
    """
    rag_client: RagClient
    pdf_reader: PDFReader
    chat_interface: ChatInterface

    def __init__(
            self,
            model_id: str,
            hf_token: str,
    ):
        self.rag_client = RagClient(
            model_id=model_id,
            hf_token=hf_token,
        )

        with gradio.Blocks() as application:
            with gradio.Row():
                with gradio.Column(scale=1):  # Prend 2/3 de la largeur
                    self.pdf_reader = PDFReader()

                with gradio.Column(scale=2):  # Prend 2/3 de la largeur
                    super().__init__()
                    super().bind_events(
                        activate_chat_events=False,
                        activate_button_events=True,
                    )

            self.application = application

    @overrides(check_signature=False)
    def bind_events(self):
        """ Bind the events for the chat interface. """

        self.input.submit(
            fn=self.echo,
            inputs=[self.input],
            outputs=[self.input, self.chat, self.pdf_reader.pdf_display, self.pdf_reader.counter]
        )

        self.submit.click(
            fn=self.echo,
            inputs=[self.input],
            outputs=[self.input, self.chat, self.pdf_reader.pdf_display, self.pdf_reader.counter]
        )

    @overrides(check_signature=False)
    def echo(self, message: str) -> (str, History, fitz.Pixmap, str):
        """ Update the history, return the response in string format and update the PDF display. """

        user_message = ChatMessage("user", message)
        assistant_message = ChatMessage("assistant", self.invoke(message))

        # Append the messages to the history
        self.history.append(user_message)
        self.history.append(assistant_message)

        # Update the PDF display (highlight text)
        image, counter_update = self.pdf_reader.navigate_pdf(direction=0)

        # Clear input, return history, update PDF display (image, counter)
        return "", self.history, image, counter_update

    def invoke(self, message: str) -> str:
        """ Invoke the RAG model and highlight the text in the PDF document. """

        file_path = self.pdf_reader.file_path

        if file_path is not None:
            self.rag_client.load_pdf(file_path)

        message, list_document_context = self.rag_client.invoke(message)

        document = fitz.open(file_path)
        for document_context in list_document_context:
            text = document_context.page_content  # noqa nique tamre
            page = document_context.metadata['page']
            document = highlight_text(document, text, page)

        self.pdf_reader.pdf_document = document
        return message


if __name__ == "__main__":
    load_dotenv(ENV_PATH)

    with gr.Blocks() as application:
        RagInterface()

    application.launch()
