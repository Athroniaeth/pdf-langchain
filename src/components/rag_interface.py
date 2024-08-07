import fitz
import gradio
import gradio as gr
from dotenv import load_dotenv
from gradio import ChatMessage
from overrides import overrides

from src import ENV_PATH
from src._pymupdf import highlight_text
from src._typing import History, Examples
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

    def __init__(
            self,
            model_id: str,
            hf_token: str,

            examples: Examples = None,
    ):
        if examples is None:
            examples = [
                ["What is the main idea of the document?"],
                ["Can you summarize the document?"],
            ]

        self.rag_client = RagClient(
            model_id=model_id,
            hf_token=hf_token,
        )

        with gradio.Row():
            with gradio.Column(scale=1):  # Prend 2/3 de la largeur
                self.pdf_reader = PDFReader()
                self.pdf_reader.bind_events()

            with gradio.Column(scale=2):  # Prend 2/3 de la largeur
                super().__init__(examples=examples)
                super().bind_events(
                    activate_chat_events=False,
                    activate_button_events=True,
                )

    @property
    def file_path(self):
        return self.pdf_reader.file_path

    @overrides(check_signature=False)
    def bind_events(self):
        """ Bind the events for the chat interface. """

        self.input.submit(
            fn=self.echo,
            inputs=[self.input],
            outputs=[self.input, self.chat, self.pdf_reader.display, self.pdf_reader.counter]
        )

        self.submit.click(
            fn=self.echo,
            inputs=[self.input],
            outputs=[self.input, self.chat, self.pdf_reader.display, self.pdf_reader.counter]
        )

        self.pdf_reader.file_input.change(
            fn=self.load_document,
            inputs=[self.pdf_reader.file_input],
            outputs=[self.pdf_reader.display]  # Show loading to the user
        )

    def load_document(self, file_path: str):
        """ Load the PDF document to the RAG client. """

        # Message to user to show that the PDF (db vector) is loading
        gr.Info("RAG is loading the PDF document...")

        if file_path is None:
            self.rag_client.clean_pdf()
        else:
            self.rag_client.load_pdf(file_path)

        # Message to user to show that the PDF (db vector) is loaded
        gr.Info("RAG successfully loaded the PDF document.")

        # Show loading to the user
        return gr.update()

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
        document = None
        message, list_document_context = self.rag_client.invoke(message)

        if self.file_path is not None:
            document = fitz.open(self.file_path)

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
