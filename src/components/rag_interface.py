import os
from typing import Optional

import fitz
import gradio
import gradio as gr
from dotenv import load_dotenv
from gradio import ChatMessage

from src import ENV_PATH
from src._pymupdf import highlight_text
from src._typing import History, Examples
from src.client import RagClient
from src.components import PDFReader
from src.components.chat_interface import ChatInterface


class RagInterface:
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

        # Global client for all users, also not gr.State
        self.rag_client = RagClient(
            model_id=model_id,
            hf_token=hf_token,
        )

        with gradio.Row():
            with gradio.Column(scale=1):  # Prend 2/3 de la largeur
                self.pdf_reader = PDFReader()
                self.pdf_reader.bind_events()

            with gradio.Column(scale=2):  # Prend 2/3 de la largeur
                self.chat_interface = ChatInterface(
                    fn=self.rag_client.invoke,
                    examples=examples
                )
                self.chat_interface.bind_events(
                    activate_chat_events=False,
                    activate_button_events=True,
                )
                """
                super().__init__(examples=examples)
                super().bind_events(
                    activate_chat_events=False,
                    activate_button_events=True,
                )
                """

    def bind_events(self):
        """ Bind the events for the chat interface. """

        self.chat_interface.input.submit(
            fn=self.echo,
            inputs=[
                self.chat_interface.state_history,
                self.pdf_reader.state_pdf,
                self.chat_interface.input
            ],
            outputs=[
                self.chat_interface.state_history,
                self.pdf_reader.state_pdf,

                self.chat_interface.input,
                self.chat_interface.chat,
                self.pdf_reader.display,
                self.pdf_reader.counter
            ]
        )

        self.chat_interface.submit.click(
            fn=self.echo,
            inputs=[
                self.chat_interface.state_history,
                self.pdf_reader.state_pdf,
                self.chat_interface.input
            ],
            outputs=[
                self.chat_interface.state_history,
                self.pdf_reader.state_pdf,

                self.chat_interface.input,
                self.chat_interface.chat,
                self.pdf_reader.display,
                self.pdf_reader.counter
            ]
        )

        self.pdf_reader.file_input.change(
            fn=self.load_document,
            inputs=[self.pdf_reader.file_input],
            outputs=[self.pdf_reader.display]
        )
        """
        self.pdf_reader.reset_button.click(
            fn=self.load_document,
            inputs=[gr.State(None)],
            outputs=[self.pdf_reader.display]
        )
        """
    def echo(
            self,
            history: History,
            document: Optional[fitz.Document],
            message: str,
    ) -> (
            History,
            Optional[fitz.Document],
            gr.update,  # input
            gr.update,  # history
            gr.update,  # image
            gr.update,  # counter
    ):
        """ Update the history, return the response in string format and update the PDF display. """

        # Start inference with the RAG client
        response = self.rag_client.invoke(message, history)
        list_document_context = self.rag_client.list_document_context

        # Highlight the context in the PDF document
        if document is not None:
            for document_context in list_document_context:
                text = document_context.page_content  # noqa nique tamre
                page = document_context.metadata['page']
                document = highlight_text(document, text, page)

        # Create the chat messages
        user_message = ChatMessage("user", message)
        assistant_message = ChatMessage("assistant", response)

        # Append the messages to the history
        history.append(user_message)
        history.append(assistant_message)

        # Update the PDF display (highlight text)
        _, _, update_image, update_label = self.pdf_reader.navigate_pdf(document, 0, 0)

        # Clear input, return history, update PDF display (image, counter)
        return (
            # states
            history,
            document,

            # updates
            gr.update(value=""),
            gr.update(value=history),
            update_image,
            update_label
        )

    def load_document(
            self,
            file_path: Optional[str]
    ) -> (
        gr.update,
    ):
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


if __name__ == "__main__":
    load_dotenv(ENV_PATH)

    with gr.Blocks() as application:
        rag_interface = RagInterface(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            hf_token=os.environ["HF_TOKEN"],
        )
        rag_interface.bind_events()

    application.launch(debug=True)
