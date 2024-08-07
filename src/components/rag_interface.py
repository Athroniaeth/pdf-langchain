import os
from typing import Optional
from uuid import UUID

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

        self.state_uuid = gr.State()

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
                    examples=examples
                )
                self.chat_interface.bind_events(
                    activate_chat_input=False,
                    activate_chat_submit=False,
                    activate_button_retry=False,
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
                self.state_uuid,
                self.pdf_reader.state_pdf,
                self.chat_interface.state_history,
                self.chat_interface.input
            ],
            outputs=[
                self.state_uuid,
                self.chat_interface.state_history,

                self.chat_interface.input,
                self.chat_interface.chat,
                self.pdf_reader.display,
                self.pdf_reader.counter
            ]
        )

        self.chat_interface.submit.click(
            fn=self.echo,
            inputs=[
                self.state_uuid,
                self.pdf_reader.state_pdf,
                self.chat_interface.state_history,
                self.chat_interface.input
            ],
            outputs=[
                self.state_uuid,
                self.chat_interface.state_history,

                self.chat_interface.input,
                self.chat_interface.chat,
                self.pdf_reader.display,
                self.pdf_reader.counter
            ]
        )

        self.chat_interface.retry_button.click(
            fn=self.retry,
            inputs=[
                self.state_uuid,
                self.pdf_reader.state_pdf,
                self.chat_interface.state_history,
            ],
            outputs=[
                self.state_uuid,
                self.chat_interface.state_history,

                self.chat_interface.chat,
                self.pdf_reader.display,
                self.pdf_reader.counter
            ]
        )

        # Refresh pdf display for highlighting text
        self.pdf_reader.file_input.change(
            fn=self.load_document,
            inputs=[self.state_uuid, self.pdf_reader.file_input],
            outputs=[self.state_uuid, self.pdf_reader.display]
        )

    def echo(
            self,
            state_uuid: Optional[UUID],
            state_document: Optional[fitz.Document],
            state_history: History,

            message: str,
    ) -> (
            UUID,
            History,

            gr.update,  # input
            gr.update,  # history
            gr.update,  # image
            gr.update,  # counter
    ):
        """ Update the history, return the response in string format and update the PDF display. """

        # Start inference with the RAG client
        state_uuid, response, list_document_context = self.rag_client.invoke(message, state_history, state_uuid)

        # Highlight the context in the PDF document
        if state_document is not None:
            for document_context in list_document_context:
                text = document_context.page_content  # noqa nique tamre
                page = document_context.metadata['page']
                state_document = highlight_text(state_document, text, page)

        # Create the chat messages
        user_message = ChatMessage("user", message)
        assistant_message = ChatMessage("assistant", response)

        # Append the messages to the history
        state_history.append(user_message)
        state_history.append(assistant_message)

        # Update the PDF display (highlight text)
        _, _, update_image, update_label = self.pdf_reader.navigate_pdf(state_document, 0, 0)

        # Clear input, return history, update PDF display (image, counter)
        return (
            # states
            state_uuid,  # state_uuid
            state_history,  # history

            # updates
            gr.update(value=""),  # input
            gr.update(value=state_history),  # chat
            update_image,  # image
            update_label  # counter
        )

    def load_document(
            self,
            state_uuid: UUID,
            file_path: Optional[str]
    ) -> (
            UUID,
            gr.update
    ):
        """ Load the PDF document to the RAG client. """

        if file_path is None:
            state_uuid = self.rag_client.clean_pdf(state_uuid)
        else:
            # Message to user to show that the PDF (db vector) is loading
            gr.Info("RAG is loading the PDF document...")

            state_uuid = self.rag_client.process_pdf(file_path, state_uuid)

            # Message to user to show that the PDF (db vector) is loaded
            gr.Info("RAG successfully loaded the PDF document.")

        # Show loading to the user
        return state_uuid, gr.update()

    def retry(
            self,
            state_uuid: UUID,
            state_document: Optional[fitz.Document],
            state_history: History
    ) -> (
            UUID,
            History,

            gr.update,  # history
            gr.update,  # image
            gr.update,  # counter
    ):
        """ Retry the last message. """

        if len(state_history) > 1:
            # Get the last user message
            message = state_history[-2].content

            # Remove the last two messages (user and assistant)
            state_history = state_history[:-2]

            # Restart the inference with the last user message
            state_uuid, state_history, _, update_chat, update_image, update_counter = self.echo(state_uuid, state_document, state_history, message)

            # Retry the last message (user)
            return (
                state_uuid,  # state_uuid
                state_history,  # history

                # updates
                update_chat,  # history
                update_image,  # image
                update_counter  # counter
            )

        # Alert the user that there is no message to retry
        gr.Warning("There is no message to retry.")

        return (
            state_uuid,  # state_uuid
            state_history,  # history

            # updates
            gr.update(value=state_history),  # history
            gr.update(),  # image
            gr.update()  # counter
        )


if __name__ == "__main__":
    load_dotenv(ENV_PATH)

    with gr.Blocks() as application:
        rag_interface = RagInterface(
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            hf_token=os.environ["HF_TOKEN"],
        )
        rag_interface.bind_events()

    application.launch(debug=True)
