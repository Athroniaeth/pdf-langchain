from typing import List

import gradio as gr
from gradio import ChatMessage

from src._typing import History, Examples


class ChatInterface:
    """
    Chat interface for the Gradio application.
    
    gr.ChatInterface don't allow to return gr.update, because 'echo'
    function must just return the response in string format. This class
    return the response in string format and update the PDF display.
    """
    application: gr.Blocks

    history: History

    chat: gr.Chatbot
    input: gr.Textbox
    submit: gr.Button
    examples: gr.Examples

    retry_button: gr.Button
    undo_button: gr.Button
    clear_button: gr.Button

    def __init__(self, examples: Examples = None):
        self.history = []

        if examples is None:
            examples = [
                ["What is the capital of France?"],
                ["What is the capital of Spain?"],
                ["What is the capital of Italy?"],
            ]

        with gr.Blocks() as application:
            with gr.Column(variant="compact"):
                self.chat = gr.Chatbot(type="messages", show_copy_button=True)

                # Retry button, Undo button and Clear button
                with gr.Row():
                    self.retry_button = gr.Button("🔄 Retry", variant="secondary")
                    self.undo_button = gr.Button("↩️ Undo", variant="secondary")
                    self.clear_button = gr.Button("🗑️ Clear", variant="secondary")

                # Input text box for the user to type the message
                with gr.Row(variant="compact"):
                    self.input = gr.Textbox(container=False, scale=2, lines=1, max_lines=1, show_label=False, placeholder="Type your message here...", interactive=True)
                    self.submit = gr.Button("Submit", variant="primary", scale=1)

                # Examples of messages to help the user
                self.examples = gr.Examples(examples, self.input, self.input)

            self.application = application

    def bind_events(
            self,
            activate_chat_events: bool = True,
            activate_button_events: bool = True,
    ):
        """ Bind the events for the chat interface. """

        if activate_chat_events:
            self.input.submit(
                fn=self.echo,
                inputs=[self.input],
                outputs=[self.input, self.chat]
            )

            self.submit.click(
                fn=self.echo,
                inputs=[self.input],
                outputs=[self.input, self.chat]
            )

        if activate_button_events:
            self.retry_button.click(
                fn=self.retry,
                outputs=[self.input, self.chat]
            )

            self.undo_button.click(
                fn=self.undo,
                outputs=[self.input, self.chat]
            )

            self.clear_button.click(
                fn=self.clear,
                outputs=[self.input, self.chat]
            )

    def retry(self) -> (str, History):
        """ Retry the last message from the user. """

        if len(self.history) > 1:
            # Get the last user message
            last_user_message = self.history[-2].content

            # Restart the inference with the last user message
            message, history = self.echo(last_user_message)  # noqa F841

            # Remove the last two messages (assistant and user)
            self.undo()

            # Retry the last message (user)
            return "", self.history

        return "", self.history

    def undo(self) -> (str, History):
        """ Undo the last message from the user. """

        if len(self.history) != 0:
            # Remove the last two messages (assistant and user)
            self.history.pop()
            self.history.pop()

        return "", self.history

    def clear(self) -> (str, History):
        """ Clear the chat history. """

        self.history.clear()
        return "", self.history

    def echo(self, message: str) -> (str, History):
        # Update the history
        user_message = ChatMessage("user", message)
        assistant_message = ChatMessage("assistant", "I don't know the answer to that question.")

        # Append the messages to the history
        self.history.append(user_message)
        self.history.append(assistant_message)

        # Clear input, return history
        return "", self.history

    def refresh_history(self) -> None:
        """
        Refresh the chat history with the new message.
        """
        _refresh_history = lambda: gr.update(value=self.history)
        self.application.load(_refresh_history, outputs=[chat.chat])


if __name__ == "__main__":
    with gr.Blocks() as application:
        chat = ChatInterface()

        # refresh chat attribute with history
        chat.refresh_history()

    application.launch()
