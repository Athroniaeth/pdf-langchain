import gradio as gr
from gradio import ChatMessage

from src._typing import History


class ChatInterface:
    """
    Chat interface for the Gradio application.
    
    gr.ChatInterface don't allow to return gr.update, because 'echo'
    function must just return the response in string format. This class
    return the response in string format and update the PDF display.
    """
    history: History

    def __init__(self):
        self.history = []

        with gr.Blocks() as application:
            with gr.Column(variant="compact"):
                self.chat = gr.Chatbot(type="messages")

                # Retry button, Undo button and Clear button
                with gr.Row():
                    self.retry_button = gr.Button("Retry", variant="secondary")
                    self.undo_button = gr.Button("Undo", variant="secondary")
                    self.clear_button = gr.Button("Clear", variant="secondary")

                # Input text box for the user to type the message
                with gr.Row(variant="compact"):
                    self.input = gr.Textbox(container=False, scale=2, lines=1, max_lines=1, show_label=False, placeholder="Type your message here...", interactive=True)
                    self.submit = gr.Button("Submit", variant="primary", scale=1)

                # Examples of messages to help the user
                self.examples = gr.Examples([
                    ["What is the capital of France?"],
                    ["What is the capital of Spain?"],
                    ["What is the capital of Italy?"],
                ], self.input, self.input)

                ## self.input.submit(
                ##     fn=self.echo,
                ##     inputs=[self.input],
                ##     outputs=[self.input, self.chat]
                ## )

                ## self.submit.click(
                ##     fn=self.echo,
                ##     inputs=[self.input],
                ##     outputs=[self.input, self.chat]
                ## )

        self.application = application

    def echo(self, message: str) -> (str, History):
        # Update the history
        chat_message = ChatMessage(
            "user",
            message,
        )

        # Return string generation LLM
        self.history.append(chat_message)

        chat_message = ChatMessage(
            "assistant",
            "I don't know the answer to that question.",
        )

        # Return string generation LLM
        self.history.append(chat_message)

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
