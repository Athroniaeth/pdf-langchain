from typing import Optional, Tuple

import gradio as gr
from gradio import ChatMessage

from src._typing import Examples, History, InferenceCallable


class ChatInterface:
    """
    Chat interface for the Gradio application.

    gr.ChatInterface don't allow to return arbitrary outputs, because 'echo'
    function must just return the response in string format. This class manage
    the history of the chat and return the arbitrary outputs.
    """

    fn: InferenceCallable
    state_history: gr.State  # History

    chat: gr.Chatbot
    input: gr.Textbox
    submit: gr.Button
    examples: gr.Examples

    retry_button: gr.Button
    undo_button: gr.Button
    clear_button: gr.Button

    def __init__(self, fn: Optional[InferenceCallable] = None, examples: Optional[Examples] = None):
        if fn is None:
            fn = lambda message, history: message  # noqa E731

        if examples is None:
            examples = [
                ["What is the capital of France?"],
                ["What is the capital of Spain?"],
                ["What is the capital of Italy?"],
            ]

        self.fn = fn
        self.state_history = gr.State([])

        with gr.Column(variant="compact"):
            self.chat = gr.Chatbot(type="messages")

            # Retry button, Undo button and Clear button
            with gr.Row():
                self.retry_button = gr.Button("🔄 Retry", variant="secondary")
                self.undo_button = gr.Button("↩️ Undo", variant="secondary")
                self.clear_button = gr.Button("🗑️ Clear", variant="secondary")

            # Input text box for the user to type the message
            with gr.Row(variant="compact"):
                self.input = gr.Textbox(
                    container=False,
                    scale=2,
                    lines=1,
                    max_lines=1,
                    show_label=False,
                    placeholder="Type your message here...",
                    interactive=True,
                )
                self.submit = gr.Button("Submit", variant="primary", scale=1)

            # Examples of messages to help the user
            self.examples = gr.Examples(examples, self.input, self.input)

    def bind_events(
        self,
        activate_chat_input: bool = True,
        activate_chat_submit: bool = True,
        activate_button_retry: bool = True,
        activate_button_undo: bool = True,
        activate_button_clear: bool = True,
    ):
        """Bind the events for the chat interface."""

        if activate_chat_input:
            self.input.submit(
                fn=self.echo,
                inputs=[self.input, self.state_history],
                outputs=[self.state_history, self.input, self.chat],
            )

        if activate_chat_submit:
            self.submit.click(
                fn=self.echo,
                inputs=[self.input, self.state_history],
                outputs=[self.state_history, self.input, self.chat],
            )

        if activate_button_retry:
            self.retry_button.click(
                fn=self.retry,
                inputs=[self.state_history],
                outputs=[self.state_history, self.chat],
            )

        if activate_button_undo:
            self.undo_button.click(
                fn=self.undo,
                inputs=[self.state_history],
                outputs=[self.state_history, self.chat],
            )

        if activate_button_clear:
            self.clear_button.click(
                fn=self.clear,
                inputs=[self.state_history],
                outputs=[self.state_history, self.chat],
            )

    def retry(
        self, history: History
    ) -> Tuple[
        History,
        gr.update,
    ]:
        """
        Retry the last message from the user.

        Args:
            history (History): The chat history.

        Returns:
            tuple: The updated history and the chatbot component.
            - History: The updated chat history.
            - gr.update: Update for the chatbot component.
        """

        if len(history) > 1:
            # Get the last user message
            message = history[-2].content

            # Restart the inference with the last user message
            response = self.fn(message, history)

            # Replace the last assistant message with the new response
            history[-1] = ChatMessage("assistant", response)

            # Retry the last message (user)
            return history, gr.update(value=history)

        # Alert the user that there is no message to retry
        gr.Warning("There is no message to retry.")

        # No user message or no assistant response (to user message)
        return history, gr.update(value=history)

    @staticmethod
    def undo(
        history: History,
    ) -> Tuple[
        History,
        gr.update,
    ]:
        """
        Undo the last message from the user.

        Args:
            history (History): The chat history.

        Returns:
            tuple: The updated history and the chatbot component.
            - History: The updated chat history.
            - gr.update: Update for the chatbot component.
        """

        # Remove the last two messages (assistant and user)
        if len(history) > 1:
            history.pop()
            history.pop()
        else:
            # Alert the user that there is no message to undo
            gr.Warning("There is no message to undo.")

        return history, gr.update(value=history)

    @staticmethod
    def clear(
        history: History,
    ) -> Tuple[
        History,
        gr.update,
    ]:
        """
        Clear the chat history.

        Args:
            history (History): The chat history.

        Returns:
            tuple: The updated history and the chatbot component.
            - History: The updated chat history.
            - gr.update: Update for the chatbot component.
        """
        history.clear()

        return (history, gr.update(value=history))

    def echo(
        self, message: str, history: History
    ) -> Tuple[
        History,
        gr.update,
        gr.update,
    ]:
        """
        Start the chatbot inference and update the chat history.

        Args:
            message (str): The user message.
            history (History): The chat history.

        Returns:
            History: The updated chat history.
            gr.update: Update for the input component.
            gr.update: Update for the chatbot component

        """
        # preprocess the message
        message = message.strip()

        # Alert the user that there is no message to echo
        if not message:
            gr.Warning("There is no message to echo.")

            return (history, gr.update(value=""), gr.update(value=history))

        # Start inference
        response = self.fn(message, history)

        # Update the history
        user_message = ChatMessage("user", message)
        assistant_message = ChatMessage("assistant", response)

        # Append the messages to the history
        history.append(user_message)
        history.append(assistant_message)

        # Clear input, return history
        return (
            # states
            history,  # history
            # components
            gr.update(value=""),  # input
            gr.update(value=history),  # chatbot
        )


if __name__ == "__main__":
    with gr.Blocks() as application:
        chat = ChatInterface()
        chat.bind_events(activate_chat_events=True, activate_button_events=True)

    application.launch()
