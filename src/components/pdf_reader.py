import io
from typing import Optional, Tuple

import fitz
import gradio as gr
from PIL import Image


class PDFReader:
    """
    Gradio component to read and display a PDF file.

    Notes:
        - This component uses PyMuPDF to read PDF files.
        - The PDF file is displayed as an image on the interface.
        - The user can navigate through the pages of the PDF using the "⬅️" and "➡️" buttons.
        - The user can reset the PDF file using the "❌" button.

    Args:
        height (int): The height of the display.
    """

    state_pdf: gr.State
    state_page: gr.State

    file_input: gr.File
    display: gr.Image
    counter: gr.Textbox

    def __init__(
        self,
        height: int = 410,
    ):
        self.state_pdf = gr.State()
        self.state_page = gr.State()

        with gr.Column(variant="compact"):
            self.file_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
            self.display = gr.Image(visible=False, height=height)
            self.counter = gr.Textbox(
                show_label=False,
                max_lines=1,
                interactive=False,
                value="No PDF loaded.",
                elem_id="counter",
            )

            with gr.Row():
                self.prev_button = gr.Button("⬅️ Prev Page")
                self.next_button = gr.Button("➡️ Next Page")

            self.reset_button = gr.Button("🗑️ Clear PDF")

    def bind_events(self):
        """Bind the events for the PDF reader."""

        self.file_input.change(
            fn=self.load_pdf,
            inputs=[self.file_input],
            outputs=[self.state_pdf, self.state_page, self.file_input, self.display, self.counter],
        )

        self.prev_button.click(
            fn=self.navigate_pdf,
            inputs=[self.state_pdf, self.state_page, gr.State(-1)],
            outputs=[self.state_pdf, self.state_page, self.display, self.counter],
        )

        self.next_button.click(
            fn=self.navigate_pdf,
            inputs=[self.state_pdf, self.state_page, gr.State(1)],
            outputs=[self.state_pdf, self.state_page, self.display, self.counter],
        )

        self.reset_button.click(
            fn=self.reset_pdf,
            outputs=[self.state_pdf, self.state_page, self.file_input, self.display, self.counter],
        )

    @staticmethod
    def load_pdf(
        file_path: Optional[str],
    ) -> Tuple[Optional[fitz.Document], int, gr.update, gr.update, gr.update]:
        """
        Load the PDF file and display the first page.

        Args:
            file_path (Optional[str]): The path to the PDF file.

        Returns:
            Optional[fitz.Document]: The loaded PDF document or None if the file path is not provided.
            int: The total number of pages in the PDF document.
            gr.update: Update for the PDF document display (Gradio component update).
            gr.update: Update for the page number display (Gradio component update).
            gr.update: Update for any additional component display or state (Gradio component update).
        """

        if file_path is None:
            return (
                None,  # state_pdf
                0,  # state_page
                gr.update(visible=True),  # file_input
                gr.update(visible=False),  # display
                gr.update(value="No PDF loaded."),  # counter
            )

        pdf_document = fitz.open(file_path)
        image = get_page_image(pdf_document, 0)
        label = counter_label(pdf_document, 0)

        return (
            pdf_document,  # state_pdf
            0,  # state_page
            gr.update(visible=False),  # file_input
            gr.update(visible=True, value=image),  # display
            gr.update(value=label),  # counter
        )

    @staticmethod
    def navigate_pdf(
        state_pdf: Optional[fitz.Document], current_page: int, direction: int
    ) -> Tuple[
        Optional[fitz.Document],
        int,
        gr.update,
        gr.update,
    ]:
        """
        Navigate through the pages of the PDF.

        Args:
            state_pdf (Optional[fitz.Document]): The current PDF document.
            current_page (int): The current page number.
            direction (int): The navigation direction (-1 for previous, 1 for next).

        Returns:
            Optional[fitz.Document]: The updated PDF document.
            int: The updated page number.
            gr.update: Update for the PDF document display (Gradio component update).
            gr.update: Update for the page number display (Gradio component update).
        """

        if state_pdf is None:
            label = counter_label()

            return (
                state_pdf,  # state_pdf
                0,  # state_page
                gr.update(visible=False),  # pdf_display
                gr.update(visible=True, value=label),  # counter
            )

        max_pages = state_pdf.page_count
        current_page = max(0, min(current_page + direction, max_pages - 1))

        image = get_page_image(state_pdf, current_page)
        label = counter_label(state_pdf, current_page)

        return (
            state_pdf,  # state_pdf
            current_page,  # state_page
            gr.update(value=image),  # display
            gr.update(visible=True, value=label),  # counter
        )

    @staticmethod
    def reset_pdf() -> Tuple[Optional[fitz.Document], int, gr.update, gr.update, gr.update]:
        """
        Reset the PDF reader to its initial state.

        Returns:
            Optional[fitz.Document]: The state PDF document.
            int: The state page number.

            gr.update: Update for the PDF document display (Gradio component update).
            gr.update: Update for the page number display (Gradio component update).
            gr.update: Update for any additional component display or state (Gradio component update).
        """
        return (
            None,  # state_pdf
            0,  # state_page
            gr.update(value=None, visible=True),  # file_input
            gr.update(value=None, visible=False),  # display
            gr.update(value="No PDF loaded."),  # counter
        )


def get_page_image(pdf_document: fitz.Document, page_number: int) -> Image:
    """Get the image of a page from a PDF document."""
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    bytes_img = pix.tobytes("png")
    bytes_io = io.BytesIO(bytes_img)
    img = Image.open(bytes_io)
    return img


def counter_label(
    pdf_document: Optional[fitz.Document] = None, current_page: Optional[int] = None
) -> str:
    """Get the counter label for the display."""
    if pdf_document is None:
        return "No PDF loaded."

    if current_page is None:
        return f"Page 1 / {pdf_document.page_count}"

    # pdf_document & current_page are not None
    return f"Page {current_page + 1} / {pdf_document.page_count}"


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                pdf_reader = PDFReader()
                pdf_reader.bind_events()

            with gr.Column(scale=2):
                gr.Markdown("")

    demo.launch()
