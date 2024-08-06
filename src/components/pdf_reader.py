import io
from functools import partial
from typing import Optional

import fitz  # PyMuPDF
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
        label (str): The label of the file input.
        height (int): The height of the PDF display.
        initial_height (int): The initial height of the PDF display.
    """
    height: int
    initial_height: int

    current_page: gr.State  # int
    file_path: gr.State  # Optional[str]
    pdf_document: Optional[fitz.Document]

    file_input: gr.File
    pdf_display: gr.Image
    counter: gr.Textbox

    prev_button: gr.Button
    next_button: gr.Button
    reset_button: gr.Button

    def __init__(
            self,
            height: int = 410,
            initial_height: int = 250,
            activate_gradio_events: bool = True
    ):
        self.height = height
        self.initial_height = initial_height

        self.file_path = gr.State(None)

        # Variables globales pour le document PDF et la page actuelle
        self.pdf_document = None
        self.current_page = gr.State(0)

        # class input must have text-align: center;
        with gr.Column(variant="compact"):
            self.file_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
            self.pdf_display = gr.Image(visible=False, height=self.initial_height)
            self.counter = gr.Textbox(show_label=False, max_lines=1, interactive=False, value="No PDF loaded.", elem_id="counter")

            with gr.Row():
                self.prev_button = gr.Button("⬅️ Prev Page")
                self.next_button = gr.Button("➡️ Next Page")

            self.reset_button = gr.Button("🗑️ Clear PDF")

        if activate_gradio_events:
            self.file_input.change(
                fn=self.load_pdf,
                inputs=[self.file_input],
                outputs=[self.current_page, self.file_path, self.file_input, self.pdf_display, self.pdf_display, self.counter]
            )

            self.prev_button.click(
                fn=partial(self.navigate_pdf),
                inputs=[gr.State(-1), self.current_page],
                outputs=[self.pdf_display, self.counter]
            )
            self.next_button.click(
                fn=self.navigate_pdf,
                inputs=[gr.State(1), self.current_page],
                outputs=[self.pdf_display, self.counter]
            )

            self.reset_button.click(
                fn=self.reset_pdf,
                outputs=[self.current_page, self.file_input, self.pdf_display, self.file_input]
            )

    def load_pdf(self, file_path: Optional[str]):
        if file_path is not None:
            self.pdf_document = fitz.open(file_path)

            return (
                0,  # gr.State Page number
                file_path,  # gr.State File path (else shared variable)

                gr.update(visible=False),  # File input
                self.get_page_image(0),  # PDF display
                gr.update(visible=True, height=self.height),  # PDF display
                gr.update(value=self.counter_label(0))  # Counter
            )

        return (
            0,  # gr.State Page number
            file_path,  # gr.State File path (else shared variable)
            gr.update(visible=True),  # File input
            gr.update(visible=False),  # PDF display
            gr.update(visible=False, height=self.initial_height),  # PDF display
            gr.update(value="No PDF loaded.")  # Counter
        )

    def get_page_image(self, page_number):
        page = self.pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        return img

    def navigate_pdf(self, direction, current_page):
        """ Navigate through the pages of the PDF. """

        if self.pdf_document is None:
            return gr.update(value="No PDF loaded."), gr.update(visible=False)

        num_pages = self.pdf_document.page_count
        current_page = max(0, min(num_pages - 1, current_page + direction))

        return (
            self.get_page_image(current_page),
            gr.update(value=self.counter_label(current_page), visible=True)
        )

    def reset_pdf(self):
        self.pdf_document = None
        return 0, gr.update(value=None, visible=True), gr.update(visible=False), gr.update(visible=True, height=self.initial_height)


    def counter_label(self, current_page):
        return f"Page {current_page + 1} / {self.pdf_document.page_count}" if self.pdf_document else "No PDF loaded."


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):  # Prend 2/3 de la largeur
                pdf_reader = PDFReader()

            with gr.Column(scale=2):  # Prend 2/3 de la largeur
                gr.Markdown("")

    demo.launch()
