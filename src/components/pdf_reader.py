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

    def __init__(self, label="Upload PDF", height=410, initial_height=250):
        self.label = label
        self.height = height
        self.initial_height = initial_height

        self.file_path = None

        # Variables globales pour le document PDF et la page actuelle
        self.pdf_document = None
        self.current_page = 0

        with gr.Column(variant="compact"):
            self.file_input = gr.File(label=self.label, type="filepath", file_types=[".pdf"])
            self.pdf_display = gr.Image(visible=False, height=self.initial_height)
            self.counter = gr.Textbox(show_label=False, max_lines=1, interactive=False, value="No PDF loaded.")

            with gr.Row():
                self.prev_button = gr.Button("⬅️")
                self.next_button = gr.Button("➡️")

            self.reset_button = gr.Button("❌", variant="primary")

        self.file_input.change(
            fn=self.load_pdf,
            inputs=[self.file_input],
            outputs=[self.file_input, self.pdf_display, self.pdf_display, self.counter]
        )

        self.prev_button.click(
            fn=partial(self.navigate_pdf, direction=-1),
            outputs=[self.pdf_display, self.counter]
        )
        self.next_button.click(
            fn=partial(self.navigate_pdf, direction=1),
            outputs=[self.pdf_display, self.counter]
        )

        self.reset_button.click(
            fn=self.reset_pdf,
            outputs=[self.file_input, self.pdf_display, self.file_input]
        )

    def load_pdf(self, file_path: Optional[str]):
        self.file_path = file_path

        if file_path is not None:
            self.current_page = 0
            self.pdf_document = fitz.open(file_path)

            return (
                gr.update(visible=False),  # File input
                self.get_page_image(self.current_page),  # PDF display
                gr.update(visible=True, height=self.height),  # PDF display
                gr.update(value=self.counter_label)  # Counter
            )

        return (
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

    def navigate_pdf(self, direction):
        num_pages = self.pdf_document.page_count
        self.current_page = max(0, min(num_pages - 1, self.current_page + direction))

        return (
            self.get_page_image(self.current_page),
            gr.update(value=self.counter_label, visible=True)
        )

    def reset_pdf(self):
        self.current_page = 0
        self.pdf_document = None
        return gr.update(value=None, visible=True), gr.update(visible=False), gr.update(visible=True, height=self.initial_height)

    @property
    def counter_label(self):
        return f"Page {self.current_page + 1} / {self.pdf_document.page_count}" if self.pdf_document else "No PDF loaded."


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):  # Prend 2/3 de la largeur
                pdf_reader = PDFReader()

            with gr.Column(scale=2):  # Prend 2/3 de la largeur
                gr.Markdown("")

    demo.launch()
