import io

import fitz
from PIL import Image


def highlight_text(document: fitz.Document, text: str, index_page: int) -> fitz.Document:
    """Highlight the text in the PDF document."""

    # Clean all old highlights
    generator = (annot for page in document for annot in page.annots())
    _ = [annot.delete() for annot in generator if annot.type[1] == 8]

    # Get the page and search for the text
    page = document[index_page]
    text_instances = page.search_for(text)

    # Highlight each occurrence of the text
    for inst in text_instances:
        highlight = page.add_highlight_annot(inst)  # Ajouter une annotation de surlignage
        highlight.set_colors(stroke=(1, 1, 0))  # Définir la couleur du surlignage en jaune (RGB)
        highlight.update()  # Mettre à jour l'annotation pour appliquer les modifications

    return document


def document_to_images(pdf_path: str) -> list[Image]:
    """Convert a PDF document to a list of images."""
    with fitz.open(pdf_path) as document:
        list_images = [page_to_image(page) for page in document]
        return list_images


def page_to_image(page: fitz.Page) -> Image:
    """Convert a PDF page to an image."""
    pix = page.get_pixmap()  # noqa: F821 (Unresolved attribute reference 'get_pixmap' for class 'Page')
    bytes_ = pix.tobytes("ppm")
    bytes_io = io.BytesIO(bytes_)
    img = Image.open(bytes_io)
    return img


def update_page(pdf_path, current_page: int = 0) -> fitz.Page:
    """Change the current page of the PDF document."""
    with fitz.open(pdf_path) as document:
        max_pages = document.page_count
        next_page = (current_page + 1) % max_pages

    return document[next_page]
