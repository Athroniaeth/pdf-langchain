import io

import fitz
from PIL import Image


def highlight_text(document: fitz.Document, text: str) -> fitz.Document:
    for page in document:  # Parcourir chaque page du document
        text_instances = page.search_for(text)

        # Surligner chaque occurrence trouvée
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)  # Ajouter une annotation de surlignage
            highlight.set_colors(stroke=(1, 1, 0))  # Définir la couleur du surlignage en jaune (RGB)
            highlight.update()  # Mettre à jour l'annotation pour appliquer les modifications

    return document


def document_to_images(pdf_path: str) -> list[Image]:
    """ Convertit le PDF charger dans Gradio en une image. """
    with fitz.open(pdf_path) as document:
        list_images = [page_to_image(page) for page in document]
        return list_images


def page_to_image(page: fitz.Page) -> Image:
    """ Convertit une page PDF en une image. """
    pix = page.get_pixmap()
    bytes_ = pix.tobytes("ppm")
    bytes_io = io.BytesIO(bytes_)
    img = Image.open(bytes_io)
    return img


def update_page(pdf_path, current_page: int = 0) -> fitz.Page:
    """ Change la page courant du PDF charger dans Gradio. """
    with fitz.open(pdf_path) as document:
        max_pages = document.page_count
        next_page = (current_page + 1) % max_pages

    return document[next_page]
