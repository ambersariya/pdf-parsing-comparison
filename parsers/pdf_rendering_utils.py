"""
Shared PDF rendering and image-extraction helpers.

Used by extract_with_amazon_textract and extract_with_tesseract, which share
the same page-rendering pipeline:
  1. Extract embedded images (pypdf) → save to disk
  2. Render pages (pypdfium2) → PIL images, optionally blanking image areas
     (pdfplumber supplies bounding boxes in PDF point coords, bottom-left origin)
"""
from __future__ import annotations

from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader


def save_embedded_images(pdf_path: Path, image_dir: Path) -> None:
    """Extract all embedded images from the PDF and write them to image_dir."""
    reader = PdfReader(pdf_path)
    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        for img_index, img in enumerate(page.images):
            stem = Path(img.name).stem
            suffix = Path(img.name).suffix or ".bin"
            out_path = image_dir / f"page{page_number:04d}_img{img_index + 1:04d}_{stem}{suffix}"
            out_path.write_bytes(img.data)


def render_pages(pdf_path: Path, dpi: int, blank_images: bool) -> list[Image.Image]:
    """
    Render each PDF page to a PIL image.

    When blank_images is True, embedded image areas are painted white using
    bounding boxes from pdfplumber (PDF point coordinates, bottom-left origin).
    When blank_images is False, pdfplumber is not opened at all.
    """
    scale = dpi / 72.0
    pdfium_doc = pdfium.PdfDocument(pdf_path)
    rendered: list[Image.Image] = []

    try:
        if blank_images:
            with pdfplumber.open(pdf_path) as plumber_pdf:
                for page_index, plumber_page in enumerate(plumber_pdf.pages):
                    pdfium_page = pdfium_doc[page_index]
                    bitmap = pdfium_page.render(scale=float(scale))  # type: ignore[arg-type]
                    pil_img = bitmap.to_pil()

                    if plumber_page.images:
                        page_height_pts = plumber_page.height
                        draw = ImageDraw.Draw(pil_img)
                        for img_info in plumber_page.images:
                            x0 = int(img_info["x0"] * scale)
                            y0 = int((page_height_pts - img_info["y1"]) * scale)
                            x1 = int(img_info["x1"] * scale)
                            y1 = int((page_height_pts - img_info["y0"]) * scale)
                            draw.rectangle([x0, y0, x1, y1], fill="white")

                    rendered.append(pil_img)
        else:
            for page_index in range(len(pdfium_doc)):
                pdfium_page = pdfium_doc[page_index]
                bitmap = pdfium_page.render(scale=float(scale))  # type: ignore[arg-type]
                rendered.append(bitmap.to_pil())
    finally:
        pdfium_doc.close()

    return rendered
