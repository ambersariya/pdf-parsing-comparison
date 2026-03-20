#!/usr/bin/env python3
"""
Extract text from a PDF using Amazon Textract via the amazon-textract-textractor library.

Each page is rendered to a PNG image (via pypdfium2) and submitted to Amazon
Textract's synchronous AnalyzeDocument API with LAYOUT and TABLES features,
preserving document semantics and outputting markdown.  This avoids the need
for an S3 bucket while still exercising Textract's full analysis pipeline.

Embedded images are extracted from each page (via pypdf) and saved to a temp
directory. Image bounding boxes are obtained via pdfplumber and painted white
on the rendered page before it is sent to Textract.

Requirements:
  - Valid AWS credentials (environment variables, ~/.aws/credentials, or IAM role)
  - pypdfium2   (page rendering)
  - pypdf       (embedded image extraction)
  - pdfplumber  (image bounding-box lookup)
  - amazon-textract-textractor
"""
from __future__ import annotations

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractor.data.markdown_linearization_config import MarkdownLinearizationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using Amazon Textract."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--region",
        default="eu-west-1",
        help="AWS region for Textract (default: eu-west-1)",
    )
    parser.add_argument(
        "--page-marker",
        default="---\n\n",
        metavar="TEMPLATE",
        help="Separator inserted between pages (default: horizontal rule). Supports {page}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Textract workers (default: 4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution used when rendering PDF pages to images (default: 200)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to save extracted page images. Defaults to a temp directory.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        default=False,
        help="Send pages to Textract with embedded images intact (default: images are blanked out).",
    )
    return parser.parse_args()


def _save_embedded_images(pdf_path: Path, image_dir: Path) -> None:
    """Extract all embedded images from the PDF and write them to image_dir."""
    reader = PdfReader(str(pdf_path))
    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        for img_index, img in enumerate(page.images):
            # img.name already carries an extension (e.g. 'Im0.png')
            stem = Path(img.name).stem
            suffix = Path(img.name).suffix or ".bin"
            out_path = image_dir / f"page{page_number:04d}_img{img_index + 1:04d}_{stem}{suffix}"
            out_path.write_bytes(img.data)


def _render_pages(pdf_path: Path, dpi: int, blank_images: bool) -> list[Image.Image]:
    """
    Render each PDF page to a PIL image.

    When blank_images is True, embedded image areas are painted white before
    the page is returned (pdfplumber supplies bounding boxes in PDF point
    coordinates with a bottom-left origin, which are converted to pixels).
    """
    scale = dpi / 72.0
    pdfium_doc = pdfium.PdfDocument(str(pdf_path))
    rendered: list[Image.Image] = []

    with pdfplumber.open(str(pdf_path)) as plumber_pdf:
        for page_index, plumber_page in enumerate(plumber_pdf.pages):
            pdfium_page = pdfium_doc[page_index]
            bitmap = pdfium_page.render(scale=float(scale))  # type: ignore[arg-type]
            pil_img = bitmap.to_pil()

            if blank_images and plumber_page.images:
                page_height_pts = plumber_page.height
                draw = ImageDraw.Draw(pil_img)
                for img_info in plumber_page.images:
                    x0 = int(img_info["x0"] * scale)
                    y0 = int((page_height_pts - img_info["y1"]) * scale)
                    x1 = int(img_info["x1"] * scale)
                    y1 = int((page_height_pts - img_info["y0"]) * scale)
                    draw.rectangle([x0, y0, x1, y1], fill="white")

            rendered.append(pil_img)

    pdfium_doc.close()
    return rendered


def _analyze_page(
    extractor: Textractor,
    page_index: int,
    img: Image.Image,
    page_marker: str,
) -> tuple[int, str]:
    document = extractor.analyze_document(
        file_source=img,
        features=[TextractFeatures.LAYOUT, TextractFeatures.TABLES],
    )
    page_text = document.get_text(config=MarkdownLinearizationConfig())
    page_number = page_index + 1
    marker = page_marker.format(page=page_number) if page_marker else ""
    return page_index, marker + page_text


def extract_pdf(
    pdf_path: Path,
    region: str = "eu-west-1",
    page_marker: str = "---\n\n",
    dpi: int = 200,
    workers: int = 20,
    image_dir: Path | None = None,
    include_images: bool = False,
) -> tuple[str, Path]:
    """
    Returns (markdown_text, image_dir) where image_dir contains the extracted
    embedded images from each page.

    When include_images is False (default) image areas are blanked out before
    the page is sent to Textract. Set to True to send pages with images intact.
    """
    if image_dir is None:
        image_dir = Path(tempfile.mkdtemp(prefix="textract_images_"))
    else:
        image_dir.mkdir(parents=True, exist_ok=True)

    # Extract embedded images to disk
    _save_embedded_images(pdf_path, image_dir)

    # Render pages, optionally blanking image areas (not thread-safe, do sequentially)
    images = _render_pages(pdf_path, dpi, blank_images=not include_images)

    extractor = Textractor(region_name=region)
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_analyze_page, extractor, i, img, page_marker): i
            for i, img in enumerate(images)
        }
        for future in as_completed(futures):
            page_index, page_text = future.result()
            results[page_index] = page_text

    pages = [results[i] for i in range(len(images))]
    return "\n".join(pages).strip() + "\n", image_dir


def main() -> None:
    args = parse_args()
    text, image_dir = extract_pdf(
        args.pdf,
        region=args.region,
        page_marker=args.page_marker,
        dpi=args.dpi,
        workers=args.workers,
        image_dir=args.image_dir,
        include_images=args.include_images,
    )
    print(f"Extracted images saved to: {image_dir}", flush=True)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote output to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
