#!/usr/bin/env python3
"""
Extract text from a PDF using Tesseract OCR via pytesseract.

Each page is rendered to a PNG image (via pypdfium2) and passed to Tesseract.
Embedded images are extracted and saved to a temp directory; their areas can
optionally be blanked out before OCR (default behaviour).

The page rendering and image-blanking pipeline mirrors extract_with_amazon_textract.py;
only the OCR step differs.

Requirements:
  - tesseract  (system install, e.g. `brew install tesseract`)
  - pytesseract
  - pypdfium2   (page rendering)
  - pypdf       (embedded image extraction)
  - pdfplumber  (image bounding-box lookup)
"""
from __future__ import annotations

import argparse
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from pypdf import PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using Tesseract OCR."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--page-marker",
        default="---\n\n",
        metavar="TEMPLATE",
        help="Separator inserted between pages (default: horizontal rule). Supports {page}.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution used when rendering PDF pages to images (default: 300). "
             "Higher DPI improves Tesseract accuracy on small text.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Tesseract workers (default: 4)",
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
        help="Send pages to Tesseract with embedded images intact (default: images are blanked out).",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract language code(s), e.g. 'eng' or 'eng+fra' (default: eng)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help="Tesseract page segmentation mode (default: 3 — fully automatic, handles multi-column).",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=30,
        metavar="0-100",
        help="Discard OCR words whose Tesseract confidence is below this threshold (default: 30).",
    )
    return parser.parse_args()


def _save_embedded_images(pdf_path: Path, image_dir: Path) -> None:
    """Extract all embedded images from the PDF and write them to image_dir."""
    reader = PdfReader(str(pdf_path))
    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1
        for img_index, img in enumerate(page.images):
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


def _otsu_threshold(img: Image.Image) -> int:
    """Compute Otsu's binarisation threshold from a greyscale PIL image."""
    # histogram() returns a 256-element count list for mode "L"
    hist = img.histogram()
    total = sum(hist)
    sum_all = sum(i * hist[i] for i in range(256))

    best_thresh, best_var = 0, 0.0
    sum_bg = count_bg = 0
    for t in range(256):
        count_bg += hist[t]
        if count_bg == 0:
            continue
        count_fg = total - count_bg
        if count_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / count_bg
        mean_fg = (sum_all - sum_bg) / count_fg
        var = count_bg * count_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var, best_thresh = var, t
    return best_thresh


def _preprocess(img: Image.Image) -> Image.Image:
    """
    Prepare a page image for Tesseract:
      1. Greyscale
      2. Sharpen edges
      3. Boost contrast (2×)
      4. Otsu binarisation → clean black-on-white bitmap

    These steps reduce noise and improve recognition accuracy, especially on
    scanned or low-contrast PDFs.
    """
    img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    threshold = _otsu_threshold(img)
    img = img.point(lambda p: 255 if p >= threshold else 0)
    return img


def _bbox_height(title: str) -> int:
    """Parse the pixel height of a bbox from an hOCR title attribute."""
    m = re.search(r"bbox\s+\d+\s+(\d+)\s+\d+\s+(\d+)", title or "")
    return (int(m.group(2)) - int(m.group(1))) if m else 0


def _word_conf(title: str) -> int:
    """Parse the Tesseract word confidence (0–100) from an hOCR title attribute."""
    m = re.search(r"x_wconf\s+(\d+)", title or "")
    return int(m.group(1)) if m else 100


def _iter_class(root: ET.Element, classname: str):
    """Yield all descendant elements whose class contains classname."""
    for el in root.iter():
        if classname in el.get("class", "").split():
            yield el


def _hocr_to_markdown(hocr_bytes: bytes, min_conf: int = 30) -> str:
    """
    Convert Tesseract hOCR output to markdown.

    - Words below min_conf confidence are discarded.
    - End-of-line hyphens (soft hyphenation) are rejoined with the next word.
    - Headings are inferred by comparing each paragraph's average line height
      against the document's body-text baseline (25th-percentile line height):
        ≥ 1.6×  →  #
        ≥ 1.35× →  ##
        ≥ 1.15× →  ###
    """
    xml = re.sub(r"<!DOCTYPE[^>]*>", "", hocr_bytes.decode("utf-8"))
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return re.sub(r"<[^>]+>", " ", xml)

    # Body-text baseline from all line heights (25th-percentile)
    all_heights = [
        _bbox_height(line.get("title", ""))
        for line in _iter_class(root, "ocr_line")
    ]
    all_heights = sorted(h for h in all_heights if h > 0)
    if not all_heights:
        return ""
    body_height = all_heights[len(all_heights) // 4]

    paragraphs: list[str] = []
    for par in _iter_class(root, "ocr_par"):
        lines = list(_iter_class(par, "ocr_line"))
        if not lines:
            continue

        # Collect confident words line-by-line (preserving line boundaries
        # so we can detect hyphenation at line ends)
        line_words: list[list[str]] = []
        for line in lines:
            words = []
            for w in _iter_class(line, "ocrx_word"):
                if _word_conf(w.get("title", "")) >= min_conf:
                    token = "".join(w.itertext()).strip()
                    if token:
                        words.append(token)
            line_words.append(words)

        # De-hyphenate: soft hyphens at line ends rejoin with the next line's
        # first word (e.g. "recogni-" + "tion" → "recognition")
        flat: list[str] = []
        for words in line_words:
            if not words:
                continue
            if flat and flat[-1].endswith("-"):
                flat[-1] = flat[-1][:-1] + words[0]
                flat.extend(words[1:])
            else:
                flat.extend(words)

        text = " ".join(flat)
        if not text:
            continue

        # Heading classification by average line height
        heights = [_bbox_height(l.get("title", "")) for l in lines]
        heights = [h for h in heights if h > 0]
        avg = sum(heights) / len(heights) if heights else 0

        if avg >= body_height * 1.6:
            paragraphs.append(f"# {text}")
        elif avg >= body_height * 1.35:
            paragraphs.append(f"## {text}")
        elif avg >= body_height * 1.15:
            paragraphs.append(f"### {text}")
        else:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def _analyze_page(
    page_index: int,
    img: Image.Image,
    page_marker: str,
    lang: str,
    psm: int,
    min_conf: int,
) -> tuple[int, str]:
    img = _preprocess(img)
    config = f"--oem 1 --psm {psm}"
    hocr = pytesseract.image_to_pdf_or_hocr(img, lang=lang, config=config, extension="hocr")
    page_text = _hocr_to_markdown(hocr, min_conf=min_conf)
    page_number = page_index + 1
    marker = page_marker.format(page=page_number) if page_marker else ""
    return page_index, marker + page_text.strip()


def extract_pdf(
    pdf_path: Path,
    page_marker: str = "---\n\n",
    dpi: int = 300,
    workers: int = 4,
    image_dir: Path | None = None,
    include_images: bool = False,
    lang: str = "eng",
    psm: int = 3,
    min_conf: int = 30,
) -> tuple[str, Path]:
    """
    Returns (markdown_text, image_dir) where image_dir contains the extracted
    embedded images from each page.

    When include_images is False (default) image areas are blanked out before
    OCR. Set to True to run Tesseract on pages with images intact.
    """
    if image_dir is None:
        image_dir = Path(tempfile.mkdtemp(prefix="tesseract_images_"))
    else:
        image_dir.mkdir(parents=True, exist_ok=True)

    _save_embedded_images(pdf_path, image_dir)

    # Render pages sequentially (pypdfium2 is not thread-safe)
    images = _render_pages(pdf_path, dpi, blank_images=not include_images)

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_analyze_page, i, img, page_marker, lang, psm, min_conf): i
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
        page_marker=args.page_marker,
        dpi=args.dpi,
        workers=args.workers,
        image_dir=args.image_dir,
        include_images=args.include_images,
        lang=args.lang,
        psm=args.psm,
        min_conf=args.min_confidence,
    )
    print(f"Extracted images saved to: {image_dir}", flush=True)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote output to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
