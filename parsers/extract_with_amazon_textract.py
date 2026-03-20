#!/usr/bin/env python3
"""
Extract text from a PDF using Amazon Textract via the amazon-textract-textractor library.

Each page is rendered to a PNG image (via PyMuPDF) and submitted to Amazon
Textract's synchronous AnalyzeDocument API with LAYOUT and TABLES features,
preserving document semantics and outputting markdown.  This avoids the need
for an S3 bucket while still exercising Textract's full analysis pipeline.

Requirements:
  - Valid AWS credentials (environment variables, ~/.aws/credentials, or IAM role)
  - pymupdf  (already a project dependency — used for page rendering)
  - amazon-textract-textractor
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium
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
        help="AWS region for Textract (default: us-east-1)",
    )
    parser.add_argument(
        "--page-marker",
        default="---\n\n## Page {page}\n\n",
        metavar="TEMPLATE",
        help="Template inserted before each page (default: '--- Page {page}')",
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
    return parser.parse_args()


def _analyze_page(extractor: Textractor, page_index: int, img, page_marker: str) -> tuple[int, str]:
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
    region: str = "us-east-1",
    page_marker: str = "---\n\n## Page {page}\n\n",
    dpi: int = 200,
    workers: int = 4,
) -> str:
    extractor = Textractor(region_name=region)

    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72  # pdfium native resolution is 72 DPI

    # Render all pages upfront (pdfium is not thread-safe)
    images = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        bitmap = page.render(scale=float(scale))  # type: ignore[arg-type]
        images.append(bitmap.to_pil())
    doc.close()

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
    return "\n".join(pages).strip() + "\n"


def main() -> None:
    args = parse_args()
    text = extract_pdf(
        args.pdf,
        region=args.region,
        page_marker=args.page_marker,
        dpi=args.dpi,
        workers=args.workers,
    )

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote output to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
