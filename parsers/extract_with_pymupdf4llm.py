#!/usr/bin/env python3
"""
Extract text from a PDF using pymupdf4llm.

pymupdf4llm wraps PyMuPDF and outputs LLM-friendly Markdown, preserving
headings, tables, code blocks, and multi-column layouts better than the
raw fitz text extraction path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf4llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using pymupdf4llm."
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
        default="",
        metavar="TEMPLATE",
        help="Template inserted before each page, e.g. '--- Page {page} ---\\n'",
    )
    return parser.parse_args()


def extract_pdf(pdf_path: Path, page_marker: str = "") -> str:
    if page_marker:
        chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, use_ocr=False)
        rendered_pages: list[str] = []
        for chunk in chunks:
            page_number = chunk["metadata"]["page"] + 1
            rendered_pages.append(page_marker.format(page=page_number) + chunk["text"])
        return "\n".join(rendered_pages).strip() + "\n"

    text = pymupdf4llm.to_markdown(str(pdf_path), use_ocr=False)
    return (text or "").strip() + "\n"


def main() -> None:
    args = parse_args()
    text = extract_pdf(args.pdf, page_marker=args.page_marker)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote output to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
