#!/usr/bin/env python3
"""
Extract text from a PDF using IBM's Docling.

Docling performs deep layout analysis (headings, tables, figures, lists) and
exports clean Markdown. It is heavier than rule-based parsers but produces
well-structured output on academic and technical documents.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using Docling."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def extract_pdf(pdf_path: Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    text = result.document.export_to_markdown()
    return (text or "").strip() + "\n"


def main() -> None:
    args = parse_args()
    text = extract_pdf(args.pdf)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote output to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
