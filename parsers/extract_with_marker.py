#!/usr/bin/env python3
"""
Extract text from a PDF using Marker.

Marker is an ML-based PDF-to-Markdown converter that uses layout detection
and OCR models. It produces high-quality structured Markdown including proper
headings, tables, and code blocks, but is significantly slower than rule-based
parsers and requires model weights on first run.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using Marker."
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
    # Marker's public API has changed across versions; try newest first.
    try:
        # marker >= 0.3 (current)
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(pdf_path))
        text = rendered.markdown
    except (ImportError, AttributeError):
        # marker < 0.3 (legacy)
        from marker.convert import convert_single_pdf  # type: ignore[import]
        from marker.models import load_all_models  # type: ignore[import]

        model_lst = load_all_models()
        text, _images, _meta = convert_single_pdf(str(pdf_path), model_lst)

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
