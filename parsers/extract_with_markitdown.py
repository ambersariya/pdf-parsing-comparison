#!/usr/bin/env python3
"""
Extract text from a PDF using Microsoft's MarkItDown.

Note: MarkItDown converts the whole document at once (no per-page control).
The output is plain Markdown but lacks rich structural formatting such as
nested headings or proper table alignment — useful as a lightweight baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from markitdown import MarkItDown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using MarkItDown."
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
    md = MarkItDown()
    result = md.convert(str(pdf_path))
    text = result.text_content or ""
    return text.strip() + "\n"


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
