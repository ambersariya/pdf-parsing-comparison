#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pypdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF using pypdf."
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
        default="\n\n--- PAGE {page} ---\n\n",
        help="Marker inserted between pages. Use an empty string to disable.",
    )
    return parser.parse_args()


def normalise(text: str | None) -> str:
    if not text:
        return ""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def extract_pdf(pdf_path: Path, page_marker: str) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    rendered_pages: list[str] = []

    for page_index, page in enumerate(reader.pages):
        page_text = normalise(page.extract_text())

        if not page_text:
            page_text = "[No extractable text on this page]"

        page_number = page_index + 1
        if page_marker:
            rendered_pages.append(page_marker.format(page=page_number) + page_text)
        else:
            rendered_pages.append(page_text)

    return "\n".join(rendered_pages).strip() + "\n"


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
