"""
PDF parser scripts grouped as a package.

Each module is also a standalone CLI script that can be invoked directly:

    python -m parsers.extract_with_pdfplumber input.pdf -o output.md
    python -m parsers.extract_with_camelot    input.pdf -o output.md
    python -m parsers.extract_with_pymupdf    input.pdf -o output.md
    python -m parsers.extract_with_pdf_oxide  input.pdf -o output.md
"""

from parsers import (  # noqa: F401
    extract_with_camelot,
    extract_with_pdf_oxide,
    extract_with_pdfplumber,
    extract_with_pymupdf,
)

__all__ = [
    "extract_with_pdfplumber",
    "extract_with_camelot",
    "extract_with_pymupdf",
    "extract_with_pdf_oxide",
]
