#!/usr/bin/env python3
"""
Generate robustness test fixture PDFs in docs/.

Usage:
  python scripts/create_test_fixtures.py

Generates:
  docs/fixture_minimal.pdf        — 1-page text-only PDF (happy-path baseline)
  docs/fixture_multipage.pdf      — 20-page PDF for memory/speed stress testing
  docs/fixture_table_complex.pdf  — PDF with a bordered multi-row table (tests table parsers)
  docs/fixture_protected.pdf      — AES-256 password-protected (user password: "test")
  docs/fixture_corrupt.pdf        — truncated/corrupt PDF (tests error handling)

Requires: pymupdf (already a project dependency)
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

DOCS = Path(__file__).parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
)


def _save(doc: fitz.Document, name: str, **kwargs) -> Path:
    out = DOCS / name
    doc.save(str(out), **kwargs)
    doc.close()
    print(f"  ✓  {out.name}")
    return out


def create_minimal() -> Path:
    """1-page plain-text PDF — the simplest possible valid document."""
    doc  = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        "Minimal Test Fixture\n\n"
        "This is a single-page plain-text PDF created for parser testing.\n\n"
        + LOREM * 3,
        fontsize=11,
    )
    return _save(doc, "fixture_minimal.pdf")


def create_multipage(n: int = 20) -> Path:
    """n-page PDF for memory and speed stress testing."""
    doc = fitz.open()
    for i in range(1, n + 1):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i} of {n}\n\n" + LOREM * 8,
            fontsize=11,
        )
    return _save(doc, "fixture_multipage.pdf")


def create_table_complex() -> Path:
    """PDF with a bordered table — stresses table-detection parsers."""
    doc  = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 50), "Complex Table Fixture", fontsize=14)

    # Table geometry
    rows, cols = 9, 5
    x0, y0     = 72.0, 90.0
    col_w, row_h = 95.0, 22.0
    x1 = x0 + cols * col_w
    y1 = y0 + rows * row_h

    # Draw grid
    for r in range(rows + 1):
        y = y0 + r * row_h
        width = 1.0 if r == 0 else 0.4
        page.draw_line((x0, y), (x1, y), color=(0, 0, 0), width=width)
    for c in range(cols + 1):
        x = x0 + c * col_w
        page.draw_line((x, y0), (x, y1), color=(0, 0, 0), width=0.4)

    # Headers
    headers = ["Library", "Speed", "Memory", "Tables", "Quality"]
    for c, h in enumerate(headers):
        page.insert_text(
            (x0 + c * col_w + 4, y0 + row_h - 6),
            h, fontsize=9, fontname="helv",
        )

    # Data rows
    data = [
        ["pdfplumber",  "2.3s",  "145 MB", "✓",  "High"],
        ["camelot",     "4.1s",  "312 MB", "✓✓", "High"],
        ["pymupdf",     "0.8s",   "89 MB", "✓",  "Medium"],
        ["pdf_oxide",   "1.2s",  "102 MB", "✓",  "Medium"],
        ["pypdf",       "0.4s",   "45 MB", "✗",  "Low"],
        ["markitdown",  "1.8s",  "134 MB", "~",  "Medium"],
        ["docling",    "12.0s",  "890 MB", "✓✓", "Very High"],
        ["marker",     "35.0s", "2100 MB", "✓",  "Very High"],
    ]
    for r, row in enumerate(data):
        for c, cell in enumerate(row):
            page.insert_text(
                (x0 + c * col_w + 4, y0 + (r + 2) * row_h - 6),
                cell, fontsize=9, fontname="helv",
            )

    return _save(doc, "fixture_table_complex.pdf")


def create_protected(user_pw: str = "test") -> Path:
    """AES-256 password-protected PDF. Tests parser error handling."""
    doc  = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        f"Password-Protected Fixture\n\nUser password: {user_pw}\n\n" + LOREM,
        fontsize=11,
    )
    out = DOCS / "fixture_protected.pdf"
    doc.save(
        str(out),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=user_pw,
        owner_pw="owner_secret",
    )
    doc.close()
    print(f"  ✓  {out.name}  (password: {user_pw!r})")
    return out


def create_corrupt(source: Path | None = None) -> Path:
    """Truncated-to-50% copy of a valid PDF. Tests graceful error handling."""
    if source is None:
        source = DOCS / "fixture_minimal.pdf"
    if not source.exists():
        create_minimal()

    data = source.read_bytes()
    out  = DOCS / "fixture_corrupt.pdf"
    out.write_bytes(data[: len(data) // 2])
    print(f"  ✓  {out.name}  (truncated from {source.name})")
    return out


if __name__ == "__main__":
    print("Creating test fixtures in docs/\n")
    create_minimal()
    create_multipage()
    create_table_complex()
    create_protected()
    create_corrupt()
    print("\nDone.")
