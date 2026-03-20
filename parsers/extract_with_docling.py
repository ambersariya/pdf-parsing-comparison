#!/usr/bin/env python3
"""
Extract text from a PDF using IBM's Docling.

Docling performs deep layout analysis (headings, tables, figures, lists) and
exports clean Markdown. It is heavier than rule-based parsers but produces
well-structured output on academic and technical documents.

OCR is performed by Tesseract (system install required) when pages have no
extractable text layer. Processing is CPU-only for portability on EKS.

Memory efficiency options:
  --batch-pages N        Process N pages at a time (default: 0 = all at once).
                         Each batch is discarded after export, capping peak RSS
                         regardless of document length.
  --parallel-batches N   How many batches to run concurrently (default: 1).
                         Models are shared across threads; only meaningful when
                         --batch-pages is also set. Trades memory for speed.
  --threads N            CPU inference threads per model (default: 4). Lower
                         values reduce parallel memory pressure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypdf import PdfReader

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import ImageRefMode


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
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        default=False,
        help="Disable OCR (default: OCR enabled via Tesseract for scanned pages).",
    )
    parser.add_argument(
        "--table-mode",
        choices=["fast", "accurate"],
        default="accurate",
        help="TableFormer mode: 'accurate' (default) or 'fast'.",
    )
    parser.add_argument(
        "--page-marker",
        default="---\n\n",
        metavar="TEMPLATE",
        help="Separator inserted between pages (default: horizontal rule).",
    )
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=0,
        metavar="N",
        help="Process N pages per batch to cap peak memory (default: 0 = all at once).",
    )
    parser.add_argument(
        "--parallel-batches",
        type=int,
        default=1,
        metavar="N",
        help="Batches to run concurrently (default: 1 = sequential). "
             "Models are shared across threads. Only meaningful with --batch-pages.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        metavar="N",
        help="CPU inference threads for model inference (default: 4).",
    )
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to cache Docling model weights (avoids re-downloading).",
    )
    return parser.parse_args()


def _make_converter(
    ocr: bool,
    table_mode: str,
    threads: int,
    artifacts_path: Path | None,
) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        do_ocr=ocr,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST,
        ),
        ocr_options=TesseractCliOcrOptions() if ocr else None,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
            num_threads=threads,
        ),
        artifacts_path=artifacts_path,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _export(result, page_marker: str) -> str:
    return result.document.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        page_break_placeholder=page_marker,
    ) or ""


def extract_pdf(
    pdf_path: Path,
    ocr: bool = True,
    table_mode: str = "accurate",
    page_marker: str = "---\n\n",
    batch_pages: int = 0,
    parallel_batches: int = 1,
    threads: int = 4,
    artifacts_path: Path | None = None,
) -> str:
    converter = _make_converter(ocr, table_mode, threads, artifacts_path)

    if batch_pages <= 0:
        result = converter.convert(str(pdf_path))
        text = _export(result, page_marker)
    else:
        total = len(PdfReader(pdf_path).pages)
        ranges = [
            (start, min(start + batch_pages - 1, total))
            for start in range(1, total + 1, batch_pages)
        ]

        ordered: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, parallel_batches)) as pool:
            futures = {
                pool.submit(converter.convert, str(pdf_path), page_range=r): i
                for i, r in enumerate(ranges)
            }
            for future in as_completed(futures):
                i = futures[future]
                ordered[i] = _export(future.result(), page_marker).strip()

        text = ("\n" + page_marker).join(ordered[i] for i in range(len(ranges)) if ordered[i])

    return text.strip() + "\n"


def main() -> None:
    args = parse_args()
    text = extract_pdf(
        args.pdf,
        ocr=not args.no_ocr,
        table_mode=args.table_mode,
        page_marker=args.page_marker,
        batch_pages=args.batch_pages,
        parallel_batches=args.parallel_batches,
        threads=args.threads,
        artifacts_path=args.artifacts_path,
    )

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote output to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
