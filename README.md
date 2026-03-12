# PDF Extraction Bake-off

A local benchmark for comparing Python PDF extraction libraries on the same documents.
Measures text quality, table detection, memory usage, CPU time, and wall-clock time side-by-side.

## Libraries compared

| Parser | Library | Approach |
|---|---|---|
| `extract_with_pdfplumber.py` | [pdfplumber](https://github.com/jsvine/pdfplumber) | Text-position heuristics |
| `extract_with_camelot.py` | [camelot-py](https://camelot-py.readthedocs.io) | Lattice (ruled lines) + stream (whitespace) |
| `extract_with_pymupdf.py` | [PyMuPDF](https://pymupdf.readthedocs.io) | Built-in `find_tables()` (v1.23+) |
| `extract_with_pdf_oxide.py` | [pdf-oxide](https://github.com/pdf-oxide/pdf-oxide) | Markdown/HTML export |
| `extract_with_pypdf.py` | [pypdf](https://pypdf.readthedocs.io) | Pure-Python text extraction |
| `extract_with_markitdown.py` | [MarkItDown](https://github.com/microsoft/markitdown) | Microsoft whole-document conversion |
| `extract_with_docling.py` | [Docling](https://github.com/DS4SD/docling) | IBM deep layout analysis (ML) |
| `extract_with_marker.py` | [Marker](https://github.com/VikParuchuri/marker) | ML-based PDF → Markdown (OCR + layout) |

All parsers live in the `parsers/` package and share the same CLI:

```bash
python -m parsers.extract_with_<library> input.pdf -o output.md
```

## Setup

Requires Python 3.11–3.x and [uv](https://docs.astral.sh/uv/).
Camelot's lattice mode also requires [Ghostscript](https://www.ghostscript.com/) at the OS level.

```bash
# macOS
brew install ghostscript

# Install Python dependencies
make install        # runs: uv sync
```

## Quick start

```bash
# Run all parsers against one PDF
make run PDF=docs/my.pdf

# Run all parsers against every PDF in docs/
make run-all

# Run a subset of parsers
make run-libs PDF=docs/my.pdf LIBS="pdfplumber pymupdf marker"

# Generate robustness test fixtures (minimal, multipage, table, protected, corrupt)
make fixtures
```

All `make` targets accept optional overrides:

| Variable | Default | Description |
|---|---|---|
| `PDF` | — | PDF path for `run` / `run-libs` |
| `LIBS` | all | Space-separated library names |
| `TIMEOUT` | `300` | Per-library timeout in seconds (ML parsers are exempt) |
| `WORKERS` | all | Max parallel library runs per PDF |
| `RESULTS` | `results` | Root output directory |

## Running directly

```bash
# Single file
python run_comparison.py my.pdf

# Glob pattern (quote to prevent premature shell expansion)
python run_comparison.py "docs/*.pdf"

# Subset of libraries
python run_comparison.py my.pdf --libraries pdfplumber camelot pymupdf

# Custom output directory and concurrency cap
python run_comparison.py "docs/*.pdf" --results-dir /tmp/runs --workers 2

# Regression comparison against a previous run
python run_comparison.py my.pdf --baseline results/20260312_143022/my-document/summary.json
```

## Output structure

All PDFs from a single invocation are grouped under one session directory named
after the time the script was launched:

```
results/
  20260312_143022/              ← session dir (timestamp = invocation time)
    my-document/
      pdfplumber/
        output.md               ← extracted markdown
        meta.json               ← timing, memory, quality metrics
        run.log                 ← stdout + stderr from the subprocess
      camelot/  …
      pymupdf/  …
      marker/   …
      summary.json              ← PDF profile + all library results in one file
      report.html               ← self-contained, sortable HTML report
    other-document/
      …
```

## Metrics collected

**PDF profile** (detected before parsers run, using pdfplumber):

| Field | Description |
|---|---|
| `page_count` | Total pages |
| `file_size_kb` | File size |
| `pages_with_text` | Pages with extractable text |
| `pages_rasterised` | Scanned pages (images only, no text layer) |
| `pages_with_images` | Pages containing embedded images |
| `pages_with_tables` | Pages where pdfplumber detects a table |
| `pages_multi_column` | Pages whose word layout looks two-column |
| `is_fully_rasterised` | `true` if no page has any extractable text |

**Per-library metrics** (written to `meta.json` and the summary table):

| Field | Description |
|---|---|
| `lib_version` | Installed package version |
| `elapsed_seconds` | Wall-clock time |
| `peak_rss_mb` | Peak resident memory of the subprocess |
| `cpu_user_s` / `cpu_sys_s` | CPU time (user + system) |
| `word_count` | Words in the output markdown |
| `char_count` | Characters in the output markdown |
| `line_count` | Lines in the output markdown |
| `table_count` | Markdown tables found in the output |
| `heading_count` | Markdown headings (`#`–`######`) |
| `encoding_errors` | Replacement characters (`\ufffd`) — indicates garbled text |
| `duplicate_line_ratio` | Fraction of non-empty lines that are duplicates |

## Per-library behaviour notes

**Timeout handling** — The global `--timeout` (default 300 s) applies to all parsers
except `marker` and `docling`, which load large ML model weights and are exempt from
the timeout.

**Apple Silicon (MPS)** — Marker runs with `PYTORCH_ENABLE_MPS_FALLBACK=1` to avoid
`AcceleratorError` crashes on large documents caused by MPS sequence-length limits in
Surya's attention layers. Operations unsupported by MPS fall back silently to CPU.

## Test fixtures

`scripts/create_test_fixtures.py` generates five synthetic PDFs in `docs/` for
robustness testing:

| File | Purpose |
|---|---|
| `fixture_minimal.pdf` | 1-page plain-text — happy-path baseline |
| `fixture_multipage.pdf` | 20-page stress test for memory and speed |
| `fixture_table_complex.pdf` | Bordered 5×9 grid — tests table-detection parsers |
| `fixture_protected.pdf` | AES-256 password-protected (`user password: "test"`) |
| `fixture_corrupt.pdf` | Truncated to 50% — tests graceful error handling |

```bash
make fixtures
# or: python scripts/create_test_fixtures.py
```

## Adding a PDF for testing

Drop any `.pdf` file into `docs/` and run:

```bash
make run-all
```

PDF files are excluded from version control via `.gitignore`.
