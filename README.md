# PDF Extraction Bake-off

A local benchmark for comparing Python PDF extraction libraries on the same documents.
Measures text quality, table detection, memory usage, and wall-clock time side-by-side.

## Libraries compared

| Script | Library | Table strategy |
|---|---|---|
| `extract_with_pdfplumber.py` | [pdfplumber](https://github.com/jsvine/pdfplumber) | Text-position heuristics |
| `extract_with_camelot.py` | [camelot-py](https://camelot-py.readthedocs.io) | Lattice (ruled lines) + stream (whitespace) |
| `extract_with_pymupdf.py` | [PyMuPDF](https://pymupdf.readthedocs.io) | Built-in `find_tables()` (v1.23+) |
| `extract_with_pdf_oxide.py` | [pdf-oxide](https://github.com/pdf-oxide/pdf-oxide) | Markdown/HTML export |

All four scripts share the same CLI:

```
python extract_with_<library>.py <input.pdf> -o output.md
```

## Setup

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).
camelot's lattice mode also requires [Ghostscript](https://www.ghostscript.com/) at the OS level.

```bash
# macOS
brew install ghostscript

# Install Python dependencies
poetry install
```

## Running a single extractor

```bash
poetry run python extract_with_pdfplumber.py my.pdf -o out.md
poetry run python extract_with_camelot.py my.pdf -o out.md --camelot-flavor lattice
poetry run python extract_with_pymupdf.py my.pdf -o out.md
poetry run python extract_with_pdf_oxide.py my.pdf -o out.md
```

## Running the full comparison

`run_comparison.py` runs every extractor in parallel against one or more PDFs,
profiles each document for features (tables, images, multi-column, rasterised pages),
and writes structured results plus a summary table.

```bash
# Single file
poetry run python run_comparison.py my.pdf

# Glob pattern (quote to prevent premature shell expansion)
poetry run python run_comparison.py "*.pdf"

# Specific subset of libraries
poetry run python run_comparison.py my.pdf --libraries pdfplumber camelot

# Custom output directory and concurrency cap
poetry run python run_comparison.py "*.pdf" --results-dir /tmp/runs --workers 2
```

### Output structure

Each run creates a timestamped directory under `results/`:

```
results/
  20260312_143022_my-document/
    pdfplumber/
      output.md       ← extracted markdown
      meta.json       ← timing, memory, quality metrics
      run.log         ← stdout + stderr from the process
    camelot/   …
    pymupdf/   …
    pdf_oxide/ …
    summary.json      ← pdf profile + all library results in one file
```

### Metrics collected

**PDF profile** (detected before the race starts):

| Field | Description |
|---|---|
| `page_count` | Total pages |
| `file_size_kb` | PDF file size |
| `pages_with_text` | Pages with extractable text |
| `pages_rasterised` | Pages that are scanned images (no text) |
| `pages_with_images` | Pages containing embedded images |
| `pages_with_tables` | Pages where pdfplumber detects a table |
| `pages_multi_column` | Pages whose layout looks two-column |
| `is_fully_rasterised` | True if no page has extractable text |

**Per-library metrics**:

| Field | Description |
|---|---|
| `elapsed_seconds` | Wall-clock time |
| `peak_rss_mb` | Peak resident memory of the subprocess |
| `word_count` | Words in the output markdown |
| `char_count` | Characters in the output markdown |
| `line_count` | Lines in the output markdown |
| `table_count` | Markdown tables found in the output |

### Example console output

```
============================================================
PDF   : my-document.pdf
Run   : results/20260312_143022_my-document
============================================================

  Profiling PDF...
  Pages       : 19  (9119.3 KB)
  Features    : text, images, tables
  Text pages  : 19
  Image pages : 19  (rasterised: 0)
  Table pages : 11
  2-col pages : 0

  Running 4 libraries with up to 4 in parallel...
  [pymupdf]    ok  0.1s  51.8 MB  1,734 words  0 tables
  [pdfplumber] ok  1.3s  76.4 MB  1,818 words  1 tables
  [pdf_oxide]  ok  0.1s  119.4 MB  2,246 words  0 tables
  [camelot]    ok  7.0s  898.5 MB  2,252 words  11 tables

  Library         Status    Time      Mem    Words  Tables     Chars
  ------------------------------------------------------------------
  pdfplumber      OK        1.3s  76.4 MB    1,818       1    12,232
  camelot         OK        7.0s  898.5 MB   2,252      11    12,987
  pymupdf         OK        0.1s  51.8 MB    1,734       0    10,942
  pdf_oxide       OK        0.1s  119.4 MB   2,246       0    12,302
```

## Adding a PDF for testing

Drop any `.pdf` file into the project root and run:

```bash
poetry run python run_comparison.py "*.pdf"
```

PDF files are excluded from version control via `.gitignore`.
