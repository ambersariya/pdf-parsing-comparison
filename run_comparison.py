#!/usr/bin/env python3
"""
PDF extraction comparison runner.

Runs every registered extractor against one or more PDF files and writes
results into a structured output directory:

  results/
    <YYYYMMDD_HHMMSS>_<pdf-stem>/
      pdfplumber/
        output.md
        meta.json
        run.log
      camelot/  …
      pymupdf/  …
      pdf_oxide/  …
      summary.json    ← timing, memory, quality metrics for every library

Usage:
  python run_comparison.py my.pdf
  python run_comparison.py *.pdf --results-dir /tmp/pdf-runs
  python run_comparison.py my.pdf --libraries pdfplumber camelot
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pdfplumber
import psutil
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Library registry
# Each entry maps a short name to the CLI command that produces a markdown file.
# {pdf} and {output} are filled in at run time.
# ---------------------------------------------------------------------------
LIBRARIES: dict[str, list[str]] = {
    "pdfplumber":  [sys.executable, "parsers/extract_with_pdfplumber.py",  "{pdf}", "-o", "{output}"],
    "camelot":     [sys.executable, "parsers/extract_with_camelot.py",     "{pdf}", "-o", "{output}"],
    "pymupdf":     [sys.executable, "parsers/extract_with_pymupdf.py",     "{pdf}", "-o", "{output}"],
    "pdf_oxide":   [sys.executable, "parsers/extract_with_pdf_oxide.py",   "{pdf}", "-o", "{output}"],
    "pypdf":       [sys.executable, "parsers/extract_with_pypdf.py",       "{pdf}", "-o", "{output}"],
    "markitdown":  [sys.executable, "parsers/extract_with_markitdown.py",  "{pdf}", "-o", "{output}"],
    "docling":     [sys.executable, "parsers/extract_with_docling.py",     "{pdf}", "-o", "{output}"],
    "marker":      [sys.executable, "parsers/extract_with_marker.py",      "{pdf}", "-o", "{output}"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple PDF extractors on the same file(s) and save results."
    )
    parser.add_argument("pdfs", nargs="+", metavar="PDF",
                        help="PDF files or glob patterns, e.g. '*.pdf' or 'docs/*.pdf'.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Root directory for all run output (default: ./results).")
    parser.add_argument("--libraries", nargs="+", choices=list(LIBRARIES),
                        default=list(LIBRARIES), metavar="LIB",
                        help=f"Libraries to run. Choices: {', '.join(LIBRARIES)}. Default: all.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-library timeout in seconds (default: 300).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel library runs per PDF. Defaults to number of libraries.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PDF feature profiling
# Runs once before the library race; gives context for interpreting results.
# ---------------------------------------------------------------------------

def profile_pdf(pdf_path: Path) -> dict:
    """
    Inspect the source PDF and return a feature profile dict.

    Features detected:
      page_count          total pages
      file_size_kb        PDF file size
      pages_with_text     pages that contain extractable text characters
      pages_rasterised    pages with no text but at least one image (scanned)
      pages_with_images   pages that contain embedded images
      pages_with_tables   pages where pdfplumber detects at least one table
      pages_multi_column  pages whose word layout looks like two columns
      has_text            any page has extractable text
      has_images          any page has embedded images
      has_tables          any page has a detectable table
      is_fully_rasterised every page is rasterised (no extractable text at all)
    """
    profile: dict = {
        "file_size_kb":       round(pdf_path.stat().st_size / 1024, 1),
        "page_count":         0,
        "pages_with_text":    0,
        "pages_rasterised":   0,
        "pages_with_images":  0,
        "pages_with_tables":  0,
        "pages_multi_column": 0,
    }

    with pdfplumber.open(str(pdf_path)) as pdf:
        profile["page_count"] = len(pdf.pages)

        for page in pdf.pages:
            words  = page.extract_words() or []
            images = page.images or []
            has_text   = len(words) > 10
            has_images = bool(images)

            if has_text:
                profile["pages_with_text"] += 1
            if has_images:
                profile["pages_with_images"] += 1
            if not has_text and has_images:
                profile["pages_rasterised"] += 1

            # Table detection (fast: text strategy only)
            try:
                tables = page.find_tables()
                if tables:
                    profile["pages_with_tables"] += 1
            except Exception:
                pass

            # Two-column heuristic: look for a sparse gutter in the middle third
            if has_text and len(words) >= 40:
                mid_left  = page.width * 0.40
                mid_right = page.width * 0.60
                centres   = [(w["x0"] + w["x1"]) / 2.0 for w in words]
                gutter    = sum(1 for x in centres if mid_left <= x <= mid_right)
                if gutter / len(centres) < 0.10:
                    profile["pages_multi_column"] += 1

    profile["has_text"]            = profile["pages_with_text"]   > 0
    profile["has_images"]          = profile["pages_with_images"] > 0
    profile["has_tables"]          = profile["pages_with_tables"] > 0
    profile["is_fully_rasterised"] = (
        profile["page_count"] > 0
        and profile["pages_with_text"] == 0
        and profile["pages_rasterised"] == profile["page_count"]
    )
    return profile



# ---------------------------------------------------------------------------
# Memory monitoring
# Polls the subprocess RSS every 100 ms in a background thread.
# ---------------------------------------------------------------------------

def _monitor_peak_rss(pid: int, result: dict, stop: threading.Event) -> None:
    """Record peak RSS (bytes) of a process until stop is set."""
    peak = 0
    try:
        proc = psutil.Process(pid)
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak:
                    peak = rss
            except psutil.NoSuchProcess:
                break
            stop.wait(0.1)
    except psutil.NoSuchProcess:
        pass
    result["peak_rss_bytes"] = peak


# ---------------------------------------------------------------------------
# Output quality metrics
# Derived from the extracted markdown after the run completes.
# ---------------------------------------------------------------------------

def _analyze_output(output_file: Path) -> dict:
    """Return text-quality metrics for the extracted markdown file."""
    if not output_file.exists():
        return {}
    text = output_file.read_text(encoding="utf-8", errors="replace")
    # Count markdown table separator rows (one per table body start)
    table_count = sum(1 for line in text.splitlines() if line.startswith("| ---"))
    return {
        "char_count":  len(text),
        "word_count":  len(text.split()),
        "line_count":  text.count("\n"),
        "table_count": table_count,
    }


# ---------------------------------------------------------------------------
# Per-library runner
# ---------------------------------------------------------------------------

def run_library(
    library: str,
    cmd_template: list[str],
    pdf_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict:
    """Run one extractor subprocess; return a metadata dict with timing, memory, and quality metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.md"

    cmd = [
        part.replace("{pdf}", str(pdf_path)).replace("{output}", str(output_file))
        for part in cmd_template
    ]

    meta: dict = {
        "library":         library,
        "pdf":             str(pdf_path),
        "output":          str(output_file),
        "command":         cmd,
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": None,
        "status":          None,
        "returncode":      None,
        "error":           None,
        # --- resource metrics ---
        "output_bytes":    None,
        "peak_rss_mb":     None,
        # --- quality metrics ---
        "char_count":      None,
        "word_count":      None,
        "line_count":      None,
        "table_count":     None,
    }

    t0 = time.perf_counter()
    mem_result: dict = {}
    stop_event = threading.Event()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent,
        )

        monitor = threading.Thread(
            target=_monitor_peak_rss,
            args=(proc.pid, mem_result, stop_event),
            daemon=True,
        )
        monitor.start()

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            meta["status"] = "timeout"
            meta["error"] = f"Exceeded {timeout}s timeout"
        else:
            meta["returncode"] = proc.returncode
            if proc.returncode == 0:
                meta["status"] = "ok"
                meta["output_bytes"] = output_file.stat().st_size if output_file.exists() else 0
                meta.update(_analyze_output(output_file))
            else:
                meta["status"] = "error"
                meta["error"] = (stderr or stdout or "").strip()[-2000:]

        (output_dir / "run.log").write_text(
            f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}", encoding="utf-8"
        )

    except Exception as exc:
        meta["status"] = "exception"
        meta["error"] = str(exc)
    finally:
        stop_event.set()
        monitor.join()
        meta["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        peak = mem_result.get("peak_rss_bytes", 0)
        meta["peak_rss_mb"] = round(peak / 1024 / 1024, 1) if peak else None

    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Parallel PDF runner
# ---------------------------------------------------------------------------

def run_pdf(
    pdf_path: Path,
    run_dir: Path,
    libraries: list[str],
    timeout: int,
    workers: int | None = None,
) -> list[dict]:
    """Run all selected libraries against one PDF in parallel; return results in library order."""
    max_workers = workers or len(libraries)
    results: dict[str, dict] = {}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description:<14}[/]"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
    )

    with Live(progress, console=console, refresh_per_second=10):
        task_ids = {lib: progress.add_task(lib, status="[dim]queued[/]") for lib in libraries}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures_to_lib = {
                pool.submit(run_library, lib, LIBRARIES[lib], pdf_path, run_dir / lib, timeout): lib
                for lib in libraries
            }
            # Mark all as running now that they've been submitted
            for lib in libraries:
                progress.update(task_ids[lib], status="[yellow]running…[/]")

            for future in as_completed(futures_to_lib):
                lib = futures_to_lib[future]
                meta = future.result()
                results[lib] = meta

                s = meta["status"]
                if s == "ok":
                    status_str = (
                        f"[green]OK[/]  "
                        f"[white]{meta['elapsed_seconds']:.1f}s[/]  "
                        f"[dim]{meta['peak_rss_mb']} MB[/]  "
                        f"[cyan]{meta['word_count']:,} words[/]  "
                        f"[blue]{meta['table_count']} tables[/]"
                    )
                elif s == "timeout":
                    status_str = f"[bold yellow]TIMEOUT[/] after {meta['elapsed_seconds']:.0f}s"
                elif s == "error":
                    status_str = f"[bold red]FAILED[/] (rc={meta['returncode']})"
                else:
                    status_str = f"[bold red]EXCEPTION[/] {meta['error']}"

                progress.update(task_ids[lib], status=status_str, completed=True)

    return [results[lib] for lib in libraries]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_STATUS_STYLE = {
    "ok":        "[bold green]OK[/]",
    "error":     "[bold red]FAILED[/]",
    "timeout":   "[bold yellow]TIMEOUT[/]",
    "exception": "[bold red]EXCEPTION[/]",
}


def write_summary(run_dir: Path, all_meta: list[dict], pdf_profile: dict) -> None:
    summary = {
        "run_dir":      str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pdf_profile":  pdf_profile,
        "results":      all_meta,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("Library",  style="bold cyan",  min_width=12)
    t.add_column("Status",   justify="center",   min_width=9)
    t.add_column("Time",     justify="right",    style="white",   min_width=7)
    t.add_column("Mem",      justify="right",    style="dim",     min_width=8)
    t.add_column("Words",    justify="right",    style="cyan",    min_width=8)
    t.add_column("Tables",   justify="right",    style="blue",    min_width=7)
    t.add_column("Chars",    justify="right",    style="dim",     min_width=9)

    for m in all_meta:
        status  = _STATUS_STYLE.get(m["status"] or "", m["status"] or "")
        elapsed = f"{m['elapsed_seconds']:.1f}s"    if m["elapsed_seconds"] is not None else "---"
        mem     = f"{m['peak_rss_mb']} MB"          if m["peak_rss_mb"]     is not None else "---"
        words   = f"{m['word_count']:,}"            if m["word_count"]      is not None else "---"
        tables  = str(m["table_count"])             if m["table_count"]     is not None else "---"
        chars   = f"{m['char_count']:,}"            if m["char_count"]      is not None else "---"
        t.add_row(m["library"], status, elapsed, mem, words, tables, chars)

    console.print()
    console.print(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_pdfs(patterns: list[str]) -> list[Path]:
    """Expand shell glob patterns and plain paths into a sorted, deduplicated list of PDFs."""
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in patterns:
        # Let Path.glob handle wildcards; fall back to a literal path otherwise
        if any(c in pattern for c in ("*", "?", "[")):
            cwd = Path.cwd()
            matches = sorted(cwd.glob(pattern))
        else:
            matches = [Path(pattern)]
        for p in matches:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(p)
    return paths


def main() -> None:
    args = parse_args()
    pdf_paths = resolve_pdfs(args.pdfs)

    if not pdf_paths:
        console.print("[bold red]ERROR[/] No PDF files matched the given patterns.", highlight=False)
        sys.exit(1)

    console.print(f"\nFound [bold]{len(pdf_paths)}[/] PDF(s) to process.\n")

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            console.print(f"[yellow]WARN[/] {pdf_path} not found — skipping")
            continue

        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.results_dir / f"{run_id}_{pdf_path.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)

        console.print(Rule(f"[bold]{pdf_path.name}[/]", style="bright_blue"))
        console.print(f"  [dim]Run dir:[/] {run_dir}")
        console.print()

        with console.status("[dim]Profiling PDF…[/]", spinner="dots"):
            pdf_profile = profile_pdf(pdf_path)
        console.print(Panel(
            _profile_renderable(pdf_profile),
            title="[bold]PDF Profile[/]",
            border_style="dim",
            padding=(0, 1),
        ))

        all_meta = run_pdf(pdf_path, run_dir, args.libraries, args.timeout, args.workers)
        write_summary(run_dir, all_meta, pdf_profile)

        console.print(f"\n  [dim]Results written to:[/] {run_dir}\n")


def _profile_renderable(profile: dict):
    """Return a Rich Table for the PDF profile suitable for use inside a Panel."""
    flags = []
    if profile.get("has_text"):            flags.append("[green]text[/]")
    if profile.get("has_images"):          flags.append("[blue]images[/]")
    if profile.get("has_tables"):          flags.append("[cyan]tables[/]")
    if profile.get("pages_multi_column"):  flags.append("[yellow]multi-column[/]")
    if profile.get("is_fully_rasterised"): flags.append("[bold red]FULLY RASTERISED[/]")

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(style="dim", min_width=14)
    t.add_column()
    t.add_row("Pages",       f"{profile['page_count']}  ({profile['file_size_kb']} KB)")
    t.add_row("Features",    "  ".join(flags) or "none detected")
    t.add_row("Text pages",  str(profile["pages_with_text"]))
    t.add_row("Image pages", f"{profile['pages_with_images']}  (rasterised: {profile['pages_rasterised']})")
    t.add_row("Table pages", str(profile["pages_with_tables"]))
    t.add_row("2-col pages", str(profile["pages_multi_column"]))
    return t


if __name__ == "__main__":
    main()
