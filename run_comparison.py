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
      report.html     ← self-contained HTML report

Usage:
  python run_comparison.py my.pdf
  python run_comparison.py *.pdf --results-dir /tmp/pdf-runs
  python run_comparison.py my.pdf --libraries pdfplumber camelot
  python run_comparison.py my.pdf --baseline results/prev_run/summary.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
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
    "pymupdf4llm": [sys.executable, "parsers/extract_with_pymupdf4llm.py", "{pdf}", "-o", "{output}"],
    "pdf_oxide":   [sys.executable, "parsers/extract_with_pdf_oxide.py",   "{pdf}", "-o", "{output}"],
    "pypdf":       [sys.executable, "parsers/extract_with_pypdf.py",       "{pdf}", "-o", "{output}"],
    "markitdown":  [sys.executable, "parsers/extract_with_markitdown.py",  "{pdf}", "-o", "{output}"],
    "docling":          [sys.executable, "parsers/extract_with_docling.py",          "{pdf}", "-o", "{output}"],
    "marker":           [sys.executable, "parsers/extract_with_marker.py",           "{pdf}", "-o", "{output}"],
    "amazon_textract":  [sys.executable, "parsers/extract_with_amazon_textract.py",  "{pdf}", "-o", "{output}"],
}

# Per-library timeout overrides (seconds).  None = no timeout.
# ML-based libraries load large model weights and can legitimately run for many
# minutes, especially on first run — so we exempt them from the global limit.
LIBRARY_TIMEOUTS: dict[str, int | None] = {
    "marker":          None,
    "docling":         None,
    "amazon_textract": None,  # network-bound; exempt from global timeout
}

# Extra environment variables injected into a library's subprocess.
# PYTORCH_ENABLE_MPS_FALLBACK=1 lets PyTorch silently fall back to CPU for any
# op the Apple MPS backend doesn't support, preventing AcceleratorError crashes
# on large PDFs where surya's attention sequence length overflows MPS limits.
LIBRARY_ENV: dict[str, dict[str, str]] = {
    "marker": {"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
}

# PyPI distribution name for each library key, used for version lookup.
_PKG_NAME: dict[str, str] = {
    "pdfplumber": "pdfplumber",
    "camelot":    "camelot-py",
    "pymupdf4llm": "pymupdf4llm",
    "pdf_oxide":  "pdf-oxide",
    "pypdf":      "pypdf",
    "markitdown": "markitdown",
    "docling":    "docling",
    "marker":          "marker-pdf",
    "amazon_textract": "amazon-textract-textractor",
}


def _lib_version(lib: str) -> str:
    """Return the installed version of the package backing *lib*, or 'unknown'."""
    try:
        return importlib.metadata.version(_PKG_NAME.get(lib, lib))
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


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
    parser.add_argument("--baseline", type=Path, default=None, metavar="SUMMARY_JSON",
                        help="Path to a previous run's summary.json to diff results against.")
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


# ---------------------------------------------------------------------------
# Process monitoring
# Polls the subprocess RSS and CPU every 100 ms in a background thread.
# ---------------------------------------------------------------------------

def _monitor_process(pid: int, result: dict, stop: threading.Event) -> None:
    """Record peak RSS (bytes) and last-known CPU times of a process until stop is set."""
    peak_rss = 0
    cpu_user = 0.0
    cpu_sys  = 0.0
    try:
        proc = psutil.Process(pid)
        while not stop.is_set():
            try:
                mem = proc.memory_info().rss
                if mem > peak_rss:
                    peak_rss = mem
                ct = proc.cpu_times()
                cpu_user = ct.user
                cpu_sys  = ct.system
            except psutil.NoSuchProcess:
                break
            stop.wait(0.1)
    except psutil.NoSuchProcess:
        pass
    result["peak_rss_bytes"] = peak_rss
    result["cpu_user_s"]     = round(cpu_user, 2)
    result["cpu_sys_s"]      = round(cpu_sys,  2)


# ---------------------------------------------------------------------------
# Output quality metrics
# Derived from the extracted markdown after the run completes.
# ---------------------------------------------------------------------------

def _analyze_output(output_file: Path) -> dict:
    """Return text-quality metrics for the extracted markdown file."""
    if not output_file.exists():
        return {}
    text  = output_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    non_empty = [ln for ln in lines if ln.strip()]

    table_count   = sum(1 for ln in lines if ln.startswith("| ---"))
    heading_count = sum(1 for ln in lines if re.match(r"^#{1,6}\s", ln))
    enc_errors    = text.count("\ufffd")
    dup_ratio     = (
        round(1.0 - len(set(non_empty)) / len(non_empty), 3)
        if non_empty else 0.0
    )

    return {
        "char_count":           len(text),
        "word_count":           len(text.split()),
        "line_count":           text.count("\n"),
        "table_count":          table_count,
        "heading_count":        heading_count,
        "encoding_errors":      enc_errors,
        "duplicate_line_ratio": dup_ratio,
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
        "library":              library,
        "lib_version":          _lib_version(library),
        "pdf":                  str(pdf_path),
        "output":               str(output_file),
        "command":              cmd,
        "started_at":           datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":      None,
        "status":               None,
        "returncode":           None,
        "error":                None,
        # --- resource metrics ---
        "output_bytes":         None,
        "peak_rss_mb":          None,
        "cpu_user_s":           None,
        "cpu_sys_s":            None,
        # --- quality metrics ---
        "char_count":           None,
        "word_count":           None,
        "line_count":           None,
        "table_count":          None,
        "heading_count":        None,
        "encoding_errors":      None,
        "duplicate_line_ratio": None,
    }

    t0 = time.perf_counter()
    proc_result: dict = {}
    stop_event = threading.Event()

    try:
        proc_env = {**os.environ, **LIBRARY_ENV.get(library, {})}
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent,
            env=proc_env,
        )

        monitor = threading.Thread(
            target=_monitor_process,
            args=(proc.pid, proc_result, stop_event),
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
        peak = proc_result.get("peak_rss_bytes", 0)
        meta["peak_rss_mb"] = round(peak / 1024 / 1024, 1) if peak else None
        meta["cpu_user_s"]  = proc_result.get("cpu_user_s")
        meta["cpu_sys_s"]   = proc_result.get("cpu_sys_s")

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
                pool.submit(
                    run_library, lib, LIBRARIES[lib], pdf_path, run_dir / lib,
                    LIBRARY_TIMEOUTS[lib] if lib in LIBRARY_TIMEOUTS else timeout,
                ): lib
                for lib in libraries
            }
            for lib in libraries:
                progress.update(task_ids[lib], status="[yellow]running…[/]")

            for future in as_completed(futures_to_lib):
                lib  = futures_to_lib[future]
                meta = future.result()
                results[lib] = meta

                s = meta["status"]
                if s == "ok":
                    enc = meta.get("encoding_errors") or 0
                    enc_str = f"  [red]{enc} enc-err[/]" if enc else ""
                    status_str = (
                        f"[green]OK[/]  "
                        f"[white]{meta['elapsed_seconds']:.1f}s[/]  "
                        f"[dim]{meta['peak_rss_mb'] or 0} MB[/]  "
                        f"[cyan]{meta['word_count'] or 0:,} words[/]  "
                        f"[blue]{meta['table_count'] or 0} tables[/]  "
                        f"[magenta]{meta.get('heading_count') or 0} hdgs[/]"
                        f"{enc_str}"
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
# Summary table + HTML report
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
    t.add_column("Library",  style="bold cyan",    min_width=11)
    t.add_column("Ver",      style="dim",          min_width=6)
    t.add_column("Status",   justify="center",     min_width=9)
    t.add_column("Wall",     justify="right",      style="white",   min_width=6)
    t.add_column("CPU",      justify="right",      style="dim",     min_width=6)
    t.add_column("Mem",      justify="right",      style="dim",     min_width=7)
    t.add_column("Words",    justify="right",      style="cyan",    min_width=7)
    t.add_column("Tables",   justify="right",      style="blue",    min_width=6)
    t.add_column("Hdgs",     justify="right",      style="magenta", min_width=5)
    t.add_column("Enc✗",     justify="right",      min_width=5)
    t.add_column("Dup%",     justify="right",      style="dim",     min_width=5)

    for m in all_meta:
        status  = _STATUS_STYLE.get(m["status"] or "", m["status"] or "")
        wall    = f"{m['elapsed_seconds']:.1f}s"  if m["elapsed_seconds"] is not None else "---"
        cpu_tot = (m.get("cpu_user_s") or 0) + (m.get("cpu_sys_s") or 0)
        cpu     = f"{cpu_tot:.1f}s"               if m.get("cpu_user_s") is not None  else "---"
        mem     = f"{m['peak_rss_mb']}MB"         if m["peak_rss_mb"]    is not None  else "---"
        words   = f"{m['word_count']:,}"          if m["word_count"]     is not None  else "---"
        tables  = str(m["table_count"])           if m["table_count"]    is not None  else "---"
        hdgs    = str(m.get("heading_count") or 0) if m.get("heading_count") is not None else "---"
        enc_val = m.get("encoding_errors")
        enc     = ("[red]" + str(enc_val) + "[/]") if enc_val else (str(enc_val) if enc_val is not None else "---")
        dup_val = m.get("duplicate_line_ratio")
        dup     = f"{dup_val * 100:.1f}%" if dup_val is not None else "---"

        t.add_row(m["library"], m.get("lib_version") or "?", status,
                  wall, cpu, mem, words, tables, hdgs, enc, dup)

    console.print()
    console.print(t)

    _generate_html_report(run_dir, all_meta, pdf_profile)
    console.print(f"  [dim]HTML report:[/] {run_dir / 'report.html'}\n")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _generate_html_report(run_dir: Path, all_meta: list[dict], pdf_profile: dict) -> None:
    """Write a self-contained, sortable HTML report to run_dir/report.html."""
    pdf_name     = Path(all_meta[0]["pdf"]).name if all_meta else "unknown"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Profile table rows
    flags = []
    if pdf_profile.get("has_text"):            flags.append("text")
    if pdf_profile.get("has_images"):          flags.append("images")
    if pdf_profile.get("has_tables"):          flags.append("tables")
    if pdf_profile.get("pages_multi_column"):  flags.append("multi-column")
    if pdf_profile.get("is_fully_rasterised"): flags.append("FULLY RASTERISED")

    profile_rows_html = ""
    for label, val in [
        ("Pages",       f"{pdf_profile.get('page_count', '?')}  ({pdf_profile.get('file_size_kb', '?')} KB)"),
        ("Features",    ", ".join(flags) or "none"),
        ("Text pages",  str(pdf_profile.get("pages_with_text",   "?"))),
        ("Image pages", f"{pdf_profile.get('pages_with_images', '?')}  (rasterised: {pdf_profile.get('pages_rasterised', '?')})"),
        ("Table pages", str(pdf_profile.get("pages_with_tables", "?"))),
        ("2-col pages", str(pdf_profile.get("pages_multi_column","?"))),
    ]:
        profile_rows_html += f"<tr><td>{label}</td><td>{val}</td></tr>\n"

    # Results table rows
    def _e(v: object) -> str:
        """HTML-escape a string value."""
        return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _fmt(val: object, spec: str) -> str:
        if val is None:
            return "—"
        try:
            return format(val, spec)
        except (TypeError, ValueError):
            return str(val)

    results_rows_html = ""
    for m in all_meta:
        s            = m.get("status", "")
        status_cls   = {"ok": "ok", "error": "err", "timeout": "warn", "exception": "err"}.get(s, "")
        cpu_tot      = (m.get("cpu_user_s") or 0) + (m.get("cpu_sys_s") or 0)
        cpu_str      = f"{cpu_tot:.1f}s" if m.get("cpu_user_s") is not None else "—"
        dup_val      = m.get("duplicate_line_ratio")
        dup_str      = f"{dup_val * 100:.1f}%" if dup_val is not None else "—"
        enc_val      = m.get("encoding_errors") or 0
        enc_cls      = "enc-err" if enc_val > 0 else ""

        results_rows_html += f"""
        <tr>
          <td><strong>{_e(m.get('library', '?'))}</strong></td>
          <td class="dim">{_e(m.get('lib_version', '?'))}</td>
          <td class="status {status_cls}">{_e(s.upper())}</td>
          <td>{_fmt(m.get('elapsed_seconds'), '.1f')}s</td>
          <td>{cpu_str}</td>
          <td>{_fmt(m.get('peak_rss_mb'), '.0f')} MB</td>
          <td>{_fmt(m.get('word_count'), ',')}</td>
          <td>{_fmt(m.get('table_count'), 'd')}</td>
          <td>{_fmt(m.get('heading_count'), 'd')}</td>
          <td class="{enc_cls}">{enc_val}</td>
          <td>{dup_str}</td>
          <td>{_fmt(m.get('char_count'), ',')}</td>
        </tr>"""

    # Output previews
    previews_html = ""
    for m in all_meta:
        lib         = m.get("library", "?")
        ver         = m.get("lib_version", "?")
        output_path = Path(m.get("output", ""))
        if output_path.exists() and m.get("status") == "ok":
            raw     = output_path.read_text(encoding="utf-8", errors="replace")[:4000]
            preview = _e(raw)
        else:
            preview = _e(m.get("error") or "No output produced.")
        previews_html += f"""
        <details>
          <summary><strong>{_e(lib)}</strong> <span class="dim">v{_e(ver)}</span></summary>
          <pre class="output">{preview}</pre>
        </details>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PDF Extraction Report — {_e(pdf_name)}</title>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d27; --border:#2d3148;
    --text:#e2e8f0; --dim:#6b7280; --accent:#60a5fa;
    --green:#34d399; --red:#f87171; --yellow:#fbbf24; --magenta:#e879f9;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0 }}
  body {{ font-family:ui-monospace,monospace; background:var(--bg); color:var(--text);
          font-size:13px; line-height:1.6; padding:2rem; max-width:1400px; margin:0 auto }}
  h1  {{ color:var(--accent); font-size:1.4rem; margin-bottom:.2rem }}
  h2  {{ color:var(--dim); font-size:.85rem; font-weight:normal; margin-bottom:1.5rem }}
  h3  {{ color:var(--text); font-size:1rem; margin:2rem 0 .75rem }}
  .meta {{ color:var(--dim); font-size:.8rem; margin-bottom:2rem }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:1.5rem }}
  th {{ background:var(--surface); color:var(--dim); padding:6px 10px; text-align:left;
        font-size:.72rem; text-transform:uppercase; letter-spacing:.05em;
        border-bottom:1px solid var(--border); cursor:pointer; user-select:none }}
  th:hover {{ color:var(--text) }}
  th.sort-asc::after  {{ content:" ▲"; color:var(--accent) }}
  th.sort-desc::after {{ content:" ▼"; color:var(--accent) }}
  td {{ padding:6px 10px; border-bottom:1px solid var(--border) }}
  tr:hover td {{ background:var(--surface) }}
  .dim {{ color:var(--dim) }}
  .status.ok   {{ color:var(--green);   font-weight:bold }}
  .status.err  {{ color:var(--red);     font-weight:bold }}
  .status.warn {{ color:var(--yellow);  font-weight:bold }}
  .enc-err {{ color:var(--red) }}
  details {{ border:1px solid var(--border); border-radius:6px; margin-bottom:.6rem; overflow:hidden }}
  summary {{ background:var(--surface); padding:8px 12px; cursor:pointer; list-style:none }}
  summary:hover {{ background:#22263a }}
  summary::marker, summary::-webkit-details-marker {{ display:none }}
  summary::before {{ content:"▶  "; color:var(--dim); font-size:.7rem }}
  details[open] summary::before {{ content:"▼  " }}
  pre.output {{ padding:1rem; font-size:12px; line-height:1.5; white-space:pre-wrap;
                word-break:break-word; max-height:420px; overflow-y:auto }}
</style>
</head>
<body>
<h1>PDF Extraction Report</h1>
<h2>{_e(pdf_name)}</h2>
<p class="meta">Run&nbsp;dir: {_e(str(run_dir))} &nbsp;|&nbsp; Generated: {generated_at}</p>

<h3>PDF Profile</h3>
<table>
  <thead><tr><th>Property</th><th>Value</th></tr></thead>
  <tbody>{profile_rows_html}</tbody>
</table>

<h3>Results <span class="dim">(click column header to sort)</span></h3>
<table id="results">
  <thead><tr>
    <th>Library</th><th>Version</th><th>Status</th>
    <th>Wall</th><th>CPU</th><th>Mem</th>
    <th>Words</th><th>Tables</th><th>Headings</th>
    <th>Enc.Errors</th><th>Dup%</th><th>Chars</th>
  </tr></thead>
  <tbody>{results_rows_html}</tbody>
</table>

<h3>Output Previews <span class="dim">(first 4,000 chars)</span></h3>
{previews_html}

<script>
document.querySelectorAll('#results th').forEach((th, col) => {{
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const rows  = [...tbody.querySelectorAll('tr')];
    const asc   = th.classList.toggle('sort-asc');
    if (!asc) th.classList.toggle('sort-desc', true); else th.classList.remove('sort-desc');
    th.closest('tr').querySelectorAll('th').forEach(o => {{ if (o !== th) {{ o.classList.remove('sort-asc','sort-desc') }} }});
    rows.sort((a, b) => {{
      const av = a.cells[col].textContent.trim();
      const bv = b.cells[col].textContent.trim();
      const an = parseFloat(av.replace(/[^0-9.-]/g, ''));
      const bn = parseFloat(bv.replace(/[^0-9.-]/g, ''));
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>"""

    (run_dir / "report.html").write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Final cross-PDF summary matrix
# ---------------------------------------------------------------------------

def print_final_matrix(all_runs: list[tuple[Path, list[dict]]]) -> None:
    """
    Print a parsers × PDFs matrix summarising every run.

    Rows  = libraries
    Cols  = PDF files (one column each)
    Cell  = status + wall time + peak memory (compact)

    Also appends two aggregate columns when more than one PDF was processed:
      Avg Wall  — mean elapsed time across successful runs
      Avg Mem   — mean peak RSS across successful runs
    """
    if not all_runs:
        return

    # Preserve library order from the first run
    libs = [m["library"] for m in all_runs[0][1]]

    # Build a (pdf_name → display_label) list, deduplicating by stem if needed
    pdf_labels: list[str] = []
    for pdf_path, _ in all_runs:
        label = pdf_path.name
        if len(label) > 22:
            label = label[:19] + "…"
        pdf_labels.append(label)

    # Index: (pdf_label, library) → meta dict
    index: dict[tuple[str, str], dict] = {}
    for (pdf_path, metas), label in zip(all_runs, pdf_labels):
        for m in metas:
            index[(label, m["library"])] = m

    multi = len(all_runs) > 1

    console.print()
    console.print(Rule("[bold]Final Summary Matrix[/]", style="bright_blue"))

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold", padding=(0, 1))
    t.add_column("Library", style="bold cyan", min_width=12, no_wrap=True)
    for label in pdf_labels:
        t.add_column(label, justify="center", min_width=16, no_wrap=False)
    if multi:
        t.add_column("Avg Wall", justify="right", style="white",  min_width=8)
        t.add_column("Avg Mem",  justify="right", style="dim",    min_width=8)

    for lib in libs:
        row: list[str] = [lib]
        wall_times: list[float] = []
        mem_values: list[float] = []

        for label in pdf_labels:
            m = index.get((label, lib))
            if m is None:
                row.append("[dim]—[/]")
                continue

            s = m.get("status")
            elapsed = m.get("elapsed_seconds")
            mem     = m.get("peak_rss_mb")

            if s == "ok":
                cell = f"[green]OK[/]  [white]{elapsed:.1f}s[/]"
                if mem is not None:
                    cell += f"\n[dim]{mem:.0f} MB[/]"
                    mem_values.append(mem)
                if elapsed is not None:
                    wall_times.append(elapsed)
            elif s == "timeout":
                cell = f"[yellow]TIMEOUT[/]\n[dim]{elapsed:.0f}s[/]" if elapsed else "[yellow]TIMEOUT[/]"
            elif s == "error":
                cell = f"[red]FAILED[/]  rc={m.get('returncode', '?')}"
            else:
                cell = "[red]EXCEPTION[/]"

            row.append(cell)

        if multi:
            row.append(f"{sum(wall_times)/len(wall_times):.1f}s" if wall_times else "—")
            row.append(f"{sum(mem_values)/len(mem_values):.0f} MB" if mem_values else "—")

        t.add_row(*row)

    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

# Metrics where a lower value is better (delta<0 = green).
_LOWER_IS_BETTER = {"elapsed_seconds", "peak_rss_mb", "encoding_errors",
                    "duplicate_line_ratio", "cpu_user_s", "cpu_sys_s"}


def compare_baseline(baseline_path: Path, all_meta: list[dict]) -> None:
    """Print a regression table diffing current results against a previous summary.json."""
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[yellow]Could not load baseline:[/] {exc}")
        return

    base_by_lib: dict[str, dict] = {m["library"]: m for m in baseline.get("results", [])}

    METRICS = [
        ("elapsed_seconds",      "Wall time (s)",  ".1f", "s"),
        ("cpu_user_s",           "CPU user (s)",   ".1f", "s"),
        ("peak_rss_mb",          "Mem (MB)",        ".0f", " MB"),
        ("word_count",           "Words",            ",",  ""),
        ("table_count",          "Tables",           "d",  ""),
        ("heading_count",        "Headings",         "d",  ""),
        ("encoding_errors",      "Enc. errors",      "d",  ""),
        ("duplicate_line_ratio", "Dup. ratio",      ".3f", ""),
    ]

    t = Table(box=box.SIMPLE, header_style="bold", title="[bold]Baseline Comparison[/]",
              title_style="bright_blue")
    t.add_column("Library",  style="bold cyan", min_width=12)
    t.add_column("Metric",   style="dim",       min_width=16)
    t.add_column("Baseline", justify="right",   min_width=10)
    t.add_column("Current",  justify="right",   min_width=10)
    t.add_column("Delta",    justify="right",   min_width=10)

    for cur in all_meta:
        lib  = cur["library"]
        base = base_by_lib.get(lib)
        if not base:
            continue
        for key, label, fmt_spec, unit in METRICS:
            b_val = base.get(key)
            c_val = cur.get(key)
            if b_val is None or c_val is None:
                continue
            try:
                delta = float(c_val) - float(b_val)
            except (TypeError, ValueError):
                continue

            threshold = max(abs(float(b_val)) * 0.05, 1e-9)
            if key in _LOWER_IS_BETTER:
                delta_style = "green" if delta < -threshold else ("red" if delta > threshold else "dim")
            else:
                delta_style = "green" if delta > threshold else ("red" if delta < -threshold else "dim")

            try:
                b_str = format(b_val, fmt_spec) + unit
                c_str = format(c_val, fmt_spec) + unit
                d_str = f"[{delta_style}]{delta:+.2g}{unit}[/]"
            except (TypeError, ValueError):
                b_str, c_str, d_str = str(b_val), str(c_val), "?"

            t.add_row(lib, label, b_str, c_str, d_str)

    console.print()
    console.print(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_pdfs(patterns: list[str]) -> list[Path]:
    """Expand shell glob patterns and plain paths into a sorted, deduplicated list of PDFs."""
    seen:  set[Path]  = set()
    paths: list[Path] = []
    for pattern in patterns:
        if any(c in pattern for c in ("*", "?", "[")):
            matches = sorted(Path.cwd().glob(pattern))
        else:
            matches = [Path(pattern)]
        for p in matches:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(p)
    return paths


def main() -> None:
    args      = parse_args()
    pdf_paths = resolve_pdfs(args.pdfs)

    if not pdf_paths:
        console.print("[bold red]ERROR[/] No PDF files matched the given patterns.", highlight=False)
        sys.exit(1)

    # Capture the session timestamp once, before any processing starts, so the
    # directory name reflects when the user invoked the script — not when the
    # (potentially slow) parsers finished.
    session_start = time.perf_counter()
    session_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir   = args.results_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\nFound [bold]{len(pdf_paths)}[/] PDF(s) to process.")
    console.print(f"  [dim]Session dir:[/] {session_dir}\n")

    all_runs: list[tuple[Path, list[dict]]] = []

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            console.print(f"[yellow]WARN[/] {pdf_path} not found — skipping")
            continue

        run_dir = session_dir / pdf_path.stem
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
        all_runs.append((pdf_path, all_meta))

        if args.baseline:
            console.print(Rule("[dim]Baseline Comparison[/]", style="dim"))
            compare_baseline(args.baseline, all_meta)

    print_final_matrix(all_runs)

    total = time.perf_counter() - session_start
    m, s  = divmod(int(total), 60)
    total_str = f"{m}m {s}s" if m else f"{s}s"
    console.print(f"  [dim]All results written to:[/] {session_dir}")
    console.print(f"  [dim]Total run time:[/] [bold white]{total_str}[/]\n")


if __name__ == "__main__":
    main()
