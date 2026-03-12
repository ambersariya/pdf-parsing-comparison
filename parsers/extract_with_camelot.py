#!/usr/bin/env python3
"""
PDF extraction using camelot (lattice + stream) for tables and
pdfplumber for text and image-region detection.

Strategy:
  1. Camelot lattice  — bordered tables (ruled lines detected via OpenCV)
  2. Camelot stream   — borderless tables (whitespace heuristics)
  3. pdfplumber       — text extraction, column detection, image exclusion

Camelot tables with accuracy >= CAMELOT_MIN_ACCURACY are preferred.
pdfplumber fills in any page regions not covered by a camelot table.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import camelot
import pdfplumber

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
CAMELOT_MIN_ACCURACY = 75.0   # % — below this a camelot table is dropped
CAMELOT_EDGE_TOL = 50         # pixels — bbox edge tolerance when merging
PADDING = 4.0                 # pts — generic bbox padding


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text and tables from a PDF using camelot + pdfplumber."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument("-o", "--output", type=Path, help="Output markdown file (default: stdout)")
    parser.add_argument(
        "--page-marker",
        default="\n\n--- PAGE {page} ---\n\n",
        help="Separator between pages. Empty string to disable.",
    )
    parser.add_argument("--x-tolerance", type=float, default=2.0)
    parser.add_argument("--y-tolerance", type=float, default=2.0)
    parser.add_argument(
        "--camelot-flavor",
        choices=["lattice", "stream", "both"],
        default="both",
        help="Which camelot flavor(s) to use for table detection.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Bbox helpers (pts)
# ---------------------------------------------------------------------------

def pad_bbox(bbox: tuple, padding: float = PADDING) -> tuple:
    x0, top, x1, bottom = bbox
    return (x0 - padding, top - padding, x1 + padding, bottom + padding)


def bbox_area(bbox: tuple) -> float:
    x0, top, x1, bottom = bbox
    return max(0.0, x1 - x0) * max(0.0, bottom - top)


def intersect_bbox(a: tuple, b: tuple) -> tuple | None:
    x0, top = max(a[0], b[0]), max(a[1], b[1])
    x1, bottom = min(a[2], b[2]), min(a[3], b[3])
    return (x0, top, x1, bottom) if x1 > x0 and bottom > top else None


def overlaps(bbox: tuple, regions: list) -> bool:
    return any(intersect_bbox(bbox, r) for r in regions)


def overlap_ratio(bbox: tuple, regions: list) -> float:
    area = bbox_area(bbox)
    if area <= 0:
        return 0.0
    total = sum(
        bbox_area(i) for r in regions if (i := intersect_bbox(bbox, r)) is not None
    )
    return min(1.0, total / area)


# ---------------------------------------------------------------------------
# camelot coordinate conversion
# Camelot uses PDF coordinate origin at bottom-left; pdfplumber uses top-left.
# ---------------------------------------------------------------------------

def camelot_bbox_to_plumber(camelot_bbox: tuple, page_height: float) -> tuple:
    """Convert camelot (x1,y1,x2,y2) bottom-left origin → pdfplumber top-left."""
    cx1, cy1, cx2, cy2 = camelot_bbox
    # camelot: y1=bottom, y2=top  →  plumber: top = height-y2, bottom = height-y1
    return (cx1, page_height - cy2, cx2, page_height - cy1)


# ---------------------------------------------------------------------------
# Markdown table rendering
# ---------------------------------------------------------------------------

def normalise_cell(cell: str | None) -> str:
    if not cell:
        return ""
    return " ".join(part for part in cell.replace("\n", " ").split())


def render_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    if not any(header):
        header = [f"Column {i}" for i in range(1, width + 1)]
    sep = ["---"] * width

    def fmt(row: list[str]) -> str:
        cells = [c.replace("|", "\\|") for c in row]
        return "| " + " | ".join(cells) + " |"

    lines = [fmt(header), fmt(sep)] + [fmt(r) for r in padded[1:]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# camelot table extraction
# ---------------------------------------------------------------------------

def extract_camelot_tables(
    pdf_path: Path,
    page_number: int,   # 1-based
    page_height: float,
    flavor: str = "both",
) -> list[dict]:
    """
    Return a list of dicts: {bbox (plumber coords), markdown, accuracy, flavor}
    Deduplicates overlapping tables from lattice vs stream, preferring lattice.
    """
    page_str = str(page_number)
    results: list[dict] = []

    flavors_to_try: list[str]
    if flavor == "both":
        flavors_to_try = ["lattice", "stream"]
    else:
        flavors_to_try = [flavor]

    for fl in flavors_to_try:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor=fl)
        except Exception:
            continue

        for table in tables:
            accuracy = table.accuracy
            if accuracy < CAMELOT_MIN_ACCURACY:
                continue

            df = table.df
            rows = [[normalise_cell(str(v)) for v in row] for row in df.values.tolist()]
            if len(rows) < 2:
                continue

            # camelot parsing_report bbox is (x1,y1,x2,y2) in bottom-left coords
            cb = table._bbox  # (x1, y1, x2, y2) bottom-left origin
            bbox = camelot_bbox_to_plumber(cb, page_height)

            results.append({
                "bbox": bbox,
                "markdown": render_markdown_table(rows),
                "accuracy": accuracy,
                "flavor": fl,
            })

    # Deduplicate: if a stream table heavily overlaps a lattice table, drop stream
    if flavor == "both":
        lattice_bboxes = [t["bbox"] for t in results if t["flavor"] == "lattice"]
        filtered: list[dict] = []
        for t in results:
            if t["flavor"] == "stream" and overlap_ratio(t["bbox"], lattice_bboxes) > 0.5:
                continue
            filtered.append(t)
        results = filtered

    return results


# ---------------------------------------------------------------------------
# pdfplumber image region detection (unchanged from existing extractor)
# ---------------------------------------------------------------------------

def image_regions(page: pdfplumber.page.Page) -> list[tuple]:
    raw: list[tuple] = []
    for img in page.images:
        bbox = (float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"]))
        if bbox_area(bbox) < 400.0:
            continue
        raw.append(pad_bbox(bbox, 10.0))

    if not raw:
        return []

    raw.sort(key=lambda b: (b[1], b[0]))
    merged: list[list] = []
    for bbox in raw:
        if not merged:
            merged.append(list(bbox))
            continue
        prev = merged[-1]
        hgap = max(0.0, bbox[0] - prev[2], prev[0] - bbox[2])
        vgap = max(0.0, bbox[1] - prev[3], prev[1] - bbox[3])
        if hgap <= 20.0 and vgap <= 24.0:
            prev[0] = min(prev[0], bbox[0])
            prev[1] = min(prev[1], bbox[1])
            prev[2] = max(prev[2], bbox[2])
            prev[3] = max(prev[3], bbox[3])
        else:
            merged.append(list(bbox))

    return [pad_bbox(tuple(r), 8.0) for r in merged]


# ---------------------------------------------------------------------------
# pdfplumber text extraction helpers
# ---------------------------------------------------------------------------

def build_filtered_page(page: pdfplumber.page.Page, excluded: list) -> pdfplumber.page.Page:
    deduped = page.dedupe_chars()
    return deduped.filter(
        lambda obj: obj.get("object_type") != "char"
        or (
            obj.get("upright", True)
            and not overlaps(
                (float(obj["x0"]), float(obj["top"]), float(obj["x1"]), float(obj["bottom"])),
                excluded,
            )
        )
    )


def group_words_into_lines(words: list[dict], y_tolerance: float) -> list[list[dict]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue
        current_top = statistics.median(w["top"] for w in lines[-1])
        if abs(word["top"] - current_top) <= max(y_tolerance, 3.0):
            lines[-1].append(word)
        else:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def infer_line_height(lines: list[list[dict]]) -> float:
    heights = [
        max(float(w["bottom"]) - float(w["top"]) for w in line)
        for line in lines if line
    ]
    return statistics.median(heights) if heights else 12.0


def render_line(line: list[dict], x_tolerance: float) -> str:
    if not line:
        return ""
    parts = [line[0]["text"]]
    char_widths = [
        (float(w["x1"]) - float(w["x0"])) / max(len(w.get("text", "") or "x"), 1)
        for w in line if w.get("text")
    ]
    char_width = statistics.median(char_widths) if char_widths else 6.0
    space_width = max(char_width * 0.9, 1.0)
    for prev, curr in zip(line, line[1:]):
        gap = float(curr["x0"]) - float(prev["x1"])
        spacer = "" if gap <= x_tolerance else " " * max(1, round(gap / space_width))
        parts.append(spacer + curr["text"])
    return "".join(parts).rstrip()


def render_words_with_layout(words: list[dict], x_tolerance: float, y_tolerance: float) -> str:
    lines = group_words_into_lines(words, y_tolerance)
    if not lines:
        return ""
    line_height = infer_line_height(lines)
    rendered: list[str] = []
    prev_bottom: float | None = None
    for line in lines:
        top = min(float(w["top"]) for w in line)
        if prev_bottom is not None:
            vgap = top - prev_bottom
            if vgap > line_height * 1.6:
                rendered.extend([""] * max(1, round(vgap / max(line_height, 1.0)) - 1))
        rendered.append(render_line(line, x_tolerance))
        prev_bottom = max(float(w["bottom"]) for w in line)
    return "\n".join(rendered).strip()


def classify_two_column(words: list[dict], page_width: float) -> bool:
    if len(words) < 40:
        return False
    centres = [(w["x0"] + w["x1"]) / 2.0 for w in words]
    lc, rc = page_width * 0.45, page_width * 0.55
    left = [x for x in centres if x < lc]
    right = [x for x in centres if x > rc]
    gutter = [x for x in centres if lc <= x <= rc]
    n = len(centres)
    if len(left) / n < 0.25 or len(right) / n < 0.25 or len(gutter) / n > 0.20:
        return False
    lt = [w["top"] for w in words if (w["x0"] + w["x1"]) / 2.0 < lc]
    rt = [w["top"] for w in words if (w["x0"] + w["x1"]) / 2.0 > rc]
    if len(lt) < 10 or len(rt) < 10:
        return False
    proxy = max((w["bottom"] for w in words), default=0.0)
    if proxy and (max(lt) - min(lt) < proxy * 0.30 or max(rt) - min(rt) < proxy * 0.30):
        return False
    return True


def extract_region_words(
    page: pdfplumber.page.Page,
    bbox: tuple,
    excluded: list,
    x_tolerance: float,
    y_tolerance: float,
) -> list[dict]:
    x0, top, x1, bottom = bbox
    if x1 <= x0 or bottom <= top:
        return []
    filtered = build_filtered_page(page, excluded)
    segment = filtered.crop(bbox)
    words = segment.extract_words(
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
        extra_attrs=["size"],
        keep_blank_chars=False,
        use_text_flow=False,
    ) or []
    words = [w for w in words if w.get("text", "").strip()]
    if not words:
        return []

    lines = group_words_into_lines(words, y_tolerance)
    filtered_lines = []
    for line in lines:
        line_top = min(float(w["top"]) for w in line)
        line_bottom = max(float(w["bottom"]) for w in line)
        line_size = statistics.median(float(w.get("size", 0)) for w in line)
        if line_top < 24 and line_size <= 10:
            continue
        if (page.height - line_bottom) < 18 and line_size <= 10:
            continue
        filtered_lines.append(line)

    return [w for line in filtered_lines for w in line]


def extract_text_region(
    page: pdfplumber.page.Page,
    bbox: tuple,
    excluded: list,
    x_tolerance: float,
    y_tolerance: float,
) -> str:
    words = extract_region_words(page, bbox, excluded, x_tolerance, y_tolerance)
    if not words:
        return ""
    x0, top, x1, bottom = bbox
    width = x1 - x0
    if classify_two_column(words, width):
        mid = x0 + width / 2.0
        left_w = extract_region_words(page, (x0, top, mid, bottom), excluded, x_tolerance, y_tolerance)
        right_w = extract_region_words(page, (mid, top, x1, bottom), excluded, x_tolerance, y_tolerance)
        parts = [render_words_with_layout(w, x_tolerance, y_tolerance) for w in [left_w, right_w] if w]
        return "\n\n".join(parts)
    return render_words_with_layout(words, x_tolerance, y_tolerance)


# ---------------------------------------------------------------------------
# Per-page extraction combining camelot tables + pdfplumber text
# ---------------------------------------------------------------------------

def extract_page(
    pdf_path: Path,
    page: pdfplumber.page.Page,
    page_number: int,
    x_tolerance: float,
    y_tolerance: float,
    camelot_flavor: str,
) -> str:
    img_bboxes = image_regions(page)

    # Get camelot tables for this page
    camelot_tables = extract_camelot_tables(
        pdf_path, page_number, page.height, flavor=camelot_flavor
    )

    # Build exclusion zones: images + camelot table bboxes
    table_bboxes = [pad_bbox(t["bbox"]) for t in camelot_tables]
    exclusions = img_bboxes + table_bboxes

    # Collect all regions sorted top-to-bottom
    regions: list[dict] = []
    for bbox in img_bboxes:
        regions.append({"kind": "image", "bbox": bbox})
    for t in camelot_tables:
        regions.append({"kind": "table", "bbox": t["bbox"], "markdown": t["markdown"], "accuracy": t["accuracy"], "flavor": t["flavor"]})
    regions.sort(key=lambda r: r["bbox"][1])

    rendered: list[str] = []
    cursor = 0.0

    for region in regions:
        top = float(region["bbox"][1])
        if top > cursor:
            text = extract_text_region(
                page, (0.0, cursor, page.width, top),
                excluded=exclusions, x_tolerance=x_tolerance, y_tolerance=y_tolerance,
            )
            if text:
                rendered.append(text)

        if region["kind"] == "table":
            label = f"<!-- camelot:{region['flavor']} accuracy:{region['accuracy']:.0f}% -->"
            rendered.append(label + "\n" + region["markdown"])

        cursor = max(cursor, float(region["bbox"][3]))

    if cursor < page.height:
        text = extract_text_region(
            page, (0.0, cursor, page.width, page.height),
            excluded=exclusions, x_tolerance=x_tolerance, y_tolerance=y_tolerance,
        )
        if text:
            rendered.append(text)

    if not rendered:
        return extract_text_region(
            page, (0.0, 0.0, page.width, page.height),
            excluded=img_bboxes, x_tolerance=x_tolerance, y_tolerance=y_tolerance,
        )

    return "\n\n".join(part for part in rendered if part)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_pdf(
    pdf_path: Path,
    x_tolerance: float,
    y_tolerance: float,
    page_marker: str,
    camelot_flavor: str,
) -> str:
    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = extract_page(
                pdf_path, page, page_number,
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
                camelot_flavor=camelot_flavor,
            )
            if not page_text:
                page_text = "[No extractable text on this page]"
            if page_marker:
                pages.append(page_marker.format(page=page_number) + page_text)
            else:
                pages.append(page_text)
    return "\n".join(pages).strip() + "\n"


def main() -> None:
    args = parse_args()
    text = extract_pdf(
        pdf_path=args.pdf,
        x_tolerance=args.x_tolerance,
        y_tolerance=args.y_tolerance,
        page_marker=args.page_marker,
        camelot_flavor=args.camelot_flavor,
    )
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
