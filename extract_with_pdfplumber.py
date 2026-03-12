#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pdfplumber


PADDING = 4.0

TABLE_SETTINGS: dict = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 3,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "intersection_tolerance": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF with a simple column-aware pdfplumber pipeline."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output text/markdown file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--page-marker",
        default="\n\n--- PAGE {page} ---\n\n",
        help="Marker inserted between pages. Use an empty string to disable.",
    )
    parser.add_argument(
        "--x-tolerance",
        type=float,
        default=2.0,
        help="pdfplumber x_tolerance passed to extract_text/extract_words.",
    )
    parser.add_argument(
        "--y-tolerance",
        type=float,
        default=2.0,
        help="pdfplumber y_tolerance passed to extract_text/extract_words.",
    )
    return parser.parse_args()


def normalise(text: str | None) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def pad_bbox(bbox: tuple[float, float, float, float], padding: float = PADDING) -> tuple[float, float, float, float]:
    x0, top, x1, bottom = bbox
    return (x0 - padding, top - padding, x1 + padding, bottom + padding)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, top, x1, bottom = bbox
    return max(0.0, x1 - x0) * max(0.0, bottom - top)


def intersect_bbox(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    x0 = max(a[0], b[0])
    top = max(a[1], b[1])
    x1 = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if x1 <= x0 or bottom <= top:
        return None
    return (x0, top, x1, bottom)


def overlaps(bbox: tuple[float, float, float, float], regions: list[tuple[float, float, float, float]]) -> bool:
    return any(intersect_bbox(bbox, region) is not None for region in regions)


def overlap_ratio(bbox: tuple[float, float, float, float], regions: list[tuple[float, float, float, float]]) -> float:
    area = bbox_area(bbox)
    if area <= 0:
        return 0.0
    overlap_area = 0.0
    for region in regions:
        intersection = intersect_bbox(bbox, region)
        if intersection is not None:
            overlap_area += bbox_area(intersection)
    return min(1.0, overlap_area / area)


def image_regions(page: pdfplumber.page.Page) -> list[tuple[float, float, float, float]]:
    raw_regions: list[tuple[float, float, float, float]] = []
    for image in page.images:
        bbox = (float(image["x0"]), float(image["top"]), float(image["x1"]), float(image["bottom"]))
        if bbox_area(bbox) < 400.0:
            continue
        raw_regions.append(pad_bbox(bbox, padding=10.0))

    if not raw_regions:
        return []

    raw_regions.sort(key=lambda bbox: (bbox[1], bbox[0]))
    merged: list[list[float]] = []
    for bbox in raw_regions:
        if not merged:
            merged.append(list(bbox))
            continue

        previous = merged[-1]
        horizontal_gap = max(0.0, bbox[0] - previous[2], previous[0] - bbox[2])
        vertical_gap = max(0.0, bbox[1] - previous[3], previous[1] - bbox[3])
        if horizontal_gap <= 20.0 and vertical_gap <= 24.0:
            previous[0] = min(previous[0], bbox[0])
            previous[1] = min(previous[1], bbox[1])
            previous[2] = max(previous[2], bbox[2])
            previous[3] = max(previous[3], bbox[3])
        else:
            merged.append(list(bbox))

    return [pad_bbox(tuple(region), padding=8.0) for region in merged]


def normalise_cell(cell: str | None) -> str:
    if not cell:
        return ""
    return " ".join(part for part in cell.replace("\n", " ").split())


def is_probable_table(table: pdfplumber.table.Table, blocked_regions: list[tuple[float, float, float, float]]) -> bool:
    bbox = tuple(float(value) for value in table.bbox)
    if overlap_ratio(bbox, blocked_regions) > 0.15:
        return False

    rows = table.extract() or []
    rows = [[normalise_cell(cell) for cell in row] for row in rows if row]
    if len(rows) < 2:
        return False

    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False

    populated = sum(1 for row in rows for cell in row if cell)
    total = sum(max(len(row), width) for row in rows)
    if total == 0 or (populated / total) < 0.5:
        return False

    avg_cell_length = sum(len(cell) for row in rows for cell in row if cell) / max(populated, 1)
    if avg_cell_length > 40:
        return False

    return True


def render_markdown_table(rows: list[list[str]]) -> str:
    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = padded_rows[0]
    if not any(header):
        header = [f"Column {index}" for index in range(1, width + 1)]
    separator = ["---"] * width

    def format_row(row: list[str]) -> str:
        cells = [cell.replace("|", "\\|") for cell in row]
        return "| " + " | ".join(cells) + " |"

    body = [format_row(header), format_row(separator)]
    for row in padded_rows[1:]:
        body.append(format_row(row))
    return "\n".join(body)


def table_regions(page: pdfplumber.page.Page, blocked_regions: list[tuple[float, float, float, float]]) -> list[dict]:
    regions: list[dict] = []
    for table in page.find_tables(table_settings=TABLE_SETTINGS):
        if not is_probable_table(table, blocked_regions):
            continue
        rows = table.extract() or []
        rows = [[normalise_cell(cell) for cell in row] for row in rows if row]
        if len(rows) < 2:
            continue
        bbox = tuple(float(value) for value in table.bbox)
        regions.append({"bbox": bbox, "markdown": render_markdown_table(rows)})
    return regions


def build_filtered_page(
    page: pdfplumber.page.Page, excluded_regions: list[tuple[float, float, float, float]]
) -> pdfplumber.page.Page:
    deduped = page.dedupe_chars()
    return deduped.filter(
        lambda obj: obj.get("object_type") != "char"
        or (
            obj.get("upright", True)
            and not overlaps(
                (
                    float(obj["x0"]),
                    float(obj["top"]),
                    float(obj["x1"]),
                    float(obj["bottom"]),
                ),
                excluded_regions,
            )
        )
    )


def group_words_into_lines(words: list[dict], y_tolerance: float) -> list[list[dict]]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: (word["top"], word["x0"]))
    lines: list[list[dict]] = []

    for word in sorted_words:
        if not lines:
            lines.append([word])
            continue

        current_line = lines[-1]
        current_top = statistics.median(item["top"] for item in current_line)
        if abs(word["top"] - current_top) <= max(y_tolerance, 3.0):
            current_line.append(word)
        else:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda word: word["x0"])

    return lines


def infer_line_height(lines: list[list[dict]]) -> float:
    heights = [
        max(float(word["bottom"]) - float(word["top"]) for word in line)
        for line in lines
        if line
    ]
    if not heights:
        return 12.0
    return statistics.median(heights)


def render_line(line: list[dict], x_tolerance: float) -> str:
    if not line:
        return ""

    parts = [line[0]["text"]]
    typical_char_widths = []
    for word in line:
        text = word.get("text", "")
        if text:
            typical_char_widths.append((float(word["x1"]) - float(word["x0"])) / max(len(text), 1))

    char_width = statistics.median(typical_char_widths) if typical_char_widths else 6.0
    space_width = max(char_width * 0.9, 1.0)

    for previous, current in zip(line, line[1:]):
        gap = float(current["x0"]) - float(previous["x1"])
        if gap <= x_tolerance:
            spacer = ""
        else:
            spacer_count = max(1, round(gap / space_width))
            spacer = " " * spacer_count
        parts.append(spacer + current["text"])

    return "".join(parts).rstrip()


def render_words_with_layout(words: list[dict], x_tolerance: float, y_tolerance: float) -> str:
    lines = group_words_into_lines(words, y_tolerance=y_tolerance)
    if not lines:
        return ""

    line_height = infer_line_height(lines)
    rendered: list[str] = []
    previous_bottom: float | None = None

    for line in lines:
        current_top = min(float(word["top"]) for word in line)
        if previous_bottom is not None:
            vertical_gap = current_top - previous_bottom
            if vertical_gap > line_height * 1.6:
                blank_lines = max(1, round(vertical_gap / max(line_height, 1.0)) - 1)
                rendered.extend([""] * blank_lines)

        rendered.append(render_line(line, x_tolerance=x_tolerance))
        previous_bottom = max(float(word["bottom"]) for word in line)

    return "\n".join(rendered).strip()


def extract_region_words(
    page: pdfplumber.page.Page,
    bbox: tuple[float, float, float, float],
    excluded_regions: list[tuple[float, float, float, float]],
    x_tolerance: float,
    y_tolerance: float,
) -> list[dict]:
    x0, top, x1, bottom = bbox
    if x1 <= x0 or bottom <= top:
        return []

    filtered_page = build_filtered_page(page, excluded_regions)
    segment = filtered_page.crop(bbox)
    words = segment.extract_words(
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
        extra_attrs=["size"],
        keep_blank_chars=False,
        use_text_flow=False,
    ) or []
    words = [word for word in words if word.get("text", "").strip()]
    if not words:
        return []

    lines = group_words_into_lines(words, y_tolerance=y_tolerance)
    filtered_lines: list[list[dict]] = []
    for line in lines:
        line_top = min(float(word["top"]) for word in line)
        line_bottom = max(float(word["bottom"]) for word in line)
        line_size = statistics.median(float(word.get("size", 0.0)) for word in line)

        if line_top < 24 and line_size <= 10:
            continue
        if (page.height - line_bottom) < 18 and line_size <= 10:
            continue

        filtered_lines.append(line)

    return [word for line in filtered_lines for word in line]


def extract_text_region(
    page: pdfplumber.page.Page,
    bbox: tuple[float, float, float, float],
    excluded_regions: list[tuple[float, float, float, float]],
    x_tolerance: float,
    y_tolerance: float,
) -> str:
    words = extract_region_words(
        page,
        bbox,
        excluded_regions=excluded_regions,
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
    )
    if not words:
        return ""

    x0, top, x1, bottom = bbox
    width = x1 - x0
    if classify_two_column(words, width):
        mid = x0 + (width / 2.0)
        left_words = extract_region_words(page, (x0, top, mid, bottom), excluded_regions=excluded_regions, x_tolerance=x_tolerance, y_tolerance=y_tolerance)
        right_words = extract_region_words(page, (mid, top, x1, bottom), excluded_regions=excluded_regions, x_tolerance=x_tolerance, y_tolerance=y_tolerance)
        parts = [render_words_with_layout(w, x_tolerance=x_tolerance, y_tolerance=y_tolerance) for w in [left_words, right_words] if w]
        return "\n\n".join(parts)

    return render_words_with_layout(words, x_tolerance=x_tolerance, y_tolerance=y_tolerance)


def extract_page_layout(page: pdfplumber.page.Page, x_tolerance: float, y_tolerance: float) -> str:
    blocked_regions = image_regions(page)
    tables = table_regions(page, blocked_regions)
    exclusions = blocked_regions + [pad_bbox(table["bbox"]) for table in tables]

    regions: list[dict] = [{"kind": "image", "bbox": bbox} for bbox in blocked_regions]
    regions.extend({"kind": "table", "bbox": table["bbox"], "markdown": table["markdown"]} for table in tables)
    regions.sort(key=lambda item: item["bbox"][1])

    rendered: list[str] = []
    cursor = 0.0
    for region in regions:
        top = float(region["bbox"][1])
        if top > cursor:
            text = extract_text_region(
                page,
                (0.0, cursor, page.width, top),
                excluded_regions=exclusions,
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
            )
            if text:
                rendered.append(text)

        if region["kind"] == "table":
            rendered.append(region["markdown"])

        cursor = max(cursor, float(region["bbox"][3]))

    if cursor < page.height:
        text = extract_text_region(
            page,
            (0.0, cursor, page.width, page.height),
            excluded_regions=exclusions,
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
        )
        if text:
            rendered.append(text)

    if not rendered:
        return extract_text_region(
            page,
            (0.0, 0.0, page.width, page.height),
            excluded_regions=blocked_regions,
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
        )

    return "\n\n".join(part for part in rendered if part)


def classify_two_column(words: list[dict], page_width: float) -> bool:
    """
    Simple heuristic:
    - need enough words to avoid overfitting tiny pages
    - look at x-centres of words
    - if we have a healthy cluster on both halves and a sparse gutter, call it two-column
    """
    if len(words) < 40:
        return False

    centres = [((w["x0"] + w["x1"]) / 2.0) for w in words]
    left_cutoff = page_width * 0.45
    right_cutoff = page_width * 0.55

    left = [x for x in centres if x < left_cutoff]
    right = [x for x in centres if x > right_cutoff]
    gutter = [x for x in centres if left_cutoff <= x <= right_cutoff]

    left_ratio = len(left) / len(centres)
    right_ratio = len(right) / len(centres)
    gutter_ratio = len(gutter) / len(centres)

    if left_ratio < 0.25 or right_ratio < 0.25:
        return False

    if gutter_ratio > 0.20:
        return False

    # Extra sanity check: both sides should have some vertical spread.
    left_tops = [w["top"] for w in words if ((w["x0"] + w["x1"]) / 2.0) < left_cutoff]
    right_tops = [w["top"] for w in words if ((w["x0"] + w["x1"]) / 2.0) > right_cutoff]
    if len(left_tops) < 10 or len(right_tops) < 10:
        return False

    left_spread = max(left_tops) - min(left_tops)
    right_spread = max(right_tops) - min(right_tops)
    page_height_proxy = max([w["bottom"] for w in words], default=0.0)
    if page_height_proxy and (left_spread < page_height_proxy * 0.30 or right_spread < page_height_proxy * 0.30):
        return False

    return True


def extract_pdf(pdf_path: Path, x_tolerance: float, y_tolerance: float, page_marker: str) -> str:
    rendered_pages: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = extract_page_layout(page, x_tolerance=x_tolerance, y_tolerance=y_tolerance)

            if not page_text:
                page_text = "[No extractable text on this page]"

            if page_marker:
                rendered_pages.append(page_marker.format(page=page_number) + page_text)
            else:
                rendered_pages.append(page_text)

    return "\n".join(rendered_pages).strip() + "\n"


def main() -> None:
    args = parse_args()
    text = extract_pdf(
        pdf_path=args.pdf,
        x_tolerance=args.x_tolerance,
        y_tolerance=args.y_tolerance,
        page_marker=args.page_marker,
    )

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote extracted text to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
