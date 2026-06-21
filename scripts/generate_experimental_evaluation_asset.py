#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the presentation-ready experimental evaluation slide asset.

Pipeline:
    python scripts/generate_experimental_evaluation_asset.py

The image is generated from experiments/results/three_model_baseline_test.json
and saved to vkr/artifacts/experimental_evaluation_comparison.png by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "experiments" / "results" / "three_model_baseline_test.json"
DEFAULT_OUTPUT = ROOT / "vkr" / "artifacts" / "experimental_evaluation_comparison.png"

WIDTH = 1240
HEIGHT = 360

WHITE = "#FFFFFF"
INK = "#000000"
MUTED = "#000000"
SUBTLE = "#000000"
GRID = "#000000"
PANEL = "#FFFFFF"
PANEL_BORDER = "#000000"
ACCENT = "#000000"
ACCENT_DARK = "#000000"
GREEN = "#000000"
GRAY = "#000000"
SLATE = "#000000"
LIGHT_BLUE = "#FFFFFF"
LIGHT_GREEN = "#FFFFFF"
HIGHLIGHT = "#E6E6E6"


MODELS = [
    {
        "key": "bm25",
        "label": "BM25",
        "subtitle": "lexical baseline",
        "kind": "алгоритм без обучения",
        "color": GRAY,
    },
    {
        "key": "cross_encoder",
        "label": "Cross-encoder",
        "subtitle": "trained RuBERT-tiny2",
        "kind": "обученная нейросетевая модель",
        "color": SLATE,
    },
    {
        "key": "bi_encoder",
        "label": "Bi-encoder",
        "subtitle": "proposed trained model",
        "kind": "предлагаемая обученная модель",
        "color": ACCENT,
    },
]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


FONT_REGULAR = str(Path("C:/Windows/Fonts/times.ttf"))
FONT_BOLD = str(Path("C:/Windows/Fonts/timesbd.ttf"))
FONT_ITALIC = str(Path("C:/Windows/Fonts/timesi.ttf"))


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def text_width(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    return bbox[2] - bbox[0]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: str = INK,
    bold: bool = False,
    italic: bool = False,
    anchor: str | None = None,
) -> None:
    if bold:
        selected = FONT_BOLD
    elif italic:
        selected = FONT_ITALIC
    else:
        selected = FONT_REGULAR
    draw.text(xy, text, font=font(selected, size), fill=fill, anchor=anchor)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, text_font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, text_font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = PANEL) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=PANEL_BORDER, width=2)


def draw_pill(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, fill: str = LIGHT_BLUE) -> int:
    label_font = font(FONT_REGULAR, 22)
    value_font = font(FONT_BOLD, 24)
    content = f"{label}: {value}"
    width = text_width(draw, content, value_font) + 34
    draw.rounded_rectangle((x, y, x + width, y + 42), radius=21, fill=fill, outline="#C9D8EC", width=1)
    draw_text(draw, (x + 17, y + 9), content, 22, fill=ACCENT_DARK, bold=True)
    return width


def draw_bar_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], metrics: dict) -> None:
    x1, y1, x2, y2 = box
    rounded_panel(draw, box)
    draw_text(draw, (x1 + 32, y1 + 26), "Average Precision по парам (↑)", 31, bold=True)
    draw_text(draw, (x1 + 32, y1 + 66), "главная метрика различения релевантных и нерелевантных пар", 20, fill=SUBTLE)

    chart_x = x1 + 280
    chart_y = y1 + 135
    chart_w = x2 - chart_x - 54
    bar_h = 44
    row_gap = 62

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx = chart_x + int(chart_w * tick)
        draw.line((tx, chart_y - 28, tx, chart_y + row_gap * 2 + bar_h + 35), fill=GRID, width=1)
        draw_text(draw, (tx, chart_y + row_gap * 2 + bar_h + 44), f"{tick:.2f}", 17, fill=SUBTLE, anchor="mm")

    for idx, model in enumerate(MODELS):
        y = chart_y + idx * row_gap
        value = metrics[model["key"]]["pair_average_precision"]
        draw_text(draw, (x1 + 32, y + 2), model["label"], 23, bold=(model["key"] == "bi_encoder"))
        draw_text(draw, (x1 + 32, y + 30), model["subtitle"], 17, fill=SUBTLE)

        draw.rounded_rectangle((chart_x, y, chart_x + chart_w, y + bar_h), radius=13, fill="#EEF2F6")
        bar_w = int(chart_w * value)
        draw.rounded_rectangle((chart_x, y, chart_x + bar_w, y + bar_h), radius=13, fill=model["color"])
        draw_text(draw, (chart_x + bar_w - 13, y + bar_h / 2), fmt4(value), 22, fill=WHITE, bold=True, anchor="rm")

    bi_ap = metrics["bi_encoder"]["pair_average_precision"]
    bm25_ap = metrics["bm25"]["pair_average_precision"]
    cross_ap = metrics["cross_encoder"]["pair_average_precision"]
    ratio = bi_ap / bm25_ap
    relative_gain = (bi_ap / cross_ap - 1.0) * 100

    summary_y = y2 - 83
    draw.rounded_rectangle((x1 + 32, summary_y, x2 - 32, y2 - 30), radius=14, fill=LIGHT_GREEN, outline="#C9E9DA", width=1)
    draw_text(
        draw,
        (x1 + 54, summary_y + 16),
        f"Bi-encoder: {ratio:.1f}× к BM25; +{relative_gain:.1f}% к cross-encoder",
        23,
        fill="#0F5F43",
        bold=True,
    )


def draw_metrics_table(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], metrics: dict) -> None:
    x1, y1, x2, y2 = box
    rounded_panel(draw, box)
    draw_text(draw, (x1 + 32, y1 + 26), "Top-3 качество ранжирования (↑)", 31, bold=True)
    draw_text(draw, (x1 + 32, y1 + 66), "оценка кандидатов внутри вакансии на test.tsv", 20, fill=SUBTLE)

    table_x = x1 + 32
    table_y = y1 + 120
    table_w = x2 - x1 - 64
    header_h = 46
    row_h = 62
    col_w = [265, 130, 150, table_w - 265 - 130 - 150]
    headers = ["Подход", "AP", "NDCG@3", "MRR@3"]

    draw.rounded_rectangle((table_x, table_y, table_x + table_w, table_y + header_h), radius=12, fill="#E9EEF5")
    cx = table_x
    for header, width in zip(headers, col_w):
        draw_text(draw, (cx + 14, table_y + 13), header, 20, bold=True, fill=ACCENT_DARK)
        cx += width

    for row_idx, model in enumerate(MODELS):
        y = table_y + header_h + row_idx * row_h
        row_fill = "#FFFFFF" if row_idx % 2 == 0 else "#F6F8FB"
        if model["key"] == "bi_encoder":
            row_fill = "#EEF6FF"
        draw.rectangle((table_x, y, table_x + table_w, y + row_h), fill=row_fill)
        draw.line((table_x, y, table_x + table_w, y), fill=PANEL_BORDER, width=1)

        cx = table_x
        draw_text(draw, (cx + 14, y + 12), model["label"], 22, bold=(model["key"] == "bi_encoder"))
        draw_text(draw, (cx + 14, y + 38), model["subtitle"], 16, fill=SUBTLE)
        cx += col_w[0]

        draw_text(draw, (cx + 14, y + 20), fmt4(metrics[model["key"]]["pair_average_precision"]), 22, bold=True)
        cx += col_w[1]
        draw_text(draw, (cx + 14, y + 20), fmt4(metrics[model["key"]]["ndcg@3"]), 22, bold=True)
        cx += col_w[2]
        draw_text(draw, (cx + 14, y + 20), fmt4(metrics[model["key"]]["mrr@3"]), 22, bold=True)

    draw.line((table_x, table_y + header_h + 3 * row_h, table_x + table_w, table_y + header_h + 3 * row_h), fill=PANEL_BORDER, width=1)

    note = "Сравнение корректное: cross/bi обучены на train.tsv; BM25 — классический алгоритмический baseline."
    note_font = font(FONT_REGULAR, 19)
    note_lines = wrap_text(draw, note, note_font, x2 - x1 - 80)
    note_y = table_y + header_h + 3 * row_h + 19
    for idx, line in enumerate(note_lines):
        draw_text(draw, (x1 + 32, note_y + idx * 24), line, 19, fill=MUTED)


def draw_footer_callout(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], metrics: dict) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=ACCENT_DARK)
    bi_ap = metrics["bi_encoder"]["pair_average_precision"]
    bm25_ap = metrics["bm25"]["pair_average_precision"]
    cross_ap = metrics["cross_encoder"]["pair_average_precision"]
    ratio = bi_ap / bm25_ap
    relative_gain = (bi_ap / cross_ap - 1.0) * 100
    text = (
        f"Итог: предложенный bi-encoder даёт лучший AP = {fmt4(bi_ap)} "
        f"(+{relative_gain:.1f}% к обученному cross-encoder; {ratio:.1f}× к BM25) "
        "при NDCG@3 = MRR@3 = 1.0000."
    )
    callout_font = font(FONT_BOLD, 25)
    lines = wrap_text(draw, text, callout_font, x2 - x1 - 72)
    line_h = 32
    start_y = y1 + math.floor((y2 - y1 - line_h * len(lines)) / 2) - 1
    for idx, line in enumerate(lines):
        draw_text(draw, (x1 + 36, start_y + idx * line_h), line, 25, fill=WHITE, bold=True)


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload


def render(metrics_path: Path, output_path: Path) -> None:
    payload = load_metrics(metrics_path)
    metrics = payload["models"]

    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)

    table_x = 30
    table_y = 24
    table_w = WIDTH - 60
    header_h = 70
    row_h = 78
    col_w = [245, 385, 210, 170, 170]
    headers = ["Подход", "Что сравнивается", "Средняя точность (AP)", "NDCG@3", "MRR@3"]
    rows = [
        {
            "key": "bm25",
            "name": "BM25",
            "desc": "лексический алгоритм без обучения",
        },
        {
            "key": "cross_encoder",
            "name": "Cross-encoder",
            "desc": "обученная модель попарного оценивания",
        },
        {
            "key": "bi_encoder",
            "name": "Bi-encoder",
            "desc": "предлагаемая обученная модель",
        },
    ]

    table_h = header_h + row_h * len(rows)
    draw.rectangle((table_x, table_y, table_x + table_w, table_y + table_h), fill=WHITE, outline=INK, width=2)

    cx = table_x
    for width in col_w[:-1]:
        cx += width
        draw.line((cx, table_y, cx, table_y + table_h), fill=INK, width=1)

    draw.line((table_x, table_y + header_h, table_x + table_w, table_y + header_h), fill=INK, width=2)
    for row_idx in range(1, len(rows)):
        y = table_y + header_h + row_idx * row_h
        draw.line((table_x, y, table_x + table_w, y), fill=INK, width=1)

    cx = table_x
    for header, width in zip(headers, col_w):
        header_font = font(FONT_BOLD, 22 if header != "Средняя точность (AP)" else 20)
        lines = wrap_text(draw, header, header_font, width - 22)
        line_h = 24
        start_y = table_y + (header_h - line_h * len(lines)) // 2 - 1
        for idx, line in enumerate(lines):
            draw_text(draw, (cx + width / 2, start_y + idx * line_h), line, 22 if header != "Средняя точность (AP)" else 20, fill=INK, bold=True, anchor="ma")
        cx += width

    best_ap_key = "bi_encoder"
    for row_idx, row in enumerate(rows):
        y = table_y + header_h + row_idx * row_h
        key = row["key"]

        cx = table_x
        is_target = key == best_ap_key
        draw_text(draw, (cx + 14, y + 25), row["name"], 22, bold=is_target)
        cx += col_w[0]

        desc_font_size = 20 if key != "cross_encoder" else 18
        draw_text(draw, (cx + 14, y + 26), row["desc"], desc_font_size, fill=INK, bold=is_target)
        cx += col_w[1]

        if is_target:
            draw.rectangle((cx + 1, y + 1, cx + col_w[2] - 1, y + row_h - 1), fill=HIGHLIGHT)
            draw.rectangle((cx + 1, y + 1, cx + col_w[2] - 1, y + row_h - 1), outline=INK, width=2)
        draw_text(draw, (cx + 18, y + 29), fmt4(metrics[key]["pair_average_precision"]), 24, bold=True)
        cx += col_w[2]
        draw_text(draw, (cx + 18, y + 29), fmt4(metrics[key]["ndcg@3"]), 24, bold=True)
        cx += col_w[3]
        draw_text(draw, (cx + 18, y + 29), fmt4(metrics[key]["mrr@3"]), 24, bold=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", dpi=(192, 192))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="Path to three-model evaluation JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.metrics.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
