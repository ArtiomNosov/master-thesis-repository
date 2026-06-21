#!/usr/bin/env python3
"""Render the recruitment screening bottleneck diagram to SVG and high-DPI PNG."""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/obsidian/thesis/diagrams/recruitment_screening_bottleneck.json"
ASSETS = ROOT / "docs/obsidian/thesis/assets"


def _font_path(bold: bool) -> Path | None:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = _font_path(bold)
    if path:
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if _text_size(draw, candidate, font)[0] <= max_width:
                line = candidate
            else:
                wrapped.append(line)
                line = word
        wrapped.append(line)
    return wrapped


def _scale_rect(rect: dict[str, Any], scale: float) -> tuple[int, int, int, int]:
    return (
        round(rect["x"] * scale),
        round(rect["y"] * scale),
        round((rect["x"] + rect["width"]) * scale),
        round((rect["y"] + rect["height"]) * scale),
    )


def _stage_center_y(stage: dict[str, Any]) -> float:
    return stage["y"] + stage["height"] / 2


def _draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    center: tuple[float, float],
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int,
) -> None:
    heights = [_text_size(draw, line, font)[1] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = center[1] - total_h / 2
    for line, h in zip(lines, heights, strict=True):
        w, _ = _text_size(draw, line, font)
        draw.text((center[0] - w / 2, y), line, font=font, fill=fill)
        y += h + line_gap


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int,
    head: int,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=fill, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    left = (
        ex - head * math.cos(angle - math.pi / 6),
        ey - head * math.sin(angle - math.pi / 6),
    )
    right = (
        ex - head * math.cos(angle + math.pi / 6),
        ey - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, left, right], fill=fill)


def _draw_warning(draw: ImageDraw.ImageDraw, x: float, y: float, size: float, style: dict[str, str]) -> None:
    points = [
        (x + size / 2, y),
        (x + size, y + size * 0.88),
        (x, y + size * 0.88),
    ]
    draw.polygon(points, outline=style["text"], fill=style["box_fill"])
    font = _load_font(round(size * 0.64), bold=True)
    w, h = _text_size(draw, "!", font)
    draw.text((x + size / 2 - w / 2, y + size * 0.36 - h / 2), "!", font=font, fill=style["text"])


def render_png(config: dict[str, Any], out_path: Path, scale: float) -> None:
    canvas = config["canvas"]
    style = config["style"]
    width = round(canvas["width"] * scale)
    height = round(canvas["height"] * scale)
    image = Image.new("RGB", (width, height), style["background"])
    draw = ImageDraw.Draw(image)

    stage_font = _load_font(round(32 * scale))
    note_title_font = _load_font(round(28 * scale), bold=True)
    note_font = _load_font(round(26 * scale))
    stroke = round(2.2 * scale)
    radius = round(15 * scale)

    stages = {stage["id"]: stage for stage in config["stages"]}
    for arrow in config["arrows"]:
        src = stages[arrow["from"]]
        dst = stages[arrow["to"]]
        y = round(_stage_center_y(src) * scale)
        start = ((src["x"] + src["width"] + 2) * scale, y)
        end = ((dst["x"] - 12) * scale, y)
        _draw_arrow(draw, start, end, style["arrow"], round(3.0 * scale), round(18 * scale))

    for stage in config["stages"]:
        x0, y0, x1, y1 = _scale_rect(stage, scale)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=radius,
            fill=style["box_fill"],
            outline=style["box_stroke"],
            width=stroke,
        )
        label_x0 = x0
        if stage.get("warning"):
            icon_size = 32 * scale
            icon_x = x0 + 28 * scale
            icon_y = y0 + (y1 - y0 - icon_size * 0.88) / 2
            _draw_warning(draw, icon_x, icon_y, icon_size, style)
            label_x0 += round(76 * scale)
        max_label_width = x1 - label_x0 - round(24 * scale)
        lines = _wrap_lines(draw, stage["label"], stage_font, max_label_width)
        _draw_centered_lines(
            draw,
            lines,
            ((label_x0 + x1) / 2, (y0 + y1) / 2),
            stage_font,
            style["text"],
            round(10 * scale),
        )

    comment = config["comment"]
    target = stages[comment["target_stage"]]
    cx = comment["x"] + comment["width"] / 2
    pointer_target = (target["x"] + target["width"] / 2, target["y"] + target["height"] + 2)
    pointer = [
        ((cx - 16) * scale, comment["y"] * scale),
        ((cx + 16) * scale, comment["y"] * scale),
        (pointer_target[0] * scale, pointer_target[1] * scale),
    ]
    draw.polygon(pointer, fill=style["note_fill"], outline=style["note_stroke"])

    x0, y0, x1, y1 = _scale_rect(comment, scale)
    fold = round(28 * scale)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=round(14 * scale), fill=style["note_fill"], outline=style["note_stroke"], width=stroke)
    draw.polygon([(x1 - fold, y0), (x1, y0 + fold), (x1 - fold, y0 + fold)], fill="#f4f1a8", outline=style["note_stroke"])

    title_lines = _wrap_lines(draw, comment["title"], note_title_font, x1 - x0 - round(50 * scale))
    body_lines = _wrap_lines(draw, comment["text"], note_font, x1 - x0 - round(50 * scale))
    all_lines = title_lines + body_lines
    line_gap = round(9 * scale)
    line_heights = [
        _text_size(draw, line, note_title_font if idx < len(title_lines) else note_font)[1]
        for idx, line in enumerate(all_lines)
    ]
    total_h = sum(line_heights) + line_gap * (len(all_lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for idx, line in enumerate(all_lines):
        font = note_title_font if idx < len(title_lines) else note_font
        w, h = _text_size(draw, line, font)
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=font, fill=style["text"])
        y += h + line_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _svg_text_lines(
    lines: list[str],
    x: float,
    y: float,
    font_size: int,
    weight: str = "normal",
    anchor: str = "middle",
    fill: str = "#222222",
) -> str:
    line_height = font_size * 1.2
    start_y = y - (len(lines) - 1) * line_height / 2
    chunks = [
        f'<text x="{x:.1f}" y="{start_y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{font_size}" '
        f'font-weight="{weight}" fill="{fill}">'
    ]
    for idx, line in enumerate(lines):
        dy = "0" if idx == 0 else f"{line_height:.1f}"
        chunks.append(f'<tspan x="{x:.1f}" dy="{dy}">{escape(line)}</tspan>')
    chunks.append("</text>")
    return "\n".join(chunks)


def _rounded_rect(x: float, y: float, width: float, height: float, radius: float, fill: str, stroke: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" ry="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>'
    )


def _svg_arrow(start: tuple[float, float], end: tuple[float, float], color: str) -> str:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    head = 18
    left = (ex - head * math.cos(angle - math.pi / 6), ey - head * math.sin(angle - math.pi / 6))
    right = (ex - head * math.cos(angle + math.pi / 6), ey - head * math.sin(angle + math.pi / 6))
    points = f"{ex:.1f},{ey:.1f} {left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}"
    return (
        f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
        f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'\n<polygon points="{points}" fill="{color}"/>'
    )


def _svg_warning(x: float, y: float, size: float, style: dict[str, str]) -> str:
    points = f"{x + size / 2:.1f},{y:.1f} {x + size:.1f},{y + size * 0.88:.1f} {x:.1f},{y + size * 0.88:.1f}"
    return (
        f'<polygon points="{points}" fill="{style["box_fill"]}" stroke="{style["text"]}" stroke-width="2"/>'
        f'\n<text x="{x + size / 2:.1f}" y="{y + size * 0.66:.1f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{size * 0.62:.1f}" font-weight="700" fill="{style["text"]}">!</text>'
    )


def render_svg(config: dict[str, Any], out_path: Path) -> None:
    canvas = config["canvas"]
    style = config["style"]
    stages = {stage["id"]: stage for stage in config["stages"]}

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg width="{canvas["width"]}" height="{canvas["height"]}" '
            f'viewBox="0 0 {canvas["width"]} {canvas["height"]}" '
            'xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{escape(config["title"])}</title>',
        f'<desc id="desc">{escape(config["description"])}</desc>',
        f'<rect width="100%" height="100%" fill="{style["background"]}"/>',
    ]

    for arrow in config["arrows"]:
        src = stages[arrow["from"]]
        dst = stages[arrow["to"]]
        y = _stage_center_y(src)
        parts.append(
            _svg_arrow(
                (src["x"] + src["width"] + 2, y),
                (dst["x"] - 12, y),
                style["arrow"],
            )
        )

    for stage in config["stages"]:
        parts.append(
            _rounded_rect(
                stage["x"],
                stage["y"],
                stage["width"],
                stage["height"],
                15,
                style["box_fill"],
                style["box_stroke"],
            )
        )
        label_x = stage["x"] + stage["width"] / 2
        available_width = stage["width"] - 24
        if stage.get("warning"):
            icon_size = 32
            icon_x = stage["x"] + 28
            icon_y = stage["y"] + (stage["height"] - icon_size * 0.88) / 2
            parts.append(_svg_warning(icon_x, icon_y, icon_size, style))
            label_x = stage["x"] + 76 + (stage["width"] - 76) / 2
            available_width = stage["width"] - 100
        lines = [line.strip() for line in stage["label"].splitlines() if line.strip()]
        if stage["id"] == "offer" and available_width < 190:
            lines = ["Приглашение", "на работу"]
        parts.append(_svg_text_lines(lines, label_x, stage["y"] + stage["height"] / 2 + 9, 32, fill=style["text"]))

    comment = config["comment"]
    target = stages[comment["target_stage"]]
    cx = comment["x"] + comment["width"] / 2
    pointer_target = (target["x"] + target["width"] / 2, target["y"] + target["height"] + 2)
    pointer_points = (
        f'{cx - 16:.1f},{comment["y"]:.1f} '
        f'{cx + 16:.1f},{comment["y"]:.1f} '
        f"{pointer_target[0]:.1f},{pointer_target[1]:.1f}"
    )
    parts.append(f'<polygon points="{pointer_points}" fill="{style["note_fill"]}" stroke="{style["note_stroke"]}" stroke-width="2"/>')
    parts.append(
        _rounded_rect(
            comment["x"],
            comment["y"],
            comment["width"],
            comment["height"],
            14,
            style["note_fill"],
            style["note_stroke"],
        )
    )
    fold = 28
    x1 = comment["x"] + comment["width"]
    fold_points = (
        f'{x1 - fold:.1f},{comment["y"]:.1f} '
        f'{x1:.1f},{comment["y"] + fold:.1f} '
        f'{x1 - fold:.1f},{comment["y"] + fold:.1f}'
    )
    parts.append(f'<polygon points="{fold_points}" fill="#f4f1a8" stroke="{style["note_stroke"]}" stroke-width="1.5"/>')
    note_lines = [comment["title"], *comment["text"].splitlines()]
    note_center = (comment["x"] + comment["width"] / 2, comment["y"] + comment["height"] / 2 + 5)
    parts.append(_svg_text_lines(note_lines[:1], note_center[0], note_center[1] - 66, 28, weight="700", fill=style["text"]))
    parts.append(_svg_text_lines(note_lines[1:], note_center[0], note_center[1] + 14, 26, fill=style["text"]))

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    ET.fromstring(out_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--scale", type=float, help="PNG scale over logical SVG coordinates")
    parser.add_argument("--out-dir", type=Path, default=ASSETS)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    scale = args.scale or float(config["canvas"].get("scale", 2.4))
    stem = config["output_stem"]
    svg_path = args.out_dir / f"{stem}.svg"
    png_path = args.out_dir / f"{stem}.png"
    render_svg(config, svg_path)
    render_png(config, png_path, scale)
    print(f"OK: {svg_path.relative_to(ROOT)}")
    print(f"OK: {png_path.relative_to(ROOT)} ({round(config['canvas']['width'] * scale)}x{round(config['canvas']['height'] * scale)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
