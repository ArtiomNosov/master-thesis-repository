#!/usr/bin/env python3
"""Render regex pattern as PDF via reportlab (minimal deps)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/latex/figures/regex-section-headers.pdf"

PATTERN = r"(?i)\b(опыт работы|образование|навыки|ключевые навыки)\b"


def main() -> None:
    from reportlab.lib.pagesizes import landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_path = Path("C:/Windows/Fonts/consola.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("Consolas", str(font_path)))
        font_name = "Consolas"
    else:
        font_name = "Courier"

    page_w, page_h = landscape((420, 72))
    c = canvas.Canvas(str(OUT), pagesize=(page_w, page_h))
    c.setFont(font_name, 11)
    text_w = c.stringWidth(PATTERN, font_name, 11)
    x = max(18, (page_w - text_w) / 2)
    c.setFillColorRGB(0.07, 0.07, 0.07)
    c.roundRect(x - 12, 18, text_w + 24, 36, 6, stroke=1, fill=0)
    c.drawString(x, 30, PATTERN)
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
