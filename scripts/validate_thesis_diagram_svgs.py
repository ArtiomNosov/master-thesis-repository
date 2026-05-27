#!/usr/bin/env python3
"""Basic layout checks for thesis SVG diagrams (XML well-formed + text in canvas)."""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVGS = [
    ROOT / "docs/obsidian/thesis/assets/application_analysis_request_flow_marked.svg",
    ROOT / "docs/obsidian/thesis/assets/c4_container_architecture_marked.svg",
    ROOT
    / "archive/semester-3-and-previous/master-thesis-obsidian/thesis-draft/2/3/as_is.svg",
    ROOT
    / "archive/semester-3-and-previous/master-thesis-obsidian/thesis-draft/2/3/to_be.svg",
]

NS = {"svg": "http://www.w3.org/2000/svg"}


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    vb = root.get("viewBox")
    if not vb:
        return None
    parts = [float(p) for p in vb.replace(",", " ").split()]
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def canvas_size(root: ET.Element, vb: tuple[float, float, float, float] | None) -> tuple[float, float]:
    if vb:
        return vb[2], vb[3]
    w = float(root.get("width", "0").replace("px", ""))
    h = float(root.get("height", "0").replace("px", ""))
    return w, h


def iter_text_positions(root: ET.Element):
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag != "text":
            continue
        x = float(el.get("x", "0"))
        y = float(el.get("y", "0"))
        text = "".join(el.itertext()).strip()
        if text:
            yield x, y, text


def main() -> int:
    errors: list[str] = []
    for path in SVGS:
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        raw = path.read_text(encoding="utf-8")
        if "+</svg>" in raw or raw.rstrip().endswith("+"):
            errors.append(f"{path.name}: stray '+' before closing tag")
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            errors.append(f"{path.name}: XML parse error: {exc}")
            continue

        vb = parse_viewbox(root)
        cw, ch = canvas_size(root, vb)
        ox, oy = (vb[0], vb[1]) if vb else (0.0, 0.0)

        # Layout bounds: only for thesis SVGs with absolute coordinates (not BPMN transforms).
        if "thesis/assets" in path.as_posix():
            for x, y, text in iter_text_positions(root):
                if x < ox - 5 or y < oy - 5 or x > ox + cw + 5 or y > oy + ch + 8:
                    preview = text[:40] + ("…" if len(text) > 40 else "")
                    errors.append(
                        f"{path.name}: text outside canvas ({x:.0f},{y:.0f}): {preview!r}"
                    )
        else:
            # BPMN exports: check SAT overlay group only.
            overlay = root.find(".//*[@id='status-markup-as-is']")
            if overlay is None:
                overlay = root.find(".//*[@id='status-markup-to-be']")
            if overlay is None:
                errors.append(f"{path.name}: missing SAT overlay group")
            else:
                for el in overlay.iter():
                    if el.tag.split("}")[-1] != "text":
                        continue
                    x = float(el.get("x", "0"))
                    y = float(el.get("y", "0"))
                    if x < ox - 5 or y < oy - 5 or x > ox + cw + 5 or y > oy + ch + 8:
                        errors.append(f"{path.name}: SAT overlay text outside viewBox")

        if "thesis/assets" in path.as_posix() and not re.search(
            r"<title[^>]*>.*</title>", raw, re.I | re.S
        ):
            errors.append(f"{path.name}: missing <title>")

    if errors:
        print("SVG validation FAILED:")
        for e in errors:
            print(" -", e)
        return 1

    print(f"OK: {len(SVGS)} SVG files passed layout/XML checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
