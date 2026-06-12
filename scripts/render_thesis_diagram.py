#!/usr/bin/env python3
"""Render thesis Graphviz .dot sources to SVG in docs/obsidian/thesis/assets/."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs/obsidian/thesis/diagrams"
ASSETS = ROOT / "docs/obsidian/thesis/assets"
SAT_STYLES = DIAGRAMS / "sat_styles.inc"

# stem -> (title, desc) for accessibility in Obsidian / validators
METADATA: dict[str, tuple[str, str]] = {
    "application_analysis_request_flow": (
        "Схема потока анализа заявки с пометками статуса компонентов",
        "Потоковая диаграмма Reqcore и bi-encoder сервиса. "
        "Статус элементов кодируется цветом рамки по легенде SAT (см. подпись к рисунку).",
    ),
    "c4_container_architecture": (
        "Диаграмма контейнеров C4 с разграничением статусов компонентов",
        "C4 Container диаграмма Reqcore и ML-сервиса. "
        "Статусы компонентов кодируются цветом рамки по легенде SAT.",
    ),
}

DIAGRAM_STEMS = list(METADATA.keys())


def find_dot() -> str:
    exe = shutil.which("dot")
    if exe:
        return exe
    for candidate in (
        Path(r"C:\Program Files\Graphviz\bin\dot.exe"),
        Path(r"C:\Program Files (x86)\Graphviz\bin\dot.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Graphviz 'dot' not found. Install: winget install Graphviz.Graphviz"
    )


def compose_dot(source: Path) -> str:
    return source.read_text(encoding="utf-8")


def inject_svg_metadata(svg: str, title: str, desc: str) -> str:
    if re.search(r"<title[^>]*>", svg, re.I):
        svg = re.sub(r"<title[^>]*>.*?</title>", "", svg, count=1, flags=re.I | re.S)
    if re.search(r"<desc[^>]*>", svg, re.I):
        svg = re.sub(r"<desc[^>]*>.*?</desc>", "", svg, count=1, flags=re.I | re.S)
    insert = (
        f'  <title id="title">{title}</title>\n'
        f'  <desc id="desc">{desc}</desc>\n'
    )
    return re.sub(r"(<svg[^>]*>)", r"\1\n" + insert, svg, count=1, flags=re.I)


def render_one(stem: str, dot_path: Path | None = None) -> Path:
    dot_path = dot_path or DIAGRAMS / f"{stem}.dot"
    if not dot_path.is_file():
        raise FileNotFoundError(dot_path)

    out_path = ASSETS / f"{stem}_marked.svg"
    ASSETS.mkdir(parents=True, exist_ok=True)

    composed = compose_dot(dot_path)
    dot_exe = find_dot()
    cmd = [
        dot_exe,
        "-Tsvg",
        "-Gdpi=96",
        "-o",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        input=composed,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if proc.returncode != 0:
        # ortho splines sometimes fail; retry without ortho
        if "ortho" in composed:
            composed = re.sub(r"splines=ortho", "splines=true", composed)
            proc = subprocess.run(
                cmd,
                input=composed,
                text=True,
                encoding="utf-8",
                capture_output=True,
            )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dot failed for {dot_path.name}:\n{proc.stderr or proc.stdout}"
        )

    raw = out_path.read_text(encoding="utf-8")
    meta = METADATA.get(stem)
    if meta:
        raw = inject_svg_metadata(raw, meta[0], meta[1])
    out_path.write_text(raw, encoding="utf-8")

    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stem",
        nargs="?",
        default="application_analysis_request_flow",
        help=f"Diagram stem (without .dot). Known: {', '.join(DIAGRAM_STEMS)}. Use 'all' for every thesis diagram.",
    )
    parser.add_argument(
        "--dot",
        type=Path,
        help="Explicit path to .dot file",
    )
    args = parser.parse_args(argv)

    stems = DIAGRAM_STEMS if args.stem == "all" else [args.stem]
    try:
        for stem in stems:
            out = render_one(stem, args.dot if len(stems) == 1 else None)
            print(f"OK: {out.relative_to(ROOT)}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
