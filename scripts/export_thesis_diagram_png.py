#!/usr/bin/env python3
"""Export thesis SVG diagrams to PNG for visual E2E audit."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAGRAMS = ROOT / "docs/obsidian/thesis/diagrams"
ASSETS = ROOT / "docs/obsidian/thesis/assets"
AUDIT_DIR = ASSETS / "_audit"
DEFAULT_STEMS = [
    "application_analysis_request_flow",
    "c4_container_architecture",
]
DPI = 120


def find_inkscape() -> str | None:
    for name in ("inkscape", "inkscape.exe"):
        p = shutil.which(name)
        if p:
            return p
    for candidate in (
        Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def export_inkscape(svg: Path, png: Path, dpi: int) -> bool:
    exe = find_inkscape()
    if not exe:
        return False
    proc = subprocess.run(
        [
            exe,
            str(svg),
            f"--export-filename={png}",
            f"--export-dpi={dpi}",
            "--export-type=png",
        ],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and png.is_file()


def export_cairosvg(svg: Path, png: Path, dpi: int) -> bool:
    try:
        import cairosvg  # type: ignore
    except ImportError:
        return False
    scale = dpi / 96.0
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=scale)
    return png.is_file()


def find_dot() -> str | None:
    for name in ("dot", "dot.exe"):
        p = shutil.which(name)
        if p:
            return p
    for candidate in (
        Path(r"C:\Program Files\Graphviz\bin\dot.exe"),
        Path(r"C:\Program Files (x86)\Graphviz\bin\dot.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def export_graphviz_dot(stem: str, png: Path, dpi: int) -> bool:
    dot_exe = find_dot()
    dot_src = DIAGRAMS / f"{stem}.dot"
    if not dot_exe or not dot_src.is_file():
        return False
    proc = subprocess.run(
        [dot_exe, "-Tpng", f"-Gdpi={dpi}", "-o", str(png), str(dot_src)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and png.is_file()


def export_magick(svg: Path, png: Path, dpi: int) -> bool:
    exe = shutil.which("magick") or shutil.which("convert")
    if not exe:
        return False
    density = f"{dpi}"
    proc = subprocess.run(
        [exe, "-density", density, str(svg), str(png)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and png.is_file()


def export_one(stem: str, svg: Path, png: Path, dpi: int) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    if export_graphviz_dot(stem, png, dpi):
        return
    for fn in (export_inkscape, export_cairosvg, export_magick):
        if fn(svg, png, dpi):
            return
    raise RuntimeError(
        f"Could not export {svg.name} to PNG. Install Graphviz (dot) or Inkscape / cairosvg."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stem",
        nargs="?",
        default="all",
        help=f"Diagram stem or 'all'. Known: {', '.join(DEFAULT_STEMS)}",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=AUDIT_DIR,
        help="Output directory for PNG files",
    )
    args = parser.parse_args(argv)

    stems = DEFAULT_STEMS if args.stem == "all" else [args.stem]
    try:
        for stem in stems:
            svg = ASSETS / f"{stem}_marked.svg"
            if not svg.is_file():
                raise FileNotFoundError(svg)
            png = args.out_dir / f"{stem}_marked.png"
            export_one(stem, svg, png, args.dpi)
            print(f"OK: {png.relative_to(ROOT)}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
