#!/usr/bin/env python3
"""Render application_analysis_request_flow.dot to thesis/latex/figures/*.pdf."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOT = ROOT / "docs/obsidian/thesis/diagrams/application_analysis_request_flow.dot"
OUT_PDF = ROOT / "thesis/latex/figures/application_analysis_request_flow.pdf"
DPI = 300


def find_dot() -> str | None:
    exe = shutil.which("dot")
    if exe:
        return exe
    for candidate in (
        Path(r"C:\Program Files\Graphviz\bin\dot.exe"),
        Path(r"C:\Program Files (x86)\Graphviz\bin\dot.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def render_with_dot(dot_exe: str) -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [dot_exe, "-Tpdf", f"-Gdpi={DPI}", "-o", str(OUT_PDF), str(DOT)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def render_with_docker() -> None:
    diagrams = DOT.parent.resolve()
    figures = OUT_PDF.parent.resolve()
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{diagrams}:/in",
        "-v",
        f"{figures}:/out",
        "nshine/dot:latest",
        "dot",
        "-Tpdf",
        f"-Gdpi={DPI}",
        "-o",
        f"/out/{OUT_PDF.name}",
        f"/in/{DOT.name}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def main() -> None:
    if not DOT.is_file():
        raise FileNotFoundError(DOT)
    dot_exe = find_dot()
    if dot_exe:
        render_with_dot(dot_exe)
    else:
        render_with_docker()
    print(OUT_PDF)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
