#!/usr/bin/env python3
"""Copy self-contained PZ LaTeX build kit from thesis/latex."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "thesis/latex"
KIT = ROOT / "thesis/pz-latex-kit"

CHAPTER_FILES = [
    "master-thesis-preamble.tex",
    "master-thesis-bibl.tex",
    "_content-pz-body.tex",
    "master-thesis-abstract.tex",
    "master-thesis-intro.tex",
    "master-thesis-chapter1.tex",
    "master-thesis-chapter2.tex",
    "master-thesis-chapter3.tex",
    "master-thesis-chapter4.tex",
    "master-thesis-conclusion.tex",
    "biblio.bib",
]


FIGURE_FILES = [
    "application_analysis_request_flow.pdf",
    "regex-section-headers.pdf",
]


def sync_chapters() -> None:
    chapters = KIT / "chapters"
    chapters.mkdir(parents=True, exist_ok=True)
    for path in chapters.iterdir():
        if path.is_file() and path.name not in CHAPTER_FILES:
            path.unlink()
    for name in CHAPTER_FILES:
        shutil.copy2(SRC / "chapters" / name, chapters / name)


def sync_figures() -> None:
    figures = KIT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for path in figures.iterdir():
        if path.is_file() and path.name not in FIGURE_FILES:
            path.unlink()
    for name in FIGURE_FILES:
        shutil.copy2(SRC / "figures" / name, figures / name)


def main() -> None:
    KIT.mkdir(parents=True, exist_ok=True)
    (KIT / "build").mkdir(exist_ok=True)

    shutil.copy2(SRC / "master-thesis-pz-body.tex", KIT / "master-thesis-pz-body.tex")
    sync_chapters()
    sync_figures()
    print(f"Synced kit -> {KIT}")


if __name__ == "__main__":
    main()
