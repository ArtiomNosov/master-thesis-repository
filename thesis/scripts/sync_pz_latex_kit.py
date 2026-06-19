#!/usr/bin/env python3
"""Copy self-contained PZ LaTeX build kit from thesis/latex."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "thesis/latex"
KIT = ROOT / "thesis/pz-latex-kit"

CHAPTER_FILES = [
    "thesis-template-macro.tex",
    "thesis-template-bibl.tex",
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


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    KIT.mkdir(parents=True, exist_ok=True)
    (KIT / "chapters").mkdir(exist_ok=True)
    (KIT / "figures").mkdir(exist_ok=True)
    (KIT / "build").mkdir(exist_ok=True)

    shutil.copy2(SRC / "master-thesis-pz-body.tex", KIT / "master-thesis-pz-body.tex")
    for name in CHAPTER_FILES:
        shutil.copy2(SRC / "chapters" / name, KIT / "chapters" / name)

    copy_tree(SRC / "figures", KIT / "figures")
    print(f"Synced kit -> {KIT}")


if __name__ == "__main__":
    main()
