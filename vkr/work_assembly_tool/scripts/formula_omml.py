"""Convert thesis formula text to Word OMML."""

from __future__ import annotations

import re
from typing import Any

from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from latex2mathml import converter as latex_converter
import mathml2omml


MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _ident_to_latex(name: str) -> str:
    if "_" not in name:
        return name
    base, sub = name.split("_", 1)
    sub = sub.replace("_", r"\_")
    return f"{base}_{{{sub}}}"


def formula_text_to_latex(text: str) -> str:
    normalized = text.strip().replace("~=", " \\approx ")
    normalized = re.sub(r"\s*\*\s*", r" \\cdot ", normalized)
    normalized = IDENT_RE.sub(lambda m: _ident_to_latex(m.group(0)), normalized)
    normalized = re.sub(
        r"ceil\s*\(([^()]+)\)",
        lambda match: f"\\operatorname{{ceil}}\\left({match.group(1)}\\right)",
        normalized,
    )
    return normalized


def formula_text_to_omml(text: str) -> Any:
    latex = formula_text_to_latex(text)
    mathml = latex_converter.convert(latex)
    omml = mathml2omml.convert(mathml)
    if not omml.startswith("<m:oMath"):
        omml = f"<m:oMath>{omml}</m:oMath>"
    return parse_xml(f'<root xmlns:m="{MATH_NS}" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">{omml}</root>')[0]


def looks_like_formula(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("@startuml") or "\n" in stripped:
        return False
    return bool("=" in stripped and re.search(r"[A-Za-z_]", stripped))


def set_table_no_borders(table: Any) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        tbl.insert(0, tbl_pr)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "nil")
        borders.append(element)
    tbl_pr.append(borders)
