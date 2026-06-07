#!/usr/bin/env python3
"""Geometry/layout checks for Graphviz thesis SVG diagrams."""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/obsidian/thesis/assets"
DEFAULT_SVGS = [
    ASSETS / "application_analysis_request_flow_marked.svg",
    ASSETS / "c4_container_architecture_marked.svg",
]

OVERLAP_AREA_THRESHOLD = 120.0
CLUSTER_MARGIN = 4.0
MIN_LEGEND_COLORED_POLYGONS = 3
MAX_ASPECT_RATIO = 4.5
GATE_NODE_IDS = frozenset({"bi_gate", "llm_gate", "flow_anchor"})
LEGEND_NODE_IDS = frozenset(
    {
        "sat_legend",
        "leg_assumption",
        "leg_approach",
        "leg_concept",
        "leg_studied",
        "leg_implemented",
        "leg_experimental",
        "leg_note",
    }
)
NEUTRAL_FILLS = frozenset(
    {
        "#ffffff",
        "#fafafa",
        "#fafcfd",
        "lightgrey",
        "none",
        "#fafcfd",
    }
)
CLUSTERS_TO_CHECK = frozenset(
    {"cluster_biencoder", "cluster_llm", "cluster_system", "cluster_future"}
)


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2

    def intersects(self, other: BBox) -> bool:
        return not (
            self.xmax <= other.xmin
            or other.xmax <= self.xmin
            or self.ymax <= other.ymin
            or other.ymax <= self.ymin
        )

    def intersection_area(self, other: BBox) -> float:
        if not self.intersects(other):
            return 0.0
        w = min(self.xmax, other.xmax) - max(self.xmin, other.xmin)
        h = min(self.ymax, other.ymax) - max(self.ymin, other.ymin)
        return max(0.0, w) * max(0.0, h)

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )

    def expand(self, margin: float) -> BBox:
        return BBox(
            self.xmin - margin,
            self.ymin - margin,
            self.xmax + margin,
            self.ymax + margin,
        )


@dataclass
class LayoutIssue:
    severity: str  # error | warn
    code: str
    message: str
    node: str | None = None
    other: str | None = None


@dataclass
class DiagramReport:
    file: str
    passed: bool
    viewbox: list[float] = field(default_factory=list)
    aspect_ratio: float = 0.0
    issues: list[LayoutIssue] = field(default_factory=list)


def _numbers_from_path(d: str) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]


def bbox_from_path_d(d: str | None) -> BBox | None:
    if not d:
        return None
    nums = _numbers_from_path(d)
    if len(nums) < 2:
        return None
    xs = nums[0::2]
    ys = nums[1::2]
    if not xs or not ys:
        return None
    return BBox(min(xs), min(ys), max(xs), max(ys))


def parse_graph_transform(graph_g: ET.Element | None) -> tuple[float, float, float]:
    """Return scale, translate_x, translate_y from graph0 transform."""
    if graph_g is None:
        return 1.0, 0.0, 0.0
    tr = graph_g.get("transform", "")
    scale = 1.0
    tx, ty = 0.0, 0.0
    m = re.search(r"scale\(([\d.]+)(?:\s+([\d.]+))?\)", tr)
    if m:
        scale = float(m.group(1))
        if m.group(2):
            scale = (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"translate\(([-\d.]+)\s+([-\d.]+)\)", tr)
    if m:
        tx, ty = float(m.group(1)), float(m.group(2))
    return scale, tx, ty


def to_canvas_coords(
    bbox: BBox, scale: float, tx: float, ty: float, view_h: float
) -> BBox:
    """Map Graphviz SVG coords (y up) to canvas/viewBox coords (y down)."""

    def map_y(y: float) -> float:
        return view_h - (y * scale + ty)

    y0 = map_y(bbox.ymin)
    y1 = map_y(bbox.ymax)
    return BBox(
        bbox.xmin * scale + tx,
        min(y0, y1),
        bbox.xmax * scale + tx,
        max(y0, y1),
    )


def parse_viewbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    vb = root.get("viewBox")
    if not vb:
        return None
    parts = [float(p) for p in vb.replace(",", " ").split()]
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def element_title(el: ET.Element) -> str | None:
    for child in el:
        if child.tag.split("}")[-1] == "title":
            return (child.text or "").strip()
    return None


def collect_elements(root: ET.Element) -> tuple[ET.Element | None, dict, dict, dict]:
    graph_g = None
    nodes: dict[str, tuple[ET.Element, BBox]] = {}
    clusters: dict[str, BBox] = {}
    texts: list[tuple[float, float, str]] = []

    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "g" and el.get("id") == "graph0":
            graph_g = el
        if tag == "g" and "node" in (el.get("class") or ""):
            title = element_title(el)
            if not title:
                continue
            bbox = None
            for sub in el.iter():
                st = sub.tag.split("}")[-1]
                if st == "path" and sub.get("d"):
                    pb = bbox_from_path_d(sub.get("d"))
                    if pb:
                        bbox = pb if bbox is None else BBox(
                            min(bbox.xmin, pb.xmin),
                            min(bbox.ymin, pb.ymin),
                            max(bbox.xmax, pb.xmax),
                            max(bbox.ymax, pb.ymax),
                        )
                if st == "ellipse":
                    cx = float(sub.get("cx", "0"))
                    cy = float(sub.get("cy", "0"))
                    rx = float(sub.get("rx", "4"))
                    ry = float(sub.get("ry", "4"))
                    pb = BBox(cx - rx, cy - ry, cx + rx, cy + ry)
                    bbox = pb if bbox is None else BBox(
                        min(bbox.xmin, pb.xmin),
                        min(bbox.ymin, pb.ymin),
                        max(bbox.xmax, pb.xmax),
                        max(bbox.ymax, pb.ymax),
                    )
            if bbox:
                nodes[title] = (el, bbox)
        if tag == "g" and "cluster" in (el.get("class") or ""):
            title = element_title(el)
            if not title:
                continue
            for sub in el:
                if sub.tag.split("}")[-1] == "path" and sub.get("d"):
                    cb = bbox_from_path_d(sub.get("d"))
                    if cb:
                        clusters[title] = cb
                    break
        if tag == "text":
            x = float(el.get("x", "0"))
            y = float(el.get("y", "0"))
            text = "".join(el.itertext()).strip()
            if text and not text.isspace():
                texts.append((x, y, text))

    return graph_g, nodes, clusters, {"texts": texts}


def count_colored_polygons_in_node(root: ET.Element, node_id: str) -> int:
    count = 0
    for el in root.iter():
        if el.tag.split("}")[-1] != "g":
            continue
        if element_title(el) != node_id:
            continue
        for sub in el.iter():
            if sub.tag.split("}")[-1] != "polygon":
                continue
            fill = (sub.get("fill") or "").lower()
            if fill and fill not in NEUTRAL_FILLS:
                count += 1
    return count


def check_diagram(path: Path) -> DiagramReport:
    raw = path.read_text(encoding="utf-8")
    root = ET.fromstring(raw)
    report = DiagramReport(file=path.name, passed=True)

    vb = parse_viewbox(root)
    if not vb:
        report.issues.append(
            LayoutIssue("error", "missing_viewbox", "SVG has no viewBox")
        )
        report.passed = False
        return report

    ox, oy, vw, vh = vb
    report.viewbox = [ox, oy, vw, vh]
    report.aspect_ratio = vw / vh if vh else 0.0
    if report.aspect_ratio > MAX_ASPECT_RATIO or (
        report.aspect_ratio > 0 and 1 / report.aspect_ratio > MAX_ASPECT_RATIO
    ):
        report.issues.append(
            LayoutIssue(
                "warn",
                "extreme_aspect_ratio",
                f"Unusual aspect ratio {report.aspect_ratio:.2f} (>{MAX_ASPECT_RATIO})",
            )
        )

    graph_g, nodes, clusters, extra = collect_elements(root)
    scale, tx, ty = parse_graph_transform(graph_g)

    canvas_nodes: dict[str, BBox] = {}
    for nid, (_, gb) in nodes.items():
        canvas_nodes[nid] = to_canvas_coords(gb, scale, tx, ty, vh)

    canvas_clusters: dict[str, BBox] = {}
    for cid, cb in clusters.items():
        canvas_clusters[cid] = to_canvas_coords(cb, scale, tx, ty, vh)

    is_graphviz = "Generated by graphviz" in raw

    # Graphviz text coords are in local Y-up space; skip naive viewBox text check.
    if not is_graphviz:
        for x, y, text in extra["texts"]:
            if x < ox - 8 or y < oy - 8 or x > ox + vw + 8 or y > oy + vh + 8:
                preview = text[:50] + ("…" if len(text) > 50 else "")
                report.issues.append(
                    LayoutIssue(
                        "error",
                        "text_outside_viewbox",
                        f"Text outside viewBox at ({x:.0f},{y:.0f}): {preview!r}",
                    )
                )
                report.passed = False

    # Node overlap
    ids = sorted(canvas_nodes.keys())
    for i, a in enumerate(ids):
        if a in GATE_NODE_IDS or a in LEGEND_NODE_IDS:
            continue
        for b in ids[i + 1 :]:
            if b in GATE_NODE_IDS or b in LEGEND_NODE_IDS:
                continue
            area = canvas_nodes[a].intersection_area(canvas_nodes[b])
            if area > OVERLAP_AREA_THRESHOLD:
                report.issues.append(
                    LayoutIssue(
                        "error",
                        "node_overlap",
                        f"Overlap area {area:.0f}px² between '{a}' and '{b}'",
                        node=a,
                        other=b,
                    )
                )
                report.passed = False

    # Cluster containment
    for cluster_id in CLUSTERS_TO_CHECK:
        if cluster_id not in canvas_clusters:
            continue
        cb = canvas_clusters[cluster_id]
        members = []
        if cluster_id == "cluster_biencoder":
            members = [
                n
                for n in canvas_nodes
                if n
                in {
                    "bi_gate",
                    "bicall",
                    "fastapi",
                    "ranker",
                    "appscore",
                    "onlyscore",
                }
            ]
        elif cluster_id == "cluster_llm":
            members = [
                n
                for n in canvas_nodes
                if n in {"llm_gate", "llmprov", "scoring", "llmwrite"}
            ]
        elif cluster_id == "cluster_system":
            members = [
                n
                for n in canvas_nodes
                if n
                not in LEGEND_NODE_IDS
                and n not in GATE_NODE_IDS
                and n != "sat_legend"
                and n != "flow_anchor"
                and not n.startswith("cluster_")
                and n
                not in {
                    "vectordb",
                    "rerank",
                }
            ]

        for nid in members:
            nb = canvas_nodes.get(nid)
            if not nb:
                continue
            for px, py in (
                (nb.xmin, nb.ymin),
                (nb.xmax, nb.ymin),
                (nb.xmin, nb.ymax),
                (nb.xmax, nb.ymax),
                (nb.cx, nb.cy),
            ):
                if not cb.contains_point(px, py, CLUSTER_MARGIN):
                    report.issues.append(
                        LayoutIssue(
                            "error",
                            "node_outside_cluster",
                            f"Node '{nid}' extends outside '{cluster_id}'",
                            node=nid,
                            other=cluster_id,
                        )
                    )
                    report.passed = False
                    break

    # Legend ink
    legend_colored = 0
    if "sat_legend" in nodes:
        legend_colored = count_colored_polygons_in_node(root, "sat_legend")
    elif "cluster_legend" in clusters:
        for nid in LEGEND_NODE_IDS:
            if nid in nodes:
                legend_colored += 1
        if legend_colored < MIN_LEGEND_COLORED_POLYGONS:
            for el in root.iter():
                if el.tag.split("}")[-1] != "g":
                    continue
                if element_title(el) not in LEGEND_NODE_IDS:
                    continue
                for sub in el.iter():
                    if sub.tag.split("}")[-1] == "path":
                        fill = (sub.get("fill") or "").lower()
                        if fill and fill not in NEUTRAL_FILLS:
                            legend_colored += 1

    if legend_colored < MIN_LEGEND_COLORED_POLYGONS:
        report.issues.append(
            LayoutIssue(
                "error",
                "empty_legend",
                f"Legend has only {legend_colored} colored items (need >= {MIN_LEGEND_COLORED_POLYGONS})",
            )
        )
        report.passed = False

    # Flow: horizontal branch heuristic (bicall vs fastapi should differ in x)
    if "bicall" in canvas_nodes and "fastapi" in canvas_nodes:
        dx = abs(canvas_nodes["bicall"].cx - canvas_nodes["fastapi"].cx)
        dy = abs(canvas_nodes["bicall"].cy - canvas_nodes["fastapi"].cy)
        if dy > dx * 1.2:
            report.issues.append(
                LayoutIssue(
                    "warn",
                    "vertical_branch_chain",
                    "bi-encoder chain looks vertical (bicall/fastapi stacked); expected horizontal row",
                    node="bicall",
                    other="fastapi",
                )
            )

    if "ui" in canvas_nodes and "analyze" in canvas_nodes:
        dx = abs(canvas_nodes["ui"].cx - canvas_nodes["analyze"].cx)
        dy = abs(canvas_nodes["ui"].cy - canvas_nodes["analyze"].cy)
        if dy > dx:
            report.issues.append(
                LayoutIssue(
                    "warn",
                    "vertical_spine",
                    "Spine ui→analyze looks vertical; expected horizontal row",
                    node="ui",
                    other="analyze",
                )
            )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "svg",
        nargs="*",
        type=Path,
        help="SVG paths (default: thesis Graphviz assets)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write combined JSON report to this path",
    )
    args = parser.parse_args(argv)

    paths = args.svg or DEFAULT_SVGS
    reports: list[DiagramReport] = []
    failed = False

    for path in paths:
        if not path.is_file():
            print(f"ERROR: missing {path}", file=sys.stderr)
            failed = True
            continue
        rep = check_diagram(path)
        reports.append(rep)
        status = "PASS" if rep.passed else "FAIL"
        print(f"{status}: {path.name}")
        for issue in rep.issues:
            prefix = "  !!" if issue.severity == "error" else "  ??"
            print(f"{prefix} [{issue.code}] {issue.message}")
        if not rep.passed:
            failed = True

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "passed": not failed,
            "diagrams": [
                {
                    **{k: v for k, v in asdict(r).items() if k != "issues"},
                    "issues": [asdict(i) for i in r.issues],
                }
                for r in reports
            ],
        }
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            rel = args.json.relative_to(ROOT)
        except ValueError:
            rel = args.json
        print(f"Report: {rel}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
