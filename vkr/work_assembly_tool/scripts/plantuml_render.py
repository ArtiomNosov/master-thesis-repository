"""Render PlantUML diagrams locally (jar) or via an HTTP PlantUML server."""

from __future__ import annotations

import re
import subprocess
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

PLANTUML_VERSION = "1.2026.5"
PLANTUML_DOWNLOAD_URL = (
    f"https://github.com/plantuml/plantuml/releases/download/v{PLANTUML_VERSION}/"
    f"plantuml-{PLANTUML_VERSION}.jar"
)
DEFAULT_JAR_RELATIVE = "tools/plantuml.jar"
DEFAULT_SERVER = "http://127.0.0.1:8080/png/"
PUBLIC_SERVER = "http://www.plantuml.com/plantuml/img/"


def normalize_plantuml_source(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("@startuml"):
        stripped = "@startuml\n" + stripped
    if not stripped.endswith("@enduml"):
        stripped = stripped + "\n@enduml"
    return stripped + "\n"


def extract_plantuml_title(content: str) -> str | None:
    match = re.search(r"^\s*title\s+(.+)$", content, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else None


def resolve_jar_path(tool_root: Path, assembly: dict[str, Any]) -> Path:
    jar_relative = assembly.get("plantuml_jar", DEFAULT_JAR_RELATIVE)
    return (tool_root / jar_relative).resolve()


def download_plantuml_jar(jar_path: Path) -> None:
    jar_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PlantUML {PLANTUML_VERSION} to {jar_path} ...")
    with urllib.request.urlopen(PLANTUML_DOWNLOAD_URL, timeout=120) as response:
        jar_path.write_bytes(response.read())
    print("PlantUML jar downloaded.")


def ensure_plantuml_jar(tool_root: Path, assembly: dict[str, Any]) -> Path:
    jar_path = resolve_jar_path(tool_root, assembly)
    if jar_path.exists():
        return jar_path
    if assembly.get("plantuml_auto_download", True):
        download_plantuml_jar(jar_path)
        return jar_path
    raise FileNotFoundError(
        f"PlantUML jar not found: {jar_path}. "
        f"Run: py vkr/work_assembly_tool/scripts/setup_plantuml.py"
    )


def render_plantuml_png_local(content: str, jar_path: Path) -> bytes:
    source = normalize_plantuml_source(content)
    completed = subprocess.run(
        ["java", "-jar", str(jar_path), "-tpng", "-pipe"],
        input=source.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"PlantUML local render failed (exit {completed.returncode}): {stderr}")
    if not completed.stdout.startswith(b"\x89PNG"):
        snippet = completed.stdout[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"PlantUML local render did not return PNG: {snippet!r}")
    return completed.stdout


def render_plantuml_png_server(content: str, server_url: str) -> bytes:
    try:
        from plantuml import PlantUML
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install plantuml: pip install plantuml") from exc

    source = normalize_plantuml_source(content)
    result = PlantUML(server_url).processes(source)
    if not isinstance(result, bytes):
        raise RuntimeError("PlantUML server did not return PNG bytes")
    if not result.startswith(b"\x89PNG"):
        raise RuntimeError("PlantUML server returned unexpected payload (expected PNG)")
    return result


def render_plantuml_png(content: str, assembly: dict[str, Any], tool_root: Path) -> bytes:
    mode = assembly.get("plantuml_mode", "local_jar")
    if mode == "local_jar":
        jar_path = ensure_plantuml_jar(tool_root, assembly)
        return render_plantuml_png_local(content, jar_path)
    if mode == "server":
        server_url = assembly.get("plantuml_server_url", DEFAULT_SERVER)
        return render_plantuml_png_server(content, server_url)
    if mode == "public_server":
        server_url = assembly.get("plantuml_server_url", PUBLIC_SERVER)
        return render_plantuml_png_server(content, server_url)
    raise ValueError(f"Unknown plantuml_mode: {mode!r}")


def render_plantuml_png_stream(content: str, assembly: dict[str, Any], tool_root: Path) -> BytesIO:
    return BytesIO(render_plantuml_png(content, assembly, tool_root))
