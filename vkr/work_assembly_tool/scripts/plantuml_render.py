"""Render PlantUML source to PNG bytes via the plantuml Python package."""

from __future__ import annotations

import re
from io import BytesIO

try:
    from plantuml import PlantUML
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install plantuml: pip install plantuml") from exc


DEFAULT_SERVER = "http://www.plantuml.com/plantuml/img/"


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


def render_plantuml_png(content: str, server_url: str = DEFAULT_SERVER) -> bytes:
    source = normalize_plantuml_source(content)
    client = PlantUML(server_url)
    result = client.processes(source)
    if not isinstance(result, bytes):
        raise RuntimeError("PlantUML server did not return PNG bytes")
    if not result.startswith(b"\x89PNG"):
        raise RuntimeError("PlantUML server returned unexpected payload (expected PNG)")
    return result


def render_plantuml_png_stream(content: str, server_url: str = DEFAULT_SERVER) -> BytesIO:
    return BytesIO(render_plantuml_png(content, server_url))
