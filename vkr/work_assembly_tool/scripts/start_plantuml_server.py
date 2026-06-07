#!/usr/bin/env python3
"""Start a local PlantUML picoweb server (optional HTTP mode)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from plantuml_render import DEFAULT_SERVER, ensure_plantuml_jar, resolve_jar_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to vkr/work_assembly_tool",
    )
    args = parser.parse_args()

    tool_root = args.config_dir.resolve()
    jar_path = ensure_plantuml_jar(tool_root, {})
    bind = f"{args.port}:{args.host}"
    server_url = f"http://{args.host}:{args.port}/png/"

    print(f"PlantUML jar: {jar_path}")
    print(f"Starting picoweb on {args.host}:{args.port}")
    print(f"Set plantuml_mode=server and plantuml_server_url={server_url!r} in vkr_format_config.json")
    print(f"Default local URL: {DEFAULT_SERVER}")
    print("Press Ctrl+C to stop.")

    try:
        subprocess.run(["java", "-jar", str(jar_path), f"-picoweb:{bind}"], check=True)
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
