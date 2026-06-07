#!/usr/bin/env python3
"""Download PlantUML jar for offline/local diagram rendering."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from plantuml_render import (
    PLANTUML_DOWNLOAD_URL,
    PLANTUML_VERSION,
    download_plantuml_jar,
    resolve_jar_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to vkr/work_assembly_tool (default: parent of scripts/)",
    )
    parser.add_argument(
        "--jar",
        type=Path,
        default=None,
        help="Override jar output path (default: tools/plantuml.jar under config-dir)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run java -jar ... -version after download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if jar already exists",
    )
    args = parser.parse_args()

    jar_path = args.jar.resolve() if args.jar else resolve_jar_path(args.config_dir, {})
    print(f"PlantUML version: {PLANTUML_VERSION}")
    print(f"Download URL: {PLANTUML_DOWNLOAD_URL}")
    if jar_path.exists() and not args.force:
        print(f"Already present: {jar_path}")
    else:
        download_plantuml_jar(jar_path)

    if args.verify:
        completed = subprocess.run(
            ["java", "-jar", str(jar_path), "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    print(f"Ready: {jar_path}")


if __name__ == "__main__":
    main()
