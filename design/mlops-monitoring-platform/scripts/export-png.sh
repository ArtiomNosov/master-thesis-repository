#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/exports/png"
CHROME_BIN="${CHROME_BIN:-google-chrome}"

mkdir -p "$OUT_DIR"

screens=(
  "dashboard:01-main-mlops-dashboard.png"
  "model-detail:02-model-detail.png"
  "data-drift:03-data-drift-monitoring.png"
  "model-performance:04-model-performance.png"
  "alerts-incidents:05-alerts-incidents.png"
  "model-registry:06-model-registry.png"
)

for item in "${screens[@]}"; do
  screen="${item%%:*}"
  file="${item##*:}"
  profile_dir="$(mktemp -d)"
  out_file="$OUT_DIR/$file"
  rm -f "$out_file"
  timeout 15s "$CHROME_BIN" \
    --headless=new \
    --no-sandbox \
    --disable-dev-shm-usage \
    --disable-gpu \
    --hide-scrollbars \
    --user-data-dir="$profile_dir" \
    --window-size=1440,1024 \
    --screenshot="$out_file" \
    "file://$ROOT_DIR/index.html?screen=$screen" || true
  rm -rf "$profile_dir"

  if [[ ! -s "$out_file" ]]; then
    echo "Failed to export $screen to $out_file" >&2
    exit 1
  fi
done

echo "PNG exports saved to $OUT_DIR"
