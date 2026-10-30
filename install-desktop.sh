#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$APP_DIR/.venv/bin/python"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"

if [[ ! -x "$PY" ]]; then
  echo "No .venv found. Run: uv venv --python 3.12 && uv sync" >&2
  exit 1
fi

install -Dm644 "$APP_DIR/assets/ChemPad_logo_pencil_128x124.svg" \
  "$DATA/icons/hicolor/scalable/apps/chempad.svg"

mkdir -p "$DATA/applications"
sed -e "s|@PYTHON@|$PY|g" -e "s|@APP_DIR@|$APP_DIR|g" \
  "$APP_DIR/packaging/chempad.desktop.in" > "$DATA/applications/chempad.desktop"
chmod 644 "$DATA/applications/chempad.desktop"

update-desktop-database "$DATA/applications" 2>/dev/null || true
gtk-update-icon-cache -f -t "$DATA/icons/hicolor" 2>/dev/null || true
echo "Installed. Search for ChemPad in Activities."