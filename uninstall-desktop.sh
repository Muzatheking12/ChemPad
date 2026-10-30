#!/usr/bin/env bash
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -f "$DATA/applications/chempad.desktop" \
      "$DATA/icons/hicolor/scalable/apps/chempad.svg"
update-desktop-database "$DATA/applications" 2>/dev/null || true