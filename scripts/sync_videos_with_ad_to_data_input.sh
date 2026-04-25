#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/videos_with_ad"
DST="$ROOT/data/input"
mkdir -p "$DST" "$SRC"
shopt -s nullglob
found=0
batch=( "$SRC"/test_*.mp4 )
idx=0
while [[ idx -lt ${#batch[@]} ]]; do
  f="${batch[idx]}"
  base="$(basename "$f")"
  ln -sf "$f" "$DST/$base"
  echo "Linked $DST/$base -> $f"
  found=1
  idx=$((idx + 1))
done
if [[ "$found" -eq 0 ]]; then
  echo "No test_*.mp4 found under $SRC"
  echo "Add your stitched videos there, then run this script again."
  exit 1
fi
