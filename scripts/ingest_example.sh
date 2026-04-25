#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:?first arg: input video path}"
WORKDIR="${2:?second arg: output directory}"
STEM=$(basename "$VIDEO" | sed 's/\.[^.]*$//')

mkdir -p "$WORKDIR/$STEM"

ffmpeg -y -i "$VIDEO" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$WORKDIR/$STEM/audio.wav"

mkdir -p "$WORKDIR/$STEM/frames"
ffmpeg -y -i "$VIDEO" -vf "fps=1" -q:v 3 "$WORKDIR/$STEM/frames/frame_%06d.jpg"

ffmpeg -y -i "$VIDEO" -vf "scale=w=320:h=-2" -c:v libx264 -preset veryfast -crf 28 -an "$WORKDIR/$STEM/proxy_320p.mp4"

echo "Outputs under $WORKDIR/$STEM: audio.wav, frames/, proxy_320p.mp4"
