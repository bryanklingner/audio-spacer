#!/bin/sh
# Pull the latest code, rebuild the image, and restart the container.
# Override the published port with PORT=nnnn.
set -e
cd "$(dirname "$0")"
git pull --ff-only
docker build -t audio-spacer .
docker rm -f audio-spacer 2>/dev/null || true
docker run -d --name audio-spacer -p "${PORT:-8931}:8000" \
    --restart unless-stopped audio-spacer
