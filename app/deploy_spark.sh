#!/usr/bin/env bash
# Build + run the hardened PosterReview container on the Spark.
# The app is isolated: non-root, read-only root fs, no host mounts, dropped
# capabilities, resource caps. It reaches the host's Ollama via host-gateway
# and is published to the host's localhost only (cloudflared is the ingress).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo "building image…"
docker build -f app/Dockerfile -t posterreview:latest .

docker rm -f posterreview 2>/dev/null || true

echo "starting hardened container…"
docker run -d --name posterreview --restart unless-stopped \
  --add-host=host.docker.internal:host-gateway \
  -p 127.0.0.1:5000:5000 \
  --memory=6g --cpus=4 --pids-limit=256 \
  --read-only \
  --tmpfs /srv/app/uploads:size=128m,mode=1777 \
  --tmpfs /tmp:size=128m \
  --tmpfs /home/appuser/.cache:size=512m \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  posterreview:latest

sleep 4
echo "--- container status ---"
docker ps --filter name=posterreview --format '{{.Names}}: {{.Status}}'
echo "--- health check (localhost:5000) ---"
curl -s -o /dev/null -w "app HTTP %{http_code}\n" http://127.0.0.1:5000/ || echo "not up yet"
