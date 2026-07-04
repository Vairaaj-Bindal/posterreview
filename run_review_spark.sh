#!/usr/bin/env bash
# Zero-cost review using the model on the DGX Spark (via an encrypted SSH tunnel).
# Usage:  ./run_review_spark.sh <poster.pdf> [review.py args...]
#
# Ensures the SSH tunnel to the Spark's Ollama is up, then runs the reviewer
# pointed at it. Nothing is exposed to the network; traffic rides SSH.
set -euo pipefail

SPARK="${SPARK_SSH:-aisuper@spark-4fb0.local}"
MODEL="${POSTERREVIEW_OLLAMA_MODEL:-qwen2.5:72b}"
PORT=11434
HERE="$(cd "$(dirname "$0")" && pwd)"

# bring the tunnel up if the Spark's Ollama isn't already reachable on localhost
if ! curl -s "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
  echo "opening SSH tunnel to ${SPARK}..." >&2
  ssh -f -N -L "${PORT}:localhost:${PORT}" "${SPARK}"
  for _ in 1 2 3 4 5; do
    curl -s "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break || sleep 1
  done
fi

OLLAMA_HOST="127.0.0.1:${PORT}" POSTERREVIEW_OLLAMA_MODEL="${MODEL}" \
  "${HERE}/.venv/bin/python" "${HERE}/research/scripts/review.py" "$@"
