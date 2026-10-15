#!/bin/sh
# Starts the web demo. With DRGRADE_DOWNLOAD_WEIGHTS=1 the model files are first downloaded into the (writable)
# DRGRADE_WEIGHTS_DIR, for hosts where nothing can be mounted (e.g. a Hugging Face Docker Space).
set -e
if [ "${DRGRADE_DOWNLOAD_WEIGHTS:-0}" = "1" ]; then
    python scripts/download_weights.py --inference-only
fi
exec python app/server.py --host "${HOST:-0.0.0.0}" --port "${PORT:-8501}"
