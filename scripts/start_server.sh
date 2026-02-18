#!/bin/bash

# Start API server
# Usage: ./scripts/start_server.sh

PORT=${1:-8000}

echo "🚀 Starting Robot Autonomy API on port $PORT..."

python -m uvicorn \
    serving.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload
