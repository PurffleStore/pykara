#!/bin/bash
# start.sh - Properly handle PORT for Railway

# Read PORT from environment, default to 5000
PORT=${PORT:-5000}

# Export for Python/gunicorn
export PORT

echo "Starting application on port: $PORT"

# Start gunicorn with the resolved port
exec gunicorn --bind "0.0.0.0:$PORT" pytrade:app