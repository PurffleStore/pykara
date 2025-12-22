#!/bin/sh
# Use PORT if provided by Railway, otherwise default to 5000
PORT=${PORT:-5000}
echo "Starting Gunicorn on port: $PORT"
# Start Gunicorn
exec gunicorn --bind 0.0.0.0:$PORT pytrade:app