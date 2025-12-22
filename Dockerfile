# Use a slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for packages like pyodbc, TA-Lib if needed)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV FLASK_APP=pytrade.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will override $PORT)
EXPOSE 5000

# Start command for Railway
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-5000}", "pytrade:app"]