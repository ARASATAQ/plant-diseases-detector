# Production container for Google Cloud Run
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from requirements-full.txt
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy all source files
COPY . .

EXPOSE 8080

# Start uvicorn binding to the port injected by Cloud Run ($PORT)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
