# Production container for Google Cloud Run (Full 38-class ResNet50 Classifier)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install full dependencies including PyTorch
COPY requirements-full.txt .
RUN pip install --no-cache-dir -r requirements-full.txt

# Copy project files
COPY . .

# Download weights during container build so the full model is pre-loaded
RUN python download_weights.py || true

# Expose container port
EXPOSE 8080

# Run FastAPI with uvicorn respecting Cloud Run's dynamic $PORT
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
