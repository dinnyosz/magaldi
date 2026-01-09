# Multi-stage build for Magaldi Web Service
# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY src/magaldi_web/frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY src/magaldi_web/frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Python application
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir -e ".[web]"

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./src/magaldi_web/static

# Create non-root user
RUN useradd -m -u 1000 magaldi
USER magaldi

# Expose port
EXPOSE 8080

# Environment variables
ENV MAGALDI_WEB_HOST=0.0.0.0
ENV MAGALDI_WEB_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/admin/health')"

# Run the web server
CMD ["magaldi-web"]
