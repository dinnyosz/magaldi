# Magaldi Docker Setup

## Overview

Complete Docker Compose setup for running all Magaldi services locally or in production.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCKER ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   MySQL     │  │ Elastic     │  │   Ollama    │  │   Redis     │   │
│  │   8.0       │  │ search 8.11 │  │   (GPU)     │  │  (cache)    │   │
│  │   :3306     │  │   :9200     │  │   :11434    │  │   :6379     │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │          │
│         └────────────────┴────────────────┴────────────────┘          │
│                                   │                                    │
│                    ┌──────────────┴──────────────┐                    │
│                    │                             │                    │
│              ┌─────┴─────┐                 ┌─────┴─────┐              │
│              │  Magaldi  │                 │  Magaldi  │              │
│              │  Workers  │                 │  Web/MCP  │              │
│              │ (AI jobs) │                 │  Server   │              │
│              └───────────┘                 └─────┬─────┘              │
│                                                  │                    │
│                                             :8080 (Web)               │
│                                             stdio (MCP)               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
magaldi/
├── docker/
│   ├── Dockerfile              # Main application image
│   ├── Dockerfile.worker       # AI worker image
│   ├── nginx.conf              # Reverse proxy config
│   └── init-db.sql             # Database initialization
├── docker-compose.yml          # Development setup
├── docker-compose.prod.yml     # Production overrides
├── .env.example                # Environment template
└── config/
    └── magaldi.yaml            # Application config
```

---

## Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # ==========================================================================
  # DATABASES
  # ==========================================================================

  mysql:
    image: percona:8.0
    container_name: magaldi-mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:-rootpassword}
      MYSQL_DATABASE: magaldi
      MYSQL_USER: magaldi
      MYSQL_PASSWORD: ${MYSQL_PASSWORD:-magaldipassword}
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
      - ./docker/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-p${MYSQL_ROOT_PASSWORD:-rootpassword}"]
      interval: 10s
      timeout: 5s
      retries: 5

  elasticsearch:
    image: elasticsearch:8.11.0
    container_name: magaldi-elasticsearch
    restart: unless-stopped
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - xpack.security.enrollment.enabled=false
      - xpack.monitoring.collection.enabled=true
      - xpack.monitoring.elasticsearch.collection.enabled=true
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: magaldi-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ==========================================================================
  # AI SERVICES
  # ==========================================================================

  ollama:
    image: ollama/ollama:latest
    container_name: magaldi-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # GPU support (uncomment for NVIDIA GPU)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:11434/api/tags || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Model initialization (runs once)
  ollama-init:
    image: curlimages/curl:latest
    container_name: magaldi-ollama-init
    depends_on:
      ollama:
        condition: service_healthy
    restart: "no"
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        echo "Pulling summarization model..."
        curl -X POST http://ollama:11434/api/pull -d '{"name": "qwen2.5-coder:7b"}'
        echo "Pulling embedding model..."
        curl -X POST http://ollama:11434/api/pull -d '{"name": "snowflake-arctic-embed2"}'
        echo "Models ready!"

  # ==========================================================================
  # MAGALDI SERVICES
  # ==========================================================================

  magaldi-web:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: magaldi-web
    restart: unless-stopped
    command: ["magaldi", "web", "serve"]
    ports:
      - "8080:8080"
    environment:
      - MAGALDI_MYSQL_HOST=mysql
      - MAGALDI_MYSQL_PASSWORD=${MYSQL_PASSWORD:-magaldipassword}
      - MAGALDI_ELASTICSEARCH_URL=http://elasticsearch:9200
      - MAGALDI_OLLAMA_URL=http://ollama:11434
      - MAGALDI_REDIS_URL=redis://redis:6379
      - MAGALDI_LOG_LEVEL=${LOG_LEVEL:-INFO}
    volumes:
      - ./config:/etc/magaldi:ro
    depends_on:
      mysql:
        condition: service_healthy
      elasticsearch:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/api/v1/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  magaldi-worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    container_name: magaldi-worker
    restart: unless-stopped
    command: ["magaldi", "worker", "start", "--all"]
    environment:
      - MAGALDI_MYSQL_HOST=mysql
      - MAGALDI_MYSQL_PASSWORD=${MYSQL_PASSWORD:-magaldipassword}
      - MAGALDI_ELASTICSEARCH_URL=http://elasticsearch:9200
      - MAGALDI_OLLAMA_URL=http://ollama:11434
      - MAGALDI_LOG_LEVEL=${LOG_LEVEL:-INFO}
      - MAGALDI_WORKER_SUMMARIZATION_COUNT=4
      - MAGALDI_WORKER_EMBEDDING_COUNT=4
    volumes:
      - ./config:/etc/magaldi:ro
    depends_on:
      mysql:
        condition: service_healthy
      elasticsearch:
        condition: service_healthy
      ollama:
        condition: service_healthy
      ollama-init:
        condition: service_completed_successfully

  # CLI tool for parsing (run manually)
  magaldi-cli:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: magaldi-cli
    profiles: ["cli"]  # Only starts with --profile cli
    environment:
      - MAGALDI_MYSQL_HOST=mysql
      - MAGALDI_MYSQL_PASSWORD=${MYSQL_PASSWORD:-magaldipassword}
      - MAGALDI_ELASTICSEARCH_URL=http://elasticsearch:9200
      - MAGALDI_OLLAMA_URL=http://ollama:11434
    volumes:
      - ./config:/etc/magaldi:ro
      - ${REPO_PATH:-.}:/repos:ro  # Mount repository to parse
    depends_on:
      mysql:
        condition: service_healthy
      elasticsearch:
        condition: service_healthy

volumes:
  mysql_data:
  es_data:
  redis_data:
  ollama_data:

networks:
  default:
    name: magaldi-network
```

---

## Dockerfiles

### Main Application Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

LABEL maintainer="Magaldi Project"
LABEL description="Magaldi Code Discovery Engine"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash magaldi
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py pyproject.toml ./

# Install application
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER magaldi

# Default command
CMD ["magaldi", "--help"]
```

### Worker Dockerfile (with ML dependencies)

```dockerfile
# docker/Dockerfile.worker
FROM python:3.11-slim

LABEL maintainer="Magaldi Project"
LABEL description="Magaldi AI Worker"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash magaldi
WORKDIR /app

# Install Python dependencies (includes ML libs for UMAP/clustering)
COPY requirements.txt requirements-worker.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-worker.txt

# Copy application code
COPY src/ ./src/
COPY setup.py pyproject.toml ./

# Install application
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER magaldi

# Default command
CMD ["magaldi", "worker", "start", "--all"]
```

### Requirements Files

```txt
# requirements.txt (base)
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
mysql-connector-python>=8.2.0
elasticsearch>=8.11.0
redis>=5.0.0
pyyaml>=6.0
requests>=2.31.0
tree-sitter>=0.20.0
tree-sitter-python>=0.20.0
tree-sitter-javascript>=0.20.0
tree-sitter-typescript>=0.20.0
tree-sitter-rust>=0.20.0
tree-sitter-php>=0.20.0
click>=8.1.0
rich>=13.0.0
```

```txt
# requirements-worker.txt (ML dependencies for workers)
numpy>=1.24.0
scikit-learn>=1.3.0
umap-learn>=0.5.4
```

---

## Database Initialization

```sql
-- docker/init-db.sql
-- Magaldi Database Schema

CREATE DATABASE IF NOT EXISTS magaldi;
USE magaldi;

-- Repositories
CREATE TABLE repositories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tags JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_scope_name (scope, name)
);

-- File states
CREATE TABLE file_states (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,

    file_hash VARCHAR(64),
    language VARCHAR(50),
    file_size INT,
    line_count INT,

    is_deleted BOOLEAN DEFAULT FALSE,

    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,

    UNIQUE KEY unique_file (scope, repository, username, relative_path(255)),
    INDEX idx_expiry (expires_at),
    INDEX idx_user (username)
);

-- Code elements
CREATE TABLE code_elements (
    element_id VARCHAR(512) PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL,
    relative_path VARCHAR(1024) NOT NULL,

    element_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    line_start INT NOT NULL,
    line_end INT,

    raw_code MEDIUMTEXT,
    signature VARCHAR(1024),
    docstring TEXT,

    level INT NOT NULL,
    parent_id VARCHAR(512),

    language VARCHAR(50),
    decorators JSON,
    visibility VARCHAR(20),
    is_async BOOLEAN DEFAULT FALSE,

    summary_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    embedding_status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    summary TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,

    INDEX idx_scope_repo (scope, repository),
    INDEX idx_username (username),
    INDEX idx_parent (parent_id),
    INDEX idx_expiry (expires_at)
);

-- Repository languages
CREATE TABLE repository_languages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scope VARCHAR(100) NOT NULL,
    repository VARCHAR(255) NOT NULL,
    language VARCHAR(50) NOT NULL,
    file_count INT DEFAULT 0,
    line_count INT DEFAULT 0,

    UNIQUE KEY unique_repo_lang (scope, repository, language)
);

-- Summarization jobs
CREATE TABLE summarization_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    element_id VARCHAR(512) NOT NULL,
    level INT NOT NULL,
    parent_element_id VARCHAR(512),

    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',
    dependencies_met BOOLEAN DEFAULT FALSE,
    priority INT DEFAULT 0,

    worker_id VARCHAR(100),
    claimed_at DATETIME,
    completed_at DATETIME,
    retry_count INT DEFAULT 0,
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_element (element_id),
    INDEX idx_status_level (status, level, dependencies_met),
    INDEX idx_worker (worker_id)
);

-- Embedding jobs
CREATE TABLE embedding_jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    element_id VARCHAR(512) NOT NULL,

    status ENUM('pending', 'running', 'completed', 'failed') DEFAULT 'pending',

    worker_id VARCHAR(100),
    claimed_at DATETIME,
    completed_at DATETIME,
    retry_count INT DEFAULT 0,
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_element (element_id),
    INDEX idx_status (status)
);

-- Element dependencies (for dependency mapping feature)
CREATE TABLE element_dependencies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_element_id VARCHAR(512) NOT NULL,
    target_element_id VARCHAR(512) NOT NULL,
    relationship_type ENUM('import', 'call', 'inherit', 'use') NOT NULL,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY unique_dep (source_element_id, target_element_id, relationship_type),
    INDEX idx_source (source_element_id),
    INDEX idx_target (target_element_id)
);

-- Grant permissions
GRANT ALL PRIVILEGES ON magaldi.* TO 'magaldi'@'%';
FLUSH PRIVILEGES;
```

---

## Environment Configuration

```bash
# .env.example
# Copy to .env and customize

# =============================================================================
# DATABASE
# =============================================================================
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_PASSWORD=magaldipassword

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL=INFO

# =============================================================================
# REPOSITORY PATH (for CLI parsing)
# =============================================================================
REPO_PATH=/path/to/your/repositories

# =============================================================================
# GPU SUPPORT (uncomment to enable)
# =============================================================================
# NVIDIA_VISIBLE_DEVICES=all
```

---

## Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  mysql:
    deploy:
      resources:
        limits:
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    deploy:
      resources:
        limits:
          memory: 4G

  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  magaldi-web:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G

  magaldi-worker:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G

  nginx:
    image: nginx:alpine
    container_name: magaldi-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - magaldi-web
```

### Nginx Configuration

```nginx
# docker/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream magaldi_web {
        server magaldi-web:8080;
    }

    server {
        listen 80;
        server_name _;

        location / {
            proxy_pass http://magaldi_web;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /api/ {
            proxy_pass http://magaldi_web;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300s;
        }
    }
}
```

---

## Usage Commands

### Development

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f magaldi-web

# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Parsing a Repository

```bash
# Parse a repository (one-time)
docker compose run --rm \
  -v /path/to/your/repo:/repos/myrepo:ro \
  magaldi-cli \
  magaldi parse /repos/myrepo --user main

# Or use the REPO_PATH env variable
REPO_PATH=/path/to/your/repo docker compose run --rm magaldi-cli \
  magaldi parse /repos --user main
```

### Check Status

```bash
# Check service health
docker compose ps

# Check Ollama models
curl http://localhost:11434/api/tags

# Check Elasticsearch cluster health
curl http://localhost:9200/_cluster/health

# Check Elasticsearch monitoring stats
curl http://localhost:9200/_nodes/stats
curl http://localhost:9200/_cluster/stats

# Check Elasticsearch index stats
curl http://localhost:9200/magaldi_code_elements/_stats

# Check Magaldi API
curl http://localhost:8080/api/v1/health
```

### Scaling Workers

```bash
# Scale up workers for large indexing jobs
docker compose up -d --scale magaldi-worker=4

# Scale back down
docker compose up -d --scale magaldi-worker=1
```

---

## GPU Support (NVIDIA)

```bash
# Install NVIDIA Container Toolkit first:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Start with GPU support
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Troubleshooting

### Ollama Models Not Loading

```bash
# Manually pull models
docker exec -it magaldi-ollama ollama pull qwen2.5-coder:7b
docker exec -it magaldi-ollama ollama pull snowflake-arctic-embed2

# Check loaded models
docker exec -it magaldi-ollama ollama list
```

### Elasticsearch Memory Issues

```bash
# Increase heap size in docker-compose.yml
environment:
  - "ES_JAVA_OPTS=-Xms2g -Xmx2g"

# Or increase host vm.max_map_count
sudo sysctl -w vm.max_map_count=262144
```

### Database Connection Issues

```bash
# Check MySQL is ready
docker exec -it magaldi-mysql mysql -u magaldi -p -e "SELECT 1"

# Reset database
docker compose down -v
docker compose up -d mysql
# Wait for healthy, then restart other services
docker compose up -d
```

---

## Data Persistence

| Service | Volume | Data |
|---------|--------|------|
| MySQL | mysql_data | All parsed elements, jobs |
| Elasticsearch | es_data | Search index, embeddings |
| Redis | redis_data | Cache (can be cleared) |
| Ollama | ollama_data | Downloaded models |

### Backup

```bash
# Backup MySQL
docker exec magaldi-mysql mysqldump -u root -p magaldi > backup.sql

# Backup Elasticsearch
curl -X PUT "localhost:9200/_snapshot/backup" -H 'Content-Type: application/json' -d'
{
  "type": "fs",
  "settings": {
    "location": "/backup"
  }
}'
```

---

## Summary

| Service | Port | Purpose |
|---------|------|---------|
| MySQL | 3306 | Metadata storage |
| Elasticsearch | 9200 | Vector search |
| Redis | 6379 | Caching |
| Ollama | 11434 | AI models |
| Magaldi Web | 8080 | Web UI + API |
| Magaldi Worker | - | Background AI jobs |
