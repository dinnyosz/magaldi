# Magaldi Makefile
# Common development commands

.PHONY: help setup services services-full services-down \
        test test-cov lint format typecheck check clean dev \
        ollama-pull db-reset logs

# Default target
help:
	@echo "Magaldi Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        Create venv and install all dependencies"
	@echo ""
	@echo "Services:"
	@echo "  make services      Start core Docker services (mysql, es, redis)"
	@echo "  make services-full Start all services including Ollama"
	@echo "  make services-down Stop all Docker services"
	@echo "  make logs          Follow Docker service logs"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Format code with ruff"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make check         Run all checks (lint, typecheck, test)"
	@echo ""
	@echo "Utilities:"
	@echo "  make ollama-pull   Pull required Ollama models (local)"
	@echo "  make db-reset      Reset database (drops all data)"
	@echo "  make clean         Remove generated files"

# =============================================================================
# SETUP
# =============================================================================

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)/bin/activate
	$(PIP) install -e .
	@echo ""
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

# =============================================================================
# DOCKER SERVICES
# =============================================================================

services:
	docker compose up -d
	@echo ""
	@echo "Core services started:"
	@echo "  MySQL:         localhost:3306"
	@echo "  Elasticsearch: localhost:9200"
	@echo "  Redis:         localhost:6379"
	@echo ""
	@echo "Waiting for services to be healthy..."
	@docker compose ps

services-full:
	docker compose --profile ollama up -d
	@echo ""
	@echo "All services started:"
	@echo "  MySQL:         localhost:3306"
	@echo "  Elasticsearch: localhost:9200"
	@echo "  Redis:         localhost:6379"
	@echo "  Ollama:        localhost:11434"
	@echo ""
	@echo "Note: Ollama models are being pulled in the background."
	@docker compose ps

services-down:
	docker compose --profile ollama down
	@echo "All services stopped."

logs:
	docker compose logs -f

# =============================================================================
# TESTING
# =============================================================================

test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src/magaldi --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow and not integration"

test-integration:
	$(PYTHON) -m pytest tests/ -v -m "integration"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
	$(PYTHON) -m ruff check --fix src/ tests/

typecheck:
	$(PYTHON) -m mypy src/magaldi

check: lint typecheck test
	@echo ""
	@echo "All checks passed!"

# =============================================================================
# UTILITIES
# =============================================================================

ollama-pull:
	@echo "Pulling Ollama models (this may take a while)..."
	ollama pull qwen2.5-coder:7b
	ollama pull snowflake-arctic-embed2
	@echo ""
	@echo "Models ready!"

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker compose down mysql
	docker volume rm magaldi_mysql_data || true
	docker compose up -d mysql
	@echo "Database reset complete."

clean:
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned generated files."

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev:
	$(PYTHON) -m uvicorn magaldi.web.app:app --reload --host 0.0.0.0 --port 8080
