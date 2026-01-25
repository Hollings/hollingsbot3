# Development tasks for hollingsbot3
# Run `make help` to see available commands

.PHONY: help install install-dev lint format test clean pre-commit

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies (includes linters, testing tools)
	pip install -r requirements.txt
	pip install ruff pytest pytest-asyncio pytest-cov pre-commit mypy bandit

lint:  ## Run linter (ruff)
	ruff check src/ tests/ scripts/

format:  ## Format code (ruff)
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

test:  ## Run tests
	PYTHONPATH=src pytest tests/ -v

test-cov:  ## Run tests with coverage report
	PYTHONPATH=src pytest tests/ -v --cov=src --cov-report=term-missing

typecheck:  ## Run type checker (mypy)
	mypy src/

security:  ## Run security scan (bandit)
	bandit -c pyproject.toml -r src/

pre-commit:  ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

pre-commit-all:  ## Run pre-commit on all files
	pre-commit run --all-files

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

check: lint typecheck security test  ## Run all checks (lint, typecheck, security, test)
