.PHONY: help install install-dev lint lint-fix format check test clean build upload upload-test

# Default target
help:
	@echo "Available commands:"
	@echo "  install              - Install the package in editable mode"
	@echo "  install-dev          - Install the package with development dependencies"
	@echo "  lint                 - Run ruff linter (check only)"
	@echo "  lint-fix             - Run ruff linter and fix errors automatically"
	@echo "  format               - Format code with ruff"
	@echo "  check                - Run all checks (lint + format check + tests)"
	@echo "  test                 - Run tests with pytest"
	@echo "  clean                - Clean build artifacts and cache files"
	@echo "  build                - Build the package for distribution"
	@echo "  upload-test          - Upload package to TestPyPI"
	@echo "  upload               - Upload package to PyPI"
	@echo "  info                 - Show package information"

# Install the package in editable mode
install:
	uv pip install -e .

# Install the package with development dependencies
install-dev:
	uv pip install -e ".[dev]"

# Run ruff linter (check only)
lint:
	uv run ruff check .

# Run ruff linter and fix errors automatically
lint-fix:
	uv run ruff check --fix .

# Format code with ruff
format:
	uv run ruff format .

# Check if code is properly formatted (without making changes)
format-check:
	uv run ruff format --check .

# Run all checks (lint + format check + tests)
check: lint format-check test

# Run tests with pytest
test:
	uv run pytest

# Clean build artifacts and cache files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Build the package for distribution
build: clean
	uv build

# Upload package to TestPyPI (requires TWINE_USERNAME and TWINE_PASSWORD env vars)
upload-test: build
	uv run twine upload --repository testpypi dist/*

# Upload package to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD env vars)
upload: build
	uv run twine upload dist/*

# Install twine for package uploading (run this once before upload commands)
install-twine:
	uv add --dev twine