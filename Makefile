.PHONY: install dev run test lint format clean

# Default Python
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install -e ".[dev]"

dev:
	$(VENV_BIN)/uvicorn job_compass.main:app --reload --port 8000

run:
	$(VENV_BIN)/python -m job_compass.main

test:
	$(VENV_BIN)/pytest tests/ -v

lint:
	$(VENV_BIN)/ruff check src/ tests/
	$(VENV_BIN)/ruff format --check src/ tests/

format:
	$(VENV_BIN)/ruff check --fix src/ tests/
	$(VENV_BIN)/ruff format src/ tests/

clean:
	rm -rf $(VENV) build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
