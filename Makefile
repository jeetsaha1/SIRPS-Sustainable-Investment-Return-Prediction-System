
PYTHON ?= python

.PHONY: install test lint format pilot check

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m black src/ tests/

pilot:
	$(PYTHON) -m sirps.pilot

check: lint test
