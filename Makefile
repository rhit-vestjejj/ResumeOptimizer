.PHONY: install run test

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

run:
	$(VENV_PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8030

test:
	$(VENV_PYTHON) -m pytest -q
