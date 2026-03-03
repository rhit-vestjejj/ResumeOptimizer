.PHONY: install run test eval check docker-build docker-up docker-down docker-logs

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
DOCKER_COMPOSE ?= docker compose

install:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

run:
	$(VENV_PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8030

test:
	$(VENV_PYTHON) -m pytest -q

eval:
	$(VENV_PYTHON) scripts/eval_selection.py

check: test eval

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up --build -d

docker-down:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f resume-optimizer
