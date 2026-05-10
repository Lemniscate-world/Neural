SHELL := /bin/bash

PYTHON = $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)
COMPOSE := docker-compose

.PHONY: help bootstrap install install-dev check-venv up down build rebuild shell test test-docker coverage bandit safety security precommit docs clean

help:
	@echo "NeuralDBG Make targets:"
	@echo "  make bootstrap     - one-command local onboarding"
	@echo "  make install       - install runtime dependencies"
	@echo "  make install-dev   - install runtime + dev dependencies"
	@echo "  make build         - build Docker dev image"
	@echo "  make up            - start Docker dev service"
	@echo "  make down          - stop Docker dev service"
	@echo "  make shell         - open shell in running dev container"
	@echo "  make test          - run pytest locally"
	@echo "  make test-docker   - run pytest in Docker container"
	@echo "  make coverage      - run coverage with gate >= 60%"
	@echo "  make bandit        - run Bandit security scan"
	@echo "  make safety        - run Safety dependency scan"
	@echo "  make security      - run Bandit + Safety"
	@echo "  make precommit     - run pre-commit hooks on all files"
	@echo "  make check-venv    - verify/recreate .venv if Python version mismatch"
	@echo "  make docs          - generate API docs to docs/api/"
	@echo "  make clean         - remove local QA artifacts"

bootstrap:
	@bash scripts/bootstrap.sh

check-venv:
	@bash scripts/ensure_venv.sh

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

build:
	$(COMPOSE) build neuraldbg-dev

rebuild:
	$(COMPOSE) build --no-cache neuraldbg-dev

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

shell:
	$(COMPOSE) exec neuraldbg-dev bash

test: check-venv
	$(PYTHON) -m pytest

test-docker:
	$(COMPOSE) run --rm neuraldbg-dev bash -lc "pytest"

coverage: check-venv
	$(PYTHON) -m coverage run --include=neuraldbg.py -m pytest
	$(PYTHON) -m coverage report --show-missing --fail-under=60

bandit: check-venv
	$(PYTHON) -m bandit -r . -ll -x tests,*/tests/*,.venv,venv,__pycache__

safety: check-venv
	$(PYTHON) -m safety check --full-report

security: bandit safety

precommit:
	$(PYTHON) -m pre_commit run --all-files

docs: check-venv
	$(PYTHON) -m pdoc neuraldbg --output-dir docs/api

clean:
	rm -rf .pytest_cache htmlcov .mypy_cache
	rm -f .coverage .coverage.*
