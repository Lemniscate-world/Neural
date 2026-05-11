# Platform Detection
ifeq ($(OS),Windows_NT)
    VENV_BIN = .venv/Scripts
    PYTHON = $(VENV_BIN)/python.exe
else
    VENV_BIN = .venv/bin
    PYTHON = $(VENV_BIN)/python
endif

# Fallback to system python if venv doesn't exist
ifeq (,$(wildcard $(PYTHON)))
    PYTHON = python3
endif

COMPOSE := docker-compose

.PHONY: help bootstrap install install-dev check-venv up down build rebuild shell test test-docker coverage bandit safety security precommit docs session-docx release clean

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
	@echo "  make check-venv    - verify/recreate .venv if Python version mismatch"
	@echo "  make coverage      - run coverage with gate >= 60%"
	@echo "  make bandit        - run Bandit security scan"
	@echo "  make safety        - run Safety dependency scan"
	@echo "  make security      - run Bandit + Safety"
	@echo "  make precommit     - run pre-commit hooks on all files"
	@echo "  make check-venv    - verify/recreate .venv if Python version mismatch"
	@echo "  make docs          - generate API docs to docs/api/"
	@echo "  make session-docx  - convert SESSION_SUMMARY.md to outputs/SESSION_SUMMARY.docx"
	@echo "  make release       - full release automation (R18, R6, R19)"
	@echo "  make clean         - remove local QA artifacts"

bootstrap:
	@$(PYTHON) infrastructure/scripts/bootstrap.py

check-venv:
	@$(PYTHON) infrastructure/scripts/ensure_venv.py

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[automation,mlops,sync,docs]"

install-dev: install
	$(PYTHON) -m pip install -e ".[dev]"

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

session-docx: check-venv
	$(PYTHON) infrastructure/scripts/session_to_docx.py

release: test security
	@if [ -z "$(version)" ]; then echo "Error: version is required (e.g. make release version=1.2.0-kuro)"; exit 1; fi
	@$(PYTHON) -c "import re; exit(0 if re.match(r'^[0-9]+\.[0-9]+\.[0-9]+-kuro$$', '$(version)') else 1)" || (echo "Error: version must match X.Y.Z-kuro (Rule 19/91)"; exit 1)
	@echo "Releasing v$(version)..."
	@$(PYTHON) infrastructure/scripts/bump_version.py pyproject.toml $(version)
	@git add pyproject.toml
	@git commit -m "release: v$(version)"
	@git tag -a v$(version) -m "Release v$(version)"
	@git push origin main --tags

clean:
	rm -rf .pytest_cache htmlcov .mypy_cache
	rm -f .coverage .coverage.*
