#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap.sh [--with-validation-sync]

One-command local onboarding for NeuralDBG:
  - verifies or recreates .venv
  - installs runtime, development, and MLOps dependencies
  - installs repository git hooks

Options:
  --with-validation-sync    Run validation sync after setup when VALIDATION_BUNDLE_TOKEN is set
  -h, --help                Show this help message
EOF
}

WITH_VALIDATION_SYNC=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-validation-sync)
      WITH_VALIDATION_SYNC=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[bootstrap] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

echo "[bootstrap] Preparing virtual environment"
bash scripts/ensure_venv.sh

if [ ! -x ".venv/bin/python" ]; then
  echo "[bootstrap] Missing .venv/bin/python after ensure_venv.sh" >&2
  exit 1
fi

VENV_PYTHON=".venv/bin/python"

echo "[bootstrap] Upgrading pip, setuptools, and wheel"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "[bootstrap] Installing runtime dependencies"
"${VENV_PYTHON}" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch
"${VENV_PYTHON}" -m pip install -r requirements.txt

echo "[bootstrap] Installing development dependencies"
"${VENV_PYTHON}" -m pip install -r requirements-dev.txt

if [ -f "requirements-mlops.txt" ]; then
  echo "[bootstrap] Installing MLOps dependencies"
  "${VENV_PYTHON}" -m pip install -r requirements-mlops.txt
fi

echo "[bootstrap] Installing project in editable mode"
"${VENV_PYTHON}" -m pip install -e .

echo "[bootstrap] Activating repository hooks"
bash scripts/install_hooks.sh

if [ "${WITH_VALIDATION_SYNC}" -eq 1 ]; then
  if [ -n "${VALIDATION_BUNDLE_TOKEN:-}" ]; then
    echo "[bootstrap] Running validation sync"
    bash scripts/validation_sync.sh
  else
    echo "[bootstrap] VALIDATION_BUNDLE_TOKEN is not set; skipping validation sync"
  fi
else
  echo "[bootstrap] Validation sync not run by default"
fi

echo "[bootstrap] Done"
echo "[bootstrap] Activate the environment with: source .venv/bin/activate"
