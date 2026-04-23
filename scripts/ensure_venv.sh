#!/usr/bin/env bash
set -euo pipefail

SYSTEM_PY="$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
VENV_PY=""
if [ -f ".venv/bin/python3" ]; then
    VENV_PY="$(.venv/bin/python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
fi

if [ "${SYSTEM_PY}" != "${VENV_PY}" ]; then
    echo "[ensure_venv] Python mismatch (venv=${VENV_PY:-none}, system=${SYSTEM_PY}). Recreating .venv..."
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch --quiet
    .venv/bin/pip install -r requirements.txt --quiet
    .venv/bin/pip install -r requirements-dev.txt --quiet
    echo "[ensure_venv] Done — .venv now uses Python ${SYSTEM_PY}."
fi
