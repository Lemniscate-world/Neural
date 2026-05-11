#!/usr/bin/env bash
set -euo pipefail

_pip_works() {
    [ -x ".venv/bin/python" ] && .venv/bin/python -c "import pip" &>/dev/null
}

_version_matches() {
    SYSTEM_PY="$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
    VENV_PY=""
    if [ -x ".venv/bin/python" ]; then
        VENV_PY="$(.venv/bin/python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
    fi
    [ "${SYSTEM_PY}" = "${VENV_PY}" ]
}

if ! _pip_works || ! _version_matches; then
    if _pip_works && ! _version_matches; then
        SYSTEM_PY="$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
        VENV_PY="$(.venv/bin/python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
        echo "[ensure_venv] Python mismatch (venv=${VENV_PY}, system=${SYSTEM_PY}). Recreating..."
    else
        echo "[ensure_venv] .venv missing or broken. Recreating..."
    fi
    rm -rf .venv

    # python3-venv peut être absent sur Debian/Ubuntu — --without-pip contourne
    python3 -m venv --without-pip .venv

    echo "[ensure_venv] Bootstrap pip via get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    .venv/bin/python /tmp/get-pip.py --quiet
    rm /tmp/get-pip.py

    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch --quiet
    .venv/bin/pip install -r requirements.txt --quiet
    .venv/bin/pip install -r requirements-dev.txt --quiet

    VENV_PY="$(.venv/bin/python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)"
    echo "[ensure_venv] Done — .venv now uses Python ${VENV_PY}."
fi
