#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks
chmod +x .githooks/post-checkout .githooks/post-merge \
         scripts/validation_sync.sh scripts/ensure_venv.sh

echo "[install_hooks] Git hooks activés depuis .githooks/"
bash scripts/ensure_venv.sh
echo "[install_hooks] Environnement prêt. Pense à définir VALIDATION_BUNDLE_TOKEN pour le sync MLO-15."
