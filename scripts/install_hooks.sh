#!/usr/bin/env bash
set -euo pipefail

git config core.hooksPath .githooks
chmod +x .githooks/post-checkout .githooks/post-merge scripts/validation_sync.sh

echo "Hooks installed. Ensure VALIDATION_BUNDLE_TOKEN is set."
