#!/usr/bin/env bash
set -euo pipefail

REPO_OWNER="${VALIDATION_BUNDLE_OWNER:-LambdaSection}"
REPO_NAME="${VALIDATION_BUNDLE_REPO:-validation-bundle}"
REPO_REF="${VALIDATION_BUNDLE_REF:-main}"
API="https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents"
TARGET_DIR="$(git rev-parse --show-toplevel)"
TOKEN="${VALIDATION_BUNDLE_TOKEN:-}"

if [ -z "${TOKEN}" ]; then
  echo "VALIDATION_BUNDLE_TOKEN not set; skipping validation sync." >&2
  exit 0
fi

PYTHON_BIN="${VALIDATION_SYNC_PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

files=(
  "mom_test_script.md"
  "mom_test_results.md"
  "decision.md"
  "validation.json"
)

for f in "${files[@]}"; do
  response="$(curl -sS \
    -H "Authorization: token ${TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "${API}/${f}?ref=${REPO_REF}")"

  content="$("${PYTHON_BIN}" - <<'PY' <<<"${response}"
import sys
import json
import base64

data = json.loads(sys.stdin.read())
if "content" not in data:
    raise SystemExit("missing content in GitHub API response")

print(base64.b64decode(data["content"]).decode("utf-8"), end="")
PY
  )"

  printf "%s" "${content}" > "${TARGET_DIR}/${f}"
  chmod 600 "${TARGET_DIR}/${f}"
done
