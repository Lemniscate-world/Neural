#!/bin/bash
# validation_sync.sh - Synchronize validation bundle from private GitHub repo
#
# Usage: ./scripts/validation_sync.sh
#
# Requirements:
#   - VALIDATION_BUNDLE_TOKEN env var (GitHub PAT with read access to validation-bundle repo)
#   - VALIDATION_BUNDLE_REPO env var (default: your-org/validation-bundle)
#
# This script downloads protected validation files from a private repo
# to ensure Mom Test artifacts are never accidentally committed to public repos.

set -e

# Configuration
REPO="${VALIDATION_BUNDLE_REPO:-LambdaSection/Validation-Bundle}"
TOKEN="${VALIDATION_BUNDLE_TOKEN:-}"
BRANCH="${VALIDATION_BUNDLE_BRANCH:-main}"

# Files to sync (these are protected and should NOT be in public repo)
PROTECTED_FILES=(
    "mom_test_results.md"
    "decision.md"
    "mom_test_script.md"
    "ideas.md"
    "architecture_notes.md"
    "validation.json"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    log_error "Not in a git repository. Please run from the repo root."
    exit 1
fi

# Check if token is available
if [ -z "$TOKEN" ]; then
    log_warn "VALIDATION_BUNDLE_TOKEN not set. Skipping sync."
    log_info "To enable sync, set VALIDATION_BUNDLE_TOKEN environment variable."
    log_info "Example: export VALIDATION_BUNDLE_TOKEN=ghp_xxxx"
    exit 0
fi

# Check if repo is configured
if [ -z "$REPO" ]; then
    log_error "VALIDATION_BUNDLE_REPO not set."
    log_info "Example: export VALIDATION_BUNDLE_REPO=your-org/validation-bundle"
    exit 1
fi

log_info "Syncing validation bundle from $REPO (branch: $BRANCH)"

# Create temp directory for download
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Download each protected file
for file in "${PROTECTED_FILES[@]}"; do
    log_info "Downloading $file..."

    # Construct GitHub raw URL with token auth
    # Format: https://TOKEN@raw.githubusercontent.com/OWNER/REPO/BRANCH/FILE
    URL="https://${TOKEN}@raw.githubusercontent.com/${REPO}/${BRANCH}/${file}"

    # Download with curl
    if curl -sSf -o "${TEMP_DIR}/${file}" "$URL" 2>/dev/null; then
        # Check if file is not empty
        if [ -s "${TEMP_DIR}/${file}" ]; then
            mv "${TEMP_DIR}/${file}" "./${file}"
            log_info "  -> Synced ${file}"
        else
            log_warn "  -> ${file} is empty, skipping"
        fi
    else
        log_warn "  -> ${file} not found in bundle (this is OK for new projects)"
    fi
done

log_info "Validation sync complete."

# Verify protected files are in .gitignore
log_info "Verifying .gitignore protection..."
MISSING_IGNORE=0
for file in "${PROTECTED_FILES[@]}"; do
    if [ -f "./${file}" ] && ! grep -q "^${file}$" .gitignore 2>/dev/null; then
        log_warn "${file} exists but not in .gitignore - adding it"
        echo "${file}" >> .gitignore
        MISSING_IGNORE=1
    fi
done

if [ $MISSING_IGNORE -eq 1 ]; then
    log_info "Updated .gitignore. Please commit this change."
fi

exit 0
