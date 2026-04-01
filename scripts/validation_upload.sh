#!/bin/bash
# validation_upload.sh - Upload protected files TO validation bundle repo
#
# Usage: ./scripts/validation_upload.sh
#
# Requirements:
#   - VALIDATION_BUNDLE_TOKEN env var (GitHub PAT with WRITE access to validation-bundle repo)
#   - VALIDATION_BUNDLE_REPO env var (default: LambdaSection/Validation-Bundle)
#
# This script uploads protected validation files TO a private repo
# Use this when you modify Mom Test files locally and want to sync them to the bundle.

set -e

# Configuration
REPO="${VALIDATION_BUNDLE_REPO:-LambdaSection/Validation-Bundle}"
TOKEN="${VALIDATION_BUNDLE_TOKEN:-}"
BRANCH="${VALIDATION_BUNDLE_BRANCH:-main}"

# Files to upload (these are protected and should NOT be in public repo)
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
    log_error "VALIDATION_BUNDLE_TOKEN not set."
    log_info "To enable upload, set VALIDATION_BUNDLE_TOKEN environment variable."
    log_info "Note: Upload requires WRITE access (contents:write permission)"
    log_info "Example: export VALIDATION_BUNDLE_TOKEN=ghp_xxxx"
    exit 1
fi

# Check if repo is configured
if [ -z "$REPO" ]; then
    log_error "VALIDATION_BUNDLE_REPO not set."
    log_info "Example: export VALIDATION_BUNDLE_REPO=your-org/validation-bundle"
    exit 1
fi

log_info "Uploading to validation bundle: $REPO (branch: $BRANCH)"
log_info ""

# Upload each protected file
for file in "${PROTECTED_FILES[@]}"; do
    if [ ! -f "./${file}" ]; then
        log_warn "Skipping ${file} - not found locally"
        continue
    fi
    
    log_info "Uploading ${file}..."
    
    # Get file content and encode to base64
    CONTENT=$(base64 -w 0 "./${file}")
    
    # Get current SHA if file exists
    SHA=$(curl -s -H "Authorization: token ${TOKEN}" \
        "https://api.github.com/repos/${REPO}/contents/${file}?ref=${BRANCH}" 2>/dev/null | \
        grep -o '"sha": "[^"]*"' | head -1 | cut -d'"' -f4)
    
    # Prepare JSON payload
    if [ -n "$SHA" ]; then
        # Update existing file
        JSON="{\"message\": \"Update ${file} from NeuralDBG\", \"content\": \"${CONTENT}\", \"sha\": \"${SHA}\", \"branch\": \"${BRANCH}\"}"
        log_info "  -> Updating existing file"
    else
        # Create new file
        JSON="{\"message\": \"Add ${file} from NeuralDBG\", \"content\": \"${CONTENT}\", \"branch\": \"${BRANCH}\"}"
        log_info "  -> Creating new file"
    fi
    
    # Upload via GitHub API
    RESPONSE=$(curl -s -X PUT \
        -H "Authorization: token ${TOKEN}" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "Content-Type: application/json" \
        -d "${JSON}" \
        "https://api.github.com/repos/${REPO}/contents/${file}")
    
    # Check response
    if echo "$RESPONSE" | grep -q '"content"'; then
        log_info "  -> Uploaded successfully"
    else
        ERROR=$(echo "$RESPONSE" | grep -o '"message": "[^"]*"' | head -1 | cut -d'"' -f4)
        if [ -n "$ERROR" ]; then
            log_error "  -> Upload failed: ${ERROR}"
        else
            log_error "  -> Upload failed (unknown error)"
        fi
    fi
done

log_info ""
log_info "Upload complete."
log_info ""
log_info "Files uploaded to: https://github.com/${REPO}/tree/${BRANCH}"

exit 0
