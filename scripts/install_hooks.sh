#!/bin/bash
# install_hooks.sh - Install git hooks for validation sync
#
# Usage: ./scripts/install_hooks.sh
#
# This script installs git hooks that automatically sync the validation bundle
# after checkout and merge operations.

set -e

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

# Create .githooks directory if it doesn't exist
mkdir -p .githooks

# Create post-checkout hook
log_info "Creating post-checkout hook..."
cat > .githooks/post-checkout << 'EOF'
#!/bin/bash
# post-checkout hook - Sync validation bundle after branch switch
#
# This hook runs after git checkout to ensure validation files are synced.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[post-checkout] Syncing validation bundle..."
cd "$SCRIPT_DIR"
./scripts/validation_sync.sh 2>/dev/null || true
EOF

# Create post-merge hook
log_info "Creating post-merge hook..."
cat > .githooks/post-merge << 'EOF'
#!/bin/bash
# post-merge hook - Sync validation bundle after pull/merge
#
# This hook runs after git pull to ensure validation files are synced.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[post-merge] Syncing validation bundle..."
cd "$SCRIPT_DIR"
./scripts/validation_sync.sh 2>/dev/null || true
EOF

# Make hooks executable
chmod +x .githooks/post-checkout
chmod +x .githooks/post-merge

# Make sync script executable
chmod +x scripts/validation_sync.sh

# Configure git to use .githooks directory
log_info "Configuring git to use .githooks..."
git config core.hooksPath .githooks

log_info "Git hooks installed successfully!"
log_info ""
log_info "Hooks installed:"
log_info "  - .githooks/post-checkout (runs on git checkout)"
log_info "  - .githooks/post-merge (runs on git pull)"
log_info ""
log_info "To enable sync, set this environment variable:"
log_info "  export VALIDATION_BUNDLE_TOKEN=ghp_xxxx"
log_info ""
log_info "Repo configured: LambdaSection/Validation-Bundle"
log_info "To change: export VALIDATION_BUNDLE_REPO=other-org/validation-bundle"
log_info ""
log_info "Or add them to your shell profile (~/.bashrc, ~/.zshrc, etc.)"

exit 0