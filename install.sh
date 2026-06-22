#!/usr/bin/env bash
# 3dfloorplan — installer
# Works on: macOS, Ubuntu/Debian VPS
# Usage:  bash install.sh

set -euo pipefail

PYTHON_MIN="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info() { echo -e "    $*"; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  3dfloorplan — Wall Extraction Engine"
echo "  Installer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Detect OS ─────────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      err "Unsupported OS: $OS" ;;
esac
info "Platform: $PLATFORM"

# ── Check Python 3.11+ ────────────────────────────────────────────────────────
find_python() {
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'  2>/dev/null || true)"
            major="${ver%%.*}"; minor="${ver##*.}"
            if [[ "$major" -ge 3 && "$minor" -ge 11 ]]; then
                echo "$cmd"; return 0
            fi
        fi
    done
    return 1
}

if PYTHON=$(find_python); then
    ok "Python $($PYTHON --version 2>&1 | awk '{print $2}') found → $PYTHON"
else
    warn "Python $PYTHON_MIN+ not found. Attempting install..."
    if [[ "$PLATFORM" == "linux" ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update -qq
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
            PYTHON="python3.11"
        elif command -v yum &>/dev/null; then
            sudo yum install -y python311 python311-pip
            PYTHON="python3.11"
        else
            err "Cannot auto-install Python. Install Python $PYTHON_MIN+ manually."
        fi
    elif [[ "$PLATFORM" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            brew install python@3.11
            PYTHON="$(brew --prefix)/bin/python3.11"
        else
            err "Homebrew not found. Install Python $PYTHON_MIN+ from https://python.org or install Homebrew first."
        fi
    fi
    ok "Python installed → $PYTHON"
fi

# ── Install system deps (Linux only) ─────────────────────────────────────────
if [[ "$PLATFORM" == "linux" ]]; then
    MISSING_DEPS=()
    for pkg in libgl1 libglib2.0-0; do
        if ! dpkg -l "$pkg" &>/dev/null 2>&1; then
            MISSING_DEPS+=("$pkg")
        fi
    done
    if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
        info "Installing system libs: ${MISSING_DEPS[*]}"
        sudo apt-get install -y "${MISSING_DEPS[@]}" -qq
        ok "System libs installed"
    else
        ok "System libs present (libgl1, libglib2.0-0)"
    fi
fi

# ── Install uv ───────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    ok "uv found → $(uv --version)"
else
    warn "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to current session path
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed → $(uv --version)"
    else
        err "uv install failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi
fi

# ── Install project dependencies ──────────────────────────────────────────────
echo ""
info "Installing Python dependencies..."
cd "$PROJECT_DIR"

uv sync --python "$PYTHON"
ok "Dependencies installed"

# ── Verify installation ───────────────────────────────────────────────────────
echo ""
info "Verifying installation..."

uv run python - <<'EOF'
import sys
errors = []

try:
    import cv2
except ImportError as e:
    errors.append(f"opencv: {e}")

try:
    import numpy
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import shapely
except ImportError as e:
    errors.append(f"shapely: {e}")

try:
    import networkx
except ImportError as e:
    errors.append(f"networkx: {e}")

try:
    import pydantic
except ImportError as e:
    errors.append(f"pydantic: {e}")

try:
    from wallgraph.detector import WallDetector
    from wallgraph.builder import WallGraphBuilder
    from wallgraph.mask_to_shape import extract_wall_shapes
except ImportError as e:
    errors.append(f"wallgraph: {e}")

if errors:
    print("FAIL: " + "; ".join(errors))
    sys.exit(1)
else:
    print("OK")
EOF

if [[ $? -eq 0 ]]; then
    ok "All imports verified"
else
    err "Verification failed. Check errors above."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Installation complete!"
echo ""
echo "  Run engine:"
echo "    uv run python running.py image.png --api -o out.json"
echo ""
echo "  Contour mode (best quality):"
echo "    uv run python running.py image.png --api --contour --close 8 --wall-thickness 6 -o out.json"
echo ""
echo "  Or activate venv manually:"
echo "    source .venv/bin/activate"
echo "    python running.py image.png --api --contour -o out.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
