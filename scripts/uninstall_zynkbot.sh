#!/bin/bash
# =============================================================================
# ZYNKBOT DATA UNINSTALL — Linux
# =============================================================================
# Removes Zynkbot's database, configuration, downloaded models, and build
# artifacts. Leaves Rust, Node.js, and all system packages in place —
# this machine will still be a working dev environment after running.
#
# Use this to wipe your Zynkbot data and start fresh without reinstalling
# system dependencies.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "   Zynkbot Data Uninstall — Linux"
echo "============================================="
echo ""
echo "This will remove:"
echo "  - Zynkbot SQLite database (~/.config/Zynkbot/)"
echo "  - Environment configuration (.env)"
echo "  - Downloaded system models (embeddings, NER, safety)"
echo "  - node_modules and frontend build"
echo "  - Rust build cache (src-tauri/target)"
echo ""
echo "This will NOT remove:"
echo "  - Rust toolchain"
echo "  - Node.js / npm"
echo "  - Any system packages"
echo "  - The project folder"
echo ""
read -p "Continue? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi
echo ""

# ============================================
# Step 1: Stop running Zynkbot processes
# ============================================
echo "============================================="
echo "Step 1: Stopping Zynkbot processes"
echo "============================================="

pkill -f "zynkbot_rust" 2>/dev/null && echo "[OK] Killed zynkbot_rust process" || echo "[--] No zynkbot_rust process running"
fuser -k 3000/tcp 2>/dev/null && echo "[OK] Stopped process on port 3000" || echo "[--] Nothing on port 3000"
fuser -k 57963/tcp 2>/dev/null && echo "[OK] Stopped process on port 57963" || echo "[--] Nothing on port 57963"
echo ""

# ============================================
# Step 2: Remove SQLite database
# ============================================
echo "============================================="
echo "Step 2: Removing SQLite database"
echo "============================================="

DB_DIR="$HOME/.config/Zynkbot"
if [ -d "$DB_DIR" ]; then
    rm -rf "$DB_DIR"
    echo "[OK] Removed database directory: $DB_DIR"
else
    echo "[--] No database directory found at $DB_DIR"
fi
echo ""

# ============================================
# Step 3: Remove environment configuration
# ============================================
echo "============================================="
echo "Step 3: Removing configuration"
echo "============================================="

ENV_FILE="$SCRIPT_DIR/zynkbot_rust/src-tauri/.env"
if [ -f "$ENV_FILE" ]; then
    rm "$ENV_FILE"
    echo "[OK] Removed .env"
else
    echo "[--] No .env found"
fi
echo ""

# ============================================
# Step 4: Remove downloaded models
# ============================================
echo "============================================="
echo "Step 4: Removing downloaded models"
echo "============================================="

MODELS_DIR="$SCRIPT_DIR/zynkbot_rust/src-tauri/models"
if [ -d "$MODELS_DIR" ]; then
    rm -rf "$MODELS_DIR"
    echo "[OK] Removed models directory"
else
    echo "[--] No models directory found"
fi
echo ""

# ============================================
# Step 5: Remove build artifacts
# ============================================
echo "============================================="
echo "Step 5: Removing build artifacts"
echo "============================================="

if [ -d "$SCRIPT_DIR/zynkbot_rust/node_modules" ]; then
    rm -rf "$SCRIPT_DIR/zynkbot_rust/node_modules"
    echo "[OK] Removed node_modules"
fi

if [ -d "$SCRIPT_DIR/zynkbot_rust/build" ]; then
    rm -rf "$SCRIPT_DIR/zynkbot_rust/build"
    echo "[OK] Removed frontend build"
fi

if [ -d "$SCRIPT_DIR/zynkbot_rust/src-tauri/target" ]; then
    echo "Removing Rust build cache (this may take a moment)..."
    rm -rf "$SCRIPT_DIR/zynkbot_rust/src-tauri/target"
    echo "[OK] Removed Rust target directory"
fi
echo ""

# ============================================
# Done
# ============================================
echo "============================================="
echo "   Zynkbot Data Uninstall Complete"
echo "============================================="
echo ""
echo "To reinstall:  ./install.sh"
echo ""
echo "To also remove Rust (standard Linux tools):"
echo "  rustup self uninstall"
echo ""
