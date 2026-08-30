#!/bin/bash

# Zynkbot Tauri App - Linux Startup Script
# This starts the Rust-powered Tauri desktop app (NOT the old Python backend)
# Usage: ./START_ZYNKBOT.sh

set -e

echo "========================================="
echo "   Zynkbot - Rust/Tauri Desktop App"
echo "========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/zynkbot_rust"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ ERROR: Node.js is not installed"
    echo ""
    echo "Install Node.js first:"
    echo "  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -"
    echo "  sudo apt-get install -y nodejs"
    exit 1
fi

# Sync npm dependencies every launch, not just when node_modules is absent: a pull
# that adds a dependency otherwise leaves node_modules stale and the frontend dies
# with an opaque "Module not found" error. This is a fast no-op once satisfied.
# Kept non-fatal (|| echo) so a network hiccup can't block an otherwise working install.
echo "📦 Syncing npm dependencies..."
npm install --no-audit --no-fund || echo "⚠️  npm install failed — continuing with existing node_modules"
echo ""

# Load Rust environment and export PATH for npm subprocesses
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Explicitly add Rust to PATH (important for npm subprocesses)
export PATH="$HOME/.cargo/bin:$PATH"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ ERROR: Rust is not installed"
    echo ""
    echo "Install Rust first:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source \$HOME/.cargo/env"
    exit 1
fi

# Create models/user directory if it doesn't exist
MODELS_DIR="$SCRIPT_DIR/zynkbot_rust/src-tauri/models/user"
if [ ! -d "$MODELS_DIR" ]; then
    echo "📁 Creating models directory..."
    mkdir -p "$MODELS_DIR"
    echo "   Created: $MODELS_DIR"
    echo "   You can download GGUF models here for local inference"
    echo ""
fi

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/zynkbot_rust/src-tauri/.env" ]; then
    echo "⚠️  WARNING: .env file not found"
    echo "   API features (Anthropic/OpenAI) will not work"
    echo "   Local offline features will still work"
    echo ""
fi


BINARY="$SCRIPT_DIR/zynkbot_rust/src-tauri/target/debug/app"
if [ ! -f "$BINARY" ]; then
    echo "⚠️  Binary not found — Rust backend has not been compiled yet."
    echo "   Run install.sh first. Starting anyway in 10 seconds... (Ctrl+C to cancel)"
    sleep 10
else
    echo "Starting Zynkbot..."
    sleep 1
fi
echo ""

# Cleanup function to kill port 3000 on exit
cleanup() {
    echo ""
    echo "🛑 Cleaning up..."
    # Kill any process using port 3000
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Detect CUDA and set features flag accordingly
TAURI_FEATURES=""
if command -v nvidia-smi &> /dev/null || ls /usr/lib/x86_64-linux-gnu/libcuda.so* &> /dev/null 2>&1; then
    if command -v nvcc &> /dev/null; then
        TAURI_FEATURES="--features cuda"
        echo "⚡ CUDA detected — building with GPU acceleration"

        # CUDA installed via apt lands in /usr/lib/x86_64-linux-gnu/ instead of the
        # standard /usr/local/cuda/ that llama-cpp-sys-2 and find_cuda_helper expect,
        # and nvcc needs -fPIC to compile CUDA code into a cdylib.
        #
        # These used to live in zynkbot_rust/src-tauri/.cargo/config.toml, but a cargo
        # [env] block applies on every platform — it was injecting CUDA_PATH=/usr and
        # -fPIC into Windows and macOS builds. Setting them here keeps them Linux-only.
        # ${VAR:-default} preserves the old behaviour: a cargo [env] entry without
        # force=true also yields to a value already present in the environment.
        export CUDA_PATH="${CUDA_PATH:-/usr}"
        export CUDA_LIBRARY_PATH="${CUDA_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu}"
        export CUDAFLAGS="${CUDAFLAGS:--Xcompiler -fPIC}"
    fi
fi

# Start Tauri app (Rust backend runs automatically inside Tauri)
echo "🚀 Starting Tauri app with Rust backend..."
npm run tauri -- dev $TAURI_FEATURES
