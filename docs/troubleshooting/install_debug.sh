#!/bin/bash
# Zynkbot Installation Debug Script for Linux
# Runs the main installer with verbose logging to help diagnose failures.
# Usage: sudo ./docs/troubleshooting/install_debug.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="$PROJECT_ROOT/install_debug_$(date +%Y%m%d_%H%M%S).log"

echo "======================================="
echo "   Zynkbot Installation Debug Mode"
echo "======================================="
echo ""
echo "Logging all output to: $LOG_FILE"
echo ""

# Run installer with verbose output, tee to log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[DEBUG] Date: $(date)"
echo "[DEBUG] User: $(whoami)"
echo "[DEBUG] OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2)"
echo "[DEBUG] Kernel: $(uname -r)"
echo "[DEBUG] RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "[DEBUG] Disk free: $(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')"
echo ""

# Check internet connectivity
echo "[DEBUG] Testing internet connectivity..."
ping -c 1 github.com >/dev/null 2>&1 && echo "[OK] GitHub reachable" || echo "[WARN] GitHub unreachable"
ping -c 1 huggingface.co >/dev/null 2>&1 && echo "[OK] Hugging Face reachable" || echo "[WARN] Hugging Face unreachable"
echo ""

# Check for Rust
echo "[DEBUG] Checking Rust..."
if command -v cargo &>/dev/null; then
    echo "[OK] Rust: $(rustc --version)"
    echo "[OK] Cargo: $(cargo --version)"
else
    echo "[INFO] Rust not installed — installer will install it"
fi
echo ""

# Check for Node
echo "[DEBUG] Checking Node.js..."
if command -v node &>/dev/null; then
    echo "[OK] Node: $(node --version)"
    echo "[OK] npm: $(npm --version)"
else
    echo "[WARN] Node.js not found — will be installed by installer"
fi
echo ""

# Check GPU
echo "[DEBUG] Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    echo "[INFO] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if command -v nvcc &>/dev/null; then
        echo "[OK] CUDA toolkit present: $(nvcc --version | grep release)"
    else
        echo "[INFO] CUDA toolkit not found — installer will run in CPU mode"
    fi
else
    echo "[INFO] No NVIDIA GPU detected — CPU mode"
fi
echo ""

# Check disk space
echo "[DEBUG] Disk space requirements:"
echo "  Available: $(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')"
echo "  Minimum needed: ~5GB"
echo ""

echo "[DEBUG] Running main installer..."
echo "======================================="
echo ""

exec "$PROJECT_ROOT/install.sh"
