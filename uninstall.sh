#!/bin/bash
# Zynkbot Uninstall Script for Linux
# Removes Zynkbot and optionally clears your memory database and Rust toolchain.

set -e

keep_open() {
    echo ""
    read -rp "Press Enter to close this window..." _
}
trap keep_open EXIT

if [ "$EUID" -eq 0 ]; then
    echo "Do not run this script as root. Run it as your normal user account."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================="
echo "   Zynkbot Uninstall"
echo "========================================="
echo ""
echo "This script will:"
echo "  - Stop any running Zynkbot processes"
echo "  - Remove the desktop entry"
echo "  - Optionally remove your memory database"
echo "  - Optionally remove the Rust toolchain"
echo "  - Optionally remove the Zynkbot project folder"
echo ""
echo "System packages installed by install.sh (curl, cmake, etc.) are"
echo "NOT removed — they may be used by other applications."
echo ""
read -rp "Continue? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
fi
echo ""

# ============================================
# Stop running processes
# ============================================
echo "Stopping any running Zynkbot processes..."
pkill -f "zynkbot_rust" 2>/dev/null || true
pkill -f "START_ZYNKBOT" 2>/dev/null || true
sleep 1
echo "Done."
echo ""

# ============================================
# Remove desktop entry
# ============================================
DESKTOP_FILE="$HOME/.local/share/applications/zynkbot.desktop"
if [ -f "$DESKTOP_FILE" ]; then
    rm -f "$DESKTOP_FILE"
    echo "Removed desktop entry."
else
    echo "No desktop entry found (already removed or never created)."
fi
echo ""

# ============================================
# Memory database (ask — this is personal data)
# ============================================
DB_DIR="$HOME/.local/share/zynkbot"
if [ -d "$DB_DIR" ]; then
    echo "Your memory database is stored at: $DB_DIR"
    echo "This contains all memories Zynkbot has learned about you."
    echo ""
    read -rp "Delete your memory database? This cannot be undone. [y/N] " del_db
    if [[ "$del_db" =~ ^[Yy]$ ]]; then
        rm -rf "$DB_DIR"
        echo "Memory database deleted."
    else
        echo "Memory database kept at: $DB_DIR"
        echo "You can delete it manually at any time."
    fi
else
    echo "No memory database found."
fi
echo ""

# ============================================
# Rust toolchain (ask — may be used elsewhere)
# ============================================
if command -v rustup &> /dev/null; then
    echo "The Rust toolchain (rustup + cargo) is installed on this machine."
    echo "Rust may be used by other projects. Only remove it if you installed"
    echo "it solely for Zynkbot."
    echo ""
    read -rp "Remove the Rust toolchain? [y/N] " del_rust
    if [[ "$del_rust" =~ ^[Yy]$ ]]; then
        rustup self uninstall -y
        echo "Rust toolchain removed."
    else
        echo "Rust toolchain kept."
    fi
else
    echo "Rust toolchain not found (already removed or not installed)."
fi
echo ""

# ============================================
# Project folder (ask — contains models)
# ============================================
echo "The Zynkbot project folder is at: $SCRIPT_DIR"
echo "This contains the app, your downloaded models, and configuration."
echo ""
read -rp "Delete the entire project folder? [y/N] " del_proj
if [[ "$del_proj" =~ ^[Yy]$ ]]; then
    # We can't delete our own parent — schedule it
    PARENT="$(dirname "$SCRIPT_DIR")"
    PROJ_NAME="$(basename "$SCRIPT_DIR")"
    echo "Scheduling deletion of $SCRIPT_DIR ..."
    # Use nohup so the script can finish before the directory is removed
    nohup bash -c "sleep 2 && rm -rf '$SCRIPT_DIR'" > /dev/null 2>&1 &
    echo "Project folder will be deleted in a moment."
else
    echo "Project folder kept."
    echo "You can delete it manually: rm -rf $SCRIPT_DIR"
fi
echo ""

echo "========================================="
echo "   Zynkbot has been uninstalled."
echo "========================================="
echo ""
echo "Thank you for trying Zynkbot!"
echo "GitHub: https://github.com/MSkill1/zynkbot"
echo ""
