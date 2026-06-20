#!/bin/bash
# Quick script to test ZynkSync locally with two instances

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/zynkbot_rust"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          ZynkSync Local Testing - Two Instances           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if avahi-daemon is running (required for mDNS)
if ! systemctl is-active --quiet avahi-daemon 2>/dev/null; then
    echo "⚠️  Warning: avahi-daemon is not running"
    echo "   mDNS discovery requires avahi-daemon on Linux"
    echo ""
    echo "   To install and start:"
    echo "   sudo apt-get install avahi-daemon"
    echo "   sudo systemctl start avahi-daemon"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📋 Instructions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will open TWO terminal windows, each running a Zynkbot instance."
echo ""
echo "In each window:"
echo "  1. Wait for app to fully load"
echo "  2. Go to ZynkSync panel"
echo "  3. Click '▶ Start ZynkSync'"
echo "  4. Wait ~10 seconds"
echo "  5. Check 'Discovered Devices' section"
echo ""
echo "Both instances should discover each other!"
echo ""
echo "Press Ctrl+C in each terminal to stop."
echo ""
read -p "Ready to start? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "🚀 Starting Instance A..."
echo ""

# Launch first instance in new terminal
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "
        cd '$SCRIPT_DIR/zynkbot_rust'
        echo '╔═══════════════════════════════════════╗'
        echo '║       Zynkbot Instance A              ║'
        echo '╚═══════════════════════════════════════╝'
        echo ''
        echo 'Starting Zynkbot Instance A...'
        echo ''
        npm run tauri:dev
        echo ''
        echo 'Instance A stopped.'
        read -p 'Press Enter to close...'
    " &
elif command -v xterm &> /dev/null; then
    xterm -hold -e "
        cd '$SCRIPT_DIR/zynkbot_rust'
        echo '╔═══════════════════════════════════════╗'
        echo '║       Zynkbot Instance A              ║'
        echo '╚═══════════════════════════════════════╝'
        echo ''
        echo 'Starting Zynkbot Instance A...'
        echo ''
        npm run tauri:dev
    " &
else
    echo "❌ No supported terminal found (gnome-terminal or xterm required)"
    exit 1
fi

echo "⏳ Waiting 5 seconds before starting Instance B..."
sleep 5

echo ""
echo "🚀 Starting Instance B..."
echo ""

# Launch second instance in new terminal
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "
        cd '$SCRIPT_DIR/zynkbot_rust'
        echo '╔═══════════════════════════════════════╗'
        echo '║       Zynkbot Instance B              ║'
        echo '╚═══════════════════════════════════════╝'
        echo ''
        echo 'Starting Zynkbot Instance B...'
        echo ''
        npm run tauri:dev
        echo ''
        echo 'Instance B stopped.'
        read -p 'Press Enter to close...'
    " &
elif command -v xterm &> /dev/null; then
    xterm -hold -e "
        cd '$SCRIPT_DIR/zynkbot_rust'
        echo '╔═══════════════════════════════════════╗'
        echo '║       Zynkbot Instance B              ║'
        echo '╚═══════════════════════════════════════╝'
        echo ''
        echo 'Starting Zynkbot Instance B...'
        echo ''
        npm run tauri:dev
    " &
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Both instances starting!                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Next steps:"
echo "   1. Wait for both apps to load (watch the terminal windows)"
echo "   2. In each app: Go to ZynkSync panel"
echo "   3. In each app: Click '▶ Start ZynkSync'"
echo "   4. Check if they discover each other!"
echo ""
echo "📊 To verify:"
echo "   - Look for console messages: [ZynkSync] Discovered peer"
echo "   - Check 'Discovered Devices' count (should be 1 in each)"
echo ""
echo "🛑 To stop: Press Ctrl+C in each terminal window"
echo ""
echo "📖 Full guide: TEST_ZYNKSYNC_LOCAL.md"
echo ""
