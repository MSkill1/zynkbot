#!/bin/bash
# Zynkbot One-Click Installation Script for Linux
# Tested on: Ubuntu 22.04+. Experimental support for Debian, Fedora, Arch.
# Usage: ./install.sh

set -e  # Exit on error

keep_open() {
    echo ""
    read -rp "Press Enter to close this window..." _
}
trap keep_open EXIT

# Block root execution — Rust and npm must install to the user's home directory
if [ "$EUID" -eq 0 ]; then
    echo "❌ Do not run this script with sudo or as root."
    echo "   Run it as your normal user account: ./install.sh"
    exit 1
fi

echo "========================================="
echo "   Zynkbot Automated Installation"
echo "========================================="
echo ""
echo "This script will:"
echo "  1. Install system dependencies"
echo "  2. Install Rust toolchain"
echo "  3. Detect GPU and configure CUDA (if available)"
echo "  4. Configure environment"
echo "  5. Install Node dependencies"
echo "  6. Create model directories"
echo "  7. Download system models (required)"
echo "  8. Download user LLM models (optional)"
echo ""
echo "Note: The database is SQLite — embedded in the app, no setup required."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

echo "📋 Detected: $PRETTY_NAME"
echo ""

# ============================================
# Step 1: Install System Dependencies
# ============================================
echo "========================================="
echo "Step 1: Installing System Dependencies"
echo "========================================="
echo ""

if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]] || [[ "$ID_LIKE" == *"ubuntu"* ]] || [[ "$ID_LIKE" == *"debian"* ]]; then
    echo "📦 Installing packages for Ubuntu/Debian/Mint..."
    sudo apt update || true  # Continue even if some repos have warnings
    sudo apt install -y \
        curl wget git build-essential cmake clang libclang-dev \
        pkg-config libssl-dev libwebkit2gtk-4.1-dev libgtk-3-dev \
        libayatana-appindicator3-dev librsvg2-dev file \
        nodejs npm \
        mesa-vulkan-drivers vulkan-tools libvulkan1 \
        || {
            echo "⚠️  Trying alternative webkit package..."
            sudo apt install -y libwebkit2gtk-4.0-dev
        }
elif [[ "$OS" == "fedora" ]]; then
    echo "📦 Installing packages for Fedora..."
    sudo dnf install -y \
        curl wget git gcc gcc-c++ cmake clang clang-devel \
        openssl-devel webkit2gtk4.1-devel gtk3-devel \
        libappindicator-gtk3-devel librsvg2-devel file \
        nodejs npm \
        mesa-vulkan-drivers vulkan-tools vulkan-loader-devel
elif [[ "$OS" == "arch" ]]; then
    echo "📦 Installing packages for Arch Linux..."
    sudo pacman -Syu --needed --noconfirm \
        curl wget git base-devel cmake clang openssl \
        webkit2gtk gtk3 libappindicator-gtk3 librsvg \
        nodejs npm \
        vulkan-icd-loader vulkan-tools
else
    echo "⚠️  Unsupported distribution: $OS"
    echo "Please install dependencies manually and re-run this script"
    exit 1
fi

echo "✅ System dependencies installed"
echo ""

# ============================================
# Step 2: Install Rust
# ============================================
echo "========================================="
echo "Step 2: Installing Rust Toolchain"
echo "========================================="
echo ""

if command -v cargo &> /dev/null; then
    echo "✅ Rust already installed: $(rustc --version)"
else
    echo "📥 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
    source "$HOME/.cargo/env" 2>/dev/null || true
    echo "✅ Rust installed: $(rustc --version)"
fi

# Ensure Rust is in PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi
echo ""

# ============================================
# Step 3: Detect GPU and Configure CUDA
# ============================================
echo "========================================="
echo "Step 3: Detecting GPU Hardware"
echo "========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Tell git to ignore install.sh's local edits to Cargo.toml on this clone.
# Cargo.toml is tracked (the project needs it), but install.sh mutates it
# based on GPU detection, and those local mutations should never be committed.
# Users who want to legitimately edit Cargo.toml deps can undo this with:
#   git update-index --no-skip-worktree zynkbot_rust/src-tauri/Cargo.toml
if [ -d "$SCRIPT_DIR/.git" ] && command -v git &> /dev/null; then
    (cd "$SCRIPT_DIR" && git update-index --skip-worktree zynkbot_rust/src-tauri/Cargo.toml 2>/dev/null) || true
fi

GPU_DETECTED=0
NVIDIA_GPU=0

if command -v nvidia-smi &> /dev/null; then
    NVIDIA_GPU=1
    echo "🎮 NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
elif ls /usr/lib/x86_64-linux-gnu/libcuda.so* &> /dev/null 2>&1; then
    NVIDIA_GPU=1
    echo "🎮 NVIDIA GPU detected (driver present, nvidia-smi not in PATH)"
fi

if [ "$NVIDIA_GPU" = "1" ]; then
    echo ""

    if command -v nvcc &> /dev/null; then
        echo "✅ CUDA toolkit found: $(nvcc --version | grep 'release' | awk '{print $5}' | tr -d ',')"
        echo "⚡ GPU acceleration will be enabled automatically when you run START_ZYNKBOT.sh"

        # Create /usr/local/cuda/lib64 symlink structure expected by the build system.
        # apt installs CUDA libraries to /usr/lib/x86_64-linux-gnu/ instead of the
        # standard /usr/local/cuda/ path that llama-cpp-sys-2 and find_cuda_helper use.
        if [ ! -d /usr/local/cuda/lib64 ]; then
            echo "   Creating /usr/local/cuda/lib64 symlinks for apt-installed CUDA..."
            sudo mkdir -p /usr/local/cuda/lib64
            sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudart_static.a  /usr/local/cuda/lib64/libcudart_static.a
            sudo ln -sf /usr/lib/x86_64-linux-gnu/libcudart.so.12      /usr/local/cuda/lib64/libcudart.so
            sudo ln -sf /usr/lib/x86_64-linux-gnu/libcublas_static.a   /usr/local/cuda/lib64/libcublas_static.a
            sudo ln -sf /usr/lib/x86_64-linux-gnu/libcublasLt_static.a /usr/local/cuda/lib64/libcublasLt_static.a
            sudo ln -sf /usr/lib/x86_64-linux-gnu/libculibos.a         /usr/local/cuda/lib64/libculibos.a
            sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so           /usr/local/cuda/lib64/libcuda.so
            echo "   ✅ CUDA symlinks created"
        fi

        GPU_DETECTED=1
    else
        echo ""
        echo "⚠️  NVIDIA GPU found, but CUDA toolkit (nvcc) is not installed."
        echo "   Zynkbot will run on CPU. To enable GPU acceleration, install the CUDA toolkit:"
        echo ""
        echo "   Ubuntu/Debian:  sudo apt install nvidia-cuda-toolkit"
        echo "   Fedora:         sudo dnf install cuda-toolkit"
        echo "   Arch:           sudo pacman -S cuda"
        echo "   All distros:    https://developer.nvidia.com/cuda-downloads"
        echo ""
        echo "   After installing the toolkit, re-run install.sh to set up symlinks, then"
        echo "   START_ZYNKBOT.sh will automatically build with GPU acceleration."
    fi
else
    echo "ℹ️  No NVIDIA GPU detected — building for CPU mode"
fi

echo ""

# ============================================
# Step 4: Configure Environment
# ============================================
echo "========================================="
echo "Step 4: Configuring Environment"
echo "========================================="
echo ""

ENV_FILE="$SCRIPT_DIR/zynkbot_rust/src-tauri/.env"

if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env file already exists, backing up..."
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
fi

echo "📝 Creating .env file..."
cat > "$ENV_FILE" << 'EOF'
# LLM Backend
ZYNK_MODEL_BACKEND=local
# LOCAL_MODEL_PATH is not needed — models are auto-discovered from models/user/

# API Keys (optional — add via the Settings UI after launching)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=

# Safety
ZYNK_CONTAINMENT_MODE=guardian

# ZynkSync
ZYNKSYNC_AUTO_SYNC=true
ZYNKSYNC_SYNC_INTERVAL=60
EOF

echo "✅ Environment configured"
echo ""

# ============================================
# Step 5: Install Node Dependencies
# ============================================
echo "========================================="
echo "Step 5: Installing Node Dependencies"
echo "========================================="
echo ""

cd "$SCRIPT_DIR/zynkbot_rust"
echo "📦 Running npm install..."
npm install

echo "✅ Node dependencies installed"
echo ""

# ============================================
# Step 6: Create Model Directories
# ============================================
echo "========================================="
echo "Step 6: Creating Model Directories"
echo "========================================="
echo ""

USER_MODELS_DIR="$SCRIPT_DIR/zynkbot_rust/src-tauri/models/user"
SYSTEM_MODELS_DIR="$SCRIPT_DIR/zynkbot_rust/src-tauri/models/system"

mkdir -p "$USER_MODELS_DIR"
echo "✅ User models directory: $USER_MODELS_DIR"

mkdir -p "$SYSTEM_MODELS_DIR"
echo "✅ System models directory: $SYSTEM_MODELS_DIR"
echo ""

# ============================================
# Step 7: Download System Models (Required)
# ============================================
echo "========================================="
echo "Step 7: Download System Models (Required)"
echo "========================================="
echo ""
echo "Downloading internal models for embeddings, safety, and entity extraction..."
echo ""

# Download a model from HuggingFace
download_hf_model() {
    local repo="$1"
    local target_dir="$2"
    local files=("${@:3}")

    echo "📥 Downloading $repo..."
    mkdir -p "$target_dir"

    for file in "${files[@]}"; do
        local url="https://huggingface.co/$repo/resolve/main/$file"
        local target="$target_dir/$file"

        if [ -f "$target" ]; then
            echo "   ✓ $file already exists, skipping"
        else
            echo "   Downloading $file..."
            wget -q --show-progress -O "$target" "$url" || {
                echo "   ✗ Failed to download $file"
                return 1
            }
            echo "   ✓ $file downloaded"
        fi
    done

    echo "✅ $repo downloaded successfully"
    return 0
}

# all-MiniLM-L6-v2 — semantic embeddings for memory search
download_hf_model \
    "sentence-transformers/all-MiniLM-L6-v2" \
    "$SYSTEM_MODELS_DIR/all-MiniLM-L6-v2" \
    "config.json" "tokenizer.json" "model.safetensors" || {
    echo "❌ Failed to download embeddings model"
    exit 1
}

echo ""

# toxic-bert — local safety classifier
download_hf_model \
    "unitary/toxic-bert" \
    "$SYSTEM_MODELS_DIR/toxic-bert" \
    "config.json" "vocab.txt" "model.safetensors" || {
    echo "❌ Failed to download safety classifier"
    exit 1
}

echo ""

# bert-base-NER — entity extraction for hybrid memory search
download_hf_model \
    "dslim/bert-base-NER" \
    "$SYSTEM_MODELS_DIR/bert-base-NER" \
    "config.json" "vocab.txt" "model.safetensors" || {
    echo "❌ Failed to download BERT NER model"
    exit 1
}

echo ""
echo "✅ All system models downloaded"
echo ""

# ============================================
# Step 8: Download User Models (Optional)
# ============================================
echo "========================================="
echo "Step 8: Download User Models (Optional)"
echo "========================================="
echo ""
echo "Would you like to download local LLM models for offline inference?"
echo "These are not required — you can use API models (OpenAI, Anthropic, xAI) instead."
echo ""
echo "Available models:"
echo "  1. Qwen3 8B (5.0GB)                     - Best all-around; recommended for new users"
echo "  2. DeepSeek R1 Distill Llama 8B (4.7GB) - Reasoning model; analytical tasks"
echo "  3. Llama 3.1 8B Lexi Uncensored (4.9GB) - Creative, unfiltered responses"
echo ""
echo "Enter model numbers to download (space-separated, e.g. '1 2 3'), or press Enter to skip."
echo ""
read -p "Model(s) to download: " model_choices

if [ -n "$model_choices" ]; then
    cd "$USER_MODELS_DIR"

    for choice in $model_choices; do
        case $choice in
            1)
                echo ""
                echo "📥 Downloading Qwen3 8B (5.0GB)..."
                if wget -c https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf -O Qwen3-8B-Q4_K_M.gguf; then
                    echo "✅ Qwen3 8B downloaded"
                else
                    echo "❌ Failed to download Qwen3 8B (check internet connection)"
                fi
                ;;
            2)
                echo ""
                echo "📥 Downloading DeepSeek R1 Distill Llama 8B (4.7GB)..."
                if wget -c https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf -O DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf; then
                    echo "✅ DeepSeek R1 Distill Llama 8B downloaded"
                else
                    echo "❌ Failed to download DeepSeek R1 Distill Llama 8B (check internet connection)"
                fi
                ;;
            3)
                echo ""
                echo "📥 Downloading Llama 3.1 8B Lexi Uncensored (4.9GB)..."
                if wget -c https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf -O Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf; then
                    echo "✅ Llama 3.1 8B Lexi Uncensored downloaded"
                else
                    echo "❌ Failed to download Llama 3.1 8B Lexi Uncensored (check internet connection)"
                fi
                ;;
            *)
                echo "⚠️  Invalid choice: $choice (skipped)"
                ;;
        esac
    done

    echo ""
    echo "✅ Model downloads complete"
    echo "   Models saved to: $USER_MODELS_DIR"
else
    echo "⏭️  Skipping model downloads"
    echo "   You can add GGUF models to $USER_MODELS_DIR at any time"
fi
echo ""

# ============================================
# Create Desktop Entry
# ============================================
echo "========================================="
echo "Creating Desktop Entry"
echo "========================================="
echo ""

mkdir -p "$HOME/.local/share/applications"
cat > "$HOME/.local/share/applications/zynkbot.desktop" << EOF
[Desktop Entry]
Type=Application
Name=Zynkbot
Comment=AI Assistant with Memory
Exec=$SCRIPT_DIR/START_ZYNKBOT.sh
Icon=$SCRIPT_DIR/zynkbot_rust/src-tauri/icons/icon.png
Terminal=true
Categories=Utility;Development;
EOF

echo "✅ Desktop entry created — Zynkbot now appears in your application menu"
echo ""

# ============================================
# Installation Complete
# ============================================
echo "========================================="
echo "   ✅ Installation Complete!"
echo "========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Start Zynkbot:"
echo "   cd $SCRIPT_DIR"
echo "   ./START_ZYNKBOT.sh"
echo ""
echo "   ⚠️  IMPORTANT: First launch compiles the Rust backend.
   This can take 10–15 minutes, or longer if CUDA is being set up.
   The build may appear to freeze (often around line 774) — this is normal.
   Do NOT close the terminal. Let it complete."
echo "   If you see 'cargo: command not found', restart your terminal"
echo "   or run: source \$HOME/.cargo/env"
echo ""
echo "2. Add API keys (optional, for cloud models):"
echo "   Click ⚙️ Settings → API Keys in the app"
echo "   - OpenAI, Anthropic, or xAI keys"
echo "   - Not required — local models work fully offline"
echo ""
echo "3. Complete onboarding:"
echo "   Click 🎯 Get to Know You to personalize your experience"
echo ""
echo "4. Add documents to Knowledge Base (optional):"
echo "   Settings → Knowledge Base → Upload Documents"
echo "   Supports: txt, md, json, code files"
echo ""
echo "For help, see: docs/LINUX_INSTALLATION_GUIDE.md"
echo ""
echo "========================================="
echo " 🎉 Ready to use Zynkbot!"
echo "========================================="
echo ""
