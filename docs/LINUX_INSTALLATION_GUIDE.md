# Zynkbot - Linux Installation Guide

**Tested on Ubuntu. Experimental support for Debian, Arch, and Fedora.**

---

## Binary Install (Recommended)

No compilation required. Download and install the `.deb` package:

```bash
sudo dpkg -i Zynkbot_0.9.0_amd64.deb
```

**[Download from GitHub Releases →](https://github.com/MSkill1/zynkbot/releases/latest)**

A first-run setup wizard handles all model downloads automatically.

> ⚠️ **Local models are CPU-only in pre-built binaries.** They work but can have 60+ second responses on some hardware. For optimized local model performance with CUDA support, clone and use the developer install below. API models are unaffected.

---

## Developer Install (from source)

## Quick Start (TL;DR)

```bash
# 1. Clone the repository
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot

# 2. Run the installer
sudo ./install.sh

# 3. Start Zynkbot
./START_ZYNKBOT.sh
```

**That's it!** The installer handles everything automatically.

**Time:** 15–40 minutes (depends on internet speed and model downloads)

---

## Table of Contents

1. [What Gets Installed](#what-gets-installed)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Starting Zynkbot](#starting-zynkbot)
5. [After Installation](#after-installation)
6. [System Requirements](#system-requirements)
7. [Performance Tips](#performance-tips)
8. [Uninstalling](#uninstalling)
9. [Troubleshooting](#troubleshooting)
10. [Manual Installation (Advanced)](#manual-installation-advanced)
11. [Getting Help](#getting-help)
12. [License](#license)

---

## What Gets Installed

The `install.sh` script automatically installs and configures:

- ✅ **System dependencies** — build tools, cmake, clang, Vulkan drivers, webkit2gtk
- ✅ **Node.js** — JavaScript runtime for the frontend
- ✅ **Rust** — detects existing installation or installs the Rust toolchain
- ✅ **CUDA / GPU acceleration** — detects NVIDIA GPU automatically; enables GPU acceleration if CUDA toolkit is present; provides install instructions if a GPU is found without the toolkit
- ✅ **System Models** — Embeddings, safety classifier, entity extraction (required)
- ✅ **Environment Config** — Sets up `.env` file
- ✅ **npm Dependencies** — Frontend packages
- ✅ **Desktop Entry** — Adds Zynkbot to your application menu

**Note:** No database server is required. Zynkbot uses an embedded SQLite database created automatically on first launch.

**Optional (prompted during installation):**
- 📦 Local LLM models (Llama, Qwen, Dolphin) for offline inference

---

## Prerequisites

Before starting, ensure you have:

- Ubuntu 20.04+ (tested and supported)
- Debian, Arch, or Fedora (experimental — installer has distro-specific branches but these have not been tested)
- At least 4GB RAM (8GB recommended)
- At least 15GB free disk space
- Internet connection
- `sudo` access

**That's all.** Everything else is installed automatically by `install.sh`.

---

## Installation Steps

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot
```

### Step 2: Run the Installer

```bash
sudo ./install.sh
```

> **Note:** `sudo` is required to install system packages. The installer will prompt you if it needs additional input.

### Step 3: Follow Installation Prompts

The installer will:

1. **Install system dependencies** (~3–5 min)
   - Build tools, cmake, clang, Vulkan drivers
   - Detects your distro (Ubuntu/Debian, Arch, Fedora) automatically

2. **Detect or install Rust** (~2–3 min)
   - Skipped automatically if Rust is already installed
   - If not found, installs via rustup — press Enter to accept defaults if prompted

3. **Configure environment** (~1 min)
   - Creates `.env` file with all required settings

4. **Install npm dependencies** (~2 min)
   - Runs `npm install`

5. **Download system models** (~5 min)
   - `all-MiniLM-L6-v2` — embeddings
   - `toxic-bert` — safety classifier
   - `bert-base-NER` — entity extraction

6. **Optional: Download local LLM models** (~5–15 min)
   - You'll be prompted to choose:
     - `1` — Qwen3 8B (5.0GB) — Best all-around; recommended for new users
     - `2` — DeepSeek R1 Distill Llama 8B (4.7GB) — Reasoning model; analytical tasks
     - `3` — Llama 3.1 8B Lexi Uncensored (4.9GB) — Creative, unfiltered responses
   - Enter numbers space-separated: `1 2 3` or just `1`
   - Press Enter to skip

7. **GPU / CUDA detection** (automatic)
   - NVIDIA GPU detected + CUDA toolkit installed → GPU acceleration enabled automatically
   - NVIDIA GPU detected but no CUDA toolkit → installer prints distro-specific install instructions; Zynkbot builds in CPU mode. To enable GPU acceleration, install the CUDA toolkit, then fully uninstall Zynkbot (including the Rust toolchain) and reinstall from scratch.
   - No NVIDIA GPU → builds in CPU mode

### Step 4: Installation Complete!

When you see:
```
=========================================
   ✅ Installation Complete!
=========================================
```

You're ready to start Zynkbot.

---

## Starting Zynkbot

After installation, start Zynkbot anytime with:

```bash
./START_ZYNKBOT.sh
```

**First launch:**
- Rust backend compiles (~3–5 min first time)
- React dev server starts
- Desktop window opens automatically

**Subsequent launches:**
- Much faster (~10–30 seconds)
- Everything is cached

**To stop:**
- Close the Zynkbot window
- Or press `Ctrl+C` in the terminal

---

## After Installation

### Add API Keys (Optional)

Zynkbot works offline with local models — API keys are not required. To add cloud provider keys:

1. Start Zynkbot
2. Click ⚙️ **Settings**
3. Click **API Keys**
4. Add keys for any providers you want:
   - Anthropic (Claude)
   - OpenAI (GPT-4)
   - xAI (Grok)

Get keys from:
- Anthropic: https://console.anthropic.com/settings/keys
- OpenAI: https://platform.openai.com/api-keys
- xAI: https://console.x.ai/

### Complete Onboarding

1. Click 🎯 **"Get to Know You"** button on first launch
2. Answer the personalization questions
3. Zynkbot learns your preferences for better responses

### Add Documents to Knowledge Base

1. Click ⚙️ Settings → **Knowledge Base**
2. Click **"Manage Documents"**
3. Add `.txt`, `.md`, `.json`, or code files (PDF support coming soon)
4. Click the 📚 **KB button** in the chat input to search them during conversation

### Next Steps

1. ✅ Start Zynkbot (`./START_ZYNKBOT.sh`)
2. 🔑 Add API keys (optional — Settings → API Keys)
3. 🎯 Complete onboarding ("Get to Know You" button)
4. 📚 Add documents to Knowledge Base (optional)
5. 💬 Start chatting — try "What can you do?"

---

## System Requirements

> These are estimated minimums based on typical usage patterns — not formally benchmarked.

**Minimum:**
- CPU: Dual-core 2GHz+
- RAM: 4GB
- Storage: 10GB free
- OS: Ubuntu 20.04+

**Recommended:**
- CPU: Quad-core 3GHz+
- RAM: 8GB
- Storage: 20GB free (for local models)
- OS: Ubuntu 22.04+

**For local LLM models:**
- RAM: 8GB+ (16GB recommended for 7B models)
- CPU: AVX2 support (most modern CPUs)
- GPU: NVIDIA GPU with CUDA toolkit gives 10–100x faster inference (optional but recommended)

**GPU Acceleration (NVIDIA CUDA):**

The installer automatically detects your GPU and enables CUDA if the toolkit is present — no manual configuration needed. **CUDA must be installed before running `install.sh`.** If you installed Zynkbot without CUDA and want GPU acceleration, you must:

1. Run `./uninstall.sh` and choose **yes** to removing the Rust toolchain (required — the binary was compiled without CUDA support)
2. Install the CUDA toolkit for your distro:

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Fedora
sudo dnf install cuda-toolkit

# Arch
sudo pacman -S cuda

# Or download from NVIDIA directly:
# https://developer.nvidia.com/cuda-downloads
```

3. Clone the repo fresh and run `./install.sh` again — the installer will detect CUDA and build with GPU support.

---

## Performance Tips

### Speed Up Compilation

Use `sccache` to cache Rust builds — dramatically speeds up recompiles after the first build:

```bash
cargo install sccache
echo 'export RUSTC_WRAPPER=sccache' >> ~/.bashrc
source ~/.bashrc
```

---

## Uninstalling

To completely remove Zynkbot:

```bash
# Remove application files
rm -rf ~/zynkbot

# Remove cache files
rm -rf ~/.cache/huggingface
rm -rf ~/.config/zynkbot

# Remove desktop launcher
rm -f ~/.local/share/applications/zynkbot.desktop

# Remove database (optional — this deletes all your memories and chat history)
rm -f ~/.local/share/zynkbot/zynkbot.db

# Remove Rust toolchain (optional — only if you don't use Rust for anything else)
rustup self uninstall
```

---

## Troubleshooting

### Blank White Window

**Symptoms:** App starts but shows only a blank white screen.

**Cause:** GPU driver issue — missing NVIDIA drivers or Vulkan problem.

**Logs may show:**
```
MESA: error: ZINK: vkQueueSubmit failed (VK_ERROR_DEVICE_LOST)
libEGL warning: DRI3 error: Could not get DRI3 device
```

**1. Check which renderer is in use:**
```bash
glxinfo | grep "OpenGL renderer"
# If it shows "llvmpipe" instead of your GPU name, drivers aren't working
```

**2. For NVIDIA GPUs:**
```bash
sudo ubuntu-drivers install
sudo reboot
nvidia-smi  # Should show your GPU after reboot
```

**3. For AMD/Intel GPUs:**
```bash
sudo apt install -y mesa-vulkan-drivers libgl1-mesa-dri
```

**4. Temporary workaround — software rendering:**
```bash
export LIBGL_ALWAYS_SOFTWARE=1
cd ~/zynkbot/zynkbot_rust
npm run tauri:dev
```

---

### "cargo: command not found"

**Cause:** Rust not in PATH after installation.

```bash
source $HOME/.cargo/env
```

To make this permanent, add it to your shell config:
```bash
echo 'source $HOME/.cargo/env' >> ~/.bashrc   # bash
echo 'source $HOME/.cargo/env' >> ~/.zshrc    # zsh
```

---

### Compilation Fails or System Freezes

**Cause:** Insufficient RAM — llama.cpp requires ~4GB to compile.

**Fix — add temporary swap space:**
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Retry compilation
cd ~/zynkbot/zynkbot_rust
npm run tauri:dev

# Remove swap after successful build
sudo swapoff /swapfile
sudo rm /swapfile
```

---

### Rust Compilation Errors

**Update Rust toolchain:**
```bash
rustup update stable
```

**Clean build cache and retry:**
```bash
cd ~/zynkbot/zynkbot_rust/src-tauri
cargo clean
cd ..
npm run tauri:dev
```

---

### "cmake not installed" Error

**Symptoms:** `failed to execute command: No such file or directory` referencing `cmake`

```bash
sudo apt install -y cmake
cmake --version
```

---

### "Unable to find libclang" Error

```bash
sudo apt install -y clang libclang-dev
```

---

### Webkit2gtk Error on Older Debian/Ubuntu

**Error:** `Package 'webkit2gtk-4.1' has no installation candidate`

```bash
sudo apt install libwebkit2gtk-4.0-dev
```

---

### Port 3000 Already in Use

```bash
lsof -ti:3000 | xargs kill -9
```

---

### ML Models Fail to Download

**Check internet and disk space:**
```bash
ping huggingface.co
df -h ~   # Need at least 2GB free
```

Models cache to `~/.cache/huggingface/hub/` — they re-download automatically on the next run if incomplete.

---

### Tauri Package Version Mismatch Warning

**Symptoms:**
```
Error Found version mismatched Tauri packages
tauri-plugin-dialog (v2.5.0) : @tauri-apps/plugin-dialog (v2.6.0)
```

This is a warning only — the app still works. To fix:
```bash
cd ~/zynkbot/zynkbot_rust
npm update @tauri-apps/plugin-dialog
```

---

## Manual Installation (Advanced)

If you prefer to install step-by-step rather than using `install.sh`, or need to debug a failed automated install:

### Step 1: System Dependencies

**Ubuntu / Debian / Pop!_OS / Linux Mint:**
```bash
sudo apt update
sudo apt install -y \
    curl wget git build-essential cmake clang libclang-dev \
    pkg-config libssl-dev libwebkit2gtk-4.1-dev libgtk-3-dev \
    libayatana-appindicator3-dev librsvg2-dev file \
    nodejs npm \
    mesa-vulkan-drivers vulkan-tools libvulkan1
```

**For NVIDIA GPU users:**
```bash
sudo ubuntu-drivers install
sudo reboot
nvidia-smi  # Verify after reboot
```

**Arch Linux / Manjaro:**
```bash
sudo pacman -S --needed \
    curl wget git base-devel cmake clang openssl \
    webkit2gtk-4.1 gtk3 libayatana-appindicator librsvg \
    nodejs npm
```

**Fedora / RHEL:**
```bash
sudo dnf install -y \
    curl wget git gcc gcc-c++ cmake clang clang-devel \
    openssl-devel webkit2gtk4.1-devel gtk3-devel \
    libappindicator-gtk3-devel librsvg2-devel \
    nodejs npm
```

**Verify versions:**
```bash
node --version    # Should be v18+
npm --version     # Should be v9+
cmake --version   # Should be 3.10+
g++ --version
```

If Node.js is too old:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Step 2: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Press 1 (or Enter) for default installation

source $HOME/.cargo/env
rustc --version   # Should show 1.70+
cargo --version
```

Add to shell startup:
```bash
echo 'source $HOME/.cargo/env' >> ~/.bashrc
```

### Step 3: Clone Repository

```bash
cd ~
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot
```

### Step 4: Configure Environment

```bash
cat > zynkbot_rust/src-tauri/.env << 'EOF'
# LLM Backend
ZYNK_MODEL_BACKEND=local

# API Keys (add via UI later)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=

# Safety
ZYNK_CONTAINMENT_MODE=guardian

# ZynkSync
ZYNKSYNC_AUTO_SYNC=true
ZYNKSYNC_SYNC_INTERVAL=60
EOF
```

### Step 5: Install Node Dependencies

```bash
cd zynkbot_rust
npm install
```

### Step 6: First Run

```bash
npm run tauri:dev
```

The first run compiles all Rust dependencies (3–5 minutes). Subsequent runs take ~10–30 seconds.

**Verify installation** — you should see in terminal logs:
```
[Candle Embeddings] ✅ Model loaded successfully from models/system/all-MiniLM-L6-v2
[Candle Safety] ✅ Safety classifier loaded successfully from models/system/toxic-bert
[HTTP Server] ✅ HTTP server started on port 57963
```

---

## Getting Help

**Documentation:**
- Features overview: [FEATURES.md](FEATURES.md)
- Architecture: [ARCHITECTURE_COMPREHENSIVE.md](architecture_and_development/ARCHITECTURE_COMPREHENSIVE.md)
- Model information: [MODELS.md](MODELS.md)
- Networking features: [NETWORKING_FEATURES.md](NETWORKING_FEATURES.md)

**Support:**
- GitHub Issues: https://github.com/MSkill1/zynkbot/issues
- Email: matt@containai.ai
- Include: your Linux distro, GPU model, and the full terminal error output

**Log locations:**
- Application logs: terminal output where you ran `npm run tauri:dev`
- HIPAA audit logs (if enabled): `logs/hipaa_audit/`

---

## License

**Zynkbot is dual-licensed:**

- **AGPL v3.0** — Free for non-commercial use, evaluation, and contribution
- **Commercial License** — Required for distribution or sale (contact: matt@containai.ai)

See the `LICENSE` file in the project root for full terms.

---

*Last Updated: May 2026 — Tested on Ubuntu 22.04*
