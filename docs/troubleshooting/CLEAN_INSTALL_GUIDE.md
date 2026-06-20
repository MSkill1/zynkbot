# Zynkbot Clean Installation Guide

**Last Updated:** 2026-03-31
**Status:** Production Release
**Tested On:** Ubuntu 24.04, Windows 11, Arch Linux

This guide covers installing Zynkbot from scratch on a clean system.

---

## 🚀 Quick Start (Recommended)

### Windows

```bash
# Clone repository
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot

# Run automated installer (as Administrator)
install.bat
```

### Linux

```bash
# Clone repository
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot

# Run automated installer
chmod +x install.sh
./install.sh
```

The automated installers handle everything:
- ✅ System dependencies
- ✅ Environment configuration
- ✅ System model downloads
- ✅ Optional user LLM downloads

**Note:** No database server is required. Zynkbot uses an embedded SQLite database created automatically on first launch.

**Estimated time:** 15-20 minutes (first time)

---

## Manual Installation (Advanced Users)

If you prefer manual control or need to troubleshoot, follow these platform-specific guides:

### Prerequisites

- **Rust**: 1.77.2 or higher
- **Node.js**: 18 or higher
- **Git**: For cloning the repository
- **Build tools**: Platform-specific (see below)

---

## Windows Manual Installation

### Step 1: Install Prerequisites

**1.1 Rust:**
```batch
# Download and run rustup-init.exe from https://rustup.rs/
# Or via winget:
winget install Rustlang.Rustup
```

**1.2 Node.js:**
```batch
# Download from https://nodejs.org/
# Or via winget:
winget install OpenJS.NodeJS.LTS
```

**1.3 Build Tools:**
```batch
# Visual Studio Build Tools 2019 or newer
winget install Microsoft.VisualStudio.2022.BuildTools

# During install, select:
# - Desktop development with C++
# - Windows 10/11 SDK
```

**1.4 LLVM (for Candle):**
```batch
# Download from https://releases.llvm.org/
# Or via winget:
winget install LLVM.LLVM
```

**1.5 CMake:**
```batch
winget install Kitware.CMake
```

### Step 2: Configure Environment

```batch
cd zynkbot_rust\src-tauri

# Edit .env with your configuration
notepad .env
```

Example `.env`:
```
ZYNK_MODEL_BACKEND=local
ZYNK_CONTAINMENT_MODE=guardian
ZYNKSYNC_AUTO_SYNC=true
ZYNKSYNC_SYNC_INTERVAL=60
```

### Step 3: Build Application

```batch
cd zynkbot_rust

# Install Node dependencies
npm install

# Build in development mode
npm run tauri:dev
```

### Step 4: Start Zynkbot

```batch
# From repository root
START_ZYNKBOT.bat
```

---

## Linux Manual Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install -y \
    curl wget git build-essential cmake clang libclang-dev \
    pkg-config libssl-dev libwebkit2gtk-4.1-dev libgtk-3-dev \
    libayatana-appindicator3-dev librsvg2-dev nodejs npm
```

**Arch Linux:**
```bash
sudo pacman -S curl wget git base-devel cmake clang pkgconf \
    openssl webkit2gtk gtk3 libappindicator-gtk3 librsvg \
    nodejs npm
```

**Fedora:**
```bash
sudo dnf install curl wget git gcc cmake clang pkgconfig \
    openssl-devel webkit2gtk4.1-devel gtk3-devel \
    libappindicator-gtk3-devel librsvg2-devel nodejs npm
```

### Step 2: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Step 3: Configure Environment

```bash
cd ~/zynkbot/zynkbot_rust/src-tauri

# Create .env file
cat > .env << 'EOF'
# LLM Backend
ZYNK_MODEL_BACKEND=local

# Safety
ZYNK_CONTAINMENT_MODE=guardian

# API Keys (optional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=

# ZynkSync
ZYNKSYNC_AUTO_SYNC=true
ZYNKSYNC_SYNC_INTERVAL=60
EOF
```

### Step 4: Build Application

```bash
cd ~/zynkbot/zynkbot_rust

# Install Node dependencies
npm install

# Build in development mode
npm run tauri:dev
```

**First run:**
- Compiles Rust backend (several minutes first time)
- Bundles React frontend
- Downloads system models if not present
- Opens Zynkbot window

**Subsequent runs:**
- Start in seconds (already compiled)

### Step 5: Start Zynkbot

```bash
cd ~/zynkbot
./START_ZYNKBOT.sh
```

---

## Post-Installation

### Verify Installation

After starting Zynkbot, verify:

1. **Application Opens** - Native window appears
2. **No Console Errors** - Check terminal for model load and startup messages
3. **Settings Accessible** - Click gear icon to open settings
4. **Memory Manager Works** - Click "Memory" to browse memories
5. **Chat Functional** - Send a test message

### Download Local Models (Optional)

```bash
cd zynkbot_rust/models/user

# Llama 3.2 3B (Recommended - 2.0GB)
curl -L -O https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Qwen 2.5 7B (Best coding - 4.4GB)
curl -L -O https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# Dolphin Mistral 7B (Creative - 4.1GB)
curl -L -O https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-GGUF/resolve/main/dolphin-2.6-mistral-7b.Q4_K_M.gguf
```

Models are auto-discovered on restart.

### Add API Keys (Optional)

1. Click ⚙️ Settings
2. Go to 🔑 API Keys tab
3. Add:
   - OpenAI API key (for GPT-4, GPT-3.5)
   - Anthropic API key (for Claude)
   - xAI API key (for Grok)

---

## Troubleshooting

### Model Download Fails

**Error:** "Failed to download model"

```bash
# Manual download
cd zynkbot_rust/models/system

# all-MiniLM-L6-v2 (embeddings)
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# toxic-bert (safety)
git clone https://huggingface.co/unitary/toxic-bert

# bert-base-NER (entity extraction)
git clone https://huggingface.co/dslim/bert-base-NER
```

### Antivirus Blocking (Windows)

**Error:** Installation freezes or fails

```batch
# Add Windows Defender exclusions
FIX_WINDOWS_DEFENDER.bat

# Or manually exclude:
# C:\Users\<YourUser>\zynkbot\
# C:\Users\<YourUser>\.cargo\
```

---

## What Gets Installed

**System Packages:**
- Build tools (gcc/MSVC, cmake, clang)
- GUI libraries (webkit2gtk, gtk3)

**Development Tools:**
- Rust toolchain (~1.5GB)
- Node.js packages (~500MB)

**Zynkbot Files:**
- Project directory (~2GB after first build)
- System models (~600MB)
- SQLite database at `~/.local/share/zynkbot/zynkbot.db` (Linux) or `%LOCALAPPDATA%\zynkbot\zynkbot.db` (Windows) — created automatically on first launch; grows with usage

**Total Disk Usage:** ~4-5GB

---

## Key Differences from Old Python Version

This guide is for the **current Rust/Tauri implementation**. If you have documentation referencing the old Python/Flask backend:

| Old (Python) | New (Rust) |
|--------------|------------|
| `setup.sh` / `setup.bat` | `install.sh` / `install.bat` |
| `python zynk_router.py` | `./START_ZYNKBOT.sh` / `START_ZYNKBOT.bat` |
| Python virtual environment | No Python needed |
| `requirements.txt` | `Cargo.toml` |
| Flask API (port 5000) | Tauri IPC (no HTTP port) |
| ONNX models | Candle (pure Rust) |

---

## Next Steps

1. **Complete Onboarding** (recommended):
   - Click 🎯 Get to Know You button
   - Answer 6 questions to build your memory profile

2. **Test Features**:
   - Send test messages
   - Open Memory Manager
   - Try Ensemble Mode (with multiple models)
   - Upload document to Knowledge Base

3. **Setup Cross-Device Sync** (optional):
   - Settings → ZynkSync
   - Pair with another device
   - Enable auto-sync

4. **Explore Documentation**:
   - [ARCHITECTURE_COMPREHENSIVE.md](../architecture_and_development/ARCHITECTURE_COMPREHENSIVE.md) - System architecture
   - [DATABASE_SCHEMA.md](../architecture_and_development/DATABASE_SCHEMA.md) - Database structure
   - [MODELS.md](../MODELS.md) - ML models explained

---

## Support

**Issues:**
- GitHub: https://github.com/MSkill1/zynkbot/issues

**Documentation:**
- `/docs` directory in repository
- In-app: Click ℹ️ About button

**Community:**
- Check GitHub Discussions when available

---

**Installation complete!** 🎉

You now have a fully functional privacy-first AI assistant with persistent memory running entirely on your machine.
