# Zynkbot - Windows Installation Guide

**Automated installation for Windows 10/11**

---

## Quick Start 

```batch
# 1. Install Visual Studio 2022 Build Tools with the "Desktop development
#    with C++" workload, then reboot if prompted:
#    https://aka.ms/vs/17/release/vs_BuildTools.exe

# 2. Clone the repository
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot

# 3. Run installer as Administrator
# Right-click install.bat -> "Run as administrator"
install.bat

# 4. Start Zynkbot
START_ZYNKBOT.bat
```

**That's it!** Aside from the C++ Build Tools — which needs a reboot, so it can't be auto-installed — the installer handles everything automatically.

---

> **⚠️ GPU Performance Note**
>
> **Local AI models run significantly slower without GPU acceleration.** If you have an NVIDIA GPU, installing CUDA will dramatically improve performance. The installer automatically detects your GPU and enables CUDA if the toolkit is installed.
>
> **To enable GPU acceleration:**
>
> **Option 1: Install CUDA before Zynkbot (Recommended)**
> 1. Download and install CUDA Toolkit 12.6: https://developer.nvidia.com/cuda-12-6-0-download-archive
> 2. Reboot your system
> 3. Run `install.bat` — it will automatically detect CUDA and build with GPU support
>
> **Option 2: Add CUDA to existing CPU-only installation**
> 1. Run `uninstall.bat` from the zynkbot directory (removes Zynkbot and Rust)
> 2. Download and install CUDA Toolkit 12.6: https://developer.nvidia.com/cuda-12-6-0-download-archive
> 3. Reboot your system
> 4. Run "install.bat" again — it will rebuild everything with GPU support
>
> **Why full reinstall?** Rust compiles different binary code for CPU vs GPU mode. Simply re-running install.bat after adding CUDA causes build cache conflicts. The uninstall script cleanly removes everything so the installer can rebuild properly with CUDA.
>
> If you don't have an NVIDIA GPU or choose to skip CUDA installation, Zynkbot will work fine in CPU mode — API-based models (Claude, GPT-4, etc.) are unaffected. Only local models will run slower.

---

## Table of Contents
1. [What Gets Installed](#what-gets-installed)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Starting Zynkbot](#starting-zynkbot)
5. [Troubleshooting](#troubleshooting)
6. [Manual Installation](#manual-installation-advanced)

---

## What Gets Installed

The `install.bat` script automatically installs and configures:

- ✅ **Chocolatey** - Windows package manager
- ✅ **Node.js** - JavaScript runtime for the frontend
- ✅ **Git** - Version control
- ✅ **wget** - File downloader for models
- ✅ **Rust** - detects existing installation or installs the Rust toolchain
- ✅ **CUDA / GPU acceleration** - detects NVIDIA GPU automatically; enables GPU acceleration if CUDA toolkit is present; provides download link if GPU is found without the toolkit
- ✅ **System Models** - Embeddings, safety classifier, entity extraction (required)
- ✅ **Environment Config** - Sets up .env file
- ✅ **npm Dependencies** - Frontend packages
- ✅ **Start Menu shortcut** - Adds Zynkbot to your Windows Start Menu

**Note:** No database server is required. Zynkbot uses an embedded SQLite database created automatically on first launch.

**Optional:**
- 📦 Local LLM models (Llama, Qwen, DeepSeek) - prompted during installation

**Time:** 15-40 minutes (depends on internet speed and model downloads)

---

## Prerequisites

**Three things are needed before installation:**

### 1. Git for Windows
- **Download:** https://git-scm.com/download/win
- **Why:** To clone the repository
- **Verify:** Open Command Prompt and run `git --version`

### 2. Visual Studio 2022 Build Tools — "Desktop development with C++"
- **Download:** https://aka.ms/vs/17/release/vs_BuildTools.exe
- **How:** Check the **"Desktop development with C++"** workload, install, and reboot if prompted.
- **Why:** Building Zynkbot compiles a C++ component (llama.cpp) for local models. It's the only dependency the installer can't add automatically (it needs a reboot); `install.bat` stops with instructions if it's missing.

### 3. Administrator Access
- **Why:** The installer needs admin rights to install software and add the antivirus exclusion
- **How:** Right-click `install.bat` → "Run as administrator"

Everything else is installed automatically.

---

## Installation Steps

### Step 1: Clone the Repository

Open **Command Prompt** and run:

```batch
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot
```

### Step 2: Windows Defender Warning

> **⚠️ Windows Defender and antivirus software will likely interfere with installation.**
>
> The Rust installer, development tools, and build outputs are routinely flagged or quarantined. The installer attempts to add a Windows Defender exclusion for the project folder automatically — but if that fails, or if you use third-party antivirus (Norton, McAfee, etc.), you will need to resolve those conflicts yourself.
>
> **If the installer fails or files go missing:**
> 1. Check your antivirus quarantine and restore any flagged files
> 2. Run `docs\troubleshooting\FIX_WINDOWS_DEFENDER.bat` as Administrator — adds exclusions for the project folder, Rust, and Git
> 3. Re-run `install.bat`

### Step 3: Run the Installer

**IMPORTANT:** Must run as Administrator!

**Option A: Right-click method (recommended)**
1. Find `install.bat` in File Explorer
2. Right-click `install.bat`
3. Select **"Run as administrator"**
4. Click "Yes" when prompted

**Option B: Administrator Command Prompt**
1. Press `Win + X`
2. Select "Command Prompt (Admin)" or "PowerShell (Admin)"
3. Navigate to project: `cd path\to\zynkbot`
4. Run: `install.bat`

### Step 4: Follow Installation Prompts

The installer will:

1. **Check/Install Dependencies** (~5 min)
   - Installs Chocolatey, Node.js, Git, wget
   - Detects or installs Rust toolchain — skipped automatically if already installed
   - You may need to press Enter to accept Rust defaults if installing fresh

2. **Configure Environment** (~1 min)
   - Creates `.env` file

3. **Install Dependencies** (~2 min)
   - Runs `npm install`

4. **Download System Models** (~5 min)
   - all-MiniLM-L6-v2 (embeddings)
   - toxic-bert (safety classifier)
   - bert-base-NER (entity extraction)

5. **GPU / CUDA detection** (automatic)
   - NVIDIA GPU detected + CUDA toolkit installed → GPU acceleration enabled automatically
   - NVIDIA GPU detected but no CUDA toolkit → installer prints a download link; Zynkbot builds in CPU mode until you install the toolkit and re-run `install.bat`
   - No NVIDIA GPU → builds in CPU mode

6. **Optional: Download LLM Models** (~5-15 min)
   - You'll be prompted to choose:
     - `1` - Qwen3 8B (5.0GB) - Best all-around; recommended for new users
     - `2` - DeepSeek R1 Distill Llama 8B (4.7GB) - Reasoning model; analytical tasks
     - `3` - Llama 3.1 8B Lexi Uncensored (4.9GB) - Creative, unfiltered responses
   - Enter numbers space-separated: `1 2 3` or just `1`
   - Press Enter to skip

### Step 5: Installation Complete! 🎉

When you see:
```
=========================================
   ✅ Installation Complete!
=========================================
```

You're ready to start Zynkbot!

---

## Starting Zynkbot

After installation, start Zynkbot anytime by:

**Double-click:** `START_ZYNKBOT.bat`

**Or from Command Prompt:**
```batch
cd zynkbot
START_ZYNKBOT.bat
```

**First Launch:**
- Rust backend compiles (~5 min first time)
- React dev server starts
- Desktop window opens automatically

**Subsequent Launches:**
- Much faster (~5-30 seconds)
- Hot reload enabled for development

**To Stop:**
- Close the Zynkbot window
- Or press `Ctrl+C` in the terminal

---

## Troubleshooting

### Installation Issues

**Problem: "This script requires Administrator privileges"**
- **Solution:** Right-click `install.bat` → "Run as administrator"

**Problem: Chocolatey installation fails**
- **Solution:**
  1. Open PowerShell as Administrator
  2. Run: `Set-ExecutionPolicy Bypass -Scope Process`
  3. Try installer again

**Problem: Rust installation hangs**
- **Solution:**
  - Press Enter when prompted to proceed with default installation
  - If stuck, close and run installer again (it will skip already-installed components)

**Problem: Model download fails**
- **Solution:**
  - Check internet connection
  - Run installer again (it skips already-downloaded models)
  - Or download models manually later (see [MODELS.md](MODELS.md))

### Startup Issues

**Problem: "Port 3000 already in use"**
- **Solution:** Kill the process:
  ```batch
  # Find the process
  netstat -ano | findstr :3000
  # Kill it (replace PID with actual number)
  taskkill /F /PID <PID>
  ```

**Problem: "cargo: command not found" after Rust installation**
- **Solution:**
  1. Close all Command Prompts
  2. Open a new Command Prompt
  3. Run `START_ZYNKBOT.bat` again

**Problem: Window opens but shows blank/white screen**
- **Solution:**
  1. Wait 30 seconds (first compile takes time)
  2. Check terminal for errors
  3. If still blank, press `Ctrl+R` in the window to refresh

---

## Manual Installation (Advanced)

If you prefer manual installation or need to troubleshoot:

### 1. Install Dependencies Manually

```batch
# Install Chocolatey (PowerShell as Admin)
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install packages
choco install -y nodejs git wget llvm cmake

# Install Visual Studio 2022 Build Tools SEPARATELY, directly from Microsoft:
#   https://aka.ms/vs/17/release/vs_BuildTools.exe
#   Select the "Desktop development with C++" workload, then reboot.
# (Don't use Chocolatey's visualstudio2022buildtools package — it leaves the
#  install incomplete until a reboot, which makes CMake fail to find a usable
#  Visual Studio instance.)

# Install Rust
# Download rustup-init.exe from https://rustup.rs/ and run it
```

### 2. Configure Environment

Create `zynkbot_rust\src-tauri\.env`:

```env
ZYNK_MODEL_BACKEND=local
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
XAI_API_KEY=
ZYNK_CONTAINMENT_MODE=guardian
ZYNKSYNC_AUTO_SYNC=true
ZYNKSYNC_SYNC_INTERVAL=60
```

### 3. Install Dependencies

```batch
cd zynkbot_rust
npm install
```

### 4. Download Models

See [MODELS.md](MODELS.md) for manual model download instructions.

---

## After Installation

### Add API Keys (Optional)

1. Start Zynkbot
2. Click ⚙️ **Settings** icon
3. Click **API Keys**
4. Add your API keys:
   - OpenAI (for GPT models)
   - Anthropic (for Claude models)
   - xAI (for Grok models)

**Not required** - Zynkbot works offline with local models!

### Complete Onboarding

1. Click 🎯 **"Get to Know You"** button
2. Answer personalization questions
3. Zynkbot learns your preferences

### Add Documents to Knowledge Base

1. Settings → **Knowledge Base**
2. Click **"Manage Documents"**
3. Add your documents (.txt, .md, .json, code files — PDF support coming soon)
4. Click 📚 **KB button** in chat to search them

---

## System Requirements

> These are estimated minimums based on typical usage patterns — not formally benchmarked.

**Minimum:**
- Windows 10 (64-bit) or Windows 11
- 8 GB RAM
- 10 GB free disk space (without local models)
- Internet connection (for installation only)

**Recommended:**
- Windows 11
- 16 GB RAM
- 20 GB free disk space (with local models)
- SSD for better performance
- NVIDIA GPU with CUDA (optional, for faster local LLM inference)

**GPU Acceleration (NVIDIA CUDA):**

The installer automatically detects your GPU and enables CUDA if the toolkit is present — no manual configuration needed. If you have an NVIDIA GPU but the installer built in CPU mode, install the CUDA toolkit and re-run `install.bat`:

- [Download NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

---

## What's Different from Linux?

The Windows installation is functionally identical to Linux:

- ✅ Same features (memory, KB, sync, safety)
- ✅ Same pure Rust backend (Candle framework)
- ✅ Same database (SQLite, embedded)
- ✅ Same models (embeddings, safety, NER)
- ✅ Same UI (React + Tauri)

**Only difference:** Installation uses `.bat` files instead of `.sh` files

---

## Need Help?

- **Installation Issues:** See [Troubleshooting](#troubleshooting) above
- **Architecture Questions:** See [ARCHITECTURE_COMPREHENSIVE.md](architecture_and_development/ARCHITECTURE_COMPREHENSIVE.md)
- **Model Information:** See [MODELS.md](MODELS.md)
- **General Questions:** Open an issue on GitHub

---

## Next Steps

After installation:

1. ✅ Start Zynkbot (`START_ZYNKBOT.bat`)
2. 🔑 Add API keys (optional - Settings → API Keys)
3. 🎯 Complete onboarding ("Get to Know You" button)
4. 📚 Add documents to Knowledge Base (optional)
5. 💬 Start chatting!

**Welcome to Zynkbot!** 🎉
