# Zynkbot Installation and Setup

## Recommended: Use the Automated Installer

The easiest way to install Zynkbot is with the provided installer, which handles all dependencies automatically.

**Linux:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```
Double-click install.bat (or right-click → Run as administrator)
```

The installer handles: SQLite database setup, schema initialization, ML model downloads (embeddings, NER, safety classifier), and optional GGUF model downloads.

After installation, start Zynkbot with `START_ZYNKBOT.sh` (Linux) or `START_ZYNKBOT.bat` (Windows).

---

## Manual Installation

If you prefer to set up manually or are contributing to development:

### System Requirements

**Minimum:**
- OS: Windows 10/11 or Linux (Ubuntu 20.04+)
- RAM: 8GB (16GB recommended for 7B local models)
- Storage: 10GB free (more for local models)
- CPU: 4+ cores recommended

**Recommended:**
- RAM: 16GB+ for 7B models
- GPU: NVIDIA with CUDA support (optional, for faster local inference)
- SSD with 50GB+ free space

### Prerequisites

**1. SQLite**

SQLite is embedded — no separate installation required. The database file is created automatically on first launch at:
- Linux: `~/.local/share/zynkbot/zynkbot.db`
- Windows: `%LOCALAPPDATA%\zynkbot\zynkbot.db`

**2. Rust toolchain**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**3. Node.js 16+**
```bash
# Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

Download from nodejs.org on Windows.

**4. C++ build toolchain** (for local GGUF models)

Zynkbot compiles llama.cpp from source to run local models, which requires a C++ compiler and CMake:
- **Windows:** Visual Studio 2022 Build Tools with the "Desktop development with C++" workload (https://aka.ms/vs/17/release/vs_BuildTools.exe). Reboot after installing. This is a manual prerequisite — the installer does not add it automatically because it requires a reboot.
- **Linux:** `build-essential` (Ubuntu/Debian) or `base-devel` (Arch) — gcc, g++, make — plus `cmake`.

### Setup Steps

**1. Clone the repository**
```bash
git clone https://github.com/MSkill1/zynkbot.git
cd zynkbot
```

**2. Apply the database schema**

The schema is applied automatically on first launch. If you need to apply it manually (e.g., for development):
```bash
sqlite3 ~/.local/share/zynkbot/zynkbot.db < scripts/db/complete_fresh_install_schema.sql
```

**3. Configure environment**

Create `zynkbot_rust/src-tauri/.env`:
```env
DATABASE_URL=sqlite:zynkbot.db

# Optional: API keys for cloud models
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
XAI_API_KEY=your_key_here
```

**4. Install Node dependencies**
```bash
cd zynkbot_rust
npm install
```

**5. Download ML models**

The embedding model (all-MiniLM-L6-v2) and NER model (bert-base-NER) are downloaded automatically on first run from HuggingFace. No manual step required.

**6. Download a local GGUF model (optional)**

Place any GGUF model file in `zynkbot_rust/src-tauri/models/user/`. Recommended models:
- DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf (~4.7GB, reasoning-distilled)
- Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf (~4.9GB, creative, unfiltered)
- Qwen3-8B-Q4_K_M.gguf (~5.0GB, coding and instruction-following)

**7. Run in development mode**
```bash
cd zynkbot_rust
npm run tauri dev
```

**Production build:**
```bash
npm run tauri build
# Output: src-tauri/target/release/bundle/
```

---

## First Launch

1. Zynkbot automatically seeds its system memories on first launch
2. The sample Knowledge Base document is copied to your KB folder automatically
3. Complete the onboarding questionnaire to build your initial profile
4. Select a model from the dropdown and start chatting

---

## Troubleshooting

**"Database connection failed"**
- The SQLite database file is created automatically — check that the app data directory exists: `~/.local/share/zynkbot/` on Linux, `%LOCALAPPDATA%\zynkbot\` on Windows
- Check your `.env` has the correct `DATABASE_URL=sqlite:zynkbot.db`
- Verify the database file exists: `ls ~/.local/share/zynkbot/zynkbot.db`

**"Model not found"**
- Ensure at least one `.gguf` file is in the models folder, OR configure an API key
- Check file permissions on the models folder

**"Failed to seed system memories"**
- Check database connection
- Ensure the schema was applied: `sqlite3 ~/.local/share/zynkbot/zynkbot.db ".tables"`

**"Embedding model not loading"**
- The all-MiniLM-L6-v2 model downloads from HuggingFace on first run
- Check internet connection on first launch
- Model is cached at `~/.cache/huggingface/` after download

**Slow local model inference**
- Use a smaller/more quantized model (Q4_K_M is a good balance)
- GPU acceleration (CUDA) significantly improves speed if available
- API models (Claude, GPT, Grok) are faster than local on most hardware

---

## Updating

```bash
git pull origin main
cd zynkbot_rust && npm install
# Re-apply any new schema changes if noted in CHANGELOG.md
```
