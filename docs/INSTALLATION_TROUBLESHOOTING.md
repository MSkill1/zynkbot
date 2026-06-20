# Zynkbot Troubleshooting Guide

**Common issues and solutions for Zynkbot installation and operation**

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Database Issues](#database-issues)
3. [Model Issues](#model-issues)
4. [Build/Compilation Errors](#buildcompilation-errors)
5. [Startup Issues](#startup-issues)
6. [Networking Issues](#networking-issues)
7. [Performance Issues](#performance-issues)
8. [Windows-Specific Issues](#windows-specific-issues)

---

## Installation Issues

### "This script requires Administrator privileges"

**Problem:** Installer won't run

**Solution:** Right-click `install.bat` → "Run as administrator"

---

### Chocolatey installation fails

**Problem:** PowerShell execution policy blocks Chocolatey

**Solution:**
1. Open PowerShell as Administrator
2. Run: `Set-ExecutionPolicy Bypass -Scope Process`
3. Try installer again

---

### Rust installation hangs

**Problem:** Rust installer appears stuck

**Solution:**
- Press **Enter** when prompted to proceed with default installation
- If stuck >5 minutes, close and run installer again (it will skip already-installed components)

---

### Model download fails

**Problem:** System models fail to download

**Solution:**
- Check internet connection
- Run installer again (it skips already-downloaded models)
- Or download models manually later (see [MODELS.md](MODELS.md))
- Check antivirus isn't blocking downloads

---

## Database Issues

Zynkbot uses SQLite — a local database file, not a server process. There is nothing to start or stop. If you have a database problem, you are dealing with a missing file, a permissions issue, or a corrupted database.

**Database file location:**
- Linux: `~/.local/share/zynkbot/zynkbot.db`
- Windows: `%LOCALAPPDATA%\zynkbot\zynkbot.db`

---

### Database file missing or not created

**Problem:** App starts but memory system is empty or errors on first run

**Solution:**
The database is created automatically on first launch. If it wasn't:
1. Check the directory exists and is writable:

   Linux:
   ```bash
   ls -la ~/.local/share/zynkbot/
   ```

   Windows:
   ```batch
   dir "%LOCALAPPDATA%\zynkbot\"
   ```

2. If the directory is missing, create it and relaunch:

   Linux:
   ```bash
   mkdir -p ~/.local/share/zynkbot
   ```

   Windows:
   ```batch
   mkdir "%LOCALAPPDATA%\zynkbot"
   ```

---

### "Database is locked"

**Problem:** Error message containing "database is locked"

**Solution:**
Another instance of Zynkbot is already running. SQLite allows only one writer at a time.
- Check for a running Zynkbot process and close it before launching another
- If no other instance is running, a previous session may have crashed and left a lock file:

  Linux:
  ```bash
  rm -f ~/.local/share/zynkbot/zynkbot.db-wal
  rm -f ~/.local/share/zynkbot/zynkbot.db-shm
  ```

  Windows:
  ```batch
  del "%LOCALAPPDATA%\zynkbot\zynkbot.db-wal"
  del "%LOCALAPPDATA%\zynkbot\zynkbot.db-shm"
  ```

---

### Database schema errors / missing tables

**Problem:** Error referencing a missing table, or the onboarding screen never appears

**Solution:**
The database schema is applied automatically via SQLx migrations on startup. If migrations failed:
1. Check terminal output for migration error messages
2. If the database is corrupted, the safest fix is to delete it and let Zynkbot recreate it:

   Linux:
   ```bash
   mv ~/.local/share/zynkbot/zynkbot.db ~/.local/share/zynkbot/zynkbot.db.bak
   ```

   Windows:
   ```batch
   rename "%LOCALAPPDATA%\zynkbot\zynkbot.db" zynkbot.db.bak
   ```

   Then relaunch Zynkbot. It will create a fresh database and run onboarding.

   ⚠️ This deletes all stored memories. Back up the file first if you want to try to recover it.

---

## Model Issues

### Local models not appearing in Settings

**Problem:** Placed `.gguf` files in models directory but don't show up

**Solution:**
1. Verify files are in correct directory: `zynkbot_rust/src-tauri/models/user/`
2. Restart Zynkbot completely (models are discovered at startup)
3. Check file permissions (should be readable)

**Download recommended models:**

Windows:
```batch
cd zynkbot_rust\src-tauri\models\user
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
wget https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf
wget https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

Linux:
```bash
cd zynkbot_rust/src-tauri/models/user
curl -L -O https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
curl -L -O https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf
curl -L -O https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

---

### System models missing

**Problem:** Embeddings or safety classifier not working

**Windows:**
```batch
cd zynkbot_rust\src-tauri
# System models should auto-download on first run
# If missing, check internet connection and restart app
```

**Linux:**
```bash
cd zynkbot_rust/src-tauri
# System models auto-download on first run
# If missing, check internet connection and restart app
./START_ZYNKBOT.sh
```

**Manual download locations:**
- Embeddings: `models/system/all-MiniLM-L6-v2/`
- Safety: `models/system/toxic-bert/`
- NER: `models/system/bert-base-NER/`

---

## Build/Compilation Errors

### "cargo: command not found"

**Problem:** Rust toolchain not installed or not in PATH

**Windows Solution:**
```batch
# Install Rust
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Close all Command Prompts and open a new one
# Verify installation
cargo --version
rustc --version
```

**Linux Solution:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
cargo --version
rustc --version
```

---

### SQLx compilation errors

**Problem:** `cargo build` fails with SQLx errors

The project uses SQLx offline mode — query metadata is cached in `.sqlx/` and committed to the repo, so you shouldn't normally need a live database to build. If the cache is stale or missing:

```bash
# Set DATABASE_URL to your local database file
export DATABASE_URL="sqlite://$HOME/.local/share/zynkbot/zynkbot.db"   # Linux
# set DATABASE_URL=sqlite:%LOCALAPPDATA%\zynkbot\zynkbot.db             # Windows

# Regenerate SQLx cache (run from src-tauri/)
cd zynkbot_rust/src-tauri
cargo sqlx prepare

# Then rebuild
cargo build --release
```

Commit the updated `.sqlx/` directory so offline builds continue to work.

---

### Missing build tools (Windows)

**Problem:** Compilation fails with "link.exe not found", "cl.exe not found", or "could not find any instance of Visual Studio"

**Solution:**

Zynkbot compiles llama.cpp from source for local models, which needs the Microsoft C++ compiler. Install **Visual Studio 2022 Build Tools** with the **"Desktop development with C++"** workload — `install.bat` does *not* install this automatically (it requires a reboot to finalize):

1. Download: https://aka.ms/vs/17/release/vs_BuildTools.exe
2. In the installer, check **"Desktop development with C++"**
3. Install, then **reboot if prompted** (required to finalize the install)
4. Re-run `install.bat`

If you've already installed it but the error persists, the install is likely incomplete. Open the **Visual Studio Installer**, choose **Repair** on the 2022 Build Tools, reboot, and try again.

---

### CMake build fails: "MSB1009: Project file does not exist" / install.vcxproj

**Problem:** The `llama-cpp-sys-2` build fails with:
```
MSBUILD : error MSB1009: Project file does not exist.
Switch: install.vcxproj
```

**Cause:** An earlier build attempt configured CMake *before* the C++ Build Tools were fully installed, leaving a stale build directory that never generated the Visual Studio project files. CMake then reports "already configured" and skips reconfiguration, so the build keeps failing.

**Solution:** Clear the stale build directory so CMake reconfigures from scratch, then rebuild:
```batch
cd zynkbot_rust\src-tauri
cargo clean -p llama-cpp-sys-2
cargo clean -p llama-cpp-2
```
Then start Zynkbot again. Make sure the C++ Build Tools are fully installed (and you've rebooted) first — see [Missing build tools (Windows)](#missing-build-tools-windows) above.

---

### Missing build essentials (Linux)

**Problem:** Compilation fails with missing libraries

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install build-essential libssl-dev pkg-config

# Arch/Manjaro
sudo pacman -S base-devel openssl pkg-config

# Fedora/RHEL
sudo dnf install gcc gcc-c++ make openssl-devel pkg-config
```

---

## Startup Issues

### "Port 3000 already in use"

**Problem:** React dev server can't bind to port

**Windows Solution:**
```batch
# Find the process
netstat -ano | findstr :3000

# Kill it (replace PID with actual number)
taskkill /F /PID <PID>
```

**Linux Solution:**
```bash
# Find the process
lsof -i :3000

# Kill it
kill -9 <PID>
```

---

### Window opens but shows blank/white screen

**Problem:** Tauri window appears but UI doesn't load

**Solution:**
1. Wait 30 seconds (first compile takes time)
2. Check terminal for errors
3. If still blank, press `Ctrl+R` in the window to refresh
4. Check if npm dependencies installed:
   ```batch
   cd zynkbot_rust
   npm install
   ```

---

### Application crashes on startup

**Problem:** App immediately closes or crashes

**Solution:**
1. Check terminal output for error messages
2. Verify the database file exists and is readable:

   Linux:
   ```bash
   ls -lh ~/.local/share/zynkbot/zynkbot.db
   ```

   Windows:
   ```batch
   dir "%LOCALAPPDATA%\zynkbot\zynkbot.db"
   ```

3. Run with verbose logging:
   ```batch
   set RUST_LOG=debug
   START_ZYNKBOT.bat
   ```

---

## Networking Issues

### ZynkSync devices can't connect

**Problem:** Devices can't pair or sync

**Basic troubleshooting:**
1. Verify both devices are on the same network (check IP addresses are in same subnet)
2. Verify pairing codes match and haven't expired (10-minute timeout)
3. Check you can ping the other device: `ping <other-device-ip>`

**If connection fails after pairing:**

ZynkSync uses port **57963**. Most home networks allow local LAN traffic by default, but if your firewall is blocking it:

**Windows Firewall:**
```batch
netsh advfirewall firewall add rule name="Zynkbot ZynkSync" dir=in action=allow protocol=TCP localport=57963
```

**Linux (UFW):**
```bash
sudo ufw allow 57963/tcp
```

**Linux (firewalld):**
```bash
sudo firewall-cmd --add-port=57963/tcp --permanent
sudo firewall-cmd --reload
```

---

### ZChat messages not delivering

**Problem:** Messages sent but not received

**Possible causes:**
- Devices not paired
- Backend not running on remote device
- Network connectivity issue
- Firewall blocking port 57963

**Solution:**
1. Check device pairing status in Settings → ZynkSync
2. Verify backend running on both devices
3. Manually trigger sync from Settings
4. Check firewall rules (see above)

---

### Ensemble mode fails or times out

**Problem:** Ensemble queries fail or take too long

**Possible causes:**
- Too many models selected
- Slow API models
- Web search timeout
- No API keys configured for API models

**Solution:**
- Use fewer models (2-3 recommended)
- Mix fast local models with API models
- Check internet connection for web search
- Verify API keys in Settings → API Keys

---

## Performance Issues

### Slow response times with local models

**Problem:** Local LLM responses take 15-60+ seconds

**Causes:**
- CPU inference (no GPU acceleration)
- Large models on limited hardware
- Quantization level too high (e.g., Q8 vs Q4)

**Solutions:**
1. **Use GPU if available:**
   - Ensure CUDA installed for NVIDIA GPUs
   - Check GPU is detected: `nvidia-smi`

2. **Use smaller/faster models:**
   - Llama 3.2 3B instead of 7B models
   - Q4_K_M quantization instead of Q8

3. **Use API models for speed:**
   - GPT-4o-mini: 1-3 second responses
   - Claude Haiku: 1-2 second responses

---

### High memory usage

**Problem:** Zynkbot using >4GB RAM

**Causes:**
- Large local models loaded
- Many memories in database
- Multiple models loaded simultaneously

**Solutions:**
- Use smaller models (3B instead of 7B)
- Close unused applications
- Restart Zynkbot periodically
- Use API models instead of local (no model loading)

---

### Slow database queries

**Problem:** Memory search or knowledge base queries are slow

**Solutions:**
1. **Rebuild database indexes:**
   ```bash
   # Open the database with sqlite3 and run:
   sqlite3 ~/.local/share/zynkbot/zynkbot.db "VACUUM; PRAGMA integrity_check;"
   ```

2. **Check database size:**

   Linux:
   ```bash
   ls -lh ~/.local/share/zynkbot/zynkbot.db
   ```

   Windows:
   ```batch
   dir "%LOCALAPPDATA%\zynkbot\zynkbot.db"
   ```

3. **Clear old data:**
   - Delete old ZChat messages via the ZChat interface
   - Remove unused knowledge base documents
   - Archive old memories via Memory Manager

---

## Windows-Specific Issues

### Windows Defender blocks installation

**Problem:** Installation fails, files quarantined

**Solution:**
```batch
# Run diagnostics
DIAGNOSE_ANTIVIRUS.bat

# Add Windows Defender exclusions (Administrator required)
FIX_WINDOWS_DEFENDER.bat
```

**Manual exclusions:**
1. Open Windows Security → Virus & threat protection → Manage settings
2. Scroll to "Exclusions" → Add or remove exclusions
3. Add folder: `C:\Zynkbot` (or wherever you extracted the project)
4. Add folder: `%USERPROFILE%\.cache\huggingface` (model cache)

---

### CUDA not detected (NVIDIA GPU)

**Problem:** Have NVIDIA GPU but Zynkbot uses CPU

**Verify CUDA installed:**
```batch
nvcc --version
```

**Check GPU recognized:**
```batch
nvidia-smi
```

**Set environment variable manually:**
```batch
setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
```
(Replace `v12.x` with your installed CUDA version)

**Rebuild Zynkbot:**
```batch
cd zynkbot\zynkbot_rust
npm run tauri:build --release
```

---

### Antivirus false positives

**Problem:** `.exe` files flagged as malicious

**Solution:**
- This is a false positive (Rust binaries are sometimes flagged)
- Restore from quarantine
- Add to antivirus exclusions
- Submit false positive report to antivirus vendor

---

## Getting Help

If you've tried these solutions and still have issues:

1. **Check logs:**
   - Terminal output when running Zynkbot
   - Tauri logs: Check console in developer tools (`F12` in app)

2. **Gather information:**
   - Operating system and version
   - Rust version: `cargo --version`
   - Node version: `node --version`
   - Error messages (full text)

3. **Open GitHub issue:**
   - Include all information above
   - Describe steps to reproduce
   - Include relevant error messages

4. **Platform-specific docs:**
   - Windows: [WINDOWS_INSTALLATION_GUIDE.md](WINDOWS_INSTALLATION_GUIDE.md)
   - Linux: [LINUX_INSTALLATION_GUIDE.md](LINUX_INSTALLATION_GUIDE.md)

---

## License

Zynkbot is dual-licensed:
- **AGPL v3** - Free for non-commercial use
- **Commercial License** - Required for commercial use

See [LICENSE](../LICENSE) for full terms.
