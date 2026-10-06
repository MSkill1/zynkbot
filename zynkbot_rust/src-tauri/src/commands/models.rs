use serde::{Deserialize, Serialize};
use tauri::Emitter;

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: String,
}

/// Get available models — scans for API keys and local GGUF files
#[tauri::command]
pub async fn get_models() -> Result<Vec<ModelInfo>, String> {
    let mut models = Vec::new();

    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        models.push(ModelInfo {
            id: "anthropic".to_string(),
            name: "Anthropic Claude".to_string(),
            model_type: "api".to_string(),
        });
    }

    if std::env::var("OPENAI_API_KEY").is_ok() {
        models.push(ModelInfo {
            id: "openai".to_string(),
            name: "OpenAI GPT".to_string(),
            model_type: "api".to_string(),
        });
    }

    if std::env::var("XAI_API_KEY").is_ok() {
        models.push(ModelInfo {
            id: "xai".to_string(),
            name: "xAI Grok".to_string(),
            model_type: "api".to_string(),
        });
    }

    if std::env::var("MISTRAL_API_KEY").is_ok() {
        models.push(ModelInfo {
            id: "mistral".to_string(),
            name: "Mistral".to_string(),
            model_type: "api".to_string(),
        });
    }

    if std::env::var("CUSTOM_API_URL").is_ok() {
        let model_name = std::env::var("CUSTOM_MODEL")
            .unwrap_or_else(|_| "custom model".to_string());
        models.push(ModelInfo {
            id: "custom".to_string(),
            name: format!("Custom / Ollama ({})", model_name),
            model_type: "api".to_string(),
        });
    }

    if let Ok(model_path) = std::env::var("LOCAL_MODEL_PATH") {
        let model_name = std::path::Path::new(&model_path)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("Local Model");

        models.push(ModelInfo {
            id: model_path.clone(),
            name: model_name.to_string(),
            model_type: "local".to_string(),
        });
    }

    let user_models_dir = crate::db::get_models_dir().join("user");

    println!("[RUST] Scanning for user models in: {}", user_models_dir.display());

    if let Ok(entries) = std::fs::read_dir(&user_models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension.eq_ignore_ascii_case("gguf") {
                        let model_name = path.file_stem()
                            .and_then(|n| n.to_str())
                            .unwrap_or("Local Model");

                        let model_path = path.to_string_lossy().to_string();

                        if !models.iter().any(|m| m.id == model_path) {
                            println!("[RUST] Found local chat model: {}", model_name);
                            models.push(ModelInfo {
                                id: model_path,
                                name: model_name.to_string(),
                                model_type: "local".to_string(),
                            });
                        }
                    }
                }
            }
        }
    } else {
        eprintln!("[RUST] User models directory not found: {}", user_models_dir.display());
        eprintln!("[RUST] Create it with: mkdir -p {}", user_models_dir.display());
    }

    Ok(models)
}

/// Open the local models/user/ folder in the system file manager
#[tauri::command]
pub async fn open_models_folder() -> Result<(), String> {
    let user_models_dir = crate::db::get_models_dir().join("user");

    if !user_models_dir.exists() {
        std::fs::create_dir_all(&user_models_dir)
            .map_err(|e| format!("Failed to create models directory: {}", e))?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(user_models_dir)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(user_models_dir)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(user_models_dir)
            .spawn()
            .map_err(|e| format!("Failed to open folder: {}", e))?;
    }

    Ok(())
}

/// List all downloaded user model filenames
#[tauri::command]
pub async fn list_user_models() -> Result<Vec<String>, String> {
    let user_models_dir = crate::db::get_models_dir().join("user");
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&user_models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext.eq_ignore_ascii_case("gguf") {
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            names.push(name.to_string());
                        }
                    }
                }
            }
        }
    }
    Ok(names)
}

/// Delete a user model file by filename
#[tauri::command]
pub async fn delete_user_model(filename: String) -> Result<(), String> {
    let user_models_dir = crate::db::get_models_dir().join("user");
    let path = user_models_dir.join(&filename);

    if !path.starts_with(&user_models_dir) {
        return Err("Invalid filename".to_string());
    }
    if !path.exists() {
        return Err(format!("Model file not found: {}", filename));
    }

    std::fs::remove_file(&path)
        .map_err(|e| format!("Failed to delete model: {}", e))?;

    println!("[RUST] Deleted user model: {}", filename);
    Ok(())
}

/// Get configured API keys (returns values for current session)
#[tauri::command]
pub async fn get_api_keys() -> Result<serde_json::Value, String> {
    let mut keys = serde_json::Map::new();

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        keys.insert("ANTHROPIC_API_KEY".to_string(), serde_json::json!(key));
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        keys.insert("OPENAI_API_KEY".to_string(), serde_json::json!(key));
    }
    if let Ok(key) = std::env::var("XAI_API_KEY") {
        keys.insert("XAI_API_KEY".to_string(), serde_json::json!(key));
    }
    if let Ok(url) = std::env::var("CUSTOM_API_URL") {
        keys.insert("CUSTOM_API_URL".to_string(), serde_json::json!(url));
    }
    if let Ok(key) = std::env::var("CUSTOM_API_KEY") {
        keys.insert("CUSTOM_API_KEY".to_string(), serde_json::json!(key));
    }
    if let Ok(model) = std::env::var("CUSTOM_MODEL") {
        keys.insert("CUSTOM_MODEL".to_string(), serde_json::json!(model));
    }
    if let Ok(model) = std::env::var("ANTHROPIC_MODEL") {
        keys.insert("ANTHROPIC_MODEL".to_string(), serde_json::json!(model));
    }
    if let Ok(model) = std::env::var("OPENAI_MODEL") {
        keys.insert("OPENAI_MODEL".to_string(), serde_json::json!(model));
    }
    if let Ok(model) = std::env::var("XAI_MODEL") {
        keys.insert("XAI_MODEL".to_string(), serde_json::json!(model));
    }
    if let Ok(key) = std::env::var("MISTRAL_API_KEY") {
        keys.insert("MISTRAL_API_KEY".to_string(), serde_json::json!(key));
    }
    if let Ok(model) = std::env::var("MISTRAL_MODEL") {
        keys.insert("MISTRAL_MODEL".to_string(), serde_json::json!(model));
    }
    if let Ok(v) = std::env::var("R2_ENDPOINT") {
        keys.insert("R2_ENDPOINT".to_string(), serde_json::json!(v));
    }
    if let Ok(v) = std::env::var("R2_ACCESS_KEY_ID") {
        keys.insert("R2_ACCESS_KEY_ID".to_string(), serde_json::json!(v));
    }
    if let Ok(v) = std::env::var("R2_SECRET_ACCESS_KEY") {
        keys.insert("R2_SECRET_ACCESS_KEY".to_string(), serde_json::json!(v));
    }
    if let Ok(v) = std::env::var("R2_BUCKET") {
        keys.insert("R2_BUCKET".to_string(), serde_json::json!(v));
    }
    Ok(serde_json::json!(keys))
}

/// Set an API key in the .env file and current session
#[tauri::command]
pub async fn set_api_key(key: String, value: String) -> Result<(), String> {
    println!("[API Keys] Saving {} (value length: {} chars)", key, value.len());
    apply_env_key(&key, &value)?;
    if PROPAGATABLE_KEYS.contains(&key.as_str()) {
        record_secret(&key, Some(&value)).await;
    }
    println!("[API Keys] ✅ Saved {}", key);
    Ok(())
}

/// Persist the model/backend the user selected in the app so paths that cannot see
/// the WebView's localStorage — the native Android voice pipeline — use the same one.
/// Stored as ZYNK_MODEL_BACKEND in the same .env set_api_key writes, and applied to the
/// process env immediately. Called by the frontend whenever the selection changes.
#[tauri::command]
pub async fn set_preferred_backend(backend: String) -> Result<(), String> {
    let backend = backend.trim().to_string();
    if backend.is_empty() {
        return Err("backend must not be empty".to_string());
    }
    let env_path = crate::db::get_app_data_dir().join(".env");
    // Quoted when needed: a bare Windows path made the whole file unreadable (KI-083).
    crate::env_file::upsert(&env_path, "ZYNK_MODEL_BACKEND", &backend)?;
    std::env::set_var("ZYNK_MODEL_BACKEND", &backend);
    println!("[Backend] Preferred backend set to '{}'", backend);
    Ok(())
}

/// Remove an API key from the .env file
#[tauri::command]
pub async fn remove_api_key(key: String) -> Result<(), String> {
    remove_env_key(&key)?;
    if PROPAGATABLE_KEYS.contains(&key.as_str()) {
        record_secret(&key, None).await;
    }
    println!("[API Keys] ✅ Removed {}", key);
    Ok(())
}

/// Push an API key to all active sync peers over the existing cert-pinned channel.
#[tauri::command]
pub async fn propagate_api_key(key: String, value: String) -> Result<serde_json::Value, String> {
    propagate_api_keys(vec![(key, value)]).await
}

/// Push several keys to every paired peer in one pass.
///
/// Name under which the backup encryption key rides along with an API-key push.
pub const BACKUP_KEY_PUSH_NAME: &str = "ZYNKBOT_BACKUP_KEY";

/// The keys that travel between a user's devices. CUSTOM_* are deliberately absent: the
/// custom endpoint is machine-local (a phone reaches Ollama through this desktop's proxy,
/// which substitutes the desktop's model), so a pushed URL or model name would only mislead.
pub const PROPAGATABLE_KEYS: &[&str] = &[
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    "OPENAI_API_KEY",    "OPENAI_MODEL",
    "XAI_API_KEY",       "XAI_MODEL",
    "MISTRAL_API_KEY",   "MISTRAL_MODEL",
    "R2_ENDPOINT",       "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
];

/// Write one key into this device's .env and the process environment. Used by the
/// Settings save, the old push route and the outbox receiver, so they cannot drift.
pub fn apply_env_key(key: &str, value: &str) -> Result<(), String> {
    let env_path = crate::db::get_app_data_dir().join(".env");
    crate::env_file::upsert(&env_path, key, value)?;
    std::env::set_var(key, value);
    Ok(())
}

/// Remove one key from this device's .env and the process environment.
pub fn remove_env_key(key: &str) -> Result<(), String> {
    let env_path = crate::db::get_app_data_dir().join(".env");
    crate::env_file::remove(&env_path, key)?;
    std::env::remove_var(key);
    Ok(())
}

/// Record a key in sync_secrets so the outbox carries it to every device (0014). Best
/// effort: the .env write has already happened, and the next drain re-seeds from the
/// environment anyway, so a failure here only delays the sync by a cycle.
pub async fn record_secret(name: &str, value: Option<&str>) {
    let Ok(pool) = sqlx::SqlitePool::connect(&crate::db::get_db_url()).await else { return };
    let _ = match value {
        Some(v) => crate::sync_outbox::record_secret(&pool, name, v).await,
        None => crate::sync_outbox::forget_secret(&pool, name).await,
    };
    pool.close().await;
}

/// The UI used to call propagate_api_key once per key, and each call re-ran the
/// whole peer loop. With the shared client's 30s timeout, a single unreachable
/// peer cost 30s x number-of-keys — around 8.5 minutes for a full key set, which
/// read as a frozen button. One pass over the peers with a short timeout instead.
#[tauri::command]
pub async fn propagate_api_keys(entries: Vec<(String, String)>) -> Result<serde_json::Value, String> {
    // The cloud-backup encryption key travels with the API keys. Without it a peer
    // receives the R2 credentials but keeps its own random backup key, so it can
    // list the backup and never decrypt it, and its Memory Manager keeps asking
    // for a passphrase that was set on another phone (OnePlus, 2026-09-07). The
    // receiver stores this one as backup.key + the acknowledged flag, not in .env.
    let mut entries = entries;
    if let Ok(key_hex) = crate::commands::backup::get_backup_key().await {
        if !key_hex.trim().is_empty() && !entries.iter().any(|(k, _)| k == BACKUP_KEY_PUSH_NAME) {
            entries.push((BACKUP_KEY_PUSH_NAME.to_string(), key_hex));
        }
    }
    // Since step 5 of the sync rebuild this records each key with the current time and
    // the outbox carries it to every paired device on the next cycle — the one that is
    // off gets it when it returns, which the old direct push never managed (KI-055).
    // Recording with "now" is also what makes a pressed button mean "this value wins".
    for (key, value) in &entries {
        record_secret(key, Some(value)).await;
    }
    let peers = {
        let guard = crate::ZYNKSYNC_SERVICE.lock().await;
        match guard.as_ref() {
            Some(service) => service.get_peers().await.into_iter().filter(|p| p.paired).count(),
            None => 0,
        }
    };
    Ok(serde_json::json!({
        "succeeded": entries.len() * peers,
        "failed": 0,
        "total": entries.len() * peers,
        "peers": peers,
        "unreachable": Vec::<String>::new(),
        "queued": true,
    }))
}

/// Fetch the list of models from a custom OpenAI-compatible endpoint (Ollama, llama-server, etc.)
#[tauri::command]
pub async fn fetch_custom_models(base_url: String, api_key: String) -> Result<Vec<String>, String> {
    let models_url = format!("{}/models", base_url.trim_end_matches('/'));
    println!("[Custom] Fetching models from: {}", models_url);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    let mut req = client.get(&models_url);
    if !api_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", api_key));
    }

    let response = req.send().await.map_err(|e| {
        format!("Can't reach {} — is the server running? ({})", base_url, e)
    })?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned {} — is this an OpenAI-compatible endpoint?",
            response.status()
        ));
    }

    let json: serde_json::Value = response.json().await
        .map_err(|e| format!("Invalid response from server: {}", e))?;

    let models = json["data"].as_array()
        .ok_or_else(|| "Unexpected response format — expected {\"data\": [...]}".to_string())?
        .iter()
        .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
        .collect::<Vec<_>>();

    println!("[Custom] Found {} model(s): {:?}", models.len(), models);
    Ok(models)
}


fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // consume ESC [ ... <final byte>
            if chars.peek() == Some(&'[') {
                chars.next();
                for ch in chars.by_ref() {
                    if ch.is_ascii_alphabetic() { break; }
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

#[derive(serde::Serialize)]
pub struct OllamaStatus {
    /// "running" | "installed_not_running" | "not_installed"
    pub state: String,
    pub models: Vec<String>,
}

/// Probe the local Ollama at startup or on demand. Uses Ollama's own HTTP API, so it works
/// identically on Windows, Linux and Mac no matter where Ollama stores its model files —
/// Zynkbot never reads that directory. "installed_not_running" is distinguished from
/// "not_installed" by looking for the CLI in the standard install locations (see ollama_binary).
#[tauri::command]
pub async fn ollama_status() -> OllamaStatus {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1500))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    match client.get("http://localhost:11434/api/tags").send().await {
        Ok(resp) if resp.status().is_success() => {
            let models = resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|j| {
                    j.get("models").and_then(|m| m.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                            .collect::<Vec<String>>()
                    })
                })
                .unwrap_or_default();
            OllamaStatus { state: "running".to_string(), models }
        }
        _ => {
            let installed = ollama_binary().is_file();
            OllamaStatus {
                state: if installed { "installed_not_running" } else { "not_installed" }.to_string(),
                models: vec![],
            }
        }
    }
}

/// Locate the `ollama` CLI. PATH first; then the stock install locations, because a
/// desktop app launched before Ollama was installed (or from a launcher with a minimal
/// environment) does not see the PATH entry the installer added, and `Command::new("ollama")`
/// fails with "program not found" even though Ollama is running (KI-046).
fn ollama_binary() -> std::path::PathBuf {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path) {
            candidates.push(dir.join(if cfg!(windows) { "ollama.exe" } else { "ollama" }));
        }
    }
    #[cfg(windows)]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        candidates.push(std::path::Path::new(&local).join("Programs").join("Ollama").join("ollama.exe"));
    }
    #[cfg(windows)]
    candidates.push(std::path::PathBuf::from(r"C:\Program Files\Ollama\ollama.exe"));
    #[cfg(not(windows))]
    for p in ["/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama",
              "/Applications/Ollama.app/Contents/Resources/ollama"] {
        candidates.push(std::path::PathBuf::from(p));
    }
    candidates.into_iter().find(|c| c.is_file())
        .unwrap_or_else(|| std::path::PathBuf::from("ollama"))
}

/// Run `ollama stop <model_name>` to unload the model from GPU/RAM.
/// Returns immediately; does not stream progress.
#[tauri::command]
pub async fn stop_ollama_model(model_name: String) -> Result<String, String> {
    if model_name.is_empty()
        || !model_name.chars().all(|c| c.is_alphanumeric() || ":.-_/".contains(c))
    {
        return Err(format!("Invalid model name: {}", model_name));
    }

    let output = std::process::Command::new(ollama_binary())
        .args(["stop", &model_name])
        .output()
        .map_err(|e| format!("Failed to run ollama: {}", e))?;

    if output.status.success() {
        Ok(format!("✅ {} stopped", model_name))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("ollama stop failed: {}", stderr.trim()))
    }
}

/// Run `ollama pull <model_name>` and stream progress via "ollama-pull-progress" events
#[tauri::command]
pub async fn pull_ollama_model(app: tauri::AppHandle, model_name: String) -> Result<(), String> {
    // Basic validation — allow alphanumeric, colon, dash, underscore, dot, slash
    if model_name.is_empty()
        || !model_name.chars().all(|c| c.is_alphanumeric() || ":.-_/".contains(c))
    {
        return Err(format!("Invalid model name: {}", model_name));
    }

    let _ = app.emit("ollama-pull-progress", format!("⬇ Pulling {}...\n", model_name));

    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<String, String>>(64);

    let name = model_name.clone();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        let mut child = match std::process::Command::new(ollama_binary())
            .args(["pull", &name])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.blocking_send(Err(format!(
                    "Failed to start ollama ({}): {} — is Ollama installed? Restart Zynkbot if you installed it while the app was open.",
                    ollama_binary().display(), e)));
                return;
            }
        };

        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(l) => {
                        let clean = strip_ansi_codes(&l);
                        if !clean.trim().is_empty() {
                            let _ = tx.blocking_send(Ok(clean));
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        match child.wait() {
            Ok(status) if status.success() => {
                let _ = tx.blocking_send(Ok(format!("✅ {} pulled successfully.", name)));
            }
            Ok(status) => {
                let code = status.code().unwrap_or(-1);
                let _ = tx.blocking_send(Err(format!("ollama pull exited with code {}", code)));
            }
            Err(e) => {
                let _ = tx.blocking_send(Err(format!("Error waiting for ollama: {}", e)));
            }
        }
    });

    while let Some(msg) = rx.recv().await {
        match msg {
            Ok(line) => {
                let _ = app.emit("ollama-pull-progress", line);
            }
            Err(err) => {
                let _ = app.emit("ollama-pull-progress", format!("❌ {}", err));
                return Err(err);
            }
        }
    }

    Ok(())
}
