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
    Ok(serde_json::json!(keys))
}

/// Set an API key in the .env file and current session
#[tauri::command]
pub async fn set_api_key(key: String, value: String) -> Result<(), String> {
    let env_path = crate::db::get_app_data_dir().join(".env");

    println!("[API Keys] Selected .env path: {:?}", env_path);
    println!("[API Keys] Saving {} (value length: {} chars)", key, value.len());

    let content = std::fs::read_to_string(&env_path)
        .unwrap_or_else(|e| {
            eprintln!("[API Keys] Warning: could not read .env ({}), starting fresh", e);
            String::new()
        });

    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let key_prefix = format!("{}=", key);

    let mut found = false;
    for line in &mut lines {
        if line.starts_with(&key_prefix) {
            *line = format!("{}={}", key, value);
            found = true;
            break;
        }
    }

    if !found {
        lines.push(format!("{}={}", key, value));
    }

    std::fs::write(&env_path, lines.join("\n"))
        .map_err(|e| format!("Failed to write .env file at {:?}: {}", env_path, e))?;

    std::env::set_var(&key, &value);

    println!("[API Keys] ✅ Successfully saved {} to .env at {:?}", key, env_path);
    Ok(())
}

/// Remove an API key from the .env file
#[tauri::command]
pub async fn remove_api_key(key: String) -> Result<(), String> {
    let env_path = crate::db::get_app_data_dir().join(".env");

    let content = std::fs::read_to_string(&env_path)
        .unwrap_or_else(|e| {
            eprintln!("[API Keys] Warning: could not read .env ({}), starting fresh", e);
            String::new()
        });

    let key_prefix = format!("{}=", key);

    let lines: Vec<String> = content
        .lines()
        .filter(|line| !line.starts_with(&key_prefix))
        .map(|s| s.to_string())
        .collect();

    std::fs::write(&env_path, lines.join("\n"))
        .map_err(|e| format!("Failed to write .env file at {:?}: {}", env_path, e))?;

    std::env::remove_var(&key);

    println!("[API Keys] ✅ Removed {} from .env at {:?}", key, env_path);
    Ok(())
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

/// Query a paired desktop's /api/ollama/info to get its configured model name.
/// Called from mobile when the user taps "Connect to Ollama on [PC]".
#[tauri::command]
pub async fn get_peer_ollama_config(host: String, port: u16) -> Result<String, String> {
    let url = format!("https://{}:{}/api/ollama/info", host, port);

    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client.get(&url).send().await
        .map_err(|e| format!("Can't reach desktop ({}): {}", host, e))?;

    let json: serde_json::Value = response.json().await
        .map_err(|e| format!("Invalid response from desktop: {}", e))?;

    // Detect self-connection: if the response comes from this device, the stored IP is wrong
    let remote_device_id = json["device_id"].as_str().unwrap_or("");
    if !remote_device_id.is_empty() {
        if let Ok(my_id) = crate::user_identity::get_device_id() {
            if remote_device_id == my_id {
                return Err(format!(
                    "This address ({}) points back to this device — the stored IP is incorrect. \
                     Re-pair with ZynkSync to fix it.",
                    host
                ));
            }
        }
    }

    match json["model"].as_str() {
        Some(m) if !m.is_empty() => Ok(m.to_string()),
        _ => Err("Desktop has no Ollama model configured. Set one in the desktop's API settings first.".to_string()),
    }
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
        let mut child = match std::process::Command::new("ollama")
            .args(["pull", &name])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                let _ = tx.blocking_send(Err(format!("Failed to start ollama: {}", e)));
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
