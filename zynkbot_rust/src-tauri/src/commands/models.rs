use serde::{Deserialize, Serialize};

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
