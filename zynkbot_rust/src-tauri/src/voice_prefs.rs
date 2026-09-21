//! Voice preferences the native (hands-free) path needs at reply time.
//!
//! The page keeps voice settings in localStorage, which the Rust core cannot read
//! and the Android assistant session never sees. Settings that change how a
//! hands-free reply is produced are mirrored here as a small JSON file in the
//! app's data folder (`voice_prefs.json`), written by the `set_voice_pref` command
//! whenever the page changes them and at startup, and read by `generate_reply`.
//!
//! First such setting (2026-09-17): `web_search_auto` — "Auto-execute web searches in
//! voice sessions". The page honoured it; the assistant-role path did not, so a
//! hands-free question that needed a search was answered with "want me to search?"
//! (GitHub #26, KI-062). Removed 2026-09-20: hands-free always searches now,
//! unconditionally — see `commands::chat::generate_reply`. The mirror mechanism
//! (`get_bool`/`set`/`set_voice_pref`) stays for whatever voice setting needs it next.

use std::path::PathBuf;

fn path() -> PathBuf { crate::db::get_app_data_dir().join("voice_prefs.json") }

fn load() -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(path()).ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

pub fn get_bool(key: &str) -> bool {
    load().get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

pub fn set(key: &str, value: serde_json::Value) -> Result<(), String> {
    let mut m = load();
    m.insert(key.to_string(), value);
    let p = path();
    if let Some(dir) = p.parent() { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
    std::fs::write(&p, serde_json::to_string_pretty(&serde_json::Value::Object(m)).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Failed to write voice prefs: {}", e))
}

#[tauri::command]
pub async fn set_voice_pref(key: String, value: serde_json::Value) -> Result<(), String> {
    set(&key, value)
}
