//! The phone's taught voice profile for the wake word (2026-11-03).
//!
//! `WakeVerifierTrainer` on Android writes `wake-word-models/hey_zynk_verifier.json`
//! under the app data folder when the owner has taught it twenty "Hey Zynk"s. Until a
//! phone holds a profile that enforces, the wake word answers nothing. The profile is
//! one small JSON file (192 weights), so it travels the way the API keys do: recorded in
//! `sync_secrets` under [`WAKE_PROFILE_PUSH_NAME`], carried to every paired device by the
//! outbox, and put in the cloud backup beside the keys. A profile that arrives through the
//! owner's own account or backup is the owner's, so the phone enforces it as it is.
//! Nothing is read from it here beyond checking it parses as a profile.
use std::path::PathBuf;

/// Name under which the profile rides in `sync_secrets` and in the backup's key list.
pub const WAKE_PROFILE_PUSH_NAME: &str = "ZYNK_WAKE_VOICE_PROFILE";

pub fn wake_profile_path() -> PathBuf {
    crate::db::get_app_data_dir().join("wake-word-models").join("hey_zynk_verifier.json")
}

/// The profile this device holds, if the file is there and is a profile.
pub fn read_wake_profile() -> Option<String> {
    let text = std::fs::read_to_string(wake_profile_path()).ok()?;
    if looks_like_profile(&text) { Some(text) } else { None }
}

/// Does this device hold a profile at all (taught here, or received)?
pub fn has_wake_profile() -> bool { read_wake_profile().is_some() }

/// Write a received profile where the wake service reads it. The service notices the
/// file's new modification time and reloads. Refuses anything that is not a profile.
pub fn install_wake_profile(text: &str) -> Result<(), String> {
    if !looks_like_profile(text) { return Err("received wake voice profile is not a profile".to_string()); }
    let path = wake_profile_path();
    if let Some(dir) = path.parent() { std::fs::create_dir_all(dir).map_err(|e| format!("wake profile folder: {}", e))?; }
    std::fs::write(&path, text).map_err(|e| format!("write wake profile: {}", e))?;
    println!("[WakeProfile] Voice profile installed at {}", path.display());
    Ok(())
}

fn looks_like_profile(text: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(v) => v.get("coef").map(|c| c.is_array()).unwrap_or(false) && v.get("intercept").is_some(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::looks_like_profile;
    #[test]
    fn a_profile_has_weights_and_an_intercept() {
        assert!(looks_like_profile(r#"{"coef":[0.1,0.2],"intercept":-1.0,"owners":["*"]}"#));
        assert!(!looks_like_profile(r#"{"coef":"no","intercept":0}"#));
        assert!(!looks_like_profile("sk-not-json"));
    }
}
