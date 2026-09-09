//! "Report a problem": assemble a plain-text bug report the user can paste
//! into a GitHub issue. Built entirely on the device from the running app's
//! own state; nothing is sent anywhere by this code.

/// `description` is the user's own words; `device` is a platform string the
/// page supplies (Android model from the Kotlin bridge, or the browser's
/// platform on desktop); `backend` is the model selection in use; `thread`
/// is the current conversation as text, included only when the user ticked
/// the box; `log_lines` caps the captured log tail.
#[tauri::command]
pub async fn build_problem_report(
    app: tauri::AppHandle,
    description: String,
    device: String,
    backend: String,
    thread: Option<String>,
    log_lines: Option<usize>,
) -> Result<String, String> {
    let version = app.package_info().version.to_string();
    let git = option_env!("ZYNKBOT_GIT_HASH").unwrap_or("unknown");
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let n = log_lines.unwrap_or(300).min(800);
    let include_conversation = thread.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false);
    let logs: Vec<String> = crate::app_log::recent(n)
        .into_iter()
        .map(|l| crate::app_log::redact(&l))
        // Without the conversation box ticked, the log tail must not carry the
        // user's words or memory titles either.
        .map(|l| if include_conversation { l } else { crate::app_log::scrub_user_text(&l) })
        .collect();

    let mut out = String::new();
    out.push_str("## Zynkbot problem report\n\n");
    out.push_str(&format!("- Version: {version} (build {git})\n"));
    out.push_str(&format!("- Platform: {} {}\n", std::env::consts::OS, std::env::consts::ARCH));
    if !device.trim().is_empty() {
        out.push_str(&format!("- Device: {}\n", device.trim()));
    }
    if !backend.trim().is_empty() {
        out.push_str(&format!("- Model backend: {}\n", backend.trim()));
    }
    out.push_str(&format!("- Reported: {now}\n\n"));
    out.push_str("### What happened\n\n");
    out.push_str(if description.trim().is_empty() { "(not described)" } else { description.trim() });
    out.push_str("\n\n");
    if let Some(t) = thread.filter(|t| !t.trim().is_empty()) {
        out.push_str("### Conversation\n\n```\n");
        out.push_str(&crate::app_log::redact(t.trim()));
        out.push_str("\n```\n\n");
    }
    out.push_str(&format!("### Last {} log lines ({})\n\n```\n", logs.len(), if include_conversation { "credentials masked" } else { "credentials masked; message text and memory titles omitted" }));
    out.push_str(&logs.join("\n"));
    out.push_str("\n```\n");
    Ok(out)
}
