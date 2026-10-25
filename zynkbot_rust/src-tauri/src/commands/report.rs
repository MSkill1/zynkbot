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
    include_wake: Option<bool>,
) -> Result<String, String> {
    let version = app.package_info().version.to_string();
    let git = option_env!("ZYNKBOT_GIT_HASH").unwrap_or("unknown");
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let n = log_lines.unwrap_or(300).min(2000);
    let include_conversation = thread.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false);
    // The on-disk log (KI-082) reaches back days; the ring buffer only holds the
    // last 800 lines and is the fallback when the file could not be opened.
    let (lines, source) = match crate::app_log::recent_from_disk(n) {
        Some(l) if !l.is_empty() => (l, "from the on-disk log"),
        _ => (crate::app_log::recent(n), "from memory; the on-disk log was not readable"),
    };
    let logs: Vec<String> = lines
        .into_iter()
        .map(|l| crate::app_log::redact(&l))
        // A line of binary read as text (a photo, 2026-10-25) is dropped, not pasted.
        .map(drop_binary_looking)
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
    out.push_str(&format!("- Reported: {now}\n"));
    out.push_str(&format!("- Full log on the device: {}\n\n", crate::app_log::file_path().display()));
    out.push_str("### What happened\n\n");
    out.push_str(if description.trim().is_empty() { "(not described)" } else { description.trim() });
    out.push_str("\n\n");
    if let Some(t) = thread.filter(|t| !t.trim().is_empty()) {
        out.push_str("### Conversation\n\n```\n");
        out.push_str(&crate::app_log::redact(t.trim()));
        out.push_str("\n```\n\n");
    }
    out.push_str(&format!("### Last {} log lines ({}, {})\n\n```\n", logs.len(), source, if include_conversation { "credentials masked" } else { "credentials masked; message text and memory titles omitted" }));
    out.push_str(&logs.join("\n"));
    out.push_str("\n```\n");
    // Hey Zynk's own log (Android; WakeLog.kt, 2026-10-24): every wake heard, kept or
    // ignored and why. Lives beside the Rust log; absent on the desktop. Included only
    // when the user ticks the box, since a chat problem has no use for it, and its
    // transcript lines (what the user said hands-free) are masked unless the conversation
    // box is ticked too.
    let wake_path = crate::app_log::file_path().with_file_name("wake.log");
    if let (true, Some(wake)) = (include_wake.unwrap_or(false), crate::app_log::tail_lines(&wake_path, 200.min(n)).filter(|l| !l.is_empty())) {
        let wake: Vec<String> = wake.into_iter()
            .map(|l| crate::app_log::redact(&l))
            .map(|l| if include_conversation { l } else { mask_wake_transcript(&l) })
            .collect();
        out.push_str(&format!("\n### Hey Zynk: last {} lines\n\n```\n", wake.len()));
        out.push_str(&wake.join("\n"));
        out.push_str("\n```\n");
    }
    Ok(out)
}

/// A wake-log line with any quoted transcript replaced: `Screen-off transcript: "..."`,
/// `Answering natively: "..."`, `Delivering transcript to app: "..."` carry the user's
/// spoken words; the scores and reasons around them are what a report needs.
fn mask_wake_transcript(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    if !(lower.contains("transcript") || lower.contains("answering natively")) { return line.to_string(); }
    match (line.find('"'), line.rfind('"')) {
        (Some(a), Some(b)) if b > a => format!("{}\"[spoken text omitted]\"{}", &line[..a], &line[b + 1..]),
        _ => line.to_string(),
    }
}

#[cfg(test)]
mod wake_mask_tests {
    use super::mask_wake_transcript;
    #[test]
    fn transcript_lines_are_masked_and_others_kept() {
        assert_eq!(mask_wake_transcript(r#"10-24 12:00:00.000 I/WakeWordService: Answering natively: "set a timer for two minutes""#),
            r#"10-24 12:00:00.000 I/WakeWordService: Answering natively: "[spoken text omitted]""#);
        let kept = "10-24 12:00:00.000 I/WakeWordService: Detection ignored: verifier says not the owner (0.012 <= 0.50)";
        assert_eq!(mask_wake_transcript(kept), kept);
    }
}

/// Very long lines, or lines full of replacement characters, are not log text.
fn drop_binary_looking(line: String) -> String {
    let n = line.chars().count();
    let bad = line.matches('\u{FFFD}').count();
    if n > 4000 || (n > 40 && bad * 10 > n) {
        format!("[line dropped: {} characters of binary-looking text]", n)
    } else { line }
}
