//! In-process log capture for "Report a problem".
//!
//! Every `println!`/`eprintln!` in this crate is routed here by the macro
//! shadows at the top of lib.rs: the line still goes to the real stdout/stderr
//! (a terminal on desktop, logcat on Android), and a copy lands in a bounded
//! ring buffer that the report command reads back. Nothing is written to disk
//! and nothing leaves the process unless the user copies a report themselves.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::Mutex;

const CAPACITY: usize = 800;

static BUFFER: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

fn push(prefix: &str, text: &str) {
    let stamp = chrono::Local::now().format("%H:%M:%S%.3f");
    let line = format!("{stamp} {prefix}{text}");
    if let Ok(mut buf) = BUFFER.lock() {
        if buf.len() >= CAPACITY {
            buf.pop_front();
        }
        buf.push_back(line);
    }
}

/// Stdout line (from `println!`).
pub fn line(text: String) {
    {
        let out = std::io::stdout();
        let mut lock = out.lock();
        let _ = writeln!(lock, "{text}");
    }
    push("", &text);
}

/// Stderr line (from `eprintln!`).
pub fn err_line(text: String) {
    {
        let err = std::io::stderr();
        let mut lock = err.lock();
        let _ = writeln!(lock, "{text}");
    }
    push("[stderr] ", &text);
}

/// The most recent `n` captured lines, oldest first.
pub fn recent(n: usize) -> Vec<String> {
    match BUFFER.lock() {
        Ok(buf) => buf.iter().rev().take(n).cloned().collect::<Vec<_>>().into_iter().rev().collect(),
        Err(_) => Vec::new(),
    }
}

/// Mask anything that looks like a credential before it goes into a report:
/// provider key prefixes, `SOMETHING_KEY=value` / `SECRET=value` pairs, long
/// hex or base64 runs, and bearer tokens. Conservative on purpose — a report
/// with a few over-masked tokens is fine, a leaked key is not.
pub fn redact(text: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;
    static PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
        [
            r"(?i)\b(sk|xai|gsk|r2|hf|ghp|gho|github_pat)[-_][A-Za-z0-9_\-]{8,}",
            r"(?i)\b([A-Z0-9_]*(KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE)[A-Z0-9_]*)\s*[=:]\s*\S+",
            r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}",
            r"\b[0-9a-fA-F]{32,}\b",
            r"\b[A-Za-z0-9+/]{40,}={0,2}\b",
        ]
        .iter()
        .map(|p| Regex::new(p).expect("redaction regex"))
        .collect()
    });
    let mut out = text.to_string();
    for re in PATTERNS.iter() {
        out = re.replace_all(&out, "[redacted]").to_string();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_keeps_the_newest_lines_in_order() {
        for i in 0..(CAPACITY + 20) {
            push("", &format!("line {i}"));
        }
        let last = recent(3);
        assert_eq!(last.len(), 3);
        assert!(last[2].ends_with(&format!("line {}", CAPACITY + 19)));
        assert!(last[0].ends_with(&format!("line {}", CAPACITY + 17)));
    }

    #[test]
    fn credentials_are_masked() {
        let r = redact("OPENAI_API_KEY=sk-abcdefghijklmnop1234 and ANTHROPIC_API_KEY: sk-ant-zzzzzzzzzzzz");
        assert!(!r.contains("sk-abc"), "{r}");
        assert!(!r.contains("sk-ant"), "{r}");
        let r = redact("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig");
        assert!(!r.contains("eyJ"), "{r}");
        let r = redact("backup key f576711a1556f576711a1556f576711a1556f576711a1556");
        assert!(r.contains("[redacted]"), "{r}");
    }

    #[test]
    fn ordinary_log_lines_pass_through() {
        let line = "[ZynkSync] ✓ Peer stored 158 sessions, 2 new messages";
        assert_eq!(redact(line), line);
        let line = "[KB RAG] outcome Found: 3 chunks returned (best: 42.1%)";
        assert_eq!(redact(line), line);
    }
}
