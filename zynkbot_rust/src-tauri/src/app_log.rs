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
            r"(?i)pairing code:?\s*\d{6}",
        ]
        .iter()
        .map(|p| Regex::new(p).expect("redaction regex"))
        .collect()
    });
    let mut out = text.to_string();
    for re in PATTERNS.iter() {
        out = re
            .replace_all(&out, |caps: &regex::Captures| {
                let m = caps.get(0).map(|m| m.as_str()).unwrap_or("");
                // A file path also matches the base64 pattern (its alphabet includes
                // '/'): "zynkbot/files/zynkbot/models/system/bert" was masked in a
                // report on 2026-09-09. Two or more slashes with only short segments
                // between them is a path, not a key; leave it readable.
                let looks_like_path = m.matches('/').count() >= 2
                    && m.split('/').all(|seg| seg.len() <= 16);
                if looks_like_path { m.to_string() } else { "[redacted]".to_string() }
            })
            .to_string();
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
        let r = redact("[ZynkSync] Generated pairing code: 423522 (expires in 10 minutes)");
        assert!(!r.contains("423522"), "{r}");
    }

    #[test]
    fn ordinary_log_lines_pass_through() {
        let line = "[ZynkSync] ✓ Peer stored 158 sessions, 2 new messages";
        assert_eq!(redact(line), line);
        let line = "[KB RAG] outcome Found: 3 chunks returned (best: 42.1%)";
        assert_eq!(redact(line), line);
    }
}

/// Remove the user's own words from a log line: the message text the chat
/// request prints, memory titles and snippets from retrieval and the decision
/// call, and anything else that quotes what was said. Applied to the log tail of
/// a problem report when the user did NOT tick "include the conversation", so
/// that choice means what the guide says it means (2026-09-09: a report with the
/// box unticked still carried the question and twenty memory snippets).
pub fn scrub_user_text(line: &str) -> String {
    use once_cell::sync::Lazy;
    use regex::Regex;
    static RULES: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
        [
            (r#"▶ .*$"#, "▶ [message text omitted]"),
            (r#"(Including: ).*$"#, "${1}[memory omitted]"),
            (r#"(Memory \d+ - Score: [\d.]+ \(\d+%\)) - .*$"#, "${1} - [title omitted]"),
            (r#"(linked #\d+) \(Some\("[^"]*"\)\)"#, "${1} ([title omitted])"),
            (r#"(Memory #\d+: \w+(?: \(confidence: [\d.]+\))?) - .*$"#, "${1} - [reason omitted]"),
            (r#"(should_remember=\w+, title=).*$"#, "${1}[omitted]"),
            (r#"(?i)((?:transcript|reply|query|remember|saved|title)\w*[:=] ?)"[^"]*""#, "${1}\"[omitted]\""),
        ]
        .iter()
        .map(|(p, r)| (Regex::new(p).expect("scrub regex"), *r))
        .collect()
    });
    let mut out = line.to_string();
    for (re, rep) in RULES.iter() {
        out = re.replace_all(&out, *rep).to_string();
    }
    out
}

#[cfg(test)]
mod redact_path_tests {
    use super::redact;
    #[test]
    fn file_paths_stay_readable_but_keys_do_not() {
        let p = "/data/data/ai.containai.zynkbot/files/zynkbot/models/system/bert-base-NER/model.safetensors";
        assert_eq!(redact(p), p);
        let key = "token abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV0123456789+/=";
        assert!(redact(key).contains("[redacted]"));
        assert!(redact("sk-ant-api03-abcdefghijklmnop").contains("[redacted]"));
    }
}

#[cfg(test)]
mod scrub_tests {
    use super::scrub_user_text;
    #[test]
    fn user_words_leave_the_log_tail() {
        let cases = [
            ("18:39:23.051 ▶ This was just a test to see if I get a response.", "▶ [message text omitted]"),
            ("[Engine]   Including: we're really actively dislike having...", "Including: [memory omitted]"),
            ("[1] Memory 1518 - Score: 0.686 (68%) - Testing code fix response time", "Memory 1518 - Score: 0.686 (68%) - [title omitted]"),
            ("memory #1518 → linked #1585 (Some(\"User confirms test success\")) via 'supports'", "linked #1585 ([title omitted]) via"),
            ("[Memory Decision]   Memory #1517: supports (confidence: 0.85) - Both describe the user", "Memory #1517: supports (confidence: 0.85) - [reason omitted]"),
            ("LLM decision: should_remember=false, title=None, 10 relationships", "should_remember=false, title=[omitted]"),
            ("Native reply: \"Yes, I received it.\"", "reply: \"[omitted]\""),
        ];
        for (line, expect) in cases {
            let got = scrub_user_text(line);
            assert!(got.contains(expect), "{line} -> {got}");
        }
        assert_eq!(scrub_user_text("[ZynkSync] ✓ Peer stored 170 sessions"), "[ZynkSync] ✓ Peer stored 170 sessions");
    }
}
