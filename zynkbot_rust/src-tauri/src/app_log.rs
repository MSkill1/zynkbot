//! In-process log capture for "Report a problem", and the on-disk log.
//!
//! Every `println!`/`eprintln!` in this crate is routed here by the macro
//! shadows at the top of lib.rs. The line goes three ways: to the real
//! stdout/stderr (a terminal on desktop, logcat on Android); into a bounded
//! ring buffer that the report command reads back; and, since 2026-10-06
//! (KI-082), appended to `<app data dir>/logs/zynkbot.log` on every platform.
//! Before that, a failure on a tester's machine left no trace once the ring
//! buffer had cycled — a `Remember:` that stored nothing on the Windows laptop on
//! 2026-10-05 could not be investigated.
//!
//! The file holds the same lines the terminal would, so it is as sensitive as
//! the terminal: it includes what the user typed. Redaction and the user-text
//! scrub apply to reports, which leave the device; the file does not. It
//! rotates at `FILE_MAX_BYTES`, keeping `FILE_KEEP` generations
//! (`zynkbot.log`, `zynkbot.log.1`, `zynkbot.log.2`).

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const CAPACITY: usize = 800;

/// Rotate the on-disk log when it would pass this size.
pub const FILE_MAX_BYTES: u64 = 5 * 1024 * 1024;
/// Generations kept on disk, the live file included.
pub const FILE_KEEP: usize = 3;

static BUFFER: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());

/// The on-disk sink: opened on the first line, `Failed` if the folder cannot be
/// created or the file cannot be opened (no retry per line — a full disk or a
/// read-only folder must not turn every log line into a syscall storm).
enum DiskState {
    Unopened,
    Open(FileSink),
    Failed,
}

static DISK: Mutex<DiskState> = Mutex::new(DiskState::Unopened);

/// An append-only log file that rotates by size.
pub struct FileSink {
    path: PathBuf,
    file: std::fs::File,
    written: u64,
    max_bytes: u64,
    keep: usize,
}

impl FileSink {
    /// Open (or create) `path` for appending; the parent folder is created.
    pub fn open(path: PathBuf, max_bytes: u64, keep: usize) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Ok(Self { path, file, written, max_bytes, keep })
    }

    /// Append one line, rotating first if the line would push the file past the limit.
    pub fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        let bytes = line.len() as u64 + 1;
        if self.written > 0 && self.written + bytes > self.max_bytes {
            self.rotate()?;
        }
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.written += bytes;
        Ok(())
    }

    /// `zynkbot.log.(keep-1)` is dropped, each older generation moves up one, the
    /// live file becomes `.1`, and a fresh live file is opened.
    fn rotate(&mut self) -> std::io::Result<()> {
        let _ = self.file.flush();
        let gen = |n: usize| -> PathBuf {
            let mut p = self.path.clone().into_os_string();
            p.push(format!(".{n}"));
            PathBuf::from(p)
        };
        if self.keep > 1 {
            let _ = std::fs::remove_file(gen(self.keep - 1));
            for n in (1..self.keep - 1).rev() {
                let _ = std::fs::rename(gen(n), gen(n + 1));
            }
            std::fs::rename(&self.path, gen(1))?;
        } else {
            std::fs::remove_file(&self.path)?;
        }
        self.file = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        self.written = 0;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Where the on-disk log lives: `<app data dir>/logs/zynkbot.log`.
pub fn file_path() -> PathBuf {
    crate::db::get_app_data_dir().join("logs").join("zynkbot.log")
}

fn append_to_disk(line: &str) {
    // Test binaries share the real data folder with a running app; they keep to
    // stdout and the ring buffer.
    if cfg!(test) {
        return;
    }
    let Ok(mut state) = DISK.lock() else { return };
    if matches!(*state, DiskState::Unopened) {
        *state = match FileSink::open(file_path(), FILE_MAX_BYTES, FILE_KEEP) {
            Ok(sink) => DiskState::Open(sink),
            Err(_) => DiskState::Failed,
        };
    }
    if let DiskState::Open(sink) = &mut *state {
        if sink.write_line(line).is_err() {
            *state = DiskState::Failed;
        }
    }
}

fn push(prefix: &str, text: &str) {
    let now = chrono::Local::now();
    let line = format!("{} {prefix}{text}", now.format("%H:%M:%S%.3f"));
    if let Ok(mut buf) = BUFFER.lock() {
        if buf.len() >= CAPACITY {
            buf.pop_front();
        }
        buf.push_back(line);
    }
    append_to_disk(&format!("{} {prefix}{text}", now.format("%Y-%m-%d %H:%M:%S%.3f")));
}

/// Said once at startup so the terminal (and the file itself) name the file.
pub fn announce() {
    line(format!("[Log] On-disk log: {} (rotates at {} MB, {} files kept)",
        file_path().display(), FILE_MAX_BYTES / (1024 * 1024), FILE_KEEP));
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

/// The last `n` lines of the on-disk log, oldest first, reaching into the
/// previous generation when the live file is short. `None` when there is no
/// file to read (first run, or the sink failed to open), so the caller can fall
/// back to the ring buffer.
pub fn recent_from_disk(n: usize) -> Option<Vec<String>> {
    tail_lines(&file_path(), n)
}

/// `recent_from_disk` for an arbitrary path (so a test can use a temp file).
pub fn tail_lines(path: &Path, n: usize) -> Option<Vec<String>> {
    let live = std::fs::read_to_string(path).ok()?;
    let mut lines: Vec<String> = live.lines().map(str::to_string).collect();
    if lines.len() < n {
        let mut prev = path.as_os_str().to_owned();
        prev.push(".1");
        if let Ok(older) = std::fs::read_to_string(PathBuf::from(prev)) {
            let mut all: Vec<String> = older.lines().map(str::to_string).collect();
            all.append(&mut lines);
            lines = all;
        }
    }
    let skip = lines.len().saturating_sub(n);
    Some(lines.into_iter().skip(skip).collect())
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
    fn disk_log_rotates_by_size_and_keeps_three_generations() {
        let dir = std::env::temp_dir().join(format!("zynkbot-applog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("logs").join("zynkbot.log");
        // 100-byte limit, 3 files: lines of 40 bytes (+ newline) → two per file.
        let mut sink = FileSink::open(path.clone(), 100, 3).expect("open");
        assert!(path.exists(), "the logs folder is created on open");
        for i in 0..7 {
            sink.write_line(&format!("{i:0>40}")).expect("write");
        }
        let read = |p: &Path| std::fs::read_to_string(p).unwrap_or_default();
        let has = |p: &Path, i: usize| read(p).lines().any(|l| l == format!("{i:0>40}"));
        let g1 = PathBuf::from(format!("{}.1", path.display()));
        let g2 = PathBuf::from(format!("{}.2", path.display()));
        let g3 = PathBuf::from(format!("{}.3", path.display()));
        assert!(has(&path, 6) && read(&path).lines().count() == 1, "newest line alone in the live file: {}", read(&path));
        assert!(has(&g1, 4) && has(&g1, 5), "{}", read(&g1));
        assert!(has(&g2, 2) && has(&g2, 3), "{}", read(&g2));
        assert!(!g3.exists(), "only three generations are kept");
        assert!(!has(&g2, 0) && !has(&g2, 1) && !has(&g1, 1), "the oldest lines are gone");
        // Reopening picks up the existing size, so the next rotation is on time.
        let sink2 = FileSink::open(path.clone(), 100, 3).expect("reopen");
        assert_eq!(sink2.written, read(&path).len() as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn report_tail_comes_from_the_file_and_reaches_into_the_previous_generation() {
        let dir = std::env::temp_dir().join(format!("zynkbot-applog-tail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("zynkbot.log");
        assert!(tail_lines(&path, 5).is_none(), "no file yet → None, so the report falls back to memory");
        let mut sink = FileSink::open(path.clone(), 100, 3).expect("open");
        for i in 0..5 {
            sink.write_line(&format!("{i:0>40}")).expect("write");
        }
        // Live file holds line 4 alone; .1 holds 2 and 3; .2 holds 0 and 1.
        let tail = tail_lines(&path, 3).expect("file exists");
        assert_eq!(tail, vec![format!("{:0>40}", 2), format!("{:0>40}", 3), format!("{:0>40}", 4)]);
        let one = tail_lines(&path, 1).expect("file exists");
        assert_eq!(one, vec![format!("{:0>40}", 4)]);
        let _ = std::fs::remove_dir_all(&dir);
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
