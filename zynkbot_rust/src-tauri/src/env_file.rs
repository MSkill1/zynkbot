//! The settings file (`.env` in the app data folder): one `KEY=value` per line.
//!
//! Written by `apply_env_key`, `remove_env_key` and `set_preferred_backend`; read
//! at startup by `load`, and by the Android side (`EnvFile.kt`, which tolerates a
//! quoted value). Before 2026-10-06 the file was read with the `dotenv` crate,
//! which treats a backslash as an escape and stops at the first line it cannot
//! parse. On Windows the local-model path (`C:\Users\...`) is the first line, so
//! the whole file was ignored at every start and the laptop showed no cloud
//! models although every key had arrived (KI-083). Now: a value that needs it is
//! written in quotes, and the loader reads line by line, skipping only the line
//! that is broken.

use std::path::Path;

/// `KEY=value`, quoting the value when a plain write would not read back as is.
pub fn format_line(key: &str, value: &str) -> String {
    format!("{}={}", key, quote_value(value))
}

fn needs_quotes(value: &str) -> bool {
    value.is_empty()
        || value != value.trim()
        || value.chars().any(|c| matches!(c, '\\' | '"' | '\'' | '$' | '#' | ' ' | '\t'))
}

/// Single quotes when the value has none (nothing inside them is interpreted);
/// double quotes with `\`, `"` and `$` escaped otherwise.
pub fn quote_value(value: &str) -> String {
    if !needs_quotes(value) {
        return value.to_string();
    }
    if !value.contains('\'') {
        return format!("'{}'", value);
    }
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '\\' | '"' | '$' => { out.push('\\'); out.push(c); }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// One line → `(key, value)`. Blank lines, comments and lines without `=` give
/// `None`. Single quotes are literal; double quotes honour `\\`, `\"`, `\$`;
/// an unquoted value is taken as is, backslashes included.
pub fn parse_line(line: &str) -> Option<(String, String)> {
    let line = line.trim_end_matches('\r');
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }
    let trimmed = trimmed.strip_prefix("export ").map(str::trim).unwrap_or(trimmed);
    let (key, raw) = trimmed.split_once('=')?;
    let key = key.trim();
    if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    let raw = raw.trim();
    let value = if let Some(inner) = raw.strip_prefix('\'').and_then(|r| r.strip_suffix('\'')) {
        inner.to_string()
    } else if let Some(inner) = raw.strip_prefix('"').and_then(|r| r.strip_suffix('"')) {
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some(n @ ('\\' | '"' | '$' | '\'')) => out.push(n),
                    Some('n') => out.push('\n'),
                    Some(n) => { out.push('\\'); out.push(n); }
                    None => out.push('\\'),
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        raw.to_string()
    };
    Some((key.to_string(), value))
}

/// Load every readable line into the process environment (never overriding a
/// variable that is already set, as `dotenv` did not). Returns how many were set
/// and the 1-based numbers of the lines that could not be read.
pub fn load(path: &Path) -> std::io::Result<(usize, Vec<usize>)> {
    let content = std::fs::read_to_string(path)?;
    let mut loaded = 0;
    let mut skipped = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        match parse_line(line) {
            Some((k, v)) => {
                if std::env::var_os(&k).is_none() {
                    std::env::set_var(&k, &v);
                }
                loaded += 1;
            }
            None => skipped.push(i + 1),
        }
    }
    Ok((loaded, skipped))
}

/// Replace or append `KEY=value` in the file (other lines untouched).
pub fn upsert(path: &Path, key: &str, value: &str) -> Result<(), String> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let prefix = format!("{}=", key);
    let line = format_line(key, value);
    match lines.iter_mut().find(|l| l.starts_with(&prefix)) {
        Some(existing) => *existing = line,
        None => lines.push(line),
    }
    std::fs::write(path, lines.join("\n"))
        .map_err(|e| format!("Failed to write .env file at {:?}: {}", path, e))
}

/// Drop every `KEY=` line from the file.
pub fn remove(path: &Path, key: &str) -> Result<(), String> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let prefix = format!("{}=", key);
    let lines: Vec<&str> = content.lines().filter(|l| !l.starts_with(&prefix)).collect();
    std::fs::write(path, lines.join("\n"))
        .map_err(|e| format!("Failed to write .env file at {:?}: {}", path, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(v: &str) {
        let line = format_line("K", v);
        let (k, back) = parse_line(&line).expect(&line);
        assert_eq!(k, "K");
        assert_eq!(back, v, "line was {line}");
    }

    #[test]
    fn values_read_back_exactly_as_written() {
        round_trip(r"C:\Users\matts\AppData\Local\zynkbot\models\user\Qwen3-8B-Q4_K_M.gguf");
        round_trip("sk-ant-api03-plainkey");
        round_trip("has a space");
        round_trip("hash#inside");
        round_trip("dollar$inside");
        round_trip("it's");
        round_trip(r#"both ' and " and \ and $"#);
        round_trip("");
        round_trip(" padded ");
    }

    #[test]
    fn plain_values_stay_plain_and_windows_paths_get_quotes() {
        assert_eq!(format_line("A", "sk-plain"), "A=sk-plain");
        assert_eq!(format_line("P", r"C:\x\y"), r"P='C:\x\y'");
    }

    #[test]
    fn an_old_unquoted_windows_path_still_reads() {
        // Files written before this module hold the path bare; it must not be lost.
        let (k, v) = parse_line(r"ZYNK_MODEL_BACKEND=C:\Users\m\model.gguf").unwrap();
        assert_eq!(k, "ZYNK_MODEL_BACKEND");
        assert_eq!(v, r"C:\Users\m\model.gguf");
    }

    #[test]
    fn comments_blanks_and_junk_are_skipped_not_fatal() {
        assert!(parse_line("# comment").is_none());
        assert!(parse_line("   ").is_none());
        assert!(parse_line("no equals sign here").is_none());
        assert!(parse_line("bad key!=x").is_none());
        assert_eq!(parse_line("export X=1"), Some(("X".into(), "1".into())));
        assert_eq!(parse_line("Y=2\r"), Some(("Y".into(), "2".into())));
    }

    #[test]
    fn load_sets_every_good_line_and_reports_the_bad_one() {
        let dir = std::env::temp_dir().join(format!("zynkbot-envfile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(".env");
        std::fs::write(&p, "ZB_T_PATH=C:\\Users\\m\\model.gguf\nthis line is broken\nZB_T_KEY=sk-test\n\n# note\nZB_T_Q='a b'\n").unwrap();
        let (loaded, skipped) = load(&p).unwrap();
        assert_eq!(loaded, 3);
        assert_eq!(skipped, vec![2]);
        assert_eq!(std::env::var("ZB_T_PATH").unwrap(), r"C:\Users\m\model.gguf");
        assert_eq!(std::env::var("ZB_T_KEY").unwrap(), "sk-test");
        assert_eq!(std::env::var("ZB_T_Q").unwrap(), "a b");
        for k in ["ZB_T_PATH", "ZB_T_KEY", "ZB_T_Q"] { std::env::remove_var(k); }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn upsert_and_remove_keep_other_lines() {
        let dir = std::env::temp_dir().join(format!("zynkbot-envfile-up-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(".env");
        upsert(&p, "A", "1").unwrap();
        upsert(&p, "B", r"C:\path").unwrap();
        upsert(&p, "A", "2").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "A=2\nB='C:\\path'");
        remove(&p, "A").unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "B='C:\\path'");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
