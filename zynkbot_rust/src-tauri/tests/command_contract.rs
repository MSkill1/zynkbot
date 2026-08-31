//! Guard against silently breaking the Tauri command wire contract.
//!
//! For a `#[tauri::command]` function, the Rust parameter name IS the JSON key that
//! the frontend must send. Renaming a parameter to `_name` to silence an
//! unused-variable warning makes it stop matching what JS sends:
//!
//!   * `Option<T>` params fail **silently** — the value is always `None`.
//!   * Required params make the invoke fail at runtime.
//!
//! This already shipped as a bug once: commit `ce07340` records that `_kb_enabled`
//! was introduced to silence a warning, so knowledge-base search was skipped on
//! every query until someone noticed by accident.
//!
//! The trap is that `rustc` *recommends* the underscore ("help: if this is
//! intentional, prefix it with an underscore") and cannot see the harm, so neither
//! the compiler nor code review catches it.
//!
//! To silence an unused command parameter, write `let _ = param;` in the body and
//! leave the parameter name alone. Never run `cargo fix` on this crate.

use std::fs;
use std::path::{Path, PathBuf};

/// Parameter types that Tauri injects rather than deserializing from JS.
/// These are not part of the wire contract, so an underscore on them is harmless.
const INJECTED_TYPES: &[&str] = &[
    "State<",
    "AppHandle",
    "Window",
    "WebviewWindow",
    "tauri::",
    "Request<",
    "Channel<",
];

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Split a parameter list on commas that are at bracket depth zero, so that
/// types like `Option<Vec<(String, u32)>>` stay in one piece.
fn split_params(params: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for ch in params.chars() {
        match ch {
            '<' | '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            '>' | ')' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_string());
    }
    parts
}

/// Given source text and the byte offset just past a `#[tauri::command...]`
/// attribute, return the function name and its raw parameter list.
fn parse_signature(src: &str, from: usize) -> Option<(String, String)> {
    let rest = &src[from..];
    let fn_rel = rest.find("fn ")?;
    let after_fn = &rest[fn_rel + 3..];

    let open_rel = after_fn.find('(')?;
    let name = after_fn[..open_rel].trim().to_string();
    if name.is_empty() || name.contains('{') {
        return None;
    }

    // Walk to the matching close paren.
    let bytes: Vec<char> = after_fn.chars().collect();
    let mut idx = open_rel;
    let mut depth = 0i32;
    let mut close = None;
    while idx < bytes.len() {
        match bytes[idx] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(idx);
                    break;
                }
            }
            _ => {}
        }
        idx += 1;
    }
    let close = close?;
    let params: String = bytes[open_rel + 1..close].iter().collect();
    Some((name, params))
}

#[test]
fn no_underscore_prefixed_tauri_command_params() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    assert!(
        !files.is_empty(),
        "found no .rs files under {} — did the layout change?",
        src_dir.display()
    );

    let mut offenders = Vec::new();
    let mut commands_checked = 0usize;

    for file in &files {
        let src = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rel = file
            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
            .unwrap_or(file)
            .display()
            .to_string();

        let mut search_from = 0usize;
        while let Some(rel_pos) = src[search_from..].find("#[tauri::command") {
            let attr_pos = search_from + rel_pos;
            search_from = attr_pos + "#[tauri::command".len();

            let (fn_name, params) = match parse_signature(&src, search_from) {
                Some(v) => v,
                None => continue,
            };
            commands_checked += 1;

            for param in split_params(&params) {
                // Strip any per-parameter attribute, e.g. `#[allow(unused)] x: T`.
                let param = param.rsplit(']').next().unwrap_or(&param).trim();
                let (name, ty) = match param.split_once(':') {
                    Some((n, t)) => (n.trim(), t.trim()),
                    None => continue,
                };
                if name == "self" || name == "&self" {
                    continue;
                }
                if INJECTED_TYPES.iter().any(|inj| ty.contains(inj)) {
                    continue;
                }
                if name.starts_with('_') && name != "_" {
                    let line = src[..attr_pos].matches('\n').count() + 1;
                    offenders.push(format!(
                        "  {rel}:{line}  fn {fn_name}(...)  param `{name}: {ty}`"
                    ));
                }
            }
        }
    }

    assert!(
        commands_checked > 0,
        "scanned {} files but found no #[tauri::command] functions — the guard is not working",
        files.len()
    );

    assert!(
        offenders.is_empty(),
        "\n\n{} #[tauri::command] parameter(s) are underscore-prefixed, which silently \
         breaks the JS wire contract (see commit ce07340):\n\n{}\n\n\
         The parameter name is the JSON key the frontend sends. To silence an unused \
         parameter, use `let _ = param;` in the body and leave the name alone.\n",
        offenders.len(),
        offenders.join("\n")
    );

    println!("[guard] {commands_checked} #[tauri::command] fns checked, all parameter names intact");
}

/// Extract the identifiers listed inside `tauri::generate_handler![ ... ]`.
fn registered_commands(lib_src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = lib_src[from..].find("generate_handler![") {
        let start = from + rel + "generate_handler![".len();
        let mut depth = 1i32;
        let mut end = start;
        for (i, ch) in lib_src[start..].char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        for entry in lib_src[start..end].split(',') {
            // Strip a `//` comment from each line individually. Truncating the whole
            // entry at the first `//` would swallow identifiers that follow a comment
            // header line, which is how this guard first reported false positives.
            let cleaned = entry
                .lines()
                .map(|l| l.split("//").next().unwrap_or("").trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            let name = cleaned
                .rsplit("::")
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                out.push(name);
            }
        }
        from = end.max(start + 1);
    }
    out
}

/// Collect literal command names passed to `invoke('name')` in the frontend.
fn invoked_names(dir: &Path, out: &mut Vec<(String, String, usize)>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().and_then(|s| s.to_str()) == Some("node_modules") {
                continue;
            }
            invoked_names(&path, out);
            continue;
        }
        let is_js = matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("js") | Some("jsx") | Some("ts") | Some("tsx")
        );
        if !is_js {
            continue;
        }
        let src = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rel = path.display().to_string();
        let mut from = 0usize;
        while let Some(rel_pos) = src[from..].find("invoke(") {
            let at = from + rel_pos;
            from = at + "invoke(".len();
            let rest = &src[from..];
            let quote = match rest.chars().next() {
                Some(c @ ('\'' | '"' | '`')) => c,
                // Dynamic command name (variable/expression) — cannot be checked statically.
                _ => continue,
            };
            let body = &rest[1..];
            if let Some(close) = body.find(quote) {
                let name = &body[..close];
                if name.is_empty() || name.contains(':') || name.contains("${") {
                    continue; // plugin: routes and interpolated names are out of scope
                }
                let line = src[..at].matches('\n').count() + 1;
                out.push((name.to_string(), rel.clone(), line));
            }
        }
    }
}

#[test]
fn every_invoked_command_exists_and_is_registered() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest.join("src");
    let frontend = manifest
        .parent()
        .expect("src-tauri should have a parent")
        .join("src");

    // Command functions that actually exist in Rust.
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    let mut defined = Vec::new();
    for file in &files {
        let src = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut from = 0usize;
        while let Some(rel) = src[from..].find("#[tauri::command") {
            let at = from + rel;
            from = at + "#[tauri::command".len();
            if let Some((name, _)) = parse_signature(&src, from) {
                defined.push(name);
            }
        }
    }

    // Commands wired into the Tauri invoke handler.
    let lib_src = fs::read_to_string(src_dir.join("lib.rs")).expect("read lib.rs");
    let registered = registered_commands(&lib_src);
    assert!(
        !registered.is_empty(),
        "found no generate_handler! entries — the guard cannot verify registration"
    );

    let mut invoked = Vec::new();
    invoked_names(&frontend, &mut invoked);
    assert!(
        !invoked.is_empty(),
        "found no invoke() calls under {} — did the frontend move?",
        frontend.display()
    );

    let mut missing = Vec::new();
    let mut unregistered = Vec::new();
    for (name, file, line) in &invoked {
        let short = file.rsplit('/').next().unwrap_or(file);
        if !defined.iter().any(|d| d == name) {
            missing.push(format!("  {short}:{line}  invoke('{name}') — no #[tauri::command] fn"));
        } else if !registered.iter().any(|r| r == name) {
            unregistered.push(format!(
                "  {short}:{line}  invoke('{name}') — defined but absent from generate_handler!"
            ));
        }
    }

    let mut problems = missing;
    problems.extend(unregistered);
    problems.sort();
    problems.dedup();

    assert!(
        problems.is_empty(),
        "\n\n{} frontend invoke() call(s) do not resolve to a registered Tauri command:\n\n{}\n\n\
         A mismatch here fails only at runtime, with no compile error.\n",
        problems.len(),
        problems.join("\n")
    );

    println!(
        "[guard] {} invoke() call sites resolve against {} defined / {} registered commands",
        invoked.len(),
        defined.len(),
        registered.len()
    );
}
