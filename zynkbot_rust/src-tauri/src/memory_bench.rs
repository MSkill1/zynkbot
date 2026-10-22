//! Which model makes the right memories? The same two tasks the app runs after a message,
//! fed with known inputs, scored, one table per backend (labs/memory_bench/cases.json,
//! 2026-10-22). Quality does not depend on the phone, so this runs on the desktop with
//! the same GGUF files; speed is measured on the phone separately.
//!
//!   MEMORY_BENCH_BACKENDS="/path/a.gguf,/path/b.gguf,anthropic" \
//!   LD_LIBRARY_PATH=lib/vosk cargo test --lib memory_bench -- --ignored --nocapture
//!
//! MEMORY_BENCH_SET overrides the case file; MEMORY_BENCH_OUT the results folder
//! (default ~/Documents/Zynk_related/memory_bench). Cloud backends need their key in the
//! environment or in the desktop app's .env, which is read if present.

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Cases { subject: String, extraction: Vec<ExtractionCase>, relationships: Vec<RelationshipCase> }
#[derive(Deserialize)]
struct ExtractionCase { id: String, message: String, expected: Vec<Expected> }
#[derive(Deserialize)]
struct Expected { must: Vec<String> }
#[derive(Deserialize)]
struct RelationshipCase { id: String, new_fact: String, existing: Vec<Existing>, expected: ExpectedRel }
#[derive(Deserialize)]
struct Existing { id: i32, content: String, title: Option<String> }
#[derive(Deserialize)]
struct ExpectedRel { memory_id: i32, #[serde(rename = "type")] kind: String }

#[derive(Serialize, Default)]
struct ExtractionResult { id: String, raw: String, facts: Vec<String>, found: usize, missed: usize, invented: usize, junk: usize, seconds: f32, error: Option<String> }
#[derive(Serialize, Default)]
struct RelationshipResult { id: String, expected: String, got: String, correct: bool, seconds: f32, error: Option<String> }
#[derive(Serialize)]
struct Report { backend: String, date: String, extraction: Vec<ExtractionResult>, relationships: Vec<RelationshipResult>, summary: Summary }
#[derive(Serialize, Default, Clone)]
struct Summary {
    cases_with_facts: usize, expected_facts: usize, found: usize, missed: usize, invented: usize,
    cases_without_facts: usize, clean: usize, junk: usize,
    relationship_cases: usize, relationships_correct: usize, errors: usize, seconds: f32,
}

/// Does a produced fact match an expected one: every `must` term present, case-insensitive.
fn matches(fact: &str, exp: &Expected) -> bool {
    let f = fact.to_lowercase();
    exp.must.iter().all(|m| f.contains(&m.to_lowercase()))
}

fn load_desktop_env() {
    let Some(home) = std::env::var_os("HOME") else { return };
    let path = std::path::PathBuf::from(home).join(".local/share/zynkbot/.env");
    let Ok(text) = std::fs::read_to_string(&path) else { return };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        if let Some((k, v)) = line.split_once('=') {
            let k = k.trim(); let v = v.trim().trim_matches('"').trim_matches('\'');
            if std::env::var_os(k).is_none() && !v.is_empty() { std::env::set_var(k, v); }
        }
    }
}

fn short_name(backend: &str) -> String {
    std::path::Path::new(backend).file_stem().and_then(|s| s.to_str()).unwrap_or(backend).to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn memory_bench() {
    load_desktop_env();
    let set = std::env::var("MEMORY_BENCH_SET").unwrap_or_else(|_| format!("{}/../../labs/memory_bench/cases.json", env!("CARGO_MANIFEST_DIR")));
    let cases: Cases = serde_json::from_str(&std::fs::read_to_string(&set).expect("case file")).expect("case file parses");
    let out_dir = std::env::var("MEMORY_BENCH_OUT").unwrap_or_else(|_| format!("{}/Documents/Zynk_related/memory_bench", std::env::var("HOME").unwrap_or_default()));
    std::fs::create_dir_all(&out_dir).ok();
    let backends: Vec<String> = std::env::var("MEMORY_BENCH_BACKENDS").expect("MEMORY_BENCH_BACKENDS").split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let date = chrono::Local::now().format("%Y%m%d-%H%M").to_string();

    for backend in &backends {
        println!("\n=== {} ===", backend);
        let mut report = Report { backend: backend.clone(), date: date.clone(), extraction: Vec::new(), relationships: Vec::new(), summary: Summary::default() };
        let mut sm = Summary::default();
        let t_all = std::time::Instant::now();

        for c in &cases.extraction {
            let prompt = crate::commands::chat::local_facts_prompt(&c.message, &cases.subject);
            let t = std::time::Instant::now();
            let mut r = ExtractionResult { id: c.id.clone(), ..Default::default() };
            match crate::call_backend_with_prompt(&prompt, backend).await {
                Ok(text) => {
                    r.raw = text.clone();
                    r.facts = crate::commands::chat::parse_local_facts(&text, &c.message);
                    if c.expected.is_empty() {
                        sm.cases_without_facts += 1;
                        if r.facts.is_empty() { sm.clean += 1; } else { r.junk = r.facts.len(); sm.junk += 1; }
                    } else {
                        sm.cases_with_facts += 1; sm.expected_facts += c.expected.len();
                        let mut used = vec![false; r.facts.len()];
                        for exp in &c.expected {
                            match r.facts.iter().position(|f| matches(f, exp)) {
                                Some(i) => { r.found += 1; used[i] = true; }
                                None => r.missed += 1,
                            }
                        }
                        r.invented = used.iter().filter(|u| !**u).count();
                        sm.found += r.found; sm.missed += r.missed; sm.invented += r.invented;
                    }
                }
                Err(e) => { r.error = Some(e); sm.errors += 1; if c.expected.is_empty() { sm.cases_without_facts += 1 } else { sm.cases_with_facts += 1; sm.expected_facts += c.expected.len(); sm.missed += c.expected.len(); } }
            }
            r.seconds = t.elapsed().as_secs_f32();
            let mark = if r.error.is_some() { "ERR" } else if c.expected.is_empty() { if r.junk == 0 { "clean" } else { "JUNK" } } else if r.missed == 0 && r.invented == 0 { "ok" } else { "part" };
            println!("  {} {:5} {:4.1}s  {:?}", c.id, mark, r.seconds, r.facts);
            report.extraction.push(r);
        }

        for c in &cases.relationships {
            let similar: Vec<(i32, String, Option<String>, f32)> = c.existing.iter().map(|e| (e.id, e.content.clone(), e.title.clone(), 0.8)).collect();
            let t = std::time::Instant::now();
            let mut r = RelationshipResult { id: c.id.clone(), expected: c.expected.kind.clone(), ..Default::default() };
            sm.relationship_cases += 1;
            match crate::ask_llm_for_relationships(&c.new_fact, &similar, backend).await {
                Ok((_title, rels, _extras)) => {
                    let got = rels.iter().find(|x| x.memory_id() == c.expected.memory_id).map(|x| x.relationship_type().to_string()).unwrap_or_else(|| "none".to_string());
                    r.got = got.clone();
                    r.correct = got.eq_ignore_ascii_case(&c.expected.kind);
                    if r.correct { sm.relationships_correct += 1; }
                }
                Err(e) => { r.error = Some(e); sm.errors += 1; r.got = "error".into(); }
            }
            r.seconds = t.elapsed().as_secs_f32();
            println!("  {} {} expected {:<11} got {:<11} {:4.1}s", c.id, if r.correct { "ok  " } else { "MISS" }, r.expected, r.got, r.seconds);
            report.relationships.push(r);
        }

        sm.seconds = t_all.elapsed().as_secs_f32();
        println!("--- {} summary: facts found {}/{}, invented {}, junk on {}/{} no-fact messages, relationships {}/{} correct, errors {}, {:.0}s",
            short_name(backend), sm.found, sm.expected_facts, sm.invented, sm.junk, sm.cases_without_facts, sm.relationships_correct, sm.relationship_cases, sm.errors, sm.seconds);
        report.summary = sm.clone();
        let path = format!("{}/{}-{}.json", out_dir, short_name(backend), date);
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
        println!("    written to {}", path);
    }
}
