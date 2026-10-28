//! The app's own prompts as fine-tuning input (labs/finetune, 2026-10-28). Writes one
//! JSONL line per memory bench case, with the prompt text built by the same functions
//! the phone uses, so what the model is trained on is what it will be asked.
//!
//!   LD_LIBRARY_PATH=lib/vosk cargo test --features cuda --lib finetune_export -- --ignored --nocapture
//!
//! FINETUNE_OUT overrides the output folder (default ~/zynkbot-finetune/data).

#[derive(serde::Deserialize)]
struct Cases { subject: String, extraction: Vec<ExtractionCase>, relationships: Vec<RelationshipCase> }
#[derive(serde::Deserialize)]
struct ExtractionCase { id: String, message: String, #[serde(default)] context: Option<String>, expected: Vec<Expected> }
#[derive(serde::Deserialize)]
struct Expected { must: Vec<String>, #[serde(default)] days_ago: Option<i64> }
#[derive(serde::Deserialize)]
struct RelationshipCase { id: String, new_fact: String, existing: Vec<Existing>, expected: ExpectedRel }
#[derive(serde::Deserialize)]
struct Existing { id: i32, content: String, title: Option<String> }
#[derive(serde::Deserialize)]
struct ExpectedRel { memory_id: i32, #[serde(rename = "type")] kind: String }

#[test]
#[ignore]
fn finetune_export() {
    let set = std::env::var("MEMORY_BENCH_SET").unwrap_or_else(|_| format!("{}/../../labs/memory_bench/cases.json", env!("CARGO_MANIFEST_DIR")));
    let cases: Cases = serde_json::from_str(&std::fs::read_to_string(&set).expect("case file")).expect("case file parses");
    let out_dir = std::env::var("FINETUNE_OUT").unwrap_or_else(|_| format!("{}/zynkbot-finetune/data", std::env::var("HOME").unwrap_or_default()));
    std::fs::create_dir_all(&out_dir).expect("out dir");
    // A fixed date, so the prompt text is the same on every export; the targets
    // resolve "yesterday" against it.
    let today = chrono::NaiveDate::from_ymd_opt(2026, 10, 28).unwrap();

    let mut lines = Vec::new();
    for c in &cases.extraction {
        // slim = the phone's prompt (fact, title, date, passing)
        let prompt = crate::memory_prompts::extraction_prompt_with(&c.message, &cases.subject, c.context.as_deref(), today, true);
        lines.push(serde_json::json!({
            "id": c.id, "task": "extraction", "prompt": prompt, "message": c.message,
            "expected": c.expected.iter().map(|e| serde_json::json!({"must": e.must, "days_ago": e.days_ago})).collect::<Vec<_>>(),
            "today": today.to_string(),
        }).to_string());
    }
    let p = format!("{}/prompts_extraction.jsonl", out_dir);
    std::fs::write(&p, lines.join("\n") + "\n").expect("write");
    println!("{} extraction prompts -> {}", lines.len(), p);

    // Real messages, when a file of them is given (FINETUNE_MESSAGES: JSONL lines of
    // {"id", "message", "context"?}; built from the user's own threads only with their
    // consent). Same prompt builder; no expectations, the teacher's answer is the target
    // and the checker only verifies the shape.
    if let Ok(path) = std::env::var("FINETUNE_MESSAGES") {
        let mut lines = Vec::new();
        for l in std::fs::read_to_string(&path).expect("messages file").lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(l).expect("message line");
            let message = v["message"].as_str().unwrap_or("").to_string();
            if message.trim().is_empty() { continue; }
            let prompt = crate::memory_prompts::extraction_prompt_with(&message, &cases.subject, v["context"].as_str(), today, true);
            lines.push(serde_json::json!({
                "id": v["id"], "task": "extraction", "prompt": prompt, "message": message,
                "expected": serde_json::Value::Null, "today": today.to_string(),
            }).to_string());
        }
        let p = format!("{}/prompts_messages.jsonl", out_dir);
        std::fs::write(&p, lines.join("\n") + "\n").expect("write");
        println!("{} real-message prompts -> {}", lines.len(), p);
    }

    let mut lines = Vec::new();
    for c in &cases.relationships {
        let similar: Vec<(i32, String, Option<String>, f32)> = c.existing.iter().map(|e| (e.id, e.content.clone(), e.title.clone(), 0.8)).collect();
        let prompt = crate::links_only_prompt(&c.new_fact, &similar);
        lines.push(serde_json::json!({
            "id": c.id, "task": "relationships", "prompt": prompt, "new_fact": c.new_fact,
            "expected": {"memory_id": c.expected.memory_id, "type": c.expected.kind},
            "existing_ids": c.existing.iter().map(|e| e.id).collect::<Vec<_>>(),
        }).to_string());
    }
    let p = format!("{}/prompts_relationships.jsonl", out_dir);
    std::fs::write(&p, lines.join("\n") + "\n").expect("write");
    println!("{} relationship prompts -> {}", lines.len(), p);
}

/// Teacher answers through the app's own Anthropic path (2026-10-28), so the key is
/// loaded the way the app loads it and the call is the one the app makes. Writes
/// raw_<task>.jsonl; `check_targets.py` filters them. FINETUNE_TASK=extraction|relationships
/// (default both), FINETUNE_TEACHER=anthropic|openai|xai (default anthropic).
#[tokio::test]
#[ignore]
async fn finetune_targets() {
    let data_dir = std::env::var("FINETUNE_OUT").unwrap_or_else(|_| format!("{}/zynkbot-finetune/data", std::env::var("HOME").unwrap_or_default()));
    // The app's own settings files. The data folder's first: a variable already set is
    // never overridden, and the dev tree's .env carries empty key lines.
    let candidates = [crate::db::get_app_data_dir().join(".env"), std::path::PathBuf::from(".env"), std::path::PathBuf::from("../../.env")];
    for p in &candidates {
        if p.exists() {
            match crate::env_file::load(p) {
                Ok((n, _)) => println!("[Dotenv] {} setting(s) from {}", n, p.display()),
                Err(e) => println!("[Dotenv] could not read {}: {}", p.display(), e),
            }
        }
    }
    let teacher = std::env::var("FINETUNE_TEACHER").unwrap_or_else(|_| "anthropic".to_string());
    let tasks: Vec<String> = std::env::var("FINETUNE_TASK").map(|t| vec![t]).unwrap_or_else(|_| vec!["extraction".into(), "relationships".into()]);
    for task in tasks {
        let src = format!("{}/prompts_{}.jsonl", data_dir, task);
        let dst = format!("{}/raw_{}.jsonl", data_dir, task);
        let done: std::collections::HashSet<String> = std::fs::read_to_string(&dst).unwrap_or_default().lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).filter_map(|v| v["id"].as_str().map(String::from)).collect();
        let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&dst).expect("raw file");
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(&src).expect("prompts file").lines().filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("prompt line")).collect();
        println!("{}: {} prompts, {} already answered, teacher {}", task, rows.len(), done.len(), teacher);
        let mut errors = 0;
        for mut r in rows {
            let id = r["id"].as_str().unwrap_or("").to_string();
            if done.contains(&id) { continue; }
            let t = std::time::Instant::now();
            match crate::call_backend_with_prompt(r["prompt"].as_str().unwrap_or(""), &teacher).await {
                Ok(text) => {
                    r["target"] = serde_json::json!(text.trim());
                    r["teacher"] = serde_json::json!(teacher);
                    use std::io::Write;
                    writeln!(out, "{}", r).expect("write");
                    println!("  {} {:4.1}s {}", id, t.elapsed().as_secs_f32(), text.trim().chars().take(90).collect::<String>().replace('\n', " "));
                }
                Err(e) => { errors += 1; println!("  {} ERROR {}", id, e.chars().take(160).collect::<String>()); if errors >= 3 { panic!("three errors in a row; stopping"); } }
            }
        }
    }
}

/// What the app's own engine returns for one extraction prompt, raw, with timing
/// (2026-10-28: the tuned model took 7.5 s a case in the app and 1.4 s outside it).
/// LOCAL_PROBE_MODEL=<gguf> [LOCAL_PROBE_N=3]
#[test]
#[ignore]
fn local_probe() {
    let model = std::env::var("LOCAL_PROBE_MODEL").expect("LOCAL_PROBE_MODEL");
    let n: usize = std::env::var("LOCAL_PROBE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let data_dir = std::env::var("FINETUNE_OUT").unwrap_or_else(|_| format!("{}/zynkbot-finetune/data", std::env::var("HOME").unwrap_or_default()));
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(format!("{}/prompts_extraction.jsonl", data_dir)).expect("prompts").lines()
        .filter(|l| !l.trim().is_empty()).take(n).map(|l| serde_json::from_str(l).unwrap()).collect();
    for r in rows {
        let prompt = r["prompt"].as_str().unwrap().to_string();
        for (label, schema) in [("free", None), ("grammar", Some(crate::memory_prompts::EXTRACTION_SCHEMA_SLIM))] {
            let t = std::time::Instant::now();
            let msgs = vec![crate::llm::Message { role: "user".into(), content: prompt.clone() }];
            let out = match schema {
                Some(s) => crate::llm::local_models::generate_with_local_model_constrained(&model, msgs, s).map(|r| r.content),
                None => crate::llm::local_models::generate_with_local_model(&model, msgs, Some(4096), Some(0.1)).map(|r| r.content),
            };
            match out {
                Ok(text) => {
                    let head: String = text.chars().take(220).collect();
                    let tail: String = text.chars().rev().take(120).collect::<Vec<_>>().into_iter().rev().collect();
                    println!("[{}] {} {:.2}s {} chars\n  head: {:?}\n  tail: {:?}", r["id"], label, t.elapsed().as_secs_f32(), text.chars().count(), head.replace('\n', "⏎"), tail.replace('\n', "⏎"));
                }
                Err(e) => println!("[{}] {} ERROR {}", r["id"], label, e),
            }
        }
    }
}
