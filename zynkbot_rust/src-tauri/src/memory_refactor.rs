//! One-off: Matt's own memories from before the fact-per-memory pipeline (2026-10-24),
//! re-extracted into short titled facts with the pipeline's own call-2 prompt and parser
//! (labs/recall_bench is the pattern). Dry mode reads a database copy and writes a table;
//! apply mode works on the app's live database (close the app first) and keeps the old
//! memory as `superseded`, so it is undoable in the Memory Manager.
//!
//!   MEMORY_REFACTOR_DB=/path/copy.db MEMORY_REFACTOR_MODE=dry \
//!   MEMORY_REFACTOR_USER=<user id> MEMORY_REFACTOR_BACKEND=anthropic \
//!   MEMORY_REFACTOR_RENAME="Maria=Marta" LD_LIBRARY_PATH=lib/vosk \
//!   cargo test --features cuda --lib memory_refactor -- --ignored --nocapture
//!
//! MEMORY_REFACTOR_OUT: report folder (default ~/Documents/Zynk_related/memory_refactor).
//! MEMORY_REFACTOR_LIMIT: stop after N candidates (a quick look).
#![cfg(test)]

use std::fmt::Write as _;

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

fn apply_renames(text: &str, renames: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (from, to) in renames {
        let re = regex::Regex::new(&format!(r"\b{}\b", regex::escape(from))).unwrap();
        out = re.replace_all(&out, to.as_str()).into_owned();
    }
    out
}

fn is_composite(content: &str) -> bool {
    let words = content.split_whitespace().count();
    let sentences = content.matches(". ").count() + 1;
    words >= 25 || sentences > 1
}

#[derive(sqlx::FromRow)]
struct Old {
    id: i32,
    title: Option<String>,
    content: String,
    original_text: Option<String>,
    source_type: Option<String>,
    session_id: Option<String>,
    created_at: String,
    namespace: Option<String>,
    is_syncable: Option<bool>,
    is_shareable: Option<bool>,
    tags: Option<String>,
}

#[tokio::test]
#[ignore]
async fn memory_refactor() {
    load_desktop_env();
    let mode = std::env::var("MEMORY_REFACTOR_MODE").unwrap_or_else(|_| "dry".into());
    let apply = mode == "apply";
    let db = if apply { crate::db::get_db_url() } else { format!("sqlite:{}?mode=ro", std::env::var("MEMORY_REFACTOR_DB").expect("MEMORY_REFACTOR_DB")) };
    let user = std::env::var("MEMORY_REFACTOR_USER").expect("MEMORY_REFACTOR_USER");
    let backend = std::env::var("MEMORY_REFACTOR_BACKEND").unwrap_or_else(|_| "anthropic".into());
    let limit: usize = std::env::var("MEMORY_REFACTOR_LIMIT").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let renames: Vec<(String, String)> = std::env::var("MEMORY_REFACTOR_RENAME").unwrap_or_default()
        .split(';').filter_map(|p| p.split_once('=')).map(|(a, b)| (a.trim().to_string(), b.trim().to_string())).collect();
    let out_dir = std::env::var("MEMORY_REFACTOR_OUT").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap().join("Documents/Zynk_related/memory_refactor"));
    std::fs::create_dir_all(&out_dir).unwrap();

    let pool = sqlx::SqlitePool::connect(&db).await.expect("db");
    let olds: Vec<Old> = sqlx::query_as(
        "SELECT id, title, content, original_text, source_type, session_id, created_at, namespace, is_syncable, is_shareable, tags \
         FROM memories WHERE user_id = ? AND created_at < '2026-10-24' AND (temporal_status IS NULL OR temporal_status != 'superseded') \
         AND COALESCE(namespace, '') != '_zynkbot' AND COALESCE(source_type, '') NOT IN ('user_explanation', 'remember', 'explicit_remember') \
         ORDER BY created_at")
        .bind(&user).fetch_all(&pool).await.expect("rows");
    let candidates: Vec<&Old> = olds.iter().filter(|o| is_composite(&o.content)).take(limit).collect();
    println!("{} memories before 2026-10-24, {} composite candidates, mode {}", olds.len(), candidates.len(), mode);

    let mut report = String::new();
    writeln!(report, "# Memory refactor {} ({} mode)\n\n{} memories before 2026-10-24; {} composite candidates (25+ words or more than one sentence). Backend {}. Renames: {:?}\n", chrono::Local::now().format("%Y-%m-%d %H:%M"), mode, olds.len(), candidates.len(), backend, renames).unwrap();
    writeln!(report, "| old id | old title | old text (start) | new facts |\n|---|---|---|---|").unwrap();
    let (mut converted, mut kept, mut facts_made, mut inserted_total) = (0usize, 0usize, 0usize, 0usize);
    let mut kept_list: Vec<String> = Vec::new();
    let mut mapping: std::collections::BTreeMap<i32, Vec<i32>> = std::collections::BTreeMap::new();

    for o in &candidates {
        // The memory's own text is the source (MEMORY_REFACTOR_SOURCE=original for the raw
        // message instead): the old pipeline's summary is the distilled version, while the
        // raw message was often a pasted conversation or a translation and re-extracted
        // into junk (dry run 2026-10-25 18:09).
        let use_original = std::env::var("MEMORY_REFACTOR_SOURCE").map(|v| v == "original").unwrap_or(false);
        let source = if use_original { o.original_text.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| o.content.clone()) } else { o.content.clone() };
        let source = apply_renames(&source, &renames);
        let date = chrono::NaiveDate::parse_from_str(&o.created_at[..10], "%Y-%m-%d").unwrap_or_else(|_| chrono::Local::now().date_naive());
        let prompt = crate::memory_prompts::extraction_prompt_with(&source, "User", None, date, false);
        let facts = match crate::call_backend_with_schema(&prompt, &backend, crate::memory_prompts::EXTRACTION_SCHEMA).await {
            Ok(text) => crate::memory_prompts::parse_extraction(&text, &source),
            Err(e) => { println!("[refactor] {} call failed: {}", o.id, e); Vec::new() }
        };
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let old_title = o.title.clone().unwrap_or_default();
        let start: String = o.content.chars().take(90).collect::<String>().replace('|', "/").replace('\n', " ");
        if facts.is_empty() {
            kept += 1;
            kept_list.push(format!("{} {}", o.id, old_title));
            writeln!(report, "| {} | {} | {} | (kept as is) |", o.id, old_title.replace('|', "/"), start).unwrap();
            continue;
        }
        converted += 1; facts_made += facts.len();
        let cell = facts.iter().map(|f| format!("**{}**: {}{}", f.title.clone().unwrap_or_default().replace('|', "/"), f.fact.replace('|', "/"), f.extras.event_date.as_ref().map(|d| format!(" ({})", d)).unwrap_or_default())).collect::<Vec<_>>().join("<br>");
        writeln!(report, "| {} | {} | {} | {} |", o.id, old_title.replace('|', "/"), start, cell).unwrap();

        if apply {
            let mut new_ids: Vec<i32> = Vec::new();
            for f in &facts {
                let title = f.title.clone().filter(|t| !t.trim().is_empty());
                let content = if f.fact.ends_with('.') { f.fact.clone() } else { format!("{}.", f.fact) };
                let entities = crate::commands::nlp::extract_entities(content.clone()).await.unwrap_or_else(|_| serde_json::json!([]));
                let etext = crate::memory::embedding_text(title.as_deref(), &content);
                let emb = tokio::task::spawn_blocking(move || crate::llm::local_embeddings::generate_local_embedding(&etext)).await.unwrap().expect("embed");
                let blob: Vec<u8> = emb.iter().flat_map(|x| x.to_le_bytes()).collect();
                let tags = f.extras.tags.as_ref().map(|t| serde_json::to_string(t).unwrap()).or_else(|| o.tags.clone());
                let ns = f.extras.namespace.clone().or_else(|| o.namespace.clone()).unwrap_or_else(|| "general".into());
                let id: i32 = sqlx::query_scalar(
                    "INSERT INTO memories (title, content, original_text, source_type, session_id, created_at, updated_at, user_id, namespace, is_syncable, is_shareable, embedding, entities_detected, event_date, tags) \
                     VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id")
                    .bind(&title).bind(&content).bind(apply_renames(o.original_text.as_deref().filter(|t| !t.trim().is_empty()).unwrap_or(&o.content), &renames)).bind(o.source_type.clone().unwrap_or_else(|| "conversation".into())).bind(&o.session_id)
                    .bind(&o.created_at).bind(&user).bind(&ns).bind(o.is_syncable.unwrap_or(true)).bind(o.is_shareable.unwrap_or(false))
                    .bind(blob).bind(entities.to_string()).bind(&f.extras.event_date).bind(tags)
                    .fetch_one(&pool).await.expect("insert");
                new_ids.push(id);
            }
            for i in 0..new_ids.len() { for j in (i + 1)..new_ids.len() {
                let _ = crate::memory::create_memory_link(&pool, new_ids[j], new_ids[i], "related", 1.0, Some("said in the same message"), "same_message").await;
            } }
            if let Some(first) = new_ids.first() {
                let _ = sqlx::query("UPDATE memory_links SET source_memory_id = ? WHERE source_memory_id = ?").bind(first).bind(o.id).execute(&pool).await;
                let _ = sqlx::query("UPDATE memory_links SET target_memory_id = ? WHERE target_memory_id = ?").bind(first).bind(o.id).execute(&pool).await;
                let _ = sqlx::query("UPDATE memories SET temporal_status = 'superseded', updated_at = datetime('now') WHERE id = ?").bind(o.id).execute(&pool).await;
            }
            inserted_total += new_ids.len();
            println!("[refactor] {} -> {:?}", o.id, new_ids);
            mapping.insert(o.id, new_ids.clone());
        }
    }
    writeln!(report, "\n## Totals\n\n- candidates: {}\n- converted: {}\n- kept as is (no facts came back): {}\n- facts {}: {}\n", candidates.len(), converted, kept, if apply { "inserted" } else { "that would be made" }, if apply { inserted_total } else { facts_made }).unwrap();
    if !kept_list.is_empty() { writeln!(report, "Kept as is: {}\n", kept_list.join("; ")).unwrap(); }
    let path = out_dir.join(format!("{}_{}.md", if apply { "APPLIED" } else { "DRY_RUN" }, chrono::Local::now().format("%Y-%m-%d_%H%M")));
    std::fs::write(&path, &report).unwrap();
    if apply {
        let mp = out_dir.join(format!("MAPPING_{}.json", chrono::Local::now().format("%Y-%m-%d_%H%M")));
        std::fs::write(&mp, serde_json::to_string_pretty(&mapping).unwrap()).unwrap();
        println!("mapping written {}", mp.display());
    }
    println!("\nwritten {}\n", path.display());
    println!("{}", report.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n"));
}
