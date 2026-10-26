//! Does recall find the right memory? (labs/recall_bench, 2026-10-25.) Questions against
//! a read-only copy of a real memory database, scored by where the first expected memory
//! ranks under the production hybrid formula (cosine, entity boost 0.6/0.4, threshold
//! 0.35). Four ways of embedding each memory are compared in one run, so the question
//! "is the new fact-per-memory pipeline worse at recall than storing the message?" has a
//! number: A the stored embedding (the fact, as the pipeline writes it today), B title +
//! fact, C fact + the first 300 characters of the source message, D the source message
//! alone (what the old pipeline stored and embedded).
//!
//!   RECALL_BENCH_DB=/path/copy.db LD_LIBRARY_PATH=lib/vosk \
//!   cargo test --features cuda --lib recall_bench -- --ignored --nocapture
//!
//! RECALL_BENCH_SET overrides the case file; RECALL_BENCH_OUT the results folder
//! (default ~/Documents/Zynk_related/recall_bench).
#![cfg(test)]

use std::collections::HashMap;

#[derive(serde::Deserialize)]
struct Case { q: String, expect: Vec<i32> }
#[derive(serde::Deserialize)]
struct Set { user_id: String, cases: Vec<Case> }

struct Row { id: i32, title: Option<String>, content: String, original_text: Option<String>, entities: Option<String>, stored: Option<Vec<f32>> }

fn entity_overlap(entities_json: &Option<String>, query_entities: &[String]) -> f64 {
    if query_entities.is_empty() { return 0.0; }
    let stored: Vec<String> = entities_json.as_ref()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.as_array().cloned())
        .map(|arr| arr.iter().filter_map(|e| e.get("word").and_then(|w| w.as_str()).map(|s| s.to_lowercase())).collect())
        .unwrap_or_default();
    let matches = query_entities.iter().filter(|qe| stored.iter().any(|se| se.contains(qe.as_str()) || qe.contains(se.as_str()))).count();
    matches as f64 / query_entities.len() as f64
}

fn rank_of(scores: &[(i32, f64)], expect: &[i32]) -> Option<usize> {
    scores.iter().position(|(id, _)| expect.contains(id)).map(|p| p + 1)
}

#[tokio::test]
#[ignore]
async fn recall_bench() {
    let db = std::env::var("RECALL_BENCH_DB").expect("RECALL_BENCH_DB");
    let set_path = std::env::var("RECALL_BENCH_SET").unwrap_or_else(|_| "../../labs/recall_bench/cases.json".into());
    let out_dir = std::env::var("RECALL_BENCH_OUT").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap().join("Documents/Zynk_related/recall_bench"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut set: Set = serde_json::from_str(&std::fs::read_to_string(&set_path).expect("case file")).expect("case json");
    // RECALL_BENCH_MAP: old memory id -> the ids of the facts that replaced it (memory_refactor
    // apply mode writes it); an expected old id then counts any of its replacements as a hit.
    if let Ok(mp) = std::env::var("RECALL_BENCH_MAP") {
        let map: std::collections::BTreeMap<i32, Vec<i32>> = serde_json::from_str(&std::fs::read_to_string(&mp).expect("map file")).expect("map json");
        for c in &mut set.cases { let extra: Vec<i32> = c.expect.iter().filter_map(|id| map.get(id)).flatten().copied().collect(); c.expect.extend(extra); }
    }

    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}?mode=ro", db)).await.expect("db");
    let raw: Vec<(i32, Option<String>, String, Option<String>, Option<String>, Option<Vec<u8>>)> = sqlx::query_as(
        "SELECT id, title, content, original_text, entities_detected, embedding FROM memories \
         WHERE user_id = ? AND (temporal_status IS NULL OR temporal_status != 'superseded')")
        .bind(&set.user_id).fetch_all(&pool).await.expect("rows");
    let rows: Vec<Row> = raw.into_iter().map(|(id, title, content, original_text, entities, emb)| Row {
        id, title, content, original_text, entities, stored: emb.map(|b| crate::memory::blob_to_f32_pub(&b)),
    }).collect();
    println!("{} memories", rows.len());

    // Variant texts
    let text_b: Vec<String> = rows.iter().map(|r| match &r.title { Some(t) if !t.is_empty() => format!("{}. {}", t, r.content), _ => r.content.clone() }).collect();
    let text_c: Vec<String> = rows.iter().map(|r| {
        let src = r.original_text.clone().unwrap_or_default();
        let src: String = src.chars().take(300).collect();
        if src.trim().is_empty() || src.trim() == r.content.trim() { r.content.clone() } else { format!("{} {}", r.content, src) }
    }).collect();
    let text_d: Vec<String> = rows.iter().map(|r| r.original_text.clone().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| r.content.clone())).collect();
    let text_a_check: Vec<String> = rows.iter().map(|r| r.content.clone()).collect();

    let embed = |texts: Vec<String>| crate::llm::local_embeddings::generate_local_embeddings_batch(texts, Some(16)).expect("embed");
    let t0 = std::time::Instant::now();
    let emb_a_fresh = embed(text_a_check);
    let emb_b = embed(text_b);
    let emb_c = embed(text_c);
    let emb_d = embed(text_d);
    println!("embedded 4 x {} texts in {:.1}s", rows.len(), t0.elapsed().as_secs_f32());

    // Sanity: the stored embedding should equal a fresh embedding of the content.
    let mut same = 0; let mut checked = 0;
    for (i, r) in rows.iter().enumerate() {
        if let Some(s) = &r.stored { checked += 1; if crate::llm::local_embeddings::cosine_similarity(s, &emb_a_fresh[i]) > 0.98 { same += 1; } }
    }
    println!("stored embedding == embed(content) for {}/{} memories", same, checked);

    let variants: Vec<(&str, Vec<Option<Vec<f32>>>)> = vec![
        ("A stored fact", rows.iter().map(|r| r.stored.clone()).collect()),
        ("B title+fact", emb_b.into_iter().map(Some).collect()),
        ("C fact+source", emb_c.into_iter().map(Some).collect()),
        ("D source only", emb_d.into_iter().map(Some).collect()),
    ];

    let enhancer = crate::nlp_enhancer::NLPEnhancer::new();
    let mut table = String::new();
    table.push_str("| # | question | A | A @0.25 | B | B @0.25 | C | C @0.25 | D | D @0.25 |\n|---|---|---|---|---|---|---|---|---|---|\n");
    let mut hits: HashMap<&str, (usize, usize, f64)> = HashMap::new(); // hit@4, hit@10, sum reciprocal rank
    let mut scored_cases = 0usize;
    for (ci, case) in set.cases.iter().enumerate() {
        if case.expect.is_empty() { continue; }
        scored_cases += 1;
        let qe = crate::llm::local_embeddings::generate_local_embedding(&case.q).expect("q embed");
        let ents: Vec<String> = enhancer.extract_entities(&case.q).into_iter().map(|e| e.word.to_lowercase()).filter(|e| e.len() > 1).collect();
        let mut cells = Vec::new();
        for (name, embs, threshold) in variants.iter().flat_map(|(n, e)| [(n.to_string(), e, 0.35f64), (format!("{} @0.25", n), e, 0.25f64)]) {
            let name: &str = Box::leak(name.into_boxed_str());
            let mut scores: Vec<(i32, f64)> = rows.iter().enumerate().filter_map(|(i, r)| {
                let semantic = embs[i].as_ref().map(|e| crate::llm::local_embeddings::cosine_similarity(&qe, e) as f64).unwrap_or(0.0);
                let entity = entity_overlap(&r.entities, &ents);
                let score = if entity > 0.0 { entity * 0.6 + semantic * 0.4 } else { semantic };
                if score >= threshold || entity > 0.0 { Some((r.id, score)) } else { None }
            }).collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let rank = rank_of(&scores, &case.expect);
            let e = hits.entry(name).or_insert((0, 0, 0.0));
            if let Some(rk) = rank { if rk <= 4 { e.0 += 1; } if rk <= 10 { e.1 += 1; } e.2 += 1.0 / rk as f64; }
            cells.push(rank.map(|r| r.to_string()).unwrap_or_else(|| "–".into()));
        }
        table.push_str(&format!("| {} | {} | {} |\n", ci + 1, case.q, cells.join(" | ")));
    }
    let mut summary = String::new();
    summary.push_str("| variant | found in top 4 (phone) | found in top 10 (cloud) | mean reciprocal rank |\n|---|---|---|---|\n");
    let names: Vec<String> = variants.iter().flat_map(|(n, _)| [n.to_string(), format!("{} @0.25", n)]).collect();
    for name in &names {
        let (h4, h10, mrr) = hits[name.as_str()];
        summary.push_str(&format!("| {} | {}/{} | {}/{} | {:.2} |\n", name, h4, scored_cases, h10, scored_cases, mrr / scored_cases as f64));
    }
    let report = format!("# Recall bench {}\n\n{} memories in the database copy; {} scored questions. Rank of the first expected memory under the production hybrid formula (cosine, entity boost, threshold 0.35); – means not returned at all.\n\n## Summary\n\n{}\n## Per question (rank)\n\n{}\n\nStored embedding equals a fresh embedding of the content for {}/{} memories (format check).\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M"), rows.len(), scored_cases, summary, table, same, checked);
    println!("\n{}", report);
    let path = out_dir.join(format!("RESULTS_{}.md", chrono::Local::now().format("%Y-%m-%d")));
    std::fs::write(&path, &report).unwrap();
    println!("written {}", path.display());
}

/// Does the 60/40 name-versus-meaning weighting earn its place? (Matt, 2026-10-26: "an
/// arbitrary test decision" nine months ago, never measured.) Same database copy, same
/// questions and stored embeddings as `recall_bench`; only the scoring formula varies.
///
///   RECALL_BENCH_DB=/path/copy.db [RECALL_BENCH_MAP=...] LD_LIBRARY_PATH=lib/vosk \
///   cargo test --features cuda --lib weight_bench -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn weight_bench() {
    let db = std::env::var("RECALL_BENCH_DB").expect("RECALL_BENCH_DB");
    let set_path = std::env::var("RECALL_BENCH_SET").unwrap_or_else(|_| "../../labs/recall_bench/cases.json".into());
    let out_dir = std::env::var("RECALL_BENCH_OUT").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap().join("Documents/Zynk_related/recall_bench"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let mut set: Set = serde_json::from_str(&std::fs::read_to_string(&set_path).expect("case file")).expect("case json");
    if let Ok(mp) = std::env::var("RECALL_BENCH_MAP") {
        let map: std::collections::BTreeMap<i32, Vec<i32>> = serde_json::from_str(&std::fs::read_to_string(&mp).expect("map file")).expect("map json");
        for c in &mut set.cases { let extra: Vec<i32> = c.expect.iter().filter_map(|id| map.get(id)).flatten().copied().collect(); c.expect.extend(extra); }
    }
    let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}?mode=ro", db)).await.expect("db");
    let raw: Vec<(i32, Option<String>, Option<Vec<u8>>)> = sqlx::query_as(
        "SELECT id, entities_detected, embedding FROM memories \
         WHERE user_id = ? AND (temporal_status IS NULL OR temporal_status != 'superseded') \
         AND (expires_at IS NULL OR substr(expires_at, 1, 10) >= date('now'))")
        .bind(&set.user_id).fetch_all(&pool).await.expect("rows");
    let rows: Vec<(i32, Option<String>, Option<Vec<f32>>)> = raw.into_iter()
        .map(|(id, ents, emb)| (id, ents, emb.map(|b| crate::memory::blob_to_f32_pub(&b)))).collect();

    // (name, score given entity overlap e and cosine s, include regardless of threshold when e > 0)
    type Formula = (&'static str, fn(f64, f64) -> f64, bool);
    let formulas: Vec<Formula> = vec![
        ("current 60/40",            |e, s| if e > 0.0 { 0.6 * e + 0.4 * s } else { s }, true),
        ("current, no forced entry", |e, s| if e > 0.0 { 0.6 * e + 0.4 * s } else { s }, false),
        ("never hurts: max(s, 60/40)", |e, s| if e > 0.0 { s.max(0.6 * e + 0.4 * s) } else { s }, true),
        ("meaning only",             |_, s| s, false),
        ("meaning + 0.3 x names",    |e, s| s + 0.3 * e, true),
        ("meaning + 0.6 x names",    |e, s| s + 0.6 * e, true),
        ("blend 30/70",              |e, s| if e > 0.0 { 0.3 * e + 0.7 * s } else { s }, true),
    ];
    const THRESHOLD: f64 = 0.35;

    let enhancer = crate::nlp_enhancer::NLPEnhancer::new();
    let mut table = String::from("| # | question | names found in the question | ");
    table.push_str(&formulas.iter().map(|f| f.0).collect::<Vec<_>>().join(" | "));
    table.push_str(" |\n|---|---|---|");
    table.push_str(&"---|".repeat(formulas.len()));
    table.push('\n');
    let mut hits: Vec<(usize, usize, f64)> = vec![(0, 0, 0.0); formulas.len()];
    let mut named_hits: Vec<(usize, usize, f64)> = vec![(0, 0, 0.0); formulas.len()];
    let mut scored_cases = 0usize;
    let mut named_cases = 0usize;
    for (ci, case) in set.cases.iter().enumerate() {
        if case.expect.is_empty() { continue; }
        scored_cases += 1;
        let qe = crate::llm::local_embeddings::generate_local_embedding(&case.q).expect("q embed");
        let ents: Vec<String> = enhancer.extract_entities(&case.q).into_iter().map(|e| e.word.to_lowercase()).filter(|e| e.len() > 1).collect();
        let named = !ents.is_empty();
        if named { named_cases += 1; }
        let mut cells = Vec::new();
        for (fi, (_, f, forced)) in formulas.iter().enumerate() {
            let mut scores: Vec<(i32, f64)> = rows.iter().filter_map(|(id, stored_ents, emb)| {
                let s = emb.as_ref().map(|e| crate::llm::local_embeddings::cosine_similarity(&qe, e) as f64).unwrap_or(0.0);
                let e = entity_overlap(stored_ents, &ents);
                let score = f(e, s);
                if score >= THRESHOLD || (*forced && e > 0.0) { Some((*id, score)) } else { None }
            }).collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let rank = rank_of(&scores, &case.expect);
            if let Some(rk) = rank {
                for h in [&mut hits[fi]].into_iter().chain(if named { Some(&mut named_hits[fi]) } else { None }) {
                    if rk <= 4 { h.0 += 1; } if rk <= 10 { h.1 += 1; } h.2 += 1.0 / rk as f64;
                }
            }
            cells.push(rank.map(|r| r.to_string()).unwrap_or_else(|| "–".into()));
        }
        table.push_str(&format!("| {} | {} | {} | {} |\n", ci + 1, case.q, ents.join(", "), cells.join(" | ")));
    }
    let summarise = |hits: &Vec<(usize, usize, f64)>, n: usize| {
        let mut s = String::from("| formula | found in top 4 (phone) | found in top 10 (cloud) | mean reciprocal rank |\n|---|---|---|---|\n");
        for (fi, f) in formulas.iter().enumerate() {
            let (h4, h10, mrr) = hits[fi];
            s.push_str(&format!("| {} | {}/{} | {}/{} | {:.2} |\n", f.0, h4, n, h10, n, if n > 0 { mrr / n as f64 } else { 0.0 }));
        }
        s
    };
    let report = format!(
        "# Weight bench {}\n\n{} memories in the database copy; {} scored questions, {} of them with a name the name-finder picked up. \
         Rank of the first expected memory under each scoring formula (e = share of the question's names found in the memory, s = cosine; threshold {}).\n\n\
         ## All questions\n\n{}\n## Questions with a name only\n\n{}\n## Per question (rank)\n\n{}",
        chrono::Local::now().format("%Y-%m-%d %H:%M"), rows.len(), scored_cases, named_cases, THRESHOLD,
        summarise(&hits, scored_cases), summarise(&named_hits, named_cases), table);
    println!("\n{}", report);
    let path = out_dir.join(format!("WEIGHTS_{}.md", chrono::Local::now().format("%Y-%m-%d")));
    std::fs::write(&path, &report).unwrap();
    println!("written {}", path.display());
}
