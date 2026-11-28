//! Does the phone's prompt make the on-device model answer well? (labs/reply_bench,
//! 2026-11-09, KI-106.) The app's own on-device prompt, built by ConversationEngine with
//! known memories and messages, is run through a GGUF file on the desktop and each reply
//! is checked against its case: a statement gets a friend's reply and no memory lookup; a
//! question is answered from the memories; a detail the memories do not hold is said to
//! be missing, not invented; news and weather ask for a web search, the date does not.
//! Two prompt texts run side by side, "before" (the text shipped up to cd0fd48, which the
//! Pixel failed on 2026-11-09) and "after" (the current one), so a change to the prompt
//! is judged by the model's answers, not by reading it. Quality does not depend on the
//! phone; speed is measured there separately.
//!
//!   REPLY_BENCH_MODEL=~/.local/share/zynkbot/models/user/Sophro-4B-memory-v1-Q4_0.gguf \
//!   LD_LIBRARY_PATH=lib/vosk cargo test --features cuda --lib reply_bench -- --ignored --nocapture
//!
//! REPLY_BENCH_MODEL takes several backends, comma-separated: GGUF paths and provider names
//! (anthropic, openai, xai, mistral; keys from the environment or the desktop app's .env).
//! A GGUF gets the on-device prompt, as in the app on every platform; a provider gets the
//! full cloud prompt; both are scored alike, so a candidate for the desktop download list is
//! read next to the cloud models (2026-11-09). REPLY_BENCH_PAUSE_MS waits between provider
//! calls (free tiers answer 429 after a burst; a 429 is retried with a growing wait anyway).
//! REPLY_BENCH_SET overrides the case file; REPLY_BENCH_OUT the results folder (default
//! ~/Documents/Zynk_related/reply_bench); REPLY_BENCH_VARIANTS picks "before", "after" or
//! both (a provider has only "after"); REPLY_BENCH_RUNS (default 3) runs each case that many times, since the app samples
//! at temperature 0.7 and one lucky answer proves nothing (the pass count moved by three
//! between two runs of the same prompt on 2026-11-09). A case passes when every run does.
//! REPLY_BENCH_SWAPS names a JSON file of trial wordings, {"name": ["text in the current
//! prompt", "text to try in its place"]}; a name listed in REPLY_BENCH_VARIANTS then runs the
//! current prompt with that one replacement, so several wordings are measured in one run
//! without a rebuild (2026-11-10, the "I've noted that" wording).
//!
//! Kits (2026-11-28, labs/reply_bench/cases_grocery.json): a case naming `kit` gets that
//! kit's block before the question label, built by kits::kit_block with no answers and the
//! case's `list` of needed items (none by default), exactly as chat.rs places it. Two kinds score the action lines kits.rs
//! parses: `kit_line` passes when every entry of `actions` ("add milk", "bought|remove
//! bananas": a verb group as kits.rs reads it, then a word of the item) is matched by a
//! line in the reply and no line names anything else; `no_kit_line` passes when the reply
//! holds no action line at all.
#![cfg(test)]

use crate::conversation_engine::{ConversationEngine, ConversationTurn, Memory};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Cases { subject: String, cases: Vec<Case> }
#[derive(Deserialize, Clone)]
struct Case {
    id: String, kind: String, message: String,
    #[serde(default)] memories: Vec<CaseMemory>,
    #[serde(default)] history: Vec<ConversationTurn>,
    #[serde(default)] must: Vec<String>,
    #[serde(default)] must_not: Vec<String>,
    #[serde(default)] no_digits: bool,
    /// A kit whose block goes before the question label while the case runs (kits.rs).
    #[serde(default)] kit: Option<String>,
    /// `kit_line`: the action lines expected, as "<verb[|verb]> <item>".
    #[serde(default)] actions: Vec<String>,
    /// Items on the kit's list (needed, section "other") while the case runs; empty by default.
    #[serde(default)] list: Vec<String>,
}
#[derive(Deserialize, Clone)]
struct CaseMemory { content: String, #[serde(default)] days_ago: i64 }

#[derive(Serialize)]
struct RunResult { reply: String, pass: bool, reasons: Vec<String>, words: usize, seconds: f32 }
#[derive(Serialize)]
struct CaseResult { id: String, kind: String, message: String, runs: Vec<RunResult>, pass: bool }
#[derive(Serialize)]
struct Report { model: String, variant: String, date: String, passed: usize, total: usize, by_kind: Vec<(String, usize, usize)>, cases: Vec<CaseResult> }

/// Replies to a statement that read it as a question to look up (what the Pixel said on
/// 2026-11-09), or that talk about the saving the app does on its own.
/// "Don't have" alone is not one of them: "you don't have a dog" echoes the user's own
/// words (first run, 2026-11-09, scored as a lookup by mistake).
const LOOKUP_PHRASES: &[&str] = &[
    "don't have that", "do not have that", "don't have this", "don't have it", "don't have any record", "don't have a record",
    "don't have any information", "don't have information", "don't have a memory", "don't currently have", "no record", "in my records",
    "in my memory", "not stored", "haven't stored", "have that stored", "have it stored", "stored as a memory", "want me to remember",
    "like me to remember", "remember now", "i've saved", "i have saved", "i'll remember", "i will remember", "noted that",
    "can't confirm", "cannot confirm", "no information", "no memory", "web_search_needed",
];
const MISSING_PHRASES: &str = "not in the memories|not in my memories|not among the memories|isn't in the memories|don't have|do not have|don't currently have|not have|no record|didn't mention|did not mention|didn't say|did not say|haven't told|haven't mentioned|haven't said|didn't tell|did not tell|not sure|don't know|do not know|no information|isn't something|not something i|no mention|nothing about|haven't shared|didn't share|never mentioned|never said";

/// The phone's instruction block as shipped up to cd0fd48 (2026-10-24 text with the lines of
/// 2026-10-25/26/29 and the two KI-106 lines of 2026-11-08), name line included.
fn before_instructions(name: &str) -> String {
    format!("Answer the newest message. The recent conversation and the memories below are background; use them only when they bear on the question. If they do not hold the number or date asked for, say you do not have it; never put another figure in its place. A memory dated before today describes something that has already happened; 'about' gives the day it happened. Keep it short: a few sentences, unless they ask for more.\n\nYou are Zynkbot, a warm, honest, direct companion. Never claim to be human or to replace the people in the user's life. Say so when you are unsure. Do not flatter; no cheering on, no therapy language. When someone shares grief or frustration, acknowledge it before solutions.\n\nStored memories below are the user's own history; treat them as facts about their life. When you list plans or things coming up, say when each was mentioned (the date in front of the memory). What the user tells you is saved for them automatically: never ask whether they want it remembered, and never say you have no record of what they just said. When a memory answers the question, state the fact plainly; do not announce that you remember.\n\nFor news, weather, prices or recent events, output: WEB_SEARCH_NEEDED: [query]. Never for the date or time, or for the user's own history.\n\nThe user's name is {name}. Use their name occasionally and naturally — not at the start of every reply. Do not address them by name as the first word of your response. Use it mid-sentence or when it adds warmth, the same way a friend would.\n\n")
}

/// The "before" prompt: the current prompt's memories, conversation, date and message with
/// the old instruction block in front and the old label.
fn as_before(after_prompt: &str, name: &str) -> String {
    let body_start = ["WHAT THE USER HAS TOLD YOU BEFORE", "USER'S STORED MEMORIES", "RECENT CONVERSATION", "Today's date is"].iter()
        .filter_map(|m| after_prompt.find(m)).min().expect("the prompt has a body");
    format!("{}{}", before_instructions(name), &after_prompt[body_start..])
        .replace("USER'S MESSAGE:", "USER'S QUESTION:").replace("WHAT THE USER HAS TOLD YOU BEFORE:", "USER'S STORED MEMORIES:")
}

fn contains_any(hay: &str, alternatives: &str) -> bool {
    alternatives.split('|').any(|a| !a.trim().is_empty() && hay.contains(&a.trim().to_lowercase()))
}

fn score(c: &Case, reply: &str) -> (bool, Vec<String>) {
    // Curly apostrophes are the model's habit ("I don’t have…" slipped past "don't" in the
    // second run of 2026-11-09).
    let low = reply.to_lowercase().replace(['\u{2019}', '\u{2018}'], "'");
    let weekday = chrono::Local::now().format("%A").to_string().to_lowercase();
    let mut reasons = Vec::new();
    let mut must: Vec<String> = c.must.iter().map(|m| m.replace("{weekday}", &weekday)).collect();
    let mut must_not: Vec<String> = c.must_not.clone();
    match c.kind.as_str() {
        "statement" => {
            must_not.extend(LOOKUP_PHRASES.iter().map(|s| s.to_string()));
            // KI-113 (OnePlus, 2026-11-28): the reply was a memory line, "(about November 28,
            // 2026) User's favorite breakfast is…". A reply that opens like one of the prompt's
            // dated lines, or repeats a recalled memory word for word, is an echo, not an answer.
            let head = low.trim_start();
            let dated_head = head.starts_with('(') && head.find(')').map_or(false, |i| i < 40 && (head[..i].contains("about") || head[..i].chars().any(|c| c.is_ascii_digit())));
            if dated_head { reasons.push("opens like a memory line".into()); }
            for m in &c.memories {
                let line = m.content.trim().trim_end_matches('.').to_lowercase();
                if line.split_whitespace().count() >= 3 && low.contains(&line) { reasons.push(format!("repeats a memory verbatim: {}", m.content)); }
            }
        }
        "kit_line" | "no_kit_line" => {
            let mut text = reply.to_string();
            let found = crate::kits::take_actions(&mut text);
            if c.kind == "no_kit_line" {
                if !found.is_empty() { reasons.push(format!("a kit line where none belongs: {}", found.iter().map(|a| format!("{} {}", a.verb, a.fields.join(" | "))).collect::<Vec<_>>().join("; "))); }
            } else {
                // The verb groups are the ones kits::apply_actions accepts.
                let group = |v: &str| -> &'static str { match v { "add" | "need" | "needed" => "add", "bought" | "got" | "done" => "bought", "remove" | "delete" | "drop" => "remove", _ => "?" } };
                let mut used = vec![false; found.len()];
                for want in &c.actions {
                    let (verbs, item) = want.split_once(' ').unwrap_or((want.as_str(), ""));
                    let item = item.trim().to_lowercase();
                    let hit = found.iter().enumerate().position(|(i, a)| !used[i] && verbs.split('|').any(|v| group(v) == group(&a.verb)) && a.fields.first().map_or(false, |f| f.to_lowercase().contains(&item)));
                    match hit { Some(i) => used[i] = true, None => reasons.push(format!("no line for: {want}")) }
                }
                for (i, a) in found.iter().enumerate() {
                    if !used[i] { reasons.push(format!("extra line: {} {}", a.verb, a.fields.join(" | "))); }
                }
                if found.is_empty() && reply.contains("[kit") { reasons.push("a kit line the app cannot parse".into()); }
            }
        }
        "recall" => must_not.extend(["web_search_needed", "don't have that", "don't have a record", "don't have any record", "no record"].iter().map(|s| s.to_string())),
        "missing" => { must.push(MISSING_PHRASES.to_string()); must_not.push("web_search_needed".into()); }
        "web" => must.push("web_search_needed".into()),
        "nosearch" => must_not.push("web_search_needed".into()),
        other => reasons.push(format!("unknown kind {other}")),
    }
    for m in &must { if !contains_any(&low, m) { reasons.push(format!("missing: {m}")); } }
    for m in &must_not { if low.contains(&m.to_lowercase()) { reasons.push(format!("says: {m}")); } }
    if c.no_digits {
        // The memories' own dates may be repeated ("you bought them on October 30, 2026");
        // any other number is one the memories do not hold.
        let mut rest = reply.to_string();
        for m in &c.memories {
            let day = chrono::Utc::now() - chrono::Duration::days(m.days_ago);
            for f in ["%B %-d, %Y", "%B %-d", "%-d %B %Y", "%Y-%m-%d", "%-m/%-d/%Y", "%-m/%-d"] { rest = rest.replace(&day.format(f).to_string(), ""); }
        }
        if rest.chars().any(|ch| ch.is_ascii_digit()) { reasons.push("a number where the memories hold none".into()); }
    }
    if reply.trim().is_empty() { reasons.push("empty reply".into()); }
    (reasons.is_empty(), reasons)
}

fn memories_of(c: &Case) -> Vec<Memory> {
    c.memories.iter().enumerate().map(|(i, m)| Memory {
        id: i as i32 + 1, content: m.content.clone(), original_text: None, title: None, similarity: Some(0.9),
        created_at: chrono::Utc::now() - chrono::Duration::days(m.days_ago), event_date: None,
    }).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn reply_bench() {
    // REPLY_BENCH_MODEL: GGUF paths and provider names (anthropic, openai, xai, mistral),
    // comma-separated. A GGUF gets the on-device prompt, which is what the app sends every
    // local model, desktop included (chat.rs builds the engine with on_device for any .gguf);
    // a provider gets the full cloud prompt. Both are scored the same way, so a candidate for
    // the desktop download list is read next to the cloud models (2026-11-09).
    crate::memory_bench::load_desktop_env();
    let home = std::env::var("HOME").unwrap_or_default();
    let backends: Vec<String> = std::env::var("REPLY_BENCH_MODEL").expect("REPLY_BENCH_MODEL=/path/model.gguf[,anthropic,…]")
        .split(',').map(|s| s.trim().replace('~', &home)).filter(|s| !s.is_empty()).collect();
    let set = std::env::var("REPLY_BENCH_SET").unwrap_or_else(|_| format!("{}/../../labs/reply_bench/cases.json", env!("CARGO_MANIFEST_DIR")));
    let cases: Cases = serde_json::from_str(&std::fs::read_to_string(&set).expect("case file")).expect("case file parses");
    let out_dir = std::env::var("REPLY_BENCH_OUT").unwrap_or_else(|_| format!("{home}/Documents/Zynk_related/reply_bench"));
    std::fs::create_dir_all(&out_dir).ok();
    let variants: Vec<String> = std::env::var("REPLY_BENCH_VARIANTS").unwrap_or_else(|_| "before,after".into()).split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let runs: usize = std::env::var("REPLY_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let pause_ms: Option<u64> = std::env::var("REPLY_BENCH_PAUSE_MS").ok().and_then(|v| v.parse().ok());
    let date = chrono::Local::now().format("%Y%m%d-%H%M").to_string();
    let swaps: std::collections::HashMap<String, (String, String)> = std::env::var("REPLY_BENCH_SWAPS").ok()
        .map(|p| serde_json::from_str(&std::fs::read_to_string(p.replace('~', &home)).expect("swap file")).expect("swap file parses"))
        .unwrap_or_default();
    for backend in &backends {
        let is_gguf = backend.ends_with(".gguf");
        let engine = ConversationEngine::new().with_on_device(is_gguf);
        // "before" is the phone's old instruction block; a provider has only the current prompt.
        let variants: Vec<&String> = variants.iter().filter(|v| is_gguf || v.as_str() != "before").collect();
        let model_name = if is_gguf { std::path::Path::new(backend).file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string() } else { backend.clone() };
        let mut reports: Vec<Report> = Vec::new();
        for variant in variants {
            println!("\n=== {} · {} · {} run(s) per case ===", model_name, variant, runs);
            let mut results = Vec::new();
            for c in &cases.cases {
                let mems = memories_of(c);
                let mut after = engine.build_prompt(&c.message, if c.history.is_empty() { None } else { Some(&c.history) }, if mems.is_empty() { None } else { Some(&mems) }, !is_gguf, Some(&cases.subject));
                if let Some(kit_id) = &c.kit {
                    // As chat.rs does, with no answers and an empty list; before the swaps, so a trial may reword the kit's text too.
                    let kit = crate::kits::get(kit_id).unwrap_or_else(|| panic!("case {} names an unknown kit {kit_id}", c.id));
                    let items: Vec<crate::kits::KitItem> = c.list.iter().enumerate().map(|(i, n)| crate::kits::KitItem {
                        id: i as i32 + 1, name: n.clone(), section: "other".into(), quantity: None, price: None, state: "needed".into(),
                        last_bought: None, bought_count: 0, created_at: String::new(), updated_at: None }).collect();
                    after = crate::kits::place_block(&after, &crate::kits::kit_block(kit, &Default::default(), &items));
                }
                let prompt = if variant == "before" { as_before(&after, &cases.subject) }
                    else if let Some((old, new)) = swaps.get(variant.as_str()) {
                        assert!(after.contains(old.as_str()), "the text the trial '{variant}' replaces is not in the prompt");
                        after.replace(old.as_str(), new)
                    } else { after };
                let mut run_results = Vec::new();
                for _ in 0..runs {
                    let t = std::time::Instant::now();
                    let reply = if is_gguf {
                        let p = prompt.clone(); let m = backend.clone();
                        tokio::task::spawn_blocking(move || {
                            crate::llm::local_models::generate_with_local_model(&m, vec![crate::llm::Message { role: "user".into(), content: p }], Some(400), None)
                        }).await.unwrap().map(|r| r.content).unwrap_or_else(|e| format!("ERROR: {e}"))
                    } else {
                        // A provider on a free tier answers 429 after a burst; wait and retry,
                        // and pause between calls when REPLY_BENCH_PAUSE_MS is set.
                        let mut attempt = 0;
                        loop {
                            match crate::call_backend_with_prompt(&prompt, backend).await {
                                Ok(r) => break r,
                                Err(e) if e.contains("429") && attempt < 6 => { attempt += 1; tokio::time::sleep(std::time::Duration::from_secs(5 * attempt)).await; }
                                Err(e) => break format!("ERROR: {e}"),
                            }
                        }
                    };
                    if let Some(ms) = pause_ms { tokio::time::sleep(std::time::Duration::from_millis(ms)).await; }
                    let raw = reply.clone();
                    let reply = crate::conversation_engine::strip_memory_marker(&reply).to_string();
                    if reply.trim().is_empty() { println!("      (raw reply: {:?})", raw.chars().take(300).collect::<String>()); }
                    let (pass, reasons) = score(c, &reply);
                    run_results.push(RunResult { words: reply.split_whitespace().count(), seconds: t.elapsed().as_secs_f32(), reply, pass, reasons });
                }
                let pass = run_results.iter().all(|r| r.pass);
                println!("{} {:<9} {} {}", if pass { "✅" } else { "❌" }, c.kind, c.id, run_results.iter().map(|r| if r.pass { "·".to_string() } else { r.reasons.join("; ") }).collect::<Vec<_>>().join(" | "));
                for r in &run_results { println!("      ↳ {}", r.reply.replace('\n', " ").chars().take(220).collect::<String>()); }
                results.push(CaseResult { id: c.id.clone(), kind: c.kind.clone(), message: c.message.clone(), runs: run_results, pass });
            }
            let mut by_kind: Vec<(String, usize, usize)> = Vec::new();
            for r in &results {
                match by_kind.iter_mut().find(|(k, _, _)| *k == r.kind) {
                    Some(e) => { e.2 += 1; if r.pass { e.1 += 1; } }
                    None => by_kind.push((r.kind.clone(), r.pass as usize, 1)),
                }
            }
            let passed = results.iter().filter(|r| r.pass).count();
            let secs: f32 = results.iter().flat_map(|r| r.runs.iter().map(|x| x.seconds)).sum::<f32>() / (results.len() * runs).max(1) as f32;
            println!("--- {} {}: {} of {} cases pass; {}; {:.1} s per reply", model_name, variant, passed, results.len(), by_kind.iter().map(|(k, p, t)| format!("{k} {p}/{t}")).collect::<Vec<_>>().join(", "), secs);
            reports.push(Report { model: model_name.clone(), variant: variant.clone(), date: date.clone(), passed, total: results.len(), by_kind, cases: results });
        }

        // One JSON per variant, and one readable file with the replies side by side.
        for r in &reports {
            std::fs::write(format!("{out_dir}/{}-{}-{date}.json", r.model, r.variant), serde_json::to_string_pretty(r).unwrap()).unwrap();
        }
        let mut md = format!("# Reply bench · {} · {}\n\nRuns per case: {}. A case passes when every run passes. A local model samples with a fixed seed, so its runs differ only when the prompt's clock minute changes.\n\n| case | kind | {} |\n|---|---|{}|\n", model_name, date, runs, reports.iter().map(|r| r.variant.as_str()).collect::<Vec<_>>().join(" | "), reports.iter().map(|_| "---").collect::<Vec<_>>().join("|"));
        for (i, c) in cases.cases.iter().enumerate() {
            md.push_str(&format!("| {} | {} | {} |\n", c.id, c.kind, reports.iter().map(|r| if r.cases[i].pass { "✅" } else { "❌" }).collect::<Vec<_>>().join(" | ")));
        }
        md.push_str(&format!("| **total** | | {} |\n\n", reports.iter().map(|r| format!("**{}/{}**", r.passed, r.total)).collect::<Vec<_>>().join(" | ")));
        for (i, c) in cases.cases.iter().enumerate() {
            md.push_str(&format!("## {} ({})\n\n**Message:** {}\n\n", c.id, c.kind, c.message));
            if !c.memories.is_empty() { md.push_str(&format!("**Memories:** {}\n\n", c.memories.iter().map(|m| format!("“{}” ({} days ago)", m.content, m.days_ago)).collect::<Vec<_>>().join("; "))); }
            for r in &reports {
                let cr = &r.cases[i];
                md.push_str(&format!("**{}** — {}\n\n", r.variant, if cr.pass { "pass" } else { "FAIL" }));
                for run in &cr.runs { md.push_str(&format!("> {}\n>\n> *{}{:.1} s, {} words*\n\n", run.reply.replace('\n', "\n> "), if run.reasons.is_empty() { String::new() } else { format!("{} · ", run.reasons.join("; ")) }, run.seconds, run.words)); }
            }
        }
        let md_path = format!("{out_dir}/RESULTS_{model_name}_{date}.md");
        std::fs::write(&md_path, md).unwrap();
        println!("\nWritten: {md_path}");
    }
}

#[cfg(test)]
mod scoring_tests {
    use super::*;
    fn case(kind: &str, must: &[&str]) -> Case {
        Case { id: "t".into(), kind: kind.into(), message: "m".into(), memories: vec![], history: vec![], must: must.iter().map(|s| s.to_string()).collect(), must_not: vec![], no_digits: false, kit: None, actions: vec![], list: vec![] }
    }
    #[test]
    fn a_statement_fails_when_the_reply_is_a_memory_line() {
        // KI-113: the OnePlus answered a statement with the shape of a recalled line.
        let mut c = case("statement", &[]);
        c.memories.push(CaseMemory { content: "Matt has a dog named Pepper.".into(), days_ago: 3 });
        assert!(!score(&c, "(about November 28, 2026) User's favorite breakfast is oatmeal with blueberries, eaten most weekdays.").0);
        assert!(!score(&c, "(November 25, 2026) Matt has a dog named Pepper.").0);
        assert!(!score(&c, "Nice. Matt has a dog named Pepper, so a walk is on the cards.").0, "a memory repeated word for word is an echo");
        assert!(score(&c, "Oatmeal with blueberries most weekdays sounds like a steady start. Does Pepper get any?").0);
        assert!(score(&c, "(Honestly) that sounds good.").0, "a bracket without a date is fine");
    }
    #[test]
    fn kit_lines_are_scored_by_verb_group_and_item() {
        let mut c = case("kit_line", &[]); c.kit = Some("grocery".into()); c.actions = vec!["add milk".into(), "add eggs".into()];
        assert!(score(&c, "Added milk and eggs.\n[kit:grocery add | milk | dairy]\n[kit:grocery need | eggs | dairy | a dozen]").0, "need is in the add group");
        assert!(!score(&c, "Added milk.\n[kit:grocery add | milk | dairy]").0, "eggs missing");
        assert!(!score(&c, "Added milk and eggs.\n[kit:grocery add | milk | dairy]\n[kit:grocery add | eggs | dairy]\n[kit:grocery add | bread | bakery]").0, "bread was not named");
        assert!(!score(&c, "Milk and eggs, got it.").0, "no lines at all");
        let mut b = case("kit_line", &[]); b.actions = vec!["bought|remove bananas".into()];
        assert!(score(&b, "Crossed off bananas.\n[kit:grocery bought | bananas]").0);
        assert!(score(&b, "[kit:grocery remove | bananas]").0);
        assert!(!score(&b, "[kit:grocery add | bananas | produce]").0);
        let n = case("no_kit_line", &[]);
        assert!(score(&n, "It's about half past three.").0);
        assert!(!score(&n, "Noted.\n[kit:grocery add | milk | dairy]").0);
    }
    #[test]
    fn a_statement_fails_when_the_reply_looks_things_up() {
        let c = case("statement", &[]);
        assert!(!score(&c, "You said you live at 1626 South Greenfield Court, but I don't have that information stored.").0);
        assert!(!score(&c, "Is that something you want me to remember?").0);
        assert!(score(&c, "Oh no, a boat trailer into the mailbox is a rough start to the day. Hope Theo is making it right.").0);
        assert!(score(&c, "So no dog now, but you had a cat named Isabel who lived to nineteen. That is a long life.").0, "echoing the user's 'don't have' is not a lookup");
        assert!(!score(&c, "I don't have that address in my records.").0);
        assert!(!score(&c, "I don\u{2019}t have that information stored for you.").0, "a curly apostrophe is still a lookup");
    }
    #[test]
    fn a_missing_detail_passes_only_when_said_to_be_missing_without_a_number() {
        let mut c = case("missing", &[]); c.no_digits = true;
        assert!(score(&c, "You didn't mention what the tires cost, so I don't have that.").0);
        assert!(!score(&c, "You paid $420 for them at Hank's Garage.").0);
        assert!(!score(&c, "They were about 400 dollars, I don't have the exact figure.").0, "a number where none is held fails even with the disclaimer");
        c.memories.push(CaseMemory { content: "Matt bought snow tires at Hank's Garage.".into(), days_ago: 10 });
        let day = (chrono::Utc::now() - chrono::Duration::days(10)).format("%B %-d, %Y").to_string();
        assert!(score(&c, &format!("You bought them at Hank's Garage on {day}, but I don't have the price.")).0, "the memory's own date is not an invented number");
    }
    #[test]
    fn recall_web_and_nosearch_rules() {
        assert!(score(&case("recall", &["captain"]), "Dana's parrot is called Captain.").0);
        assert!(!score(&case("recall", &["captain"]), "I don't have a record of a parrot.").0);
        assert!(score(&case("web", &[]), "WEB_SEARCH_NEEDED: weather tomorrow").0);
        assert!(!score(&case("nosearch", &["canberra"]), "WEB_SEARCH_NEEDED: capital of Australia").0);
        let weekday = chrono::Local::now().format("%A").to_string();
        assert!(score(&case("nosearch", &["{weekday}"]), &format!("Today is {weekday}.")).0);
    }
    #[test]
    fn the_before_prompt_is_the_old_block_on_the_same_body() {
        let e = ConversationEngine::new().with_on_device(true);
        let after = e.build_prompt("I live at 12 Oak St.", None, None, false, Some("Matt"));
        let before = as_before(&after, "Matt");
        assert!(before.starts_with("Answer the newest message."));
        assert!(before.contains("USER'S QUESTION: I live at 12 Oak St.") && !before.contains("USER'S MESSAGE"));
        let mem = vec![Memory { id: 1, content: "Matt lives on Maple Street.".into(), original_text: None, title: None, similarity: Some(0.9), created_at: chrono::Utc::now(), event_date: None }];
        let with_mem = e.build_prompt("I moved.", None, Some(&mem), false, Some("Matt"));
        assert!(with_mem.contains("WHAT THE USER HAS TOLD YOU BEFORE:\n") && !with_mem.contains("STORED"));
        assert!(as_before(&with_mem, "Matt").contains("USER'S STORED MEMORIES:\n1. ("));
        assert!(!before.contains("reply to it briefly, as a friend would"));
        assert!(after.contains("USER'S MESSAGE: I live at 12 Oak St."));
    }
}
