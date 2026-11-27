//! Kits: packaged, opt-in uses of the memory system (labs/kits/KITS_DESIGN.md, 2026-11-27).
//!
//! A kit is a JSON file shipped with the app (`src-tauri/kits/*.json`): standing
//! instructions that go in front of the prompt while the kit is on, a memory namespace
//! its items live in, a few opening questions whose answers the model is told, and two or
//! three chips above the message box. A kit has no code of its own. The one piece of
//! mechanism here is *actions*: a kit may ask the model to end its reply with lines such as
//! `[kit:grocery add | milk | dairy | 1 gallon]`; those lines are taken off the reply and
//! turned into memory rows (one row per item, its state in tags), so "we're out of milk"
//! becomes a list item the same way on every backend, the phone's small model included,
//! without a round through the memory extractor. The memory rows sync, back up and show in
//! the Memory Manager like any other, and "forget this kit" deletes the namespace.

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KitQuestion { pub id: String, pub text: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KitPrompt {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KitActions {
    #[serde(default)] pub note: String,
    #[serde(default)] pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kit {
    pub id: String,
    pub name: String,
    #[serde(default)] pub version: u32,
    #[serde(default)] pub icon: String,
    pub tagline: String,
    pub namespace: String,
    pub instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")] pub actions: Option<KitActions>,
    #[serde(default)] pub questions: Vec<KitQuestion>,
    #[serde(default)] pub prompts: Vec<KitPrompt>,
    #[serde(default)] pub sections: Vec<String>,
}

const KIT_FILES: &[&str] = &[
    include_str!("../kits/grocery.json"),
    include_str!("../kits/kitchen.json"),
];

/// Kits that do not exist yet, shown greyed so the shape of the idea is visible.
pub const FUTURE_KITS: &[(&str, &str)] = &[("Garden", "🌱"), ("Workouts", "🏃"), ("Reading", "📖"), ("Household", "🏠")];

static KITS: Lazy<Vec<Kit>> = Lazy::new(|| {
    KIT_FILES.iter().filter_map(|s| match serde_json::from_str::<Kit>(s) {
        Ok(k) => Some(k),
        Err(e) => { eprintln!("[Kits] a kit file does not parse: {}", e); None }
    }).collect()
});

pub fn all() -> &'static [Kit] { &KITS }
pub fn get(id: &str) -> Option<&'static Kit> { KITS.iter().find(|k| k.id == id) }

// ── settings ─────────────────────────────────────────────────────────────────

fn enabled_key(id: &str) -> String { format!("kit_{}_enabled", id) }
fn answers_key(id: &str) -> String { format!("kit_{}_answers", id) }

async fn setting(pool: &SqlitePool, key: &str) -> Option<String> {
    sqlx::query_scalar("SELECT value FROM app_settings WHERE key = ?").bind(key)
        .fetch_optional(pool).await.ok().flatten()
}

async fn put_setting(pool: &SqlitePool, key: &str, value: &str) -> Result<(), String> {
    sqlx::query("INSERT INTO app_settings (key, value, updated_at) VALUES (?, ?, datetime('now')) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at")
        .bind(key).bind(value).execute(pool).await.map_err(|e| format!("Could not save the kit setting: {}", e))?;
    Ok(())
}

/// Off unless switched on.
pub async fn is_enabled(pool: &SqlitePool, id: &str) -> bool {
    setting(pool, &enabled_key(id)).await.as_deref() == Some("1")
}

pub async fn set_enabled(pool: &SqlitePool, id: &str, on: bool) -> Result<(), String> {
    put_setting(pool, &enabled_key(id), if on { "1" } else { "0" }).await
}

/// The answers to the opening questions, by question id.
pub async fn answers(pool: &SqlitePool, id: &str) -> serde_json::Map<String, serde_json::Value> {
    setting(pool, &answers_key(id)).await
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

pub async fn set_answers(pool: &SqlitePool, id: &str, answers: &serde_json::Map<String, serde_json::Value>) -> Result<(), String> {
    put_setting(pool, &answers_key(id), &serde_json::Value::Object(answers.clone()).to_string()).await
}

pub async fn enabled_kits(pool: &SqlitePool) -> Vec<&'static Kit> {
    let mut out = Vec::new();
    for k in all() { if is_enabled(pool, &k.id).await { out.push(k); } }
    out
}

// ── the prompt block ─────────────────────────────────────────────────────────

/// What goes in front of the prompt while kits are on: each kit's instructions, what the
/// user answered when switching it on, the items it currently holds, and the action
/// lines the model is asked to end with. None when no kit is on.
pub async fn prompt_block(pool: &SqlitePool, user_id: &str) -> Option<String> {
    let kits = enabled_kits(pool).await;
    if kits.is_empty() { return None; }
    let mut parts = Vec::new();
    for kit in kits {
        let mut s = format!("{} kit is on. {}", kit.name, kit.instructions.trim());
        let answers = answers(pool, &kit.id).await;
        let said: Vec<String> = kit.questions.iter().filter_map(|q| {
            let a = answers.get(&q.id)?.as_str()?.trim().to_string();
            if a.is_empty() { None } else { Some(format!("{} {}", q.text.trim_end_matches('?').trim_end_matches(')').split('(').next().unwrap_or("").trim(), a)) }
        }).collect();
        if !said.is_empty() { s.push_str(&format!("\nWhen the kit was switched on the user said: {}.", said.join("; "))); }
        if kit.actions.is_some() {
            let items = list_items(pool, user_id, &kit.id).await;
            let needed: Vec<&KitItem> = items.iter().filter(|i| i.state == "needed").collect();
            if needed.is_empty() {
                s.push_str("\nThe list is empty right now.");
            } else {
                let mut by_section: Vec<(String, Vec<String>)> = Vec::new();
                for i in &needed {
                    let label = match &i.quantity { Some(q) => format!("{} ({})", i.name, q), None => i.name.clone() };
                    match by_section.iter_mut().find(|(s, _)| *s == i.section) {
                        Some((_, v)) => v.push(label),
                        None => by_section.push((i.section.clone(), vec![label])),
                    }
                }
                let line = by_section.iter().map(|(s, v)| format!("{}: {}", s, v.join(", "))).collect::<Vec<_>>().join("; ");
                s.push_str(&format!("\nOn the list now (needed): {}.", line));
            }
        }
        if let Some(a) = &kit.actions {
            s.push_str(&format!("\n{}\n{}", a.note.trim(), a.examples.join("\n")));
        }
        parts.push(s);
    }
    Some(parts.join("\n\n"))
}

// ── actions ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct KitAction { pub kit: String, pub verb: String, pub fields: Vec<String> }

/// Take every `[kit:<id> <verb> | field | field]` line out of the reply and return them.
/// Markdown dressing around a line is ignored; anything else stays as it was.
pub fn take_actions(reply: &mut String) -> Vec<KitAction> {
    let mut actions = Vec::new();
    let mut kept: Vec<&str> = Vec::new();
    for line in reply.lines() {
        let t = line.trim().trim_matches(|c: char| matches!(c, '*' | '_' | '`'));
        match parse_action(t) {
            Some(a) => actions.push(a),
            None => kept.push(line),
        }
    }
    if !actions.is_empty() {
        *reply = kept.join("\n").trim_end().to_string();
    }
    actions
}

fn parse_action(line: &str) -> Option<KitAction> {
    let inner = line.strip_prefix("[kit:")?.strip_suffix(']')?;
    let mut parts = inner.split('|').map(|p| p.trim().to_string());
    let head = parts.next()?;
    let mut head_words = head.split_whitespace();
    let kit = head_words.next()?.to_ascii_lowercase();
    let verb = head_words.next()?.to_ascii_lowercase();
    let fields: Vec<String> = parts.filter(|p| !p.is_empty()).collect();
    if kit.is_empty() || verb.is_empty() { return None; }
    Some(KitAction { kit, verb, fields })
}

/// Apply the actions to the memory rows. Returns one plain line per change, for the log.
pub async fn apply_actions(pool: &SqlitePool, user_id: &str, actions: &[KitAction]) -> Vec<String> {
    let mut done = Vec::new();
    for a in actions {
        let Some(kit) = get(&a.kit) else { done.push(format!("unknown kit {}", a.kit)); continue };
        let item = a.fields.first().map(|s| s.trim()).unwrap_or("");
        if item.is_empty() { done.push(format!("{} {}: no item named", kit.id, a.verb)); continue; }
        let r = match a.verb.as_str() {
            "add" | "need" | "needed" => add_item(pool, user_id, kit, item, a.fields.get(1).map(|s| s.as_str()), a.fields.get(2).map(|s| s.as_str()), a.fields.get(3).map(|s| s.as_str())).await.map(|_| format!("{}: added {}", kit.id, item)),
            "bought" | "got" | "done" => set_state_by_name(pool, user_id, kit, item, "bought").await.map(|n| format!("{}: {} bought ({} row)", kit.id, item, n)),
            "remove" | "delete" | "drop" => remove_by_name(pool, user_id, kit, item).await.map(|n| format!("{}: removed {} ({} row)", kit.id, item, n)),
            other => Err(format!("{}: unknown action '{}'", kit.id, other)),
        };
        match r { Ok(s) => done.push(s), Err(e) => done.push(format!("{} failed: {}", a.verb, e)) }
    }
    done
}

// ── items as memory rows ─────────────────────────────────────────────────────
// One row per distinct item in the kit's namespace. Tags carry everything the list needs:
// `kit:<id>` marks the row as a list item (other memories in the namespace are not),
// `needed` or `bought` is the state, `section:<s>`, `qty:<q>`, `price:<p>`,
// `last_bought:<date>` and `bought_count:<n>` keep the history for "what do we usually buy".

#[derive(Debug, Clone, Serialize)]
pub struct KitItem {
    pub id: i32,
    pub name: String,
    pub section: String,
    pub quantity: Option<String>,
    pub price: Option<String>,
    pub state: String,
    pub last_bought: Option<String>,
    pub bought_count: u32,
    pub created_at: String,
    pub updated_at: Option<String>,
}

fn kit_tag(kit: &Kit) -> String { format!("kit:{}", kit.id) }

fn tag_value<'a>(tags: &'a [String], prefix: &str) -> Option<&'a str> {
    tags.iter().find_map(|t| t.strip_prefix(prefix))
}

fn parse_tags(raw: &str) -> Vec<String> { serde_json::from_str(raw).unwrap_or_default() }

fn item_from_row(id: i32, title: Option<String>, tags: &[String], created_at: String, updated_at: Option<String>) -> KitItem {
    KitItem {
        id,
        name: title.unwrap_or_default(),
        section: tag_value(tags, "section:").unwrap_or("other").to_string(),
        quantity: tag_value(tags, "qty:").map(str::to_string),
        price: tag_value(tags, "price:").map(str::to_string),
        state: if tags.iter().any(|t| t == "bought") { "bought".into() } else { "needed".into() },
        last_bought: tag_value(tags, "last_bought:").map(str::to_string),
        bought_count: tag_value(tags, "bought_count:").and_then(|v| v.parse().ok()).unwrap_or(0),
        created_at,
        updated_at,
    }
}

/// The kit's items, newest first.
pub async fn list_items(pool: &SqlitePool, user_id: &str, kit_id: &str) -> Vec<KitItem> {
    let Some(kit) = get(kit_id) else { return Vec::new() };
    let rows: Vec<(i32, Option<String>, String, String, Option<String>)> = sqlx::query_as(
        "SELECT id, title, tags, created_at, updated_at FROM memories \
         WHERE user_id = ? AND namespace = ? AND tags LIKE ? ORDER BY id DESC")
        .bind(user_id).bind(&kit.namespace).bind(format!("%\"{}\"%", kit_tag(kit)))
        .fetch_all(pool).await.unwrap_or_default();
    rows.into_iter().map(|(id, title, tags, c, u)| item_from_row(id, title, &parse_tags(&tags), c, u)).collect()
}

/// The section as the kit names it, or "other".
fn normalise_section(kit: &Kit, section: Option<&str>) -> String {
    let s = section.unwrap_or("").trim().to_ascii_lowercase();
    if kit.sections.is_empty() { return if s.is_empty() { "other".into() } else { s }; }
    kit.sections.iter().find(|k| **k == s)
        .or_else(|| kit.sections.iter().find(|k| !s.is_empty() && (k.contains(&s) || s.contains(k.as_str()))))
        .cloned().unwrap_or_else(|| "other".to_string())
}

fn clean_name(item: &str) -> String {
    let t = item.trim().trim_end_matches('.').trim();
    let mut c = t.chars();
    match c.next() { Some(f) => f.to_uppercase().collect::<String>() + c.as_str(), None => String::new() }
}

async fn find_item(pool: &SqlitePool, user_id: &str, kit: &Kit, name: &str) -> Option<(i32, Vec<String>)> {
    let row: Option<(i32, String)> = sqlx::query_as(
        "SELECT id, tags FROM memories WHERE user_id = ? AND namespace = ? AND tags LIKE ? AND lower(trim(title)) = lower(trim(?)) ORDER BY id DESC LIMIT 1")
        .bind(user_id).bind(&kit.namespace).bind(format!("%\"{}\"%", kit_tag(kit))).bind(name)
        .fetch_optional(pool).await.ok().flatten();
    row.map(|(id, tags)| (id, parse_tags(&tags)))
}

async fn write_tags(pool: &SqlitePool, id: i32, tags: &[String]) -> Result<(), String> {
    sqlx::query("UPDATE memories SET tags = ?, updated_at = datetime('now') WHERE id = ?")
        .bind(serde_json::to_string(tags).unwrap_or_else(|_| "[]".into())).bind(id)
        .execute(pool).await.map_err(|e| e.to_string())?;
    Ok(())
}

fn replace_tag(tags: &mut Vec<String>, prefix: &str, value: Option<&str>) {
    tags.retain(|t| !t.starts_with(prefix));
    if let Some(v) = value { if !v.trim().is_empty() { tags.push(format!("{}{}", prefix, v.trim())); } }
}

fn set_state_tags(tags: &mut Vec<String>, state: &str) {
    tags.retain(|t| t != "needed" && t != "bought");
    tags.push(state.to_string());
    if state == "bought" {
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let count: u32 = tag_value(tags, "bought_count:").and_then(|v| v.parse().ok()).unwrap_or(0) + 1;
        replace_tag(tags, "last_bought:", Some(&today));
        replace_tag(tags, "bought_count:", Some(&count.to_string()));
    }
}

/// Add an item, or mark an existing one needed again (keeping its buying history).
pub async fn add_item(pool: &SqlitePool, user_id: &str, kit: &Kit, item: &str, section: Option<&str>, quantity: Option<&str>, price: Option<&str>) -> Result<i32, String> {
    let name = clean_name(item);
    if name.is_empty() { return Err("no item named".into()); }
    let section = normalise_section(kit, section);
    if let Some((id, mut tags)) = find_item(pool, user_id, kit, &name).await {
        set_state_tags(&mut tags, "needed");
        if section != "other" || tag_value(&tags, "section:").is_none() { replace_tag(&mut tags, "section:", Some(&section)); }
        if quantity.is_some() { replace_tag(&mut tags, "qty:", quantity); }
        if price.is_some() { replace_tag(&mut tags, "price:", price); }
        write_tags(pool, id, &tags).await?;
        return Ok(id);
    }
    let content = match quantity { Some(q) if !q.trim().is_empty() => format!("{} ({}) — {}", name, q.trim(), section), _ => format!("{} — {}", name, section) };
    let id = crate::memory::insert_memory(pool, Some(&name), &content, Some("kit"), None, None, None, None,
        Some(user_id), &kit.namespace, true, false, None, None, None, None).await.map_err(|e| e.to_string())?;
    let mut tags = vec![kit_tag(kit), "needed".to_string(), format!("section:{}", section)];
    replace_tag(&mut tags, "qty:", quantity);
    replace_tag(&mut tags, "price:", price);
    write_tags(pool, id, &tags).await?;
    Ok(id)
}

/// Mark an item by id as needed or bought.
pub async fn set_state(pool: &SqlitePool, memory_id: i32, state: &str) -> Result<(), String> {
    if state != "needed" && state != "bought" { return Err(format!("unknown state '{}'", state)); }
    let raw: Option<String> = sqlx::query_scalar("SELECT tags FROM memories WHERE id = ?").bind(memory_id)
        .fetch_optional(pool).await.map_err(|e| e.to_string())?;
    let Some(raw) = raw else { return Err("no such item".into()) };
    let mut tags = parse_tags(&raw);
    if !tags.iter().any(|t| t.starts_with("kit:")) { return Err("not a kit item".into()); }
    set_state_tags(&mut tags, state);
    write_tags(pool, memory_id, &tags).await
}

async fn set_state_by_name(pool: &SqlitePool, user_id: &str, kit: &Kit, item: &str, state: &str) -> Result<usize, String> {
    match find_item(pool, user_id, kit, &clean_name(item)).await {
        Some((id, _)) => { set_state(pool, id, state).await?; Ok(1) }
        None => Ok(0),
    }
}

async fn remove_by_name(pool: &SqlitePool, user_id: &str, kit: &Kit, item: &str) -> Result<usize, String> {
    match find_item(pool, user_id, kit, &clean_name(item)).await {
        Some((id, _)) => { delete_rows(pool, &[id]).await?; Ok(1) }
        None => Ok(0),
    }
}

/// Delete memory rows with the tombstones that keep a synced copy from coming back.
async fn delete_rows(pool: &SqlitePool, ids: &[i32]) -> Result<usize, String> {
    use sha2::{Digest, Sha256};
    let mut hashes = Vec::new();
    let mut n = 0usize;
    for id in ids {
        let content: Option<String> = sqlx::query_scalar("SELECT content FROM memories WHERE id = ?").bind(id)
            .fetch_optional(pool).await.map_err(|e| e.to_string())?;
        let Some(content) = content else { continue };
        hashes.push(format!("{:x}", Sha256::digest(content.as_bytes())));
        n += sqlx::query("DELETE FROM memories WHERE id = ?").bind(id).execute(pool).await.map_err(|e| e.to_string())?.rows_affected() as usize;
    }
    if !hashes.is_empty() {
        if let Some(service) = crate::ZYNKSYNC_SERVICE.lock().await.as_ref() {
            let _ = service.record_tombstones(&hashes).await;
        }
    }
    Ok(n)
}

pub async fn remove_item(pool: &SqlitePool, memory_id: i32) -> Result<(), String> {
    delete_rows(pool, &[memory_id]).await.map(|_| ())
}

/// "Forget everything from this kit": every memory in its namespace, on every device.
pub async fn forget(pool: &SqlitePool, user_id: &str, kit_id: &str) -> Result<usize, String> {
    let Some(kit) = get(kit_id) else { return Err("unknown kit".into()) };
    let ids: Vec<i32> = sqlx::query_scalar("SELECT id FROM memories WHERE user_id = ? AND namespace = ?")
        .bind(user_id).bind(&kit.namespace).fetch_all(pool).await.map_err(|e| e.to_string())?;
    let n = delete_rows(pool, &ids).await?;
    put_setting(pool, &answers_key(kit_id), "{}").await?;
    println!("[Kits] {}: forgot {} memories", kit.id, n);
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_kits_parse() {
        let ids: Vec<&str> = all().iter().map(|k| k.id.as_str()).collect();
        assert_eq!(ids, vec!["grocery", "kitchen"]);
        let g = get("grocery").unwrap();
        assert!(g.actions.is_some() && !g.sections.is_empty() && g.prompts.iter().any(|p| p.action.as_deref() == Some("open_list")));
        assert!(get("kitchen").unwrap().actions.is_none());
    }

    #[test]
    fn action_lines_are_taken_off_the_reply() {
        let mut reply = "Added milk and bananas.\n\n[kit:grocery add | milk | dairy | 1 gallon]\n**[kit:grocery add | bananas | produce]**\n".to_string();
        let acts = take_actions(&mut reply);
        assert_eq!(reply, "Added milk and bananas.");
        assert_eq!(acts.len(), 2);
        assert_eq!(acts[0], KitAction { kit: "grocery".into(), verb: "add".into(), fields: vec!["milk".into(), "dairy".into(), "1 gallon".into()] });
        assert_eq!(acts[1].fields, vec!["bananas".to_string(), "produce".to_string()]);
    }

    #[test]
    fn a_reply_without_actions_is_untouched() {
        let mut reply = "Nothing on the list yet. [kit] is not an action, nor is [kit:].".to_string();
        let before = reply.clone();
        assert!(take_actions(&mut reply).is_empty());
        assert_eq!(reply, before);
    }

    #[test]
    fn sections_are_normalised_to_the_kits_names() {
        let g = get("grocery").unwrap();
        assert_eq!(normalise_section(g, Some("Dairy")), "dairy");
        assert_eq!(normalise_section(g, Some("meat")), "meat and fish");
        assert_eq!(normalise_section(g, Some("cleaning")), "other");
        assert_eq!(normalise_section(g, None), "other");
    }

    #[test]
    fn state_tags_keep_the_buying_history() {
        let mut tags = vec!["kit:grocery".to_string(), "needed".to_string(), "section:dairy".to_string()];
        set_state_tags(&mut tags, "bought");
        assert!(tags.contains(&"bought".to_string()) && !tags.contains(&"needed".to_string()));
        assert_eq!(tag_value(&tags, "bought_count:"), Some("1"));
        assert!(tag_value(&tags, "last_bought:").is_some());
        set_state_tags(&mut tags, "needed");
        set_state_tags(&mut tags, "bought");
        assert_eq!(tag_value(&tags, "bought_count:"), Some("2"));
        let item = item_from_row(1, Some("Milk".into()), &tags, "now".into(), None);
        assert_eq!((item.state.as_str(), item.section.as_str(), item.bought_count), ("bought", "dairy", 2));
    }
}
