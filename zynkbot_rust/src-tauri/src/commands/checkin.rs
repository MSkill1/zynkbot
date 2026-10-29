//! Check-in questions: "A while back you said … Did that happen?" (labs/check_in,
//! 2026-10-29, Matt). A passing fact that was an intention ("I need to get a water
//! filter") is asked about once, just as it is about to leave recall, at most one a day,
//! at the start of a conversation. The user closes it: done, not yet, or never mind.
//! Life plans are never asked about because extraction does not mark them passing.
//! Experimental in beta 5; the switch is on the System Controls page.

use serde_json::json;
use sqlx::SqlitePool;

/// First-person future or obligation: the wording of something the user meant to do.
const INTENTION_CUES: &[&str] = &[
    "i need to", "i've got to", "i have to", "i gotta", "i must", "i should",
    "i'm going to", "i am going to", "i'm gonna", "i will", "i'll",
    "i want to", "i'd like to", "i plan to", "i'm planning to", "planning to",
    "need to get", "have to get", "got to get", "going to get", "going to pick up",
    "remind me to", "i still have to", "i still need to",
];

/// Cues in the stored fact, which extraction writes in the third person ("User needs to…").
const FACT_CUES: &[&str] = &[
    "needs to", "has to", "must", "should", "plans to", "is planning to", "intends to",
    "is going to", "will ", "wants to", "would like to", "means to", "hopes to", "still has to",
];

/// Whether what the user said reads as something they meant to do. A question never does.
pub fn intention_text(text: &str) -> bool {
    intention_sentence(text).is_some()
}

/// The sentence of the user's words that carries the intention, if any: the question quotes
/// that sentence, not a whole message. A question mark anywhere in the sentence rules it out.
pub fn intention_sentence(text: &str) -> Option<String> {
    for raw in text.split(|c: char| matches!(c, '.' | '!' | '?' | '\n')) {
        let sentence = raw.trim();
        if sentence.is_empty() { continue; }
        let t = sentence.to_lowercase();
        if INTENTION_CUES.iter().any(|c| t.contains(c)) {
            // A question ends the sentence with "?", which split() removed; look it up in the text.
            let start = text.find(sentence).unwrap_or(0);
            let end = start + sentence.len();
            if text[end..].trim_start().starts_with('?') { continue; }
            return Some(sentence.to_string());
        }
    }
    None
}

/// The stored fact must read as an intention too ("User needs to…"), or the message was
/// about something else ("This is a test. I just want to get a response." is not a plan).
pub fn fact_is_intention(fact: &str) -> bool {
    let t = fact.to_lowercase();
    FACT_CUES.iter().any(|c| t.contains(c))
}

/// The question, from the user's own words. A template on purpose: no model call, the
/// same on every device, and the user sees exactly what was stored.
pub fn question_for(said: &str) -> String {
    let s = said.trim().trim_end_matches(|c: char| matches!(c, '.' | '!' | ' '));
    format!("A while back you said \"{}\". Did that happen?", crate::conversation_engine::cut_chars(s, 160))
}

async fn pool() -> Result<SqlitePool, String> {
    SqlitePool::connect(&crate::db::get_db_url()).await.map_err(|e| format!("Database connection failed: {}", e))
}

async fn enabled(pool: &SqlitePool) -> bool {
    let v: Option<String> = sqlx::query_scalar("SELECT value FROM app_settings WHERE key = 'checkin_enabled'")
        .fetch_optional(pool).await.ok().flatten();
    v.as_deref() != Some("0")
}

/// The switch on the System Controls page. On unless turned off (beta 5: testers see it).
#[tauri::command]
pub async fn get_checkin_enabled() -> Result<bool, String> {
    let pool = pool().await?;
    let on = enabled(&pool).await;
    pool.close().await;
    Ok(on)
}

#[tauri::command]
pub async fn set_checkin_enabled(enabled: bool) -> Result<(), String> {
    let pool = pool().await?;
    sqlx::query("INSERT INTO app_settings (key, value, updated_at) VALUES ('checkin_enabled', ?, datetime('now')) \
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at")
        .bind(if enabled { "1" } else { "0" })
        .execute(&pool).await.map_err(|e| format!("Failed to save the setting: {}", e))?;
    pool.close().await;
    println!("[Check-in] questions {}", if enabled { "on" } else { "off" });
    Ok(())
}

/// The one question to ask now, if any: `{memory_id, question, said}`. Records that it
/// was asked. The page calls this when a conversation starts; the rules live here so
/// every device applies the same ones.
#[tauri::command]
pub async fn next_memory_checkin(user_id: String, containment_mode: Option<String>, force: Option<bool>) -> Result<Option<serde_json::Value>, String> {
    let pool = pool().await?;
    let result = next_checkin(&pool, &user_id, containment_mode.as_deref().unwrap_or(""), force.unwrap_or(false)).await;
    pool.close().await;
    result
}

/// `force` is the "Try it now" button on the Experimental page: no daily limit and no
/// expiry window, so a tester can see a question today. Still once per memory.
async fn next_checkin(pool: &SqlitePool, user_id: &str, containment_mode: &str, force: bool) -> Result<Option<serde_json::Value>, String> {
    if !enabled(pool).await { return Ok(None); }
    // Not for a child, and nothing about a HIPAA user's errands either (they may be medical).
    let mode = containment_mode.to_lowercase();
    if mode == "child" || mode == "hipaa" { return Ok(None); }
    // One a day.
    let today: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_checkins WHERE asked_at > datetime('now', '-1 day')")
        .fetch_one(pool).await.map_err(|e| e.to_string())?;
    if today > 0 && !force { return Ok(None); }
    // Passing facts at their expiry (a week past to two days ahead), never asked before.
    let rows: Vec<(i64, String, Option<String>)> = sqlx::query_as(
        "SELECT id, content, original_text FROM memories \
         WHERE user_id = ? AND is_ephemeral = 1 AND expires_at IS NOT NULL AND temporal_status IS NULL \
           AND namespace != '_zynkbot' \
           AND (? OR date(substr(expires_at, 1, 10)) BETWEEN date('now', '-7 days') AND date('now', '+2 days')) \
           AND id NOT IN (SELECT memory_id FROM memory_checkins) \
         ORDER BY expires_at ASC LIMIT 40")
        .bind(user_id).bind(force).fetch_all(pool).await.map_err(|e| e.to_string())?;
    for (id, content, original) in rows {
        if !fact_is_intention(&content) { continue; }
        let words = original.filter(|t| !t.trim().is_empty()).unwrap_or_else(|| content.clone());
        let Some(said) = intention_sentence(&words) else { continue };
        // A later memory linked to it means the user is still on it: no need to ask.
        let newer: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM memory_links WHERE relation_type IN ('updates', 'related') \
             AND ((target_memory_id = ? AND source_memory_id > ?) OR (source_memory_id = ? AND target_memory_id > ?))")
            .bind(id).bind(id).bind(id).bind(id).fetch_one(pool).await.unwrap_or(0);
        if newer > 0 { continue; }
        sqlx::query("INSERT INTO memory_checkins (memory_id) VALUES (?)").bind(id)
            .execute(pool).await.map_err(|e| e.to_string())?;
        println!("[Check-in] Asking about memory #{}", id);
        return Ok(Some(json!({ "memory_id": id, "question": question_for(&said), "said": said })));
    }
    Ok(None)
}

/// The user's answer: "done", "later", "never" or anything else (left alone).
#[tauri::command]
pub async fn answer_memory_checkin(memory_id: i32, answer: String) -> Result<String, String> {
    let pool = pool().await?;
    let note = match answer.as_str() {
        "done" => {
            crate::memory::set_temporal_status(&pool, memory_id, Some("completed")).await.map_err(|e| e.to_string())?;
            "Noted as done. It stays in Memory Manager as history."
        }
        "later" => {
            // Another two weeks, and it may be asked about once more at the end of them.
            sqlx::query("UPDATE memories SET expires_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '+14 days'), updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                .bind(memory_id).execute(&pool).await.map_err(|e| e.to_string())?;
            sqlx::query("DELETE FROM memory_checkins WHERE memory_id = ?").bind(memory_id).execute(&pool).await.map_err(|e| e.to_string())?;
            "Kept for another two weeks."
        }
        "never" => {
            pool.close().await;
            crate::commands::memory::delete_memory(memory_id).await?;
            println!("[Check-in] memory #{} dropped at the user's word", memory_id);
            return Ok("Forgotten.".to_string());
        }
        _ => "Left as it is.",
    };
    sqlx::query("UPDATE memory_checkins SET answer = ?, answered_at = datetime('now') WHERE memory_id = ?")
        .bind(&answer).bind(memory_id).execute(&pool).await.map_err(|e| e.to_string())?;
    pool.close().await;
    println!("[Check-in] memory #{}: {}", memory_id, answer);
    Ok(note.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intentions_are_things_the_user_meant_to_do() {
        assert!(intention_text("I need to get a water filter for the kitchen sink."));
        assert!(intention_text("I'm going to paint and seal the mudroom steps this weekend"));
        assert!(intention_text("I still have to call the clerk"));
        assert!(!intention_text("I parked on level three at the mall."));
        assert!(!intention_text("Had chicken wings last night."));
        assert!(!intention_text("Do I need to get a water filter?"));
        assert_eq!(intention_sentence("Let's forget about going out. I need to get a water filter. Anyway.").as_deref(), Some("I need to get a water filter"));
        assert!(fact_is_intention("User needs to get a water filter for the kitchen sink."));
        assert!(!fact_is_intention("User is conducting a test to receive a response."));
    }

    #[test]
    fn the_question_quotes_the_users_words() {
        assert_eq!(question_for("I need to get a water filter."), "A while back you said \"I need to get a water filter\". Did that happen?");
    }
}
