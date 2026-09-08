//! "What do you know about me?" — a report built from the local memory graph.
//!
//! Everything here is a read over tables the app already keeps; nothing is
//! inferred by a model and nothing leaves the device. First slice (2026-09-08):
//! totals, categories, tags, people and places, a timeline of dated memories,
//! the belief-change log, tone, and where extraction stopped.

use sqlx::Row;

fn s(row: &sqlx::sqlite::SqliteRow, col: &str) -> String {
    row.try_get::<Option<String>, _>(col).ok().flatten().unwrap_or_default()
}
fn n(row: &sqlx::sqlite::SqliteRow, col: &str) -> i64 {
    row.try_get::<i64, _>(col).unwrap_or(0)
}

#[tauri::command]
pub async fn get_memory_report(user_id: String) -> Result<serde_json::Value, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("DB connect failed: {}", e))?;
    let r = build(&pool, &user_id).await.map_err(|e| format!("Report query failed: {}", e));
    pool.close().await;
    r
}

async fn build(pool: &sqlx::SqlitePool, user_id: &str) -> Result<serde_json::Value, sqlx::Error> {
    // "Mine" = not the app's own self-description and not demo data.
    const MINE: &str = "user_id = ? AND namespace != '_zynkbot' AND COALESCE(source_type,'') != 'demo_data'";

    let totals = sqlx::query(&format!(
        "SELECT COUNT(*) AS memories, MIN(created_at) AS first_at, MAX(created_at) AS last_at,
                SUM(CASE WHEN event_date IS NOT NULL THEN 1 ELSE 0 END) AS dated,
                SUM(CASE WHEN tags != '[]' THEN 1 ELSE 0 END) AS tagged
         FROM memories WHERE {MINE}"))
        .bind(user_id).fetch_one(pool).await?;
    let links: i64 = sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM memory_links l JOIN memories m ON m.id = l.source_memory_id WHERE {MINE}"))
        .bind(user_id).fetch_one(pool).await?;
    let entities_total: i64 = sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM memory_entities e JOIN memories m ON m.id = e.memory_id WHERE {MINE}"))
        .bind(user_id).fetch_one(pool).await?;
    let sessions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_sessions WHERE user_id = ?")
        .bind(user_id).fetch_one(pool).await?;
    let messages: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_messages WHERE user_id = ?")
        .bind(user_id).fetch_one(pool).await?;

    let namespaces: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT namespace, COUNT(*) AS c FROM memories WHERE {MINE} GROUP BY namespace ORDER BY c DESC"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"name": s(r, "namespace"), "count": n(r, "c")})).collect();

    let tags: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT lower(t.value) AS tag, COUNT(*) AS c FROM memories m, json_each(m.tags) t
         WHERE {MINE} AND json_valid(m.tags) GROUP BY lower(t.value) ORDER BY c DESC LIMIT 20"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"tag": s(r, "tag"), "count": n(r, "c")})).collect();

    let mut entities = serde_json::Map::new();
    for kind in ["person", "place", "org", "thing"] {
        let rows = sqlx::query(&format!(
            "SELECT e.name AS name, COUNT(*) AS c FROM memory_entities e JOIN memories m ON m.id = e.memory_id
             WHERE {MINE} AND e.kind = ? GROUP BY e.canonical ORDER BY c DESC, e.name LIMIT 12"))
            .bind(user_id).bind(kind).fetch_all(pool).await?;
        entities.insert(kind.to_string(), serde_json::Value::Array(
            rows.iter().map(|r| serde_json::json!({"name": s(r, "name"), "count": n(r, "c")})).collect()));
    }

    let timeline: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT id, title, substr(event_date, 1, 10) AS day, namespace FROM memories
         WHERE {MINE} AND event_date IS NOT NULL ORDER BY event_date DESC LIMIT 30"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day"), "namespace": s(r, "namespace")})).collect();

    let belief_changes: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT l.relation_type AS kind, substr(l.created_at, 1, 10) AS day, l.notes AS notes,
                a.title AS new_title, b.title AS old_title, a.id AS new_id, b.id AS old_id
         FROM memory_links l
         JOIN memories a ON a.id = l.source_memory_id
         JOIN memories b ON b.id = l.target_memory_id
         WHERE l.relation_type IN ('contradicts', 'resolves') AND a.{MINE}
         ORDER BY l.created_at DESC LIMIT 15"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({
            "kind": s(r, "kind"), "day": s(r, "day"), "notes": s(r, "notes"),
            "new_title": s(r, "new_title"), "old_title": s(r, "old_title"),
            "new_id": n(r, "new_id"), "old_id": n(r, "old_id")})).collect();

    let tone: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT sentiment_label AS label, COUNT(*) AS c FROM memories WHERE {MINE} GROUP BY sentiment_label ORDER BY c DESC"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"label": s(r, "label"), "count": n(r, "c")})).collect();

    let recent: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT id, title, substr(created_at, 1, 10) AS day, namespace, tags FROM memories
         WHERE {MINE} ORDER BY created_at DESC LIMIT 10"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day"), "namespace": s(r, "namespace"), "tags": s(r, "tags")})).collect();

    let extraction: Vec<serde_json::Value> = sqlx::query(
        "SELECT COALESCE(last_extraction, 'not recorded') AS outcome, COUNT(*) AS c
         FROM conversation_sessions WHERE user_id = ? GROUP BY outcome ORDER BY c DESC")
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"outcome": s(r, "outcome"), "count": n(r, "c")})).collect();

    let kitchen: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT id, title, substr(COALESCE(event_date, created_at), 1, 10) AS day FROM memories
         WHERE {MINE} AND namespace = 'kitchen' ORDER BY COALESCE(event_date, created_at) DESC LIMIT 10"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day")})).collect();

    Ok(serde_json::json!({
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "totals": {
            "memories": n(&totals, "memories"),
            "dated": n(&totals, "dated"),
            "tagged": n(&totals, "tagged"),
            "links": links,
            "entities": entities_total,
            "sessions": sessions,
            "messages": messages,
            "first_at": s(&totals, "first_at"),
            "last_at": s(&totals, "last_at"),
        },
        "namespaces": namespaces,
        "tags": tags,
        "entities": entities,
        "timeline": timeline,
        "belief_changes": belief_changes,
        "tone": tone,
        "recent": recent,
        "extraction": extraction,
        "kitchen": kitchen,
    }))
}
