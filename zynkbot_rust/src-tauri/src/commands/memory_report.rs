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

/// Ids of memories the user stored with "Remember: ..." (marked in provenance_json at
/// write time). The Memory Manager uses it for its "Remembered on request" filter.
#[tauri::command]
pub async fn list_requested_memory_ids(user_id: String) -> Result<Vec<i64>, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("DB connect failed: {}", e))?;
    let rows = sqlx::query("SELECT id FROM memories WHERE user_id = ? AND provenance_json LIKE '%\"requested\":true%'")
        .bind(&user_id).fetch_all(&pool).await.map_err(|e| format!("Query failed: {}", e));
    pool.close().await;
    Ok(rows?.iter().map(|r| n(r, "id")).collect())
}

#[allow(non_snake_case)]
#[tauri::command]
pub async fn get_memory_report(user_id: String) -> Result<serde_json::Value, String> {
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| format!("DB connect failed: {}", e))?;
    let r = build(&pool, &user_id).await.map_err(|e| format!("Report query failed: {}", e));
    pool.close().await;
    r
}

#[allow(non_snake_case)]
async fn build(pool: &sqlx::SqlitePool, user_id: &str) -> Result<serde_json::Value, sqlx::Error> {
    // "Mine" = not the app's own self-description and not demo data. Always
    // qualified with the memories alias: two of the queries join memories twice
    // and an unqualified `namespace` was ambiguous (first run, 2026-09-08).
    let mine = |alias: &str| format!(
        "{a}.user_id = ? AND {a}.namespace != '_zynkbot' AND COALESCE({a}.source_type,'') != 'demo_data'", a = alias);
    let MINE = mine("m");

    let totals = sqlx::query(&format!(
        "SELECT COUNT(*) AS memories, MIN(created_at) AS first_at, MAX(created_at) AS last_at,
                SUM(CASE WHEN event_date IS NOT NULL THEN 1 ELSE 0 END) AS dated,
                SUM(CASE WHEN tags != '[]' THEN 1 ELSE 0 END) AS tagged
         FROM memories m WHERE {MINE}"))
        .bind(user_id).fetch_one(pool).await?;
    let links: i64 = sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM memory_links l JOIN memories m ON m.id = l.source_memory_id WHERE {MINE}"))
        .bind(user_id).fetch_one(pool).await?;
    let entities_total: i64 = sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM memory_entities e JOIN memories m ON m.id = e.memory_id WHERE {MINE}"))
        .bind(user_id).fetch_one(pool).await?;
    let pending_enrichment: i64 = sqlx::query_scalar(&format!(
        "SELECT COUNT(*) FROM memories m WHERE {MINE} AND m.provenance_json IS NULL AND COALESCE(m.source_type,'') IN ('conversation','hands_free')"))
        .bind(user_id).fetch_one(pool).await?;
    let sessions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_sessions WHERE user_id = ?")
        .bind(user_id).fetch_one(pool).await?;
    let messages: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_messages WHERE user_id = ?")
        .bind(user_id).fetch_one(pool).await?;

    let namespaces: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT m.namespace AS namespace, COUNT(*) AS c FROM memories m WHERE {MINE} GROUP BY m.namespace ORDER BY c DESC"))
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
        "SELECT m.id AS id, m.title AS title, substr(m.event_date, 1, 10) AS day, m.namespace AS namespace FROM memories m
         WHERE {MINE} AND m.event_date IS NOT NULL ORDER BY m.event_date DESC LIMIT 30"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day"), "namespace": s(r, "namespace")})).collect();

    let MINE_A = mine("a");
    let belief_changes: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT l.relation_type AS kind, substr(l.created_at, 1, 10) AS day, l.notes AS notes,
                a.title AS new_title, b.title AS old_title, a.id AS new_id, b.id AS old_id
         FROM memory_links l
         JOIN memories a ON a.id = l.source_memory_id
         JOIN memories b ON b.id = l.target_memory_id
         WHERE l.relation_type IN ('contradicts', 'resolves') AND {MINE_A}
         ORDER BY l.created_at DESC LIMIT 15"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({
            "kind": s(r, "kind"), "day": s(r, "day"), "notes": s(r, "notes"),
            "new_title": s(r, "new_title"), "old_title": s(r, "old_title"),
            "new_id": n(r, "new_id"), "old_id": n(r, "old_id")})).collect();

    let tone: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT m.sentiment_label AS label, COUNT(*) AS c FROM memories m WHERE {MINE} GROUP BY m.sentiment_label ORDER BY c DESC"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"label": s(r, "label"), "count": n(r, "c")})).collect();

    let recent: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT m.id AS id, m.title AS title, substr(m.created_at, 1, 10) AS day, m.namespace AS namespace, m.tags AS tags FROM memories m
         WHERE {MINE} ORDER BY m.created_at DESC LIMIT 10"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day"), "namespace": s(r, "namespace"), "tags": s(r, "tags")})).collect();

    let extraction: Vec<serde_json::Value> = sqlx::query(
        "SELECT COALESCE(last_extraction, 'not recorded') AS outcome, COUNT(*) AS c
         FROM conversation_sessions WHERE user_id = ? GROUP BY outcome ORDER BY c DESC")
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"outcome": s(r, "outcome"), "count": n(r, "c")})).collect();

    let kitchen: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT m.id AS id, m.title AS title, substr(COALESCE(m.event_date, m.created_at), 1, 10) AS day FROM memories m
         WHERE {MINE} AND m.namespace = 'kitchen' ORDER BY COALESCE(m.event_date, m.created_at) DESC LIMIT 10"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day")})).collect();

    let hands_free: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT m.id AS id, m.title AS title, substr(m.created_at, 1, 10) AS day FROM memories m
         WHERE {MINE} AND m.source_type = 'hands_free' ORDER BY m.created_at DESC LIMIT 30"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day")})).collect();

    // "Remember: ..." memories, newest first. Marked at write time in provenance_json
    // (2026-09-09); anything stored before that date has no mark and is not listed.
    let requested: Vec<serde_json::Value> = sqlx::query(&format!(
        "SELECT m.id AS id, m.title AS title, substr(m.created_at, 1, 10) AS day FROM memories m
         WHERE {MINE} AND m.provenance_json LIKE '%\"requested\":true%' ORDER BY m.created_at DESC LIMIT 100"))
        .bind(user_id).fetch_all(pool).await?
        .iter().map(|r| serde_json::json!({"id": n(r, "id"), "title": s(r, "title"), "day": s(r, "day")})).collect();

    Ok(serde_json::json!({
        "requested": requested,
        "hands_free": hands_free,
        "generated_at": chrono::Utc::now().to_rfc3339(),
        "totals": {
            "memories": n(&totals, "memories"),
            "pending_enrichment": pending_enrichment,
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


/// One-time enrichment of memories stored before the decision call returned
/// event date / namespace / tags / tone / entities. Runs in the background,
/// one memory every ~1.5 s, and marks each in `provenance_json` so it is never
/// repeated. Started automatically by the app after launch; there is no button,
/// because once every memory is marked there is nothing left for it to do.
/// Only the new fields change; content and title are untouched.
#[tauri::command]
pub async fn enrich_memory_backlog(user_id: String, backend: String) -> Result<serde_json::Value, String> {
    static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return Ok(serde_json::json!({ "queued": 0, "status": "already running" }));
    }
    let pool = sqlx::SqlitePool::connect(&crate::db::get_db_url())
        .await
        .map_err(|e| { RUNNING.store(false, std::sync::atomic::Ordering::SeqCst); format!("DB connect failed: {}", e) })?;
    let rows = sqlx::query(
        "SELECT id, content, substr(created_at, 1, 10) AS day, namespace FROM memories
         WHERE user_id = ? AND provenance_json IS NULL
           AND COALESCE(source_type, '') IN ('conversation', 'hands_free')
           AND namespace != '_zynkbot'
         ORDER BY id ASC LIMIT 2000",
    )
    .bind(&user_id)
    .fetch_all(&pool)
    .await
    .map_err(|e| { RUNNING.store(false, std::sync::atomic::Ordering::SeqCst); e.to_string() })?;
    let total = rows.len();
    if total == 0 {
        pool.close().await;
        RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
        return Ok(serde_json::json!({ "queued": 0, "status": "nothing to do" }));
    }
    println!("[Enrich] {} older memories to annotate with {}", total, backend);

    tokio::spawn(async move {
        let mut done = 0usize;
        let mut failed = 0usize;
        for row in rows {
            let id: i64 = row.try_get("id").unwrap_or(0);
            let content: String = row.try_get("content").unwrap_or_default();
            let day: String = row.try_get("day").unwrap_or_default();
            let old_ns: String = row.try_get("namespace").unwrap_or_else(|_| "personal".to_string());
            let said_on = chrono::NaiveDate::parse_from_str(&day, "%Y-%m-%d")
                .unwrap_or_else(|_| chrono::Utc::now().date_naive());
            let stamp = serde_json::json!({ "enriched": chrono::Utc::now().to_rfc3339(), "by": backend }).to_string();

            match crate::ask_llm_for_extras(&content, said_on, &backend).await {
                Ok(extras) => {
                    let ns = crate::memory_extras::resolve_namespace(extras.namespace.as_deref(), &old_ns);
                    let event_date = crate::memory_extras::validate_event_date(extras.event_date.as_deref(), &content, said_on + chrono::Duration::days(1))
                        .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc());
                    let tags = crate::memory_extras::clean_tags(extras.tags.as_deref());
                    let (label, score) = crate::memory_extras::tone_to_sentiment(extras.tone.as_deref());
                    let ents = crate::memory_extras::clean_entities(extras.entities.as_deref());
                    let r = sqlx::query(
                        "UPDATE memories SET namespace = ?, event_date = COALESCE(?, event_date), tags = ?,
                                sentiment_label = ?, sentiment_score = ?, provenance_json = ? WHERE id = ?")
                        .bind(&ns)
                        .bind(event_date.map(|d| d.format("%Y-%m-%dT%H:%M:%S+00:00").to_string()))
                        .bind(serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into()))
                        .bind(label).bind(score).bind(&stamp).bind(id)
                        .execute(&pool).await;
                    if r.is_ok() {
                        let _ = crate::memory::insert_memory_entities(&pool, id as i32, &ents).await;
                        done += 1;
                    } else {
                        failed += 1;
                    }
                }
                Err(e) => {
                    failed += 1;
                    println!("[Enrich] memory {} skipped: {}", id, e);
                    // A dead backend (no key, model not loaded) fails every row the same
                    // way; stop after a run of failures rather than hammering it.
                    if failed >= 5 && done == 0 {
                        println!("[Enrich] backend not usable — will retry on a later launch");
                        break;
                    }
                }
            }
            if (done + failed) % 10 == 0 {
                println!("[Enrich] {}/{} done ({} failed)", done + failed, total, failed);
            }
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        }
        println!("[Enrich] finished: {} annotated, {} failed, {} total", done, failed, total);
        pool.close().await;
        RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
    });

    Ok(serde_json::json!({ "queued": total, "status": "started" }))
}
