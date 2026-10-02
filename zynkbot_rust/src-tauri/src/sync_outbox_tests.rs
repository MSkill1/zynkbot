//! The outbox triggers (migration 0013), step 1 of the ZynkSync rebuild.
//!
//! Every write to memories, conversation_sessions or conversation_messages must queue
//! exactly one sync_outbox row naming the row's sync_id, and applying a peer's changes
//! under sync_suppress must queue nothing. These are contract tests in the sense of
//! docs/TESTING.md: they describe what the database owes the sync layer, so the step 2
//! drain worker can be written against them.
//!
//! Run: `LD_LIBRARY_PATH=$PWD/lib/vosk cargo test --lib sync_outbox`

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};

/// One connection only: every `sqlite::memory:` connection is its own database, so a
/// larger pool would scatter these statements across several empty ones.
async fn pool_with_migrations_below(version: i64) -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .expect("in-memory DB");
    sqlx::query("PRAGMA foreign_keys=ON").execute(&pool).await.unwrap();
    for m in sqlx::migrate!("./migrations").iter() {
        if m.migration_type.is_down_migration() || m.version >= version {
            continue;
        }
        sqlx::raw_sql(&m.sql)
            .execute(&pool)
            .await
            .unwrap_or_else(|e| panic!("migration {} failed: {}", m.version, e));
    }
    pool
}

async fn pool() -> SqlitePool {
    pool_with_migrations_below(i64::MAX).await
}

async fn apply_0013(pool: &SqlitePool) {
    let m = sqlx::migrate!("./migrations");
    let mig = m.iter().find(|m| m.version == 13).expect("0013 missing");
    sqlx::raw_sql(&mig.sql).execute(pool).await.expect("0013 failed");
}

/// Every queued row, oldest first: (table_name, op, row_sync_id, payload).
async fn outbox(pool: &SqlitePool) -> Vec<(String, String, String, Option<String>)> {
    sqlx::query("SELECT table_name, op, row_sync_id, payload FROM sync_outbox ORDER BY id")
        .fetch_all(pool)
        .await
        .unwrap()
        .iter()
        .map(|r| (r.get(0), r.get(1), r.get(2), r.get(3)))
        .collect()
}

async fn sync_id_of(pool: &SqlitePool, table: &str, id: i64) -> String {
    sqlx::query(&format!("SELECT sync_id FROM {} WHERE id = ?", table))
        .bind(id)
        .fetch_one(pool)
        .await
        .unwrap()
        .get::<Option<String>, _>(0)
        .expect("sync_id was never set")
}

async fn add_memory(pool: &SqlitePool, content: &str) -> i64 {
    sqlx::query("INSERT INTO memories (content, namespace) VALUES (?, 'personal')")
        .bind(content)
        .execute(pool)
        .await
        .unwrap()
        .last_insert_rowid()
}

async fn add_session(pool: &SqlitePool, session_id: &str) -> i64 {
    sqlx::query("INSERT INTO conversation_sessions (session_id, user_id, title) VALUES (?, 'u1', 'A thread')")
        .bind(session_id)
        .execute(pool)
        .await
        .unwrap()
        .last_insert_rowid()
}

async fn add_message(pool: &SqlitePool, session_id: &str, content: &str) -> i64 {
    sqlx::query("INSERT INTO conversation_messages (session_id, user_id, role, content) VALUES (?, 'u1', 'user', ?)")
        .bind(session_id)
        .bind(content)
        .execute(pool)
        .await
        .unwrap()
        .last_insert_rowid()
}

async fn clear_outbox(pool: &SqlitePool) {
    sqlx::query("DELETE FROM sync_outbox").execute(pool).await.unwrap();
}

// ---------------------------------------------------------------- memories

#[tokio::test]
async fn o01_adding_a_memory_queues_one_insert_naming_it() {
    let pool = pool().await;
    let id = add_memory(&pool, "The boiler is serviced in October").await;
    let sync_id = sync_id_of(&pool, "memories", id).await;

    let rows = outbox(&pool).await;
    assert_eq!(rows.len(), 1, "expected one queued row, got {:?}", rows);
    assert_eq!(rows[0].0, "memories");
    assert_eq!(rows[0].1, "insert");
    assert_eq!(rows[0].2, sync_id, "the queued row must name the memory's sync_id");
    assert_eq!(rows[0].3, None, "an insert carries no payload");
}

#[tokio::test]
async fn o02_editing_a_memory_queues_one_update() {
    let pool = pool().await;
    let id = add_memory(&pool, "The boiler is serviced in October").await;
    let sync_id = sync_id_of(&pool, "memories", id).await;
    clear_outbox(&pool).await;

    sqlx::query("UPDATE memories SET content = ? WHERE id = ?")
        .bind("The boiler is serviced in November")
        .bind(id)
        .execute(&pool)
        .await
        .unwrap();

    let rows = outbox(&pool).await;
    assert_eq!(rows.len(), 1, "expected one queued row, got {:?}", rows);
    assert_eq!((rows[0].0.as_str(), rows[0].1.as_str()), ("memories", "update"));
    assert_eq!(rows[0].2, sync_id, "an edit keeps the sync_id it had (KI-060)");
}

#[tokio::test]
async fn o03_deleting_a_memory_queues_a_delete_keeping_the_old_content() {
    let pool = pool().await;
    let id = add_memory(&pool, "The boiler is serviced in October").await;
    let sync_id = sync_id_of(&pool, "memories", id).await;
    clear_outbox(&pool).await;

    sqlx::query("DELETE FROM memories WHERE id = ?").bind(id).execute(&pool).await.unwrap();

    let rows = outbox(&pool).await;
    assert_eq!(rows.len(), 1, "expected one queued row, got {:?}", rows);
    assert_eq!((rows[0].0.as_str(), rows[0].1.as_str()), ("memories", "delete"));
    assert_eq!(rows[0].2, sync_id);
    // The tombstone is sha256(content) and SQLite has no sha256, so the content is kept
    // for the sender to hash. The row itself is gone by then.
    assert_eq!(rows[0].3.as_deref(), Some("The boiler is serviced in October"));
}

// ------------------------------------------------------ conversation rows

#[tokio::test]
async fn o04_a_session_queues_an_insert_an_update_and_a_delete() {
    let pool = pool().await;
    let id = add_session(&pool, "sess-aaa").await;
    let sync_id = sync_id_of(&pool, "conversation_sessions", id).await;

    sqlx::query("UPDATE conversation_sessions SET title = 'Renamed' WHERE id = ?")
        .bind(id).execute(&pool).await.unwrap();
    sqlx::query("DELETE FROM conversation_sessions WHERE id = ?")
        .bind(id).execute(&pool).await.unwrap();

    let rows = outbox(&pool).await;
    let ops: Vec<&str> = rows.iter().map(|r| r.1.as_str()).collect();
    assert_eq!(ops, vec!["insert", "update", "delete"], "got {:?}", rows);
    for row in &rows {
        assert_eq!(row.0, "conversation_sessions");
        assert_eq!(row.2, sync_id);
    }
}

#[tokio::test]
async fn o05_a_message_queues_an_insert_an_update_and_a_delete() {
    let pool = pool().await;
    add_session(&pool, "sess-bbb").await;
    let id = add_message(&pool, "sess-bbb", "What time is the appointment?").await;
    let sync_id = sync_id_of(&pool, "conversation_messages", id).await;
    clear_outbox(&pool).await;

    sqlx::query("UPDATE conversation_messages SET content = 'Edited' WHERE id = ?")
        .bind(id).execute(&pool).await.unwrap();
    sqlx::query("DELETE FROM conversation_messages WHERE id = ?")
        .bind(id).execute(&pool).await.unwrap();

    let rows = outbox(&pool).await;
    assert_eq!(rows.len(), 2, "expected update then delete, got {:?}", rows);
    assert_eq!((rows[0].1.as_str(), rows[0].2.as_str()), ("update", sync_id.as_str()));
    assert_eq!((rows[1].1.as_str(), rows[1].2.as_str()), ("delete", sync_id.as_str()));
    assert_eq!(rows[1].3.as_deref(), Some("Edited"));
}

#[tokio::test]
async fn o06_two_messages_in_the_same_second_are_two_rows() {
    // The old key was (session, second, role), which collapsed these into one (KI-028).
    let pool = pool().await;
    add_session(&pool, "sess-ccc").await;
    let a = add_message(&pool, "sess-ccc", "First").await;
    let b = add_message(&pool, "sess-ccc", "Second").await;

    let sa = sync_id_of(&pool, "conversation_messages", a).await;
    let sb = sync_id_of(&pool, "conversation_messages", b).await;
    assert_ne!(sa, sb, "two messages must never share a sync_id");

    let queued: Vec<String> = outbox(&pool).await.iter()
        .filter(|r| r.0 == "conversation_messages").map(|r| r.2.clone()).collect();
    assert_eq!(queued, vec![sa, sb]);
}

#[tokio::test]
async fn o07_deleting_a_session_queues_its_messages_too() {
    let pool = pool().await;
    add_session(&pool, "sess-ddd").await;
    add_message(&pool, "sess-ddd", "One").await;
    add_message(&pool, "sess-ddd", "Two").await;
    clear_outbox(&pool).await;

    sqlx::query("DELETE FROM conversation_sessions WHERE session_id = 'sess-ddd'")
        .execute(&pool).await.unwrap();

    let rows = outbox(&pool).await;
    let msg_deletes = rows.iter().filter(|r| r.0 == "conversation_messages" && r.1 == "delete").count();
    let sess_deletes = rows.iter().filter(|r| r.0 == "conversation_sessions" && r.1 == "delete").count();
    assert_eq!(sess_deletes, 1, "the session itself: {:?}", rows);
    assert_eq!(msg_deletes, 2, "the cascade must queue each message: {:?}", rows);
}

// ------------------------------------------------------------- suppression

#[tokio::test]
async fn o08_nothing_is_queued_while_suppressed() {
    let pool = pool().await;
    sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&pool).await.unwrap();

    let id = add_memory(&pool, "Arrived from the desktop").await;
    add_session(&pool, "sess-eee").await;
    add_message(&pool, "sess-eee", "Also from the desktop").await;
    sqlx::query("UPDATE memories SET content = 'Changed by the peer' WHERE id = ?")
        .bind(id).execute(&pool).await.unwrap();
    sqlx::query("DELETE FROM memories WHERE id = ?").bind(id).execute(&pool).await.unwrap();

    assert!(outbox(&pool).await.is_empty(), "a peer's changes must not be queued back");
}

#[tokio::test]
async fn o09_a_row_applied_under_suppression_still_gets_a_sync_id() {
    // The suppress flag stops the queue, not the naming: a row written while a peer's
    // batch is being applied must still be nameable when this device later changes it.
    let pool = pool().await;
    sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&pool).await.unwrap();
    let id = add_memory(&pool, "Arrived from the desktop").await;
    assert!(!sync_id_of(&pool, "memories", id).await.is_empty());
}

#[tokio::test]
async fn o10_queuing_resumes_once_the_flag_is_cleared() {
    let pool = pool().await;
    sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&pool).await.unwrap();
    add_memory(&pool, "From the peer").await;
    sqlx::query("DELETE FROM sync_suppress").execute(&pool).await.unwrap();

    add_memory(&pool, "Typed here").await;
    let rows = outbox(&pool).await;
    assert_eq!(rows.len(), 1, "only the local write belongs in the queue: {:?}", rows);
}

// ------------------------------------------------------------- sync_id rules

#[tokio::test]
async fn o11_an_incoming_sync_id_is_kept_not_replaced() {
    let pool = pool().await;
    sqlx::query("INSERT INTO memories (content, namespace, sync_id) VALUES ('From the phone', 'personal', 'abc123')")
        .execute(&pool).await.unwrap();
    let got: String = sqlx::query("SELECT sync_id FROM memories WHERE content = 'From the phone'")
        .fetch_one(&pool).await.unwrap().get(0);
    assert_eq!(got, "abc123", "a row from a peer keeps the name it arrived with");
}

#[tokio::test]
async fn o12_a_sessions_sync_id_is_its_session_id() {
    // A session already has a cross-device name, so two devices holding the same session
    // agree on its sync_id without reconciling.
    let pool = pool().await;
    let id = add_session(&pool, "sess-fff").await;
    assert_eq!(sync_id_of(&pool, "conversation_sessions", id).await, "sess-fff");
}

#[tokio::test]
async fn o13_every_memory_gets_a_different_sync_id() {
    let pool = pool().await;
    let a = add_memory(&pool, "One").await;
    let b = add_memory(&pool, "Two").await;
    assert_ne!(sync_id_of(&pool, "memories", a).await, sync_id_of(&pool, "memories", b).await);
}

#[tokio::test]
async fn o14_rows_written_before_the_migration_get_a_sync_id() {
    // The real case: a database with years of memories in it when 0013 first runs.
    let pool = pool_with_migrations_below(13).await;
    add_memory(&pool, "Written long before the outbox").await;
    add_memory(&pool, "And another").await;
    add_session(&pool, "sess-old").await;
    add_message(&pool, "sess-old", "An old message").await;

    apply_0013(&pool).await;

    let missing: i64 = sqlx::query(
        "SELECT (SELECT COUNT(*) FROM memories WHERE sync_id IS NULL)
              + (SELECT COUNT(*) FROM conversation_sessions WHERE sync_id IS NULL)
              + (SELECT COUNT(*) FROM conversation_messages WHERE sync_id IS NULL)")
        .fetch_one(&pool).await.unwrap().get(0);
    assert_eq!(missing, 0, "the backfill must name every existing row");

    let distinct: i64 = sqlx::query("SELECT COUNT(DISTINCT sync_id) FROM memories")
        .fetch_one(&pool).await.unwrap().get(0);
    assert_eq!(distinct, 2, "each existing memory gets its own name");

    assert!(outbox(&pool).await.is_empty(), "the backfill itself must not queue anything");
}

// ------------------------------------------------------------- keys (0014)

#[tokio::test]
async fn o15_a_key_queues_an_insert_an_update_and_a_delete_named_by_the_key() {
    let pool = pool().await;
    sqlx::query("INSERT INTO sync_secrets (name, value) VALUES ('OPENAI_API_KEY', 'sk-one')").execute(&pool).await.unwrap();
    sqlx::query("UPDATE sync_secrets SET value = 'sk-two' WHERE name = 'OPENAI_API_KEY'").execute(&pool).await.unwrap();
    sqlx::query("DELETE FROM sync_secrets WHERE name = 'OPENAI_API_KEY'").execute(&pool).await.unwrap();
    let rows = outbox(&pool).await;
    let ops: Vec<&str> = rows.iter().map(|r| r.1.as_str()).collect();
    assert_eq!(ops, vec!["insert", "update", "delete"], "got {:?}", rows);
    for r in &rows { assert_eq!((r.0.as_str(), r.2.as_str()), ("sync_secrets", "OPENAI_API_KEY")); }
}

#[tokio::test]
async fn o16_a_key_applied_under_suppression_queues_nothing() {
    let pool = pool().await;
    sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&pool).await.unwrap();
    sqlx::query("INSERT INTO sync_secrets (name, value) VALUES ('XAI_API_KEY', 'from-the-desktop')").execute(&pool).await.unwrap();
    assert!(outbox(&pool).await.is_empty());
    let sync_id: String = sqlx::query_scalar("SELECT sync_id FROM sync_secrets WHERE name = 'XAI_API_KEY'").fetch_one(&pool).await.unwrap();
    assert_eq!(sync_id, "XAI_API_KEY", "a key is named by its own name");
}

// ------------------------------------------------------------- links (0016)

#[tokio::test]
async fn o17_a_link_queues_an_insert_an_update_and_a_delete_named_by_its_own_sync_id() {
    let pool = pool().await;
    let a = add_memory(&pool, "Max is my dog").await;
    let b = add_memory(&pool, "Max likes the park").await;
    clear_outbox(&pool).await;
    sqlx::query("INSERT INTO memory_links (source_memory_id, target_memory_id, relation_type, confidence) VALUES (?, ?, 'elaborates', 0.9)")
        .bind(a).bind(b).execute(&pool).await.unwrap();
    let link_sync_id: String = sqlx::query_scalar("SELECT sync_id FROM memory_links").fetch_one(&pool).await.unwrap();
    sqlx::query("UPDATE memory_links SET confidence = 0.5").execute(&pool).await.unwrap();
    sqlx::query("DELETE FROM memory_links").execute(&pool).await.unwrap();
    let rows: Vec<_> = outbox(&pool).await.into_iter().filter(|r| r.0 == "memory_links").collect();
    let ops: Vec<&str> = rows.iter().map(|r| r.1.as_str()).collect();
    assert_eq!(ops, vec!["insert", "update", "delete"], "got {:?}", rows);
    for r in &rows { assert_eq!(r.2, link_sync_id); }
}

#[tokio::test]
async fn o18_links_that_predate_the_link_migration_are_queued_once_by_0017() {
    // Rows written before 0016: named by its backfill, never queued. 0017 queues them.
    let pool = pool_with_migrations_below(16).await;
    let a = add_memory(&pool, "Max is my dog").await;
    let b = add_memory(&pool, "Max likes the park").await;
    sqlx::query("INSERT INTO memory_links (source_memory_id, target_memory_id, relation_type, confidence) VALUES (?, ?, 'elaborates', 0.9)")
        .bind(a).bind(b).execute(&pool).await.unwrap();
    clear_outbox(&pool).await;
    for m in sqlx::migrate!("./migrations").iter() {
        if m.version == 16 || m.version == 17 {
            sqlx::raw_sql(&m.sql).execute(&pool).await.unwrap_or_else(|e| panic!("migration {} failed: {}", m.version, e));
        }
    }
    let rows: Vec<_> = outbox(&pool).await.into_iter().filter(|r| r.0 == "memory_links").collect();
    assert_eq!(rows.len(), 1, "exactly one queued row for the pre-existing link: {:?}", rows);
    assert_eq!(rows[0].1, "insert");
    let named: String = sqlx::query_scalar("SELECT sync_id FROM memory_links").fetch_one(&pool).await.unwrap();
    assert_eq!(rows[0].2, named);
}
