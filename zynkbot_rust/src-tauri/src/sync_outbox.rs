//! The ZynkSync outbox: drain and receive (step 2 of the rebuild, 2026-10-01).
//!
//! Migration 0013 made every write to memories, conversation_sessions and
//! conversation_messages queue one row in `sync_outbox`, naming the row by its
//! `sync_id`. This module moves that queue between devices:
//!
//! * **Drain** (`drain_outbox_to`): for one peer, read the queued rows past that peer's
//!   cursor, collapse them to one entry per row, read each live row at send time and
//!   serialise every column, send the batch to the peer's `/api/zynksync/outbox`, and
//!   move the cursor only once the peer has said it applied the batch. A device that is
//!   off simply has a longer queue waiting.
//! * **Receive** (`apply_outbox_batch`): inside one transaction, with a `sync_suppress`
//!   row present so the triggers do not queue what just arrived, apply each entry by
//!   `sync_id`: upsert, last-write-wins by `updated_at`; honour tombstones. A row whose
//!   `sync_id` is unknown here is first matched by content (a memory) or by the 0011 key
//!   (a message) and adopts the incoming `sync_id` — that is how copies that predate the
//!   migration converge, and it stays on permanently: two memories with the same text
//!   are one memory (Matt, 2026-10-01).
//! * **Prune** (`prune_outbox`): rows every paired peer has acknowledged are dropped, as
//!   are rows older than `OUTBOX_RETENTION_DAYS`. A peer whose cursor points before the
//!   oldest surviving row missed something and gets a full re-send of the live tables.
//!
//! `sync_bidirectional` (zynksync.rs) is a drain in each direction. The routes and
//! handlers live beside the other sync routes in zynksync.rs.

use crate::zynksync::{
    SyncConversationMessage, SyncConversationSession, SyncMemory, ZynkSyncService,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use tauri::Emitter;
use std::collections::HashMap;

/// Most outbox rows read per drain batch. Matches the history push cap (KI-068): the
/// receiver limits a request to a few megabytes.
pub const OUTBOX_BATCH_ROWS: i64 = 300;

/// Queued rows older than this are dropped; a peer that has been away longer than this
/// gets a full re-send instead. A week, not the plan's thirty days (Matt, 2026-10-01): a
/// deleted memory's text sits in the queue until pruned, and the only cost of a shorter
/// window is a full re-send, a few megabytes on the LAN.
pub const OUTBOX_RETENTION_DAYS: i64 = 7;

/// A deletion, named by what the receiver needs to find and forget the row.
/// `content_hash` is sha256(content) for a memory: the key of `deleted_memory_hashes`,
/// which the old sync paths still honour until step 5 removes them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncTombstone {
    pub table_name: String,
    pub row_sync_id: String,
    #[serde(default)]
    pub content_hash: Option<String>,
}

/// An API key or the backup key, as it travels (step 3, migration 0014). `updated_at` is
/// when the sending device first saw or last changed it; the newest wins everywhere.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSecret {
    pub name: String,
    pub value: String,
    pub updated_at: DateTime<Utc>,
    /// Goes up by one on every local change (0018, KI-081); the higher revision wins,
    /// time and then the value break ties. Absent from older senders: treated as 0.
    #[serde(default)]
    pub revision: i64,
}

/// A relationship between two memories (memory_links), naming them by sync_id (0016).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncLink {
    pub sync_id: String,
    pub source_sync_id: String,
    pub target_sync_id: String,
    pub relation_type: String,
    pub confidence: f64,
    pub notes: Option<String>,
    pub created_by: String,
    pub created_at: String,
}

/// One drained batch. `through` is the highest outbox id it covers on the sender; the
/// receiver echoes it back as the acknowledgement and the sender records it as the cursor.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutboxBatch {
    pub through: i64,
    #[serde(default)]
    pub memories: Vec<SyncMemory>,
    #[serde(default)]
    pub sessions: Vec<SyncConversationSession>,
    #[serde(default)]
    pub messages: Vec<SyncConversationMessage>,
    #[serde(default)]
    pub deletes: Vec<SyncTombstone>,
    #[serde(default)]
    pub secrets: Vec<SyncSecret>,
    #[serde(default)]
    pub links: Vec<SyncLink>,
}

impl OutboxBatch {
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty() && self.sessions.is_empty() && self.messages.is_empty()
            && self.deletes.is_empty() && self.secrets.is_empty() && self.links.is_empty()
    }
    pub fn len(&self) -> usize {
        self.memories.len() + self.sessions.len() + self.messages.len() + self.deletes.len() + self.secrets.len() + self.links.len()
    }
}

/// What the receiver reports back. `known_before` is the highest batch it had applied
/// from this sender before this one (None: nothing — a fresh or restored install), so the
/// sender can tell when its own cursor is ahead of what the peer actually holds.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutboxReceipt {
    pub through: i64,
    pub applied: usize,
    #[serde(default)]
    pub known_before: Option<i64>,
}

/// What a device asking to be drained to says about itself.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PullRequest {
    /// The highest batch this device has applied from the peer it is asking (None: nothing).
    #[serde(default)]
    pub known_through: Option<i64>,
}

/// What one drain to one peer did.
#[derive(Debug, Clone, Default)]
pub struct DrainOutcome {
    pub batches: usize,
    pub entries_sent: usize,
    pub applied_by_peer: usize,
    /// The peer was skipped this cycle (refused a connection within the last two minutes
    /// and has sent no heartbeat). Not an error: reporting it as one re-stamped the
    /// "recently refused" clock and the skip never expired (2026-10-02).
    pub skipped: bool,
}

pub fn content_hash(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}

// ============================================================================ build

/// Read the live memory rows for the given sync_ids and serialise every column.
async fn load_memories(pool: &SqlitePool, sync_ids: &[String]) -> Result<Vec<SyncMemory>, String> {
    if sync_ids.is_empty() { return Ok(Vec::new()); }
    let in_clause = sync_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT id, user_id, session_id, content, title, source_type, created_at, updated_at,
                parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                embedding, link_count, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                event_type, event_date, entities_detected, original_text,
                collection_id, memory_placement, external_id, temporal_status, provenance_json,
                tags, sync_id
         FROM memories WHERE sync_id IN ({})", in_clause);
    let mut q = sqlx::query(&sql);
    for s in sync_ids { q = q.bind(s); }
    let rows = q.fetch_all(pool).await.map_err(|e| format!("outbox: read memories: {}", e))?;
    Ok(rows.iter().map(|row| {
        let embedding: Option<Vec<f32>> = row.try_get::<Option<Vec<u8>>, _>("embedding").ok().flatten()
            .map(|blob| blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
        SyncMemory {
            id: row.get("id"),
            user_id: row.get("user_id"),
            session_id: row.get("session_id"),
            content: row.get("content"),
            title: row.get("title"),
            source_type: row.get("source_type"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            parent_scroll_id: row.get("parent_scroll_id"),
            chunk_index: row.get("chunk_index"),
            namespace: row.get("namespace"),
            is_syncable: row.get("is_syncable"),
            is_shareable: row.get("is_shareable"),
            embedding,
            link_count: row.get("link_count"),
            is_ephemeral: row.get("is_ephemeral"),
            expires_at: row.get("expires_at"),
            sentiment_score: row.get("sentiment_score"),
            sentiment_label: row.get("sentiment_label"),
            event_type: row.get("event_type"),
            event_date: row.get("event_date"),
            entities_detected: row.get("entities_detected"),
            original_text: row.get("original_text"),
            collection_id: row.get("collection_id"),
            memory_placement: row.get("memory_placement"),
            external_id: row.get("external_id"),
            temporal_status: row.get("temporal_status"),
            provenance_json: row.get("provenance_json"),
            tags: row.get("tags"),
            sync_id: row.get("sync_id"),
        }
    }).collect())
}

async fn load_sessions(pool: &SqlitePool, sync_ids: &[String]) -> Result<Vec<SyncConversationSession>, String> {
    if sync_ids.is_empty() { return Ok(Vec::new()); }
    let in_clause = sync_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT session_id, user_id, title, started_at, last_active, message_count,
                model_backend, containment_mode, sync_id
         FROM conversation_sessions WHERE sync_id IN ({})", in_clause);
    let mut q = sqlx::query(&sql);
    for s in sync_ids { q = q.bind(s); }
    let rows = q.fetch_all(pool).await.map_err(|e| format!("outbox: read sessions: {}", e))?;
    Ok(rows.iter().map(|row| SyncConversationSession {
        session_id: row.get("session_id"),
        user_id: row.get("user_id"),
        title: row.get("title"),
        started_at: row.get("started_at"),
        last_active: row.get("last_active"),
        message_count: row.get("message_count"),
        model_backend: row.get("model_backend"),
        containment_mode: row.get("containment_mode"),
        sync_id: row.get("sync_id"),
    }).collect())
}

async fn load_messages(pool: &SqlitePool, sync_ids: &[String]) -> Result<Vec<SyncConversationMessage>, String> {
    if sync_ids.is_empty() { return Ok(Vec::new()); }
    let in_clause = sync_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT session_id, user_id, role, content, created_at,
                model_backend, containment_mode, entry_hash, prev_hash, sync_id
         FROM conversation_messages WHERE sync_id IN ({}) ORDER BY created_at ASC, id ASC", in_clause);
    let mut q = sqlx::query(&sql);
    for s in sync_ids { q = q.bind(s); }
    let rows = q.fetch_all(pool).await.map_err(|e| format!("outbox: read messages: {}", e))?;
    Ok(rows.iter().map(|row| SyncConversationMessage {
        session_id: row.get("session_id"),
        user_id: row.get("user_id"),
        role: row.get("role"),
        content: row.get("content"),
        created_at: row.get("created_at"),
        model_backend: row.get("model_backend"),
        containment_mode: row.get("containment_mode"),
        entry_hash: row.get("entry_hash"),
        prev_hash: row.get("prev_hash"),
        sync_id: row.get("sync_id"),
    }).collect())
}

async fn load_secrets(pool: &SqlitePool, names: &[String]) -> Result<Vec<SyncSecret>, String> {
    if names.is_empty() { return Ok(Vec::new()); }
    let in_clause = names.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!("SELECT name, value, updated_at, revision FROM sync_secrets WHERE name IN ({})", in_clause);
    let mut q = sqlx::query(&sql);
    for n in names { q = q.bind(n); }
    let rows = q.fetch_all(pool).await.map_err(|e| format!("outbox: read secrets: {}", e))?;
    Ok(rows.iter().map(|r| SyncSecret { name: r.get("name"), value: r.get("value"), updated_at: r.get("updated_at"), revision: r.get("revision") }).collect())
}

async fn load_links(pool: &SqlitePool, sync_ids: &[String]) -> Result<Vec<SyncLink>, String> {
    if sync_ids.is_empty() { return Ok(Vec::new()); }
    let in_clause = sync_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
    let sql = format!(
        "SELECT l.sync_id, s.sync_id AS source_sync_id, t.sync_id AS target_sync_id, l.relation_type, l.confidence,
                l.notes, l.created_by, l.created_at
         FROM memory_links l
         JOIN memories s ON s.id = l.source_memory_id
         JOIN memories t ON t.id = l.target_memory_id
         WHERE l.sync_id IN ({})", in_clause);
    let mut q = sqlx::query(&sql);
    for id in sync_ids { q = q.bind(id); }
    let rows = q.fetch_all(pool).await.map_err(|e| format!("outbox: read links: {}", e))?;
    Ok(rows.iter().map(|r| SyncLink {
        sync_id: r.get("sync_id"), source_sync_id: r.get("source_sync_id"), target_sync_id: r.get("target_sync_id"),
        relation_type: r.get("relation_type"), confidence: r.get("confidence"), notes: r.get("notes"),
        created_by: r.get("created_by"), created_at: r.get("created_at"),
    }).collect())
}

/// This device changed (or first saw) a key: the row gets the current time and a revision
/// one above whatever this device last held for it, so it outranks what it followed on
/// every device, whatever their clocks say (KI-081).
pub async fn record_secret(pool: &SqlitePool, name: &str, value: &str) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO sync_secrets (name, value, updated_at, revision) VALUES (?, ?, ?, 1)
         ON CONFLICT (name) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at,
             revision = sync_secrets.revision + 1")
        .bind(name).bind(value).bind(Utc::now())
        .execute(pool).await.map_err(|e| format!("outbox: record secret: {}", e))?;
    Ok(())
}

pub async fn forget_secret(pool: &SqlitePool, name: &str) -> Result<(), String> {
    sqlx::query("DELETE FROM sync_secrets WHERE name = ?").bind(name)
        .execute(pool).await.map_err(|e| format!("outbox: forget secret: {}", e))?;
    Ok(())
}

/// Bring sync_secrets up to date with what this device actually holds: each propagatable
/// key in the environment (.env is loaded into it at startup), and the backup key if one
/// exists. A key the table lacks, or whose value differs (edited by hand, or set before
/// this table existed), is recorded with the current time. Keys the receiver wrote arrive
/// through apply_secret, which sets both the environment and the row, so they match and
/// are left alone. Called at the start of every drain; cheap.
pub async fn seed_secrets_from_env(pool: &SqlitePool) -> Result<usize, String> {
    let mut held: Vec<(String, String)> = crate::commands::models::PROPAGATABLE_KEYS.iter()
        .filter_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()).map(|v| (k.to_string(), v)))
        .collect();
    let key_file = crate::db::get_app_data_dir().join("backup.key");
    if let Ok(hex) = std::fs::read_to_string(&key_file) {
        if !hex.trim().is_empty() {
            held.push((crate::commands::models::BACKUP_KEY_PUSH_NAME.to_string(), hex.trim().to_ascii_lowercase()));
        }
    }
    let mut recorded = 0;
    for (name, value) in held {
        let current: Option<String> = sqlx::query_scalar("SELECT value FROM sync_secrets WHERE name = ?")
            .bind(&name).fetch_optional(pool).await.map_err(|e| format!("outbox: read secret: {}", e))?;
        if current.as_deref() != Some(value.as_str()) {
            record_secret(pool, &name, &value).await?;
            recorded += 1;
        }
    }
    Ok(recorded)
}

/// Collapse the queued rows past `after` into one batch: one entry per (table, sync_id),
/// a delete winning over anything queued before it for the same row. Returns None when
/// nothing is queued. The batch's `through` is the highest outbox id read, so rows that
/// collapsed to nothing (inserted then deleted) still move the cursor.
pub async fn build_batch(pool: &SqlitePool, after: i64) -> Result<Option<OutboxBatch>, String> {
    let rows = sqlx::query(
        "SELECT id, table_name, row_sync_id, op, payload FROM sync_outbox
         WHERE id > ? ORDER BY id ASC LIMIT ?")
        .bind(after).bind(OUTBOX_BATCH_ROWS)
        .fetch_all(pool).await.map_err(|e| format!("outbox: read queue: {}", e))?;
    if rows.is_empty() { return Ok(None); }

    let through: i64 = rows.last().map(|r| r.get("id")).unwrap_or(after);
    // (table, sync_id) -> (is_delete, delete payload)
    let mut latest: HashMap<(String, String), (bool, Option<String>)> = HashMap::new();
    for r in &rows {
        let key = (r.get::<String, _>("table_name"), r.get::<String, _>("row_sync_id"));
        let op: String = r.get("op");
        let payload: Option<String> = r.get("payload");
        if op == "delete" {
            latest.insert(key, (true, payload));
        } else {
            latest.entry(key).or_insert((false, None));
        }
    }

    let mut want_memories = Vec::new();
    let mut want_sessions = Vec::new();
    let mut want_messages = Vec::new();
    let mut want_secrets = Vec::new();
    let mut want_links = Vec::new();
    let mut deletes = Vec::new();
    for ((table, sync_id), (is_delete, payload)) in latest {
        if is_delete {
            let content_hash = if table == "memories" { payload.as_deref().map(content_hash) } else { None };
            deletes.push(SyncTombstone { table_name: table, row_sync_id: sync_id, content_hash });
        } else {
            match table.as_str() {
                "memories" => want_memories.push(sync_id),
                "conversation_sessions" => want_sessions.push(sync_id),
                "conversation_messages" => want_messages.push(sync_id),
                "sync_secrets" => want_secrets.push(sync_id),
                "memory_links" => want_links.push(sync_id),
                other => eprintln!("[ZynkSync] outbox: unknown table {} queued, skipped", other),
            }
        }
    }
    // A row queued as insert/update but gone by now was deleted after the window closed;
    // its delete is queued further on and goes next batch. Reading finds nothing: fine.
    let batch = OutboxBatch {
        through,
        memories: load_memories(pool, &want_memories).await?,
        sessions: load_sessions(pool, &want_sessions).await?,
        messages: load_messages(pool, &want_messages).await?,
        deletes,
        secrets: load_secrets(pool, &want_secrets).await?,
        links: load_links(pool, &want_links).await?,
    };
    Ok(Some(batch))
}

/// One capped slice of the live tables for `user_id`: sessions first (messages point at
/// them), then memories, then messages, in a fixed order so successive calls with a
/// growing `offset` walk the whole set. Returns the batch and how many rows exist in
/// all. For a peer that has never been drained to, or one that has been away longer
/// than the queue remembers; the receiver's upsert-by-sync_id makes re-sending safe.
pub async fn build_full_resend(pool: &SqlitePool, user_id: &str, through: i64, offset: usize, limit: usize)
    -> Result<(OutboxBatch, usize, usize), String>
{
    let session_ids: Vec<String> = sqlx::query_scalar(
        "SELECT sync_id FROM conversation_sessions WHERE user_id = ? AND sync_id IS NOT NULL ORDER BY id")
        .bind(user_id).fetch_all(pool).await.map_err(|e| e.to_string())?;
    let memory_ids: Vec<String> = sqlx::query_scalar(
        "SELECT sync_id FROM memories WHERE user_id = ? AND sync_id IS NOT NULL ORDER BY id")
        .bind(user_id).fetch_all(pool).await.map_err(|e| e.to_string())?;
    let message_ids: Vec<String> = sqlx::query_scalar(
        "SELECT sync_id FROM conversation_messages WHERE user_id = ? AND sync_id IS NOT NULL ORDER BY id")
        .bind(user_id).fetch_all(pool).await.map_err(|e| e.to_string())?;
    // Links last: both their memories have gone before them.
    let link_ids: Vec<String> = sqlx::query_scalar(
        "SELECT l.sync_id FROM memory_links l JOIN memories m ON m.id = l.source_memory_id
         WHERE m.user_id = ? AND l.sync_id IS NOT NULL ORDER BY l.id")
        .bind(user_id).fetch_all(pool).await.map_err(|e| e.to_string())?;
    let total = session_ids.len() + memory_ids.len() + message_ids.len() + link_ids.len();
    let all = session_ids.iter().map(|s| ("s", s)).chain(memory_ids.iter().map(|m| ("m", m)))
        .chain(message_ids.iter().map(|x| ("x", x))).chain(link_ids.iter().map(|l| ("l", l)));
    let slice: Vec<(&str, &String)> = all.skip(offset).take(limit).collect();
    let pick = |kind: &str| slice.iter().filter(|(k, _)| *k == kind).map(|(_, id)| (*id).clone()).collect::<Vec<_>>();
    // Keys are few and small, and a device with no key cannot answer: all of them ride
    // with the first slice (KI-055).
    let secrets = if offset == 0 {
        let names: Vec<String> = sqlx::query_scalar("SELECT name FROM sync_secrets ORDER BY name")
            .fetch_all(pool).await.map_err(|e| e.to_string())?;
        load_secrets(pool, &names).await?
    } else { Vec::new() };
    // How many rows of the walk this slice covers — not the batch size, which also
    // carries the keys on the first slice. Advancing by the batch size skipped one row
    // per key and a message never arrived (b08d under load, 2026-10-02).
    let covered = slice.len();
    let batch = OutboxBatch {
        through,
        sessions: load_sessions(pool, &pick("s")).await?,
        memories: load_memories(pool, &pick("m")).await?,
        messages: load_messages(pool, &pick("x")).await?,
        deletes: Vec::new(),
        secrets,
        links: load_links(pool, &pick("l")).await?,
    };
    Ok((batch, total, covered))
}

/// How far a full send to each (this device, peer) has got, in rows, and where the
/// queue stood when it began. In memory only: an app restart starts the full send
/// over, which the receiver's upsert makes harmless.
///
/// The queue position is fixed at the first slice on purpose. The slices walk the
/// live tables section by section (sessions, memories, messages, links, each by id),
/// so a row written while the send is running lands behind the offset if its section
/// has already been passed, and the slices never reach it. The cursor set at the end
/// must therefore point to the queue's end *at the start*, so the queue path delivers
/// everything written during the send. Taking the end position instead skipped two
/// messages the laptop wrote during a 26-minute first contact (2026-10-06, D11).
static FULL_SEND_OFFSET: std::sync::LazyLock<std::sync::Mutex<HashMap<(String, String), (usize, i64)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

// ============================================================================ cursor

/// None when this peer has never acknowledged a batch: first contact.
pub async fn cursor_row(pool: &SqlitePool, peer_device_id: &str) -> Result<Option<i64>, String> {
    sqlx::query_scalar(
        "SELECT last_outbox_id FROM sync_outbox_cursor WHERE peer_device_id = ?")
        .bind(peer_device_id).fetch_optional(pool).await
        .map_err(|e| format!("outbox: read cursor: {}", e))
}

pub async fn cursor_for(pool: &SqlitePool, peer_device_id: &str) -> Result<i64, String> {
    Ok(cursor_row(pool, peer_device_id).await?.unwrap_or(0))
}

pub async fn set_cursor(pool: &SqlitePool, peer_device_id: &str, through: i64) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO sync_outbox_cursor (peer_device_id, last_outbox_id, updated_at)
         VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (peer_device_id) DO UPDATE SET
            last_outbox_id = excluded.last_outbox_id, updated_at = excluded.updated_at")
        .bind(peer_device_id).bind(through)
        .execute(pool).await.map_err(|e| format!("outbox: write cursor: {}", e))?;
    Ok(())
}

/// Drop what every paired peer has acknowledged, and anything older than the retention
/// window. Returns the lowest outbox id still queued (None when the queue is empty), so a
/// caller can tell whether a peer's cursor fell behind what remains.
pub async fn prune_outbox(pool: &SqlitePool, paired_peer_ids: &[String]) -> Result<Option<i64>, String> {
    if !paired_peer_ids.is_empty() {
        let in_clause = paired_peer_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        // A paired peer with no cursor row yet has acknowledged nothing: min is 0.
        let sql = format!(
            "SELECT COALESCE(MIN(COALESCE(c.last_outbox_id, 0)), 0)
             FROM (SELECT ? AS peer_device_id{}) p
             LEFT JOIN sync_outbox_cursor c ON c.peer_device_id = p.peer_device_id",
            " UNION ALL SELECT ?".repeat(paired_peer_ids.len() - 1));
        let _ = in_clause;
        let mut q = sqlx::query_scalar::<_, i64>(&sql);
        for p in paired_peer_ids { q = q.bind(p); }
        let acknowledged_by_all = q.fetch_one(pool).await.map_err(|e| format!("outbox: prune: {}", e))?;
        sqlx::query("DELETE FROM sync_outbox WHERE id <= ?")
            .bind(acknowledged_by_all)
            .execute(pool).await.map_err(|e| format!("outbox: prune acknowledged: {}", e))?;
    }
    let cutoff = Utc::now() - Duration::days(OUTBOX_RETENTION_DAYS);
    sqlx::query("DELETE FROM sync_outbox WHERE created_at < ?")
        .bind(cutoff.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .execute(pool).await.map_err(|e| format!("outbox: prune old: {}", e))?;
    let oldest: Option<i64> = sqlx::query_scalar("SELECT MIN(id) FROM sync_outbox")
        .fetch_one(pool).await.map_err(|e| format!("outbox: oldest: {}", e))?;
    Ok(oldest)
}

pub async fn inbox_cursor(pool: &SqlitePool, peer_device_id: &str) -> Result<Option<i64>, String> {
    sqlx::query_scalar("SELECT through FROM sync_inbox_cursor WHERE peer_device_id = ?")
        .bind(peer_device_id).fetch_optional(pool).await.map_err(|e| format!("outbox: read inbox cursor: {}", e))
}

async fn set_inbox_cursor(tx: &mut Transaction<'_, Sqlite>, peer_device_id: &str, through: i64) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO sync_inbox_cursor (peer_device_id, through, updated_at) VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
         ON CONFLICT (peer_device_id) DO UPDATE SET through = MAX(sync_inbox_cursor.through, excluded.through), updated_at = excluded.updated_at")
        .bind(peer_device_id).bind(through)
        .execute(&mut **tx).await.map_err(|e| format!("outbox: write inbox cursor: {}", e))?;
    Ok(())
}

/// The sender's cursor for `peer` is ahead of what the peer says it holds: forget it, so
/// the next drain treats the peer as first contact and sends the live tables.
pub async fn reset_cursor_if_ahead(pool: &SqlitePool, peer_device_id: &str, peer_knows: Option<i64>) -> Result<bool, String> {
    let ours = cursor_row(pool, peer_device_id).await?;
    let ahead = match (ours, peer_knows) { (Some(c), None) => c > 0, (Some(c), Some(k)) => c > k, (None, _) => false };
    if ahead {
        sqlx::query("DELETE FROM sync_outbox_cursor WHERE peer_device_id = ?").bind(peer_device_id)
            .execute(pool).await.map_err(|e| format!("outbox: reset cursor: {}", e))?;
        println!("[ZynkSync] outbox: {}… holds less than our cursor says (theirs {:?}, ours {:?}) — starting it over", &peer_device_id[..8.min(peer_device_id.len())], peer_knows, ours);
    }
    Ok(ahead)
}

// ============================================================================ receive

/// Apply one batch from a peer. One transaction; the suppress row keeps the triggers from
/// queuing any of it back. Returns the number of rows written or removed.
pub async fn apply_outbox_batch(pool: &SqlitePool, local_user_id: &str, batch: &OutboxBatch) -> Result<usize, String> {
    apply_outbox_batch_from(pool, local_user_id, None, batch).await.map(|(n, _)| n)
}

/// As above, from a known sender: records how far this device has now heard from it and
/// returns (applied, what it knew before).
pub async fn apply_outbox_batch_from(pool: &SqlitePool, local_user_id: &str, sender_device_id: Option<&str>, batch: &OutboxBatch) -> Result<(usize, Option<i64>), String> {
    let known_before = match sender_device_id { Some(id) => inbox_cursor(pool, id).await?, None => None };
    let mut tx = pool.begin().await.map_err(|e| format!("outbox: begin: {}", e))?;
    sqlx::query("INSERT OR IGNORE INTO sync_suppress (flag) VALUES (1)")
        .execute(&mut *tx).await.map_err(|e| format!("outbox: suppress: {}", e))?;

    let mut applied = 0usize;
    let mut touched_sessions: Vec<String> = Vec::new();

    for m in &batch.memories {
        if apply_memory(&mut tx, local_user_id, m).await? { applied += 1; }
    }
    for s in &batch.sessions {
        if apply_session(&mut tx, local_user_id, s).await? { applied += 1; }
        touched_sessions.push(s.session_id.clone());
    }
    for msg in &batch.messages {
        if apply_message(&mut tx, local_user_id, msg).await? { applied += 1; }
        if !touched_sessions.contains(&msg.session_id) { touched_sessions.push(msg.session_id.clone()); }
    }
    for d in &batch.deletes {
        if apply_delete(&mut tx, d).await? { applied += 1; }
    }
    let mut keys_changed = false;
    for sec in &batch.secrets {
        if apply_secret(&mut tx, sec).await? { applied += 1; keys_changed = true; }
    }
    for link in &batch.links {
        if apply_link(&mut tx, link).await? { applied += 1; }
    }
    for sid in &touched_sessions {
        sqlx::query(
            "UPDATE conversation_sessions
             SET message_count = (SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?)
             WHERE session_id = ?")
            .bind(sid).bind(sid)
            .execute(&mut *tx).await.map_err(|e| format!("outbox: recount: {}", e))?;
    }

    if let Some(id) = sender_device_id {
        set_inbox_cursor(&mut tx, id, batch.through).await?;
    }
    sqlx::query("DELETE FROM sync_suppress")
        .execute(&mut *tx).await.map_err(|e| format!("outbox: unsuppress: {}", e))?;
    tx.commit().await.map_err(|e| format!("outbox: commit: {}", e))?;

    if keys_changed {
        if let Ok(guard) = crate::APP_HANDLE.lock() {
            if let Some(app) = guard.as_ref() {
                let _ = app.emit("api-keys-updated", serde_json::json!({}));
                let _ = app.emit("backup-key-updated", serde_json::json!({}));
            }
        }
    }
    // If the deletions emptied the device, drop the Einstein demo persona too, as Clear
    // All does; the old sync path did this and the model otherwise kept addressing the
    // user as "Albert" with no demo memories left (KI-048 follow-up, 2026-09-12).
    if batch.deletes.iter().any(|d| d.table_name == "memories") {
        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories")
            .fetch_one(pool).await.unwrap_or(1);
        if remaining == 0 {
            crate::db::remove_demo_persona_profile();
        }
    }
    Ok((applied, known_before))
}

/// Upsert a memory by sync_id. Unknown sync_id: match by content and adopt the incoming
/// name (permanent — same text is the same memory), else insert under a fresh local id.
/// Last-write-wins by updated_at. A memory this device deleted (tombstoned) and has not
/// since restored is not brought back by a stale copy.
async fn apply_memory(tx: &mut Transaction<'_, Sqlite>, local_user_id: &str, m: &SyncMemory) -> Result<bool, String> {
    let Some(sync_id) = m.sync_id.as_deref() else {
        return Err("outbox: memory without a sync_id".into());
    };
    let embedding: Option<Vec<u8>> = m.embedding.as_ref().map(|v| v.iter().flat_map(|f| f.to_le_bytes()).collect());
    let tags = m.tags.clone().unwrap_or_else(|| "[]".to_string());

    let existing: Option<(i32, Option<DateTime<Utc>>)> = sqlx::query_as(
        "SELECT id, updated_at FROM memories WHERE sync_id = ?")
        .bind(sync_id).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: find memory: {}", e))?;

    let existing = match existing {
        Some(e) => Some(e),
        None => {
            // Reconcile by content and adopt the incoming name.
            let by_content: Option<(i32, Option<DateTime<Utc>>)> = sqlx::query_as(
                "SELECT id, updated_at FROM memories WHERE content = ? ORDER BY id LIMIT 1")
                .bind(&m.content).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: find by content: {}", e))?;
            if let Some((id, _)) = by_content {
                sqlx::query("UPDATE memories SET sync_id = ? WHERE id = ?")
                    .bind(sync_id).bind(id)
                    .execute(&mut **tx).await.map_err(|e| format!("outbox: adopt sync_id: {}", e))?;
            }
            by_content
        }
    };

    match existing {
        Some((id, local_updated)) => {
            if let (Some(theirs), Some(ours)) = (m.updated_at, local_updated) {
                if theirs < ours { return Ok(false); }
            }
            sqlx::query(
                "UPDATE memories SET
                    session_id = ?, content = ?, title = ?, source_type = ?, created_at = ?, updated_at = ?,
                    parent_scroll_id = ?, chunk_index = ?, namespace = ?, is_syncable = ?, is_shareable = ?,
                    embedding = ?, is_ephemeral = ?, expires_at = ?, sentiment_score = ?, sentiment_label = ?,
                    event_type = ?, event_date = ?, entities_detected = ?, original_text = ?,
                    collection_id = ?, memory_placement = ?, external_id = ?, temporal_status = ?,
                    provenance_json = ?, tags = ?
                 WHERE id = ?")
                .bind(&m.session_id).bind(&m.content).bind(&m.title).bind(&m.source_type)
                .bind(m.created_at).bind(m.updated_at.unwrap_or(m.created_at))
                .bind(m.parent_scroll_id).bind(m.chunk_index).bind(&m.namespace).bind(m.is_syncable)
                .bind(m.is_shareable.unwrap_or(false))
                .bind(&embedding).bind(m.is_ephemeral.unwrap_or(false)).bind(m.expires_at)
                .bind(m.sentiment_score.unwrap_or(0.0)).bind(m.sentiment_label.clone().unwrap_or_else(|| "neutral".into()))
                .bind(&m.event_type).bind(m.event_date)
                .bind(m.entities_detected.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "[]".into()))
                .bind(&m.original_text).bind(&m.collection_id).bind(&m.memory_placement).bind(&m.external_id)
                .bind(&m.temporal_status).bind(&m.provenance_json).bind(&tags)
                .bind(id)
                .execute(&mut **tx).await.map_err(|e| format!("outbox: update memory: {}", e))?;
            Ok(true)
        }
        None => {
            // Deleted here and not restored since: a stale copy does not bring it back.
            let tombstoned: Option<String> = sqlx::query_scalar(
                "SELECT deleted_at FROM deleted_memory_hashes WHERE content_hash = ?")
                .bind(content_hash(&m.content))
                .fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: tombstone check: {}", e))?;
            if let Some(deleted_at) = tombstoned {
                let deleted_at = deleted_at.parse::<DateTime<Utc>>().ok();
                let theirs = m.updated_at.unwrap_or(m.created_at);
                if deleted_at.map_or(true, |d| theirs <= d) { return Ok(false); }
            }
            sqlx::query(
                "INSERT INTO memories (user_id, session_id, content, title, source_type, created_at, updated_at,
                    parent_scroll_id, chunk_index, namespace, is_syncable, is_shareable,
                    embedding, is_ephemeral, expires_at, sentiment_score, sentiment_label,
                    event_type, event_date, entities_detected, original_text,
                    collection_id, memory_placement, external_id, temporal_status, provenance_json,
                    tags, sync_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(local_user_id).bind(&m.session_id).bind(&m.content).bind(&m.title).bind(&m.source_type)
                .bind(m.created_at).bind(m.updated_at.unwrap_or(m.created_at))
                .bind(m.parent_scroll_id).bind(m.chunk_index).bind(&m.namespace).bind(m.is_syncable)
                .bind(m.is_shareable.unwrap_or(false))
                .bind(&embedding).bind(m.is_ephemeral.unwrap_or(false)).bind(m.expires_at)
                .bind(m.sentiment_score.unwrap_or(0.0)).bind(m.sentiment_label.clone().unwrap_or_else(|| "neutral".into()))
                .bind(&m.event_type).bind(m.event_date)
                .bind(m.entities_detected.as_ref().map(|v| v.to_string()).unwrap_or_else(|| "[]".into()))
                .bind(&m.original_text).bind(&m.collection_id).bind(&m.memory_placement).bind(&m.external_id)
                .bind(&m.temporal_status).bind(&m.provenance_json).bind(&tags).bind(sync_id)
                .execute(&mut **tx).await.map_err(|e| format!("outbox: insert memory: {}", e))?;
            Ok(true)
        }
    }
}

/// A session's sync_id is its session_id, so the upsert key is the same either way.
async fn apply_session(tx: &mut Transaction<'_, Sqlite>, local_user_id: &str, s: &SyncConversationSession) -> Result<bool, String> {
    let sync_id = s.sync_id.clone().unwrap_or_else(|| s.session_id.clone());
    sqlx::query(
        "INSERT INTO conversation_sessions
             (session_id, user_id, title, started_at, last_active, message_count,
              model_backend, containment_mode, sync_id)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (session_id) DO UPDATE SET
             title            = CASE WHEN EXCLUDED.last_active >= conversation_sessions.last_active
                                     THEN COALESCE(EXCLUDED.title, conversation_sessions.title)
                                     ELSE conversation_sessions.title END,
             last_active      = MAX(EXCLUDED.last_active, conversation_sessions.last_active),
             model_backend    = COALESCE(EXCLUDED.model_backend, conversation_sessions.model_backend),
             containment_mode = COALESCE(EXCLUDED.containment_mode, conversation_sessions.containment_mode),
             sync_id          = COALESCE(conversation_sessions.sync_id, EXCLUDED.sync_id)")
        .bind(&s.session_id).bind(local_user_id).bind(&s.title).bind(s.started_at).bind(s.last_active)
        .bind(s.message_count).bind(&s.model_backend).bind(&s.containment_mode).bind(&sync_id)
        .execute(&mut **tx).await.map_err(|e| format!("outbox: upsert session {}: {}", s.session_id, e))?;
    Ok(true)
}

/// Upsert a message by sync_id. Unknown sync_id: match by the 0011 key (session, role,
/// second, content) and adopt the incoming name; else insert. The local copy may already
/// carry a name of its own (the 0013 backfill named every row), so the match must not
/// require it to be unnamed — that mistake returned 500s on the first device pass. The thread must exist here
/// (its session travels in the same or an earlier batch); if it does not, the message
/// waits for the next batch rather than failing the whole one.
async fn apply_message(tx: &mut Transaction<'_, Sqlite>, local_user_id: &str, msg: &SyncConversationMessage) -> Result<bool, String> {
    let Some(sync_id) = msg.sync_id.as_deref() else {
        return Err("outbox: message without a sync_id".into());
    };
    let existing: Option<i64> = sqlx::query_scalar("SELECT id FROM conversation_messages WHERE sync_id = ?")
        .bind(sync_id).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: find message: {}", e))?;
    let existing = match existing {
        Some(id) => Some(id),
        None => {
            let by_key: Option<i64> = sqlx::query_scalar(
                "SELECT id FROM conversation_messages
                 WHERE session_id = ? AND role = ? AND created_at = ? AND content = ?
                 LIMIT 1")
                .bind(&msg.session_id).bind(&msg.role).bind(msg.created_at).bind(&msg.content)
                .fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: find by key: {}", e))?;
            if let Some(id) = by_key {
                sqlx::query("UPDATE conversation_messages SET sync_id = ? WHERE id = ?")
                    .bind(sync_id).bind(id)
                    .execute(&mut **tx).await.map_err(|e| format!("outbox: adopt message sync_id: {}", e))?;
            }
            by_key
        }
    };
    match existing {
        Some(id) => {
            sqlx::query(
                "UPDATE conversation_messages SET content = ?, model_backend = ?, containment_mode = ?,
                    entry_hash = ?, prev_hash = ?
                 WHERE id = ?")
                .bind(&msg.content).bind(&msg.model_backend).bind(&msg.containment_mode)
                .bind(&msg.entry_hash).bind(&msg.prev_hash).bind(id)
                .execute(&mut **tx).await.map_err(|e| format!("outbox: update message: {}", e))?;
            Ok(true)
        }
        None => {
            let thread_exists: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_sessions WHERE session_id = ?")
                .bind(&msg.session_id).fetch_one(&mut **tx).await.map_err(|e| e.to_string())?;
            if thread_exists == 0 {
                eprintln!("[ZynkSync] outbox: message for unknown thread {} held for a later batch", msg.session_id);
                return Ok(false);
            }
            sqlx::query(
                "INSERT INTO conversation_messages
                     (session_id, user_id, role, content, created_at,
                      model_backend, containment_mode, entry_hash, prev_hash, sync_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(&msg.session_id).bind(local_user_id).bind(&msg.role).bind(&msg.content).bind(msg.created_at)
                .bind(&msg.model_backend).bind(&msg.containment_mode).bind(&msg.entry_hash).bind(&msg.prev_hash)
                .bind(sync_id)
                .execute(&mut **tx).await.map_err(|e| format!("outbox: insert message: {}", e))?;
            Ok(true)
        }
    }
}

/// Newest wins (Matt, 2026-10-01): a key arriving with a later updated_at than this
/// device's row replaces it, in the table and in the environment; an older one is ignored.
/// A key this device has never held is taken. The backup key goes to its file, not .env.
async fn apply_secret(tx: &mut Transaction<'_, Sqlite>, sec: &SyncSecret) -> Result<bool, String> {
    let is_backup_key = sec.name == crate::commands::models::BACKUP_KEY_PUSH_NAME;
    if !is_backup_key && !crate::commands::models::PROPAGATABLE_KEYS.contains(&sec.name.as_str()) {
        eprintln!("[ZynkSync] outbox: key '{}' is not propagatable, ignored", sec.name);
        return Ok(false);
    }
    let ours: Option<(String, DateTime<Utc>, i64)> = sqlx::query_as("SELECT value, updated_at, revision FROM sync_secrets WHERE name = ?")
        .bind(&sec.name).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: read secret: {}", e))?;
    if let Some((value, at, rev)) = &ours {
        // Ours stands when it is the higher revision; at equal revisions the later time;
        // at equal time the greater value, so two devices never flip-flop (KI-081).
        let theirs_wins = (sec.revision, sec.updated_at, &sec.value) > (*rev, *at, value);
        if !theirs_wins {
            // Ours stands. If it is the same value, make sure the environment agrees — a
            // row recorded from a key file that was later edited by hand would otherwise
            // leave the two apart. (Also what makes b02 deterministic: both harness peers
            // share one process environment, so the phone "already holds" the key.)
            if value == &sec.value && !is_backup_key {
                let have = std::env::var(&sec.name).ok();
                if have.as_deref() != Some(sec.value.as_str()) {
                    crate::commands::models::apply_env_key(&sec.name, &sec.value)?;
                } else {
                    let env_path = crate::db::get_app_data_dir().join(".env");
                    let on_disk = std::fs::read_to_string(&env_path).unwrap_or_default();
                    if !on_disk.lines().any(|l| l == crate::env_file::format_line(&sec.name, &sec.value)) {
                        crate::commands::models::apply_env_key(&sec.name, &sec.value)?;
                    }
                }
            }
            return Ok(false);
        }
    }
    sqlx::query(
        "INSERT INTO sync_secrets (name, value, updated_at, sync_id, revision) VALUES (?, ?, ?, ?, ?)
         ON CONFLICT (name) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at, revision = excluded.revision")
        .bind(&sec.name).bind(&sec.value).bind(sec.updated_at).bind(&sec.name).bind(sec.revision)
        .execute(&mut **tx).await.map_err(|e| format!("outbox: store secret: {}", e))?;
    if is_backup_key {
        crate::commands::backup::install_pushed_backup_key(&sec.value)?;
    } else {
        crate::commands::models::apply_env_key(&sec.name, &sec.value)?;
    }
    println!("[ZynkSync] ✓ Key {} received ({})", sec.name, if ours.is_some() { "newer than ours" } else { "new here" });
    Ok(true)
}

/// Upsert a link by sync_id, resolving its two memories by theirs. If either memory is
/// not here yet (it may be in a later slice), the link waits for a later batch. An unknown
/// sync_id is matched by (source, target, relation) and adopts the incoming name.
async fn apply_link(tx: &mut Transaction<'_, Sqlite>, link: &SyncLink) -> Result<bool, String> {
    let source: Option<i64> = sqlx::query_scalar("SELECT id FROM memories WHERE sync_id = ?")
        .bind(&link.source_sync_id).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: link source: {}", e))?;
    let target: Option<i64> = sqlx::query_scalar("SELECT id FROM memories WHERE sync_id = ?")
        .bind(&link.target_sync_id).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: link target: {}", e))?;
    let (Some(source), Some(target)) = (source, target) else {
        eprintln!("[ZynkSync] outbox: link {} waits — one of its memories is not here yet", &link.sync_id[..8.min(link.sync_id.len())]);
        return Ok(false);
    };
    if source == target { return Ok(false); }
    let existing: Option<i64> = sqlx::query_scalar("SELECT id FROM memory_links WHERE sync_id = ?")
        .bind(&link.sync_id).fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: find link: {}", e))?;
    let existing = match existing {
        Some(id) => Some(id),
        None => {
            let by_key: Option<i64> = sqlx::query_scalar(
                "SELECT id FROM memory_links WHERE source_memory_id = ? AND target_memory_id = ? AND relation_type = ? LIMIT 1")
                .bind(source).bind(target).bind(&link.relation_type)
                .fetch_optional(&mut **tx).await.map_err(|e| format!("outbox: find link by key: {}", e))?;
            if let Some(id) = by_key {
                sqlx::query("UPDATE memory_links SET sync_id = ? WHERE id = ?").bind(&link.sync_id).bind(id)
                    .execute(&mut **tx).await.map_err(|e| format!("outbox: adopt link sync_id: {}", e))?;
            }
            by_key
        }
    };
    match existing {
        Some(id) => {
            sqlx::query("UPDATE memory_links SET confidence = ?, notes = ?, created_by = ? WHERE id = ?")
                .bind(link.confidence).bind(&link.notes).bind(&link.created_by).bind(id)
                .execute(&mut **tx).await.map_err(|e| format!("outbox: update link: {}", e))?;
        }
        None => {
            sqlx::query(
                "INSERT INTO memory_links (source_memory_id, target_memory_id, relation_type, confidence, notes, created_by, created_at, sync_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(source).bind(target).bind(&link.relation_type).bind(link.confidence).bind(&link.notes)
                .bind(&link.created_by).bind(&link.created_at).bind(&link.sync_id)
                .execute(&mut **tx).await.map_err(|e| format!("outbox: insert link: {}", e))?;
        }
    }
    Ok(true)
}

/// Forget a row by sync_id. A memory's tombstone hash is recorded as well, so the old
/// sync paths (until step 5) and a stale copy arriving later both respect the deletion.
async fn apply_delete(tx: &mut Transaction<'_, Sqlite>, d: &SyncTombstone) -> Result<bool, String> {
    let affected = match d.table_name.as_str() {
        "memories" => {
            if let Some(h) = &d.content_hash {
                sqlx::query("INSERT OR IGNORE INTO deleted_memory_hashes (content_hash) VALUES (?)")
                    .bind(h).execute(&mut **tx).await.map_err(|e| format!("outbox: record tombstone: {}", e))?;
            }
            sqlx::query("DELETE FROM memories WHERE sync_id = ?")
                .bind(&d.row_sync_id).execute(&mut **tx).await
                .map_err(|e| format!("outbox: delete memory: {}", e))?.rows_affected()
        }
        "conversation_sessions" => sqlx::query("DELETE FROM conversation_sessions WHERE sync_id = ?")
            .bind(&d.row_sync_id).execute(&mut **tx).await
            .map_err(|e| format!("outbox: delete session: {}", e))?.rows_affected(),
        "conversation_messages" => sqlx::query("DELETE FROM conversation_messages WHERE sync_id = ?")
            .bind(&d.row_sync_id).execute(&mut **tx).await
            .map_err(|e| format!("outbox: delete message: {}", e))?.rows_affected(),
        "memory_links" => sqlx::query("DELETE FROM memory_links WHERE sync_id = ?")
            .bind(&d.row_sync_id).execute(&mut **tx).await
            .map_err(|e| format!("outbox: delete link: {}", e))?.rows_affected(),
        "sync_secrets" => {
            let n = sqlx::query("DELETE FROM sync_secrets WHERE name = ?")
                .bind(&d.row_sync_id).execute(&mut **tx).await
                .map_err(|e| format!("outbox: delete secret: {}", e))?.rows_affected();
            if n > 0 && d.row_sync_id != crate::commands::models::BACKUP_KEY_PUSH_NAME {
                crate::commands::models::remove_env_key(&d.row_sync_id)?;
            }
            n
        }
        other => { eprintln!("[ZynkSync] outbox: delete for unknown table {}", other); 0 }
    };
    Ok(affected > 0)
}

// ============================================================================ drain

impl ZynkSyncService {
    /// Send this device's queue to one peer, one batch per call, moving the cursor only
    /// on acknowledgement. Returns what was sent and what the peer reported applying.
    pub async fn drain_outbox_to(&self, peer_device_id: &str, user_id: &str) -> Result<DrainOutcome, String> {
        let peer = {
            let peers = self.transport.peers.read().await;
            peers.get(peer_device_id).cloned().ok_or_else(|| format!("Peer {} not found", peer_device_id))?
        };
        if !peer.paired {
            return Err(format!("Device {} is not paired", peer.device_name));
        }
        // A peer that refused a connection in the last two minutes and has sent no
        // heartbeat is off; building it a 300-row slice every cycle only to fail the
        // connect again costs a phone CPU for nothing (the closed laptop, 2026-10-02).
        if !peer.is_online {
            let recent = self.transport.last_conn_error_logged.read().await.get(peer_device_id)
                .map(|t| Utc::now().signed_duration_since(*t).num_seconds() < 120).unwrap_or(false);
            if recent {
                return Ok(DrainOutcome { skipped: true, ..Default::default() });
            }
        }
        let endpoint = format!("{}/api/zynksync/outbox", peer.url);
        let client = self.transport.http_client.read().await.clone();
        let mut outcome = DrainOutcome::default();
        if let Err(e) = seed_secrets_from_env(&self.db_pool).await {
            eprintln!("[ZynkSync] outbox: could not record this device's keys: {}", e);
        }

        // Two cases get the live tables rather than the queue: a peer this device has
        // never drained to (first contact — the queue only holds changes made since the
        // outbox existed, and rows that predate it would otherwise never move; found on
        // the first device pass, 2026-10-01), and a peer whose cursor points before what
        // the queue still holds (it missed rows that were pruned). Either way the cursor
        // then starts at the queue's current end, and the queue takes over from there.
        let cursor = cursor_row(&self.db_pool, peer_device_id).await?;
        let (oldest, newest): (Option<i64>, Option<i64>) = sqlx::query_as("SELECT MIN(id), MAX(id) FROM sync_outbox")
            .fetch_one(&self.db_pool).await.map_err(|e| e.to_string())?;
        let behind = match (cursor, oldest) { (Some(c), Some(o)) => c + 1 < o, _ => false };
        if cursor.is_none() || behind {
            // Capped like every other batch, one slice per sync cycle (KI-068); the
            // cursor is written only after the last slice, so a restart starts over.
            let key = (self.identity().device_id, peer_device_id.to_string());
            let (offset, through) = FULL_SEND_OFFSET.lock().unwrap().get(&key).copied()
                .unwrap_or((0, newest.unwrap_or(0)));
            let (batch, total, covered) = build_full_resend(&self.db_pool, user_id, through, offset, OUTBOX_BATCH_ROWS as usize).await?;
            println!("[ZynkSync] outbox: {} to {} — live tables rows {}..{} of {}{} (queue {:?}..{:?})",
                if cursor.is_none() { "first contact" } else { "behind the queue" }, peer.device_name,
                offset, offset + covered, total, if batch.secrets.is_empty() { String::new() } else { format!(" + {} key(s)", batch.secrets.len()) }, oldest, newest);
            let receipt = if batch.is_empty() { OutboxReceipt { through, applied: 0, known_before: Some(through) } } else { post_batch(&client, &endpoint, peer_device_id, &batch).await? };
            outcome.batches += 1;
            outcome.entries_sent += batch.len();
            outcome.applied_by_peer += receipt.applied;
            let next = offset + covered;
            if next >= total {
                FULL_SEND_OFFSET.lock().unwrap().remove(&key);
                set_cursor(&self.db_pool, peer_device_id, through).await?;
            } else {
                FULL_SEND_OFFSET.lock().unwrap().insert(key, (next, through));
                prune_outbox(&self.db_pool, &self.paired_peer_ids().await).await?;
                return Ok(outcome); // the rest of the live tables goes next cycle
            }
        }

        // One batch per call, as the history push was capped before (KI-068): a sync
        // cycle carries at most OUTBOX_BATCH_ROWS queued rows and the rest goes next
        // cycle, so one request can never outgrow the receiver's limit and one sync call
        // never runs unbounded. The auto-sync timer drains a long queue over its cycles.
        let cursor = cursor_for(&self.db_pool, peer_device_id).await?;
        if let Some(batch) = build_batch(&self.db_pool, cursor).await? {
            let through = batch.through;
            let sent = batch.len();
            let receipt = if batch.is_empty() {
                OutboxReceipt { through, applied: 0, known_before: Some(through) } // everything collapsed away; just advance
            } else {
                post_batch(&client, &endpoint, peer_device_id, &batch).await?
            };
            // The peer applied this batch, but if it knew less beforehand than our cursor
            // claimed (a restored install), what we skipped must go again: start it over
            // next cycle rather than record this batch as the new cursor.
            if receipt.known_before.map_or(true, |k| k < cursor) && cursor > 0 {
                println!("[ZynkSync] outbox: {} knew {:?} of our batches, our cursor said {} — starting it over", peer.device_name, receipt.known_before, cursor);
                sqlx::query("DELETE FROM sync_outbox_cursor WHERE peer_device_id = ?").bind(peer_device_id)
                    .execute(&self.db_pool).await.map_err(|e| e.to_string())?;
            } else {
                set_cursor(&self.db_pool, peer_device_id, receipt.through.max(through)).await?;
            }
            outcome.batches += 1;
            outcome.entries_sent += sent;
            outcome.applied_by_peer += receipt.applied;
        }

        prune_outbox(&self.db_pool, &self.paired_peer_ids().await).await?;
        Ok(outcome)
    }

    async fn paired_peer_ids(&self) -> Vec<String> {
        let peers = self.transport.peers.read().await;
        peers.values().filter(|p| p.paired).map(|p| p.device_id.clone()).collect()
    }

    /// Ask a peer to drain its queue to this device. The peer runs `drain_outbox_to` for
    /// us on its side, so the drain code lives in one place.
    pub async fn pull_outbox_from(&self, peer_device_id: &str) -> Result<DrainOutcome, String> {
        let peer = {
            let peers = self.transport.peers.read().await;
            peers.get(peer_device_id).cloned().ok_or_else(|| format!("Peer {} not found", peer_device_id))?
        };
        let endpoint = format!("{}/api/zynksync/outbox/pull", peer.url);
        let client = self.transport.http_client.read().await.clone();
        let known_through = inbox_cursor(&self.db_pool, peer_device_id).await?;
        let response = client.post(&endpoint)
            .header("x-target-device-id", peer_device_id)
            .json(&PullRequest { known_through })
            .timeout(std::time::Duration::from_secs(120))
            .send().await
            .map_err(|e| format!("outbox pull from {} failed: {}", peer.device_name, describe(&e)))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("outbox pull from {} rejected: {} {}", peer.device_name, status, body.chars().take(300).collect::<String>()));
        }
        let v: serde_json::Value = response.json().await.map_err(|e| format!("outbox pull: bad reply: {}", e))?;
        Ok(DrainOutcome {
            batches: v["batches"].as_u64().unwrap_or(0) as usize,
            entries_sent: v["entries_sent"].as_u64().unwrap_or(0) as usize,
            applied_by_peer: v["applied"].as_u64().unwrap_or(0) as usize,
            skipped: false,
        })
    }
}

/// reqwest's Display hides the cause ("error sending request for url"); the chain has it.
fn describe(e: &reqwest::Error) -> String {
    let mut out = e.to_string();
    let mut src = std::error::Error::source(e);
    while let Some(s) = src { out.push_str(" <- "); out.push_str(&s.to_string()); src = s.source(); }
    out
}

async fn post_batch(client: &reqwest::Client, endpoint: &str, target_device_id: &str, batch: &OutboxBatch) -> Result<OutboxReceipt, String> {
    let response = client.post(endpoint)
        .header("x-target-device-id", target_device_id)
        .json(batch)
        .timeout(std::time::Duration::from_secs(120))
        .send().await
        .map_err(|e| format!("outbox send failed: {}", describe(&e)))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("outbox batch rejected: {} {}", status, body.chars().take(300).collect::<String>()));
    }
    response.json::<OutboxReceipt>().await.map_err(|e| format!("outbox: bad receipt: {}", e))
}
