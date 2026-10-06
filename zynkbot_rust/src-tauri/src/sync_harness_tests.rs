//! ZynkSync two-peer harness (2026-09-17; the rebuild's finish line reached 2026-10-02).
//!
//! Two (or more) real `ZynkSyncService` instances in one process, each with its own
//! SQLite file in a temp dir, its own TLS certificate and identity, listening on a
//! free loopback port. Peers talk over the real router and the real mTLS client —
//! nothing is mocked — so these tests describe what two devices must end up with,
//! not how the code gets there. They were the acceptance tests for the outbox rebuild
//! (docs/TESTING.md, "Sync behaviours the harness must cover"): seven of them failed on
//! the old code and were `#[ignore]`d with their known-issue numbers; all pass now, and
//! every bug the first device pass found got a test here first (b16–b22).
//!
//! Run: `LD_LIBRARY_PATH=$PWD/lib/vosk cargo test --lib sync_harness -- --nocapture`
//!
//! Keys: every peer here shares one process environment, so a key one test applies can be
//! re-seeded by another test's peer later. Each key test therefore owns one key name
//! (b02 OPENAI_API_KEY, b20 XAI_API_KEY, b23 MISTRAL_API_KEY) and holds `hold_env()`.

use crate::zynksync::{SyncIdentity, ZynkSyncService};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};
use std::path::PathBuf;
use std::sync::Arc;

/// One simulated device.
struct Peer {
    name: &'static str,
    svc: Arc<ZynkSyncService>,
    pool: SqlitePool,
    dir: PathBuf,
    port: u16,
}

impl Peer {
    async fn spawn(name: &'static str) -> Peer {
        // Peers write keys to get_app_data_dir()/.env (step 3). Never the developer's real
        // one: point XDG_DATA_HOME at a harness directory for the whole test process
        // unless a test (b02, b20) has already set its own.
        if std::env::var_os("XDG_DATA_HOME").is_none() {
            let shared = std::env::temp_dir().join(format!("zynkbot-harness-xdg-{}", std::process::id()));
            std::fs::create_dir_all(shared.join("zynkbot")).unwrap();
            std::env::set_var("XDG_DATA_HOME", &shared);
        }
        let dir = std::env::temp_dir().join(format!("zynkbot-harness-{}-{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let url = format!("sqlite://{}?mode=rwc", dir.join("zynkbot.db").display());
        let pool = SqlitePoolOptions::new()
            .max_connections(8)
            .after_connect(|conn, _| Box::pin(async move {
                sqlx::query("PRAGMA foreign_keys=ON").execute(&mut *conn).await?;
                sqlx::query("PRAGMA busy_timeout=15000").execute(&mut *conn).await?;
                Ok(())
            }))
            .connect(&url).await.expect("open peer db");
        sqlx::migrate!("./migrations").run(&pool).await.expect("migrate peer db");

        let (cert_pem, key_pem, cert_der) = crate::tls::load_or_generate_cert(&dir).expect("cert");
        let identity = SyncIdentity {
            user_id: uuid::Uuid::new_v4().to_string(),
            device_id: uuid::Uuid::new_v4().to_string(),
            device_name: name.to_string(),
        };
        let svc = Arc::new(ZynkSyncService::new(identity, Some(0), pool.clone(), Some(3600), cert_pem, key_pem, cert_der));
        let port = svc.clone().start_http_server().await.expect("start server");
        // The listener task needs a moment to be accepting.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Peer { name, svc, pool, dir, port }
    }

    /// The same device after an app restart: same folder, database, certificate and
    /// identity; a fresh service with nothing in memory. Keep the original alive until
    /// the test ends — dropping it deletes the folder.
    /// A fresh install that restored `bundle` (its identity, certificate and peers)
    /// before first start — a wiped phone brought back.
    async fn spawn_restored(name: &'static str, bundle: &serde_json::Value) -> Peer {
        let dir = std::env::temp_dir().join(format!("zynkbot-harness-{}-restored-{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let url = format!("sqlite://{}?mode=rwc", dir.join("zynkbot.db").display());
        let pool = SqlitePoolOptions::new().max_connections(8)
            .after_connect(|conn, _| Box::pin(async move {
                sqlx::query("PRAGMA foreign_keys=ON").execute(&mut *conn).await?;
                sqlx::query("PRAGMA busy_timeout=15000").execute(&mut *conn).await?;
                Ok(())
            }))
            .connect(&url).await.expect("open restored db");
        sqlx::migrate!("./migrations").run(&pool).await.expect("migrate restored db");
        let (user_id, device_id, device_name) = crate::commands::backup::apply_device_bundle(&pool, &dir, bundle).await.expect("apply bundle");
        let (cert_pem, key_pem, cert_der) = crate::tls::load_or_generate_cert(&dir).expect("restored cert");
        let identity = SyncIdentity { user_id, device_id, device_name: if device_name.is_empty() { name.to_string() } else { device_name } };
        let svc = Arc::new(ZynkSyncService::new(identity, Some(0), pool.clone(), Some(3600), cert_pem, key_pem, cert_der));
        let port = svc.clone().start_http_server().await.expect("start server");
        svc.load_devices().await.expect("load restored peers");
        svc.rebuild_http_client().await.expect("client");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Peer { name, svc, pool, dir, port }
    }

    async fn respawn(&self) -> Peer {
        let (cert_pem, key_pem, cert_der) = crate::tls::load_or_generate_cert(&self.dir).expect("cert");
        let svc = Arc::new(ZynkSyncService::new(self.svc.identity(), Some(0), self.pool.clone(), Some(3600), cert_pem, key_pem, cert_der));
        let port = svc.clone().start_http_server().await.expect("start server");
        svc.load_devices().await.expect("reload devices");
        svc.rebuild_http_client().await.expect("rebuild client");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Peer { name: self.name, svc, pool: self.pool.clone(), dir: self.dir.clone(), port }
    }

    fn user_id(&self) -> String { self.svc.user_id().unwrap() }
    fn device_id(&self) -> String { self.svc.identity().device_id }
    fn addr(&self) -> String { format!("127.0.0.1:{}", self.port) }

    /// What the UI does when the user enters a pairing code: pair, rebuild the pinned
    /// client, then adopt the host's user id (ZynkSyncPanel → set_user_identity).
    async fn pair_with(&self, host: &Peer) {
        let code = host.svc.generate_pairing_code().await.expect("pairing code");
        let peer = self.svc.add_device(&host.addr(), &code).await.expect("add_device");
        self.svc.rebuild_http_client().await.expect("rebuild client");
        assert_eq!(peer.device_id, host.device_id(), "{}: paired with the wrong device", self.name);
        if let Some(uid) = peer.user_id.as_deref() {
            if uid != self.user_id() { self.svc.adopt_user_id(uid).await.expect("adopt user id"); }
        }
        host.svc.load_devices().await.expect("host reloads devices");
    }

    /// One full sync round from this peer's point of view (what the auto-sync timer does).
    async fn sync_with(&self, other: &Peer) -> crate::zynksync::SyncResult {
        self.svc.sync_bidirectional(&other.device_id(), &self.user_id()).await
            .unwrap_or_else(|e| panic!("{} sync with {} failed: {}", self.name, other.name, e))
    }

    async fn add_memory(&self, content: &str) -> i32 {
        crate::memory::insert_memory(
            &self.pool, Some(&content.chars().take(40).collect::<String>()), content, Some("conversation"), None,
            Some(vec![0.1; 384]), None, None, Some(&self.user_id()), "personal", true, false, Some(serde_json::json!([])), None, None, Some(content),
        ).await.expect("insert memory")
    }

    async fn memory_contents(&self) -> Vec<String> {
        sqlx::query("SELECT content FROM memories WHERE user_id = ? ORDER BY content")
            .bind(self.user_id()).fetch_all(&self.pool).await.unwrap()
            .into_iter().map(|r| r.get::<String, _>("content")).collect()
    }

    async fn device_rows(&self) -> Vec<(String, String, i64, i64)> {
        sqlx::query("SELECT device_id, device_name, port, sync_paired FROM zynk_devices ORDER BY device_name")
            .fetch_all(&self.pool).await.unwrap().into_iter()
            .map(|r| (r.get("device_id"), r.get("device_name"), r.get::<i64, _>("port"), r.get::<i64, _>("sync_paired"))).collect()
    }
}

impl Drop for Peer {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.dir); }
}

fn rt_test<F: std::future::Future<Output = ()>>(f: F) {
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap().block_on(f)
}

// ---------------------------------------------------------------------------
// 1. Pair with a code: both sides hold each other's row and certificate; the joining
//    device adopts the host's user id.
// ---------------------------------------------------------------------------
#[test]
fn b01_pairing_gives_both_sides_a_pinned_peer_and_one_user_id() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        let b_original_user = b.user_id();
        b.pair_with(&a).await;

        assert_eq!(b.user_id(), a.user_id(), "the joining device adopts the host's user id");
        assert_ne!(b.user_id(), b_original_user);

        let a_rows = a.device_rows().await;
        let b_rows = b.device_rows().await;
        assert!(a_rows.iter().any(|(id, name, port, paired)| id == &b.device_id() && name == "phone" && *port as u16 == b.port && *paired == 1),
            "host must hold the phone with its real port; rows: {:?}", a_rows);
        assert!(b_rows.iter().any(|(id, name, port, paired)| id == &a.device_id() && name == "desktop" && *port as u16 == a.port && *paired == 1),
            "phone must hold the desktop with its real port; rows: {:?}", b_rows);

        // Each side pinned the other's certificate (the mTLS client is built from these rows).
        for (p, other) in [(&a, &b), (&b, &a)] {
            let cert: Option<Vec<u8>> = sqlx::query_scalar("SELECT tls_cert_der FROM zynk_devices WHERE device_id = ?")
                .bind(other.device_id()).fetch_optional(&p.pool).await.unwrap().flatten();
            assert!(cert.map(|c| !c.is_empty()).unwrap_or(false), "{} has no pinned cert for {}", p.name, other.name);
        }
    });
}

// ---------------------------------------------------------------------------
// 3/4. Memory added on A appears on B; deleted on A disappears on B and stays gone.
// ---------------------------------------------------------------------------
#[test]
fn b03_memory_added_on_one_device_appears_on_the_other() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;

        a.add_memory("My car is a 2019 Outback").await;
        a.sync_with(&b).await;
        assert_eq!(b.memory_contents().await, vec!["My car is a 2019 Outback".to_string()]);

        // and the other direction
        b.add_memory("I have a cat named Pickles").await;
        b.sync_with(&a).await;
        assert_eq!(a.memory_contents().await, vec!["I have a cat named Pickles".to_string(), "My car is a 2019 Outback".to_string()]);
    });
}

#[test]
fn b04_memory_deleted_on_one_device_is_gone_on_the_other_and_stays_gone() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let id = a.add_memory("Temporary fact").await;
        a.sync_with(&b).await;
        assert_eq!(b.memory_contents().await.len(), 1);

        // Delete on A the way the Memory Manager does: propagate (tombstone + peers), then local.
        // As the app does since step 5: tombstone locally, delete; the outbox carries it.
        a.svc.record_tombstones(&[crate::sync_outbox::content_hash("Temporary fact")]).await.unwrap();
        sqlx::query("DELETE FROM memories WHERE id = ?").bind(id).execute(&a.pool).await.unwrap();
        a.sync_with(&b).await;
        assert!(b.memory_contents().await.is_empty(), "phone still has the deleted memory");

        // A later sync from the phone must not resurrect it on the desktop (tombstone).
        b.sync_with(&a).await;
        a.sync_with(&b).await;
        assert!(a.memory_contents().await.is_empty(), "deleted memory came back on the desktop");
        assert!(b.memory_contents().await.is_empty(), "deleted memory came back on the phone");
    });
}

// ---------------------------------------------------------------------------
// 5. Edit on A: B ends with the new text only.   (KI-060: the old version returns)
// ---------------------------------------------------------------------------
#[test]
fn b05_memory_edited_on_one_device_does_not_come_back_old() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let id = a.add_memory("My dentist is Dr Alvarez").await;
        a.sync_with(&b).await;

        sqlx::query("UPDATE memories SET content = ?, updated_at = datetime('now') WHERE id = ?")
            .bind("My dentist is Dr Alvarez on Main Street").bind(id).execute(&a.pool).await.unwrap();

        // Both directions of sync afterwards: the phone must not push the old text back.
        b.sync_with(&a).await;
        a.sync_with(&b).await;
        assert_eq!(a.memory_contents().await, vec!["My dentist is Dr Alvarez on Main Street".to_string()]);
        assert_eq!(b.memory_contents().await, vec!["My dentist is Dr Alvarez on Main Street".to_string()]);
    });
}

// ---------------------------------------------------------------------------
// 8/9. Conversation history: a thread with N messages arrives once; a second sync
//      sends nothing new; deleting the thread on A removes it on B. (KI-028)
// ---------------------------------------------------------------------------
async fn thread_counts(p: &Peer, session: &str) -> (i64, i64) {
    let sessions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_sessions WHERE session_id = ?").bind(session).fetch_one(&p.pool).await.unwrap();
    let messages: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?").bind(session).fetch_one(&p.pool).await.unwrap();
    (sessions, messages)
}

#[test]
fn b08_history_thread_arrives_once_with_all_its_messages() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let uid = a.user_id();
        for i in 0..3 {
            crate::conversation_history::log_exchange(&a.pool, "thread-1", &uid, &format!("question {}", i), &format!("answer {}", i), "anthropic", "guardian", true, "typed").await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await; // real exchanges are seconds apart
        }
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-1").await, (1, 6), "phone should have the thread once with 6 messages");
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-1").await, (1, 6), "a second sync must not duplicate anything");
    });
}

#[test]
fn b08b_two_messages_in_the_same_second_both_arrive() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let uid = a.user_id();
        crate::conversation_history::log_exchange(&a.pool, "thread-3", &uid, "first", "one", "anthropic", "guardian", true, "typed").await.unwrap();
        crate::conversation_history::log_exchange(&a.pool, "thread-3", &uid, "second", "two", "anthropic", "guardian", true, "typed").await.unwrap();
        assert_eq!(thread_counts(&a, "thread-3").await, (1, 4));
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-3").await, (1, 4), "phone lost messages that shared a timestamp");
    });
}

// 8c. The "sent up to here" marker for history survives a restart, so the whole
//     history is not pushed again every time the app starts (KI-068), and what is
//     logged after the restart still arrives.
#[test]
fn b08c_history_marker_survives_a_restart() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let uid = a.user_id();
        for i in 0..2 {
            crate::conversation_history::log_exchange(&a.pool, "thread-4", &uid, &format!("q{}", i), &format!("a{}", i), "anthropic", "guardian", true, "typed").await.unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        }
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-4").await, (1, 4));

        let a2 = a.respawn().await;
        // The per-peer cursor is persisted (sync_outbox_cursor, 0013), so a restart offers
        // the phone nothing it already has. Before the outbox the marker lived in memory
        // and every app start re-sent the whole history (KI-068).
        let cursor = crate::sync_outbox::cursor_for(&a2.pool, &b.device_id()).await.unwrap();
        let pending = crate::sync_outbox::build_batch(&a2.pool, cursor).await.unwrap();
        assert!(pending.is_none(), "after a restart nothing already sent may be offered again, got {:?} entries", pending.map(|p| p.len()));

        tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
        crate::conversation_history::log_exchange(&a2.pool, "thread-4", &uid, "q2", "a2", "anthropic", "guardian", true, "typed").await.unwrap();
        a2.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-4").await, (1, 6), "an exchange logged after the restart must still arrive");
        drop(a2);
    });
}

// 8d. A history bigger than one push (a desktop's first sync to a new phone) arrives
//     over a few cycles, capped per request, with nothing duplicated (KI-068).
#[test]
fn b08d_a_history_bigger_than_one_push_arrives_over_cycles() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let uid = a.user_id();
        let cap = crate::sync_outbox::OUTBOX_BATCH_ROWS as usize;
        let total = cap + 50;
        // Two threads, one message per second, written the way the app writes them (RFC 3339).
        let t0 = chrono::Utc::now() - chrono::Duration::seconds(total as i64 + 60);
        for s in ["big-1", "big-2"] {
            sqlx::query("INSERT INTO conversation_sessions (session_id, user_id, title, started_at, last_active) VALUES (?, ?, ?, ?, ?)")
                .bind(s).bind(&uid).bind(s).bind(t0).bind(t0 + chrono::Duration::seconds(total as i64))
                .execute(&a.pool).await.unwrap();
        }
        for i in 0..total {
            let session = if i % 2 == 0 { "big-1" } else { "big-2" };
            let role = if i % 4 < 2 { "user" } else { "assistant" };
            sqlx::query("INSERT INTO conversation_messages (session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)")
                .bind(session).bind(&uid).bind(role).bind(format!("message {}", i)).bind(t0 + chrono::Duration::seconds(i as i64))
                .execute(&a.pool).await.unwrap();
        }
        let count = |p: &Peer| {
            let pool = p.pool.clone();
            async move { sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM conversation_messages").fetch_one(&pool).await.unwrap() }
        };
        a.sync_with(&b).await;
        let after_one = count(&b).await;
        assert!(after_one as usize <= cap && after_one > 0, "one push is capped at {} messages, got {}", cap, after_one);
        a.sync_with(&b).await;
        assert_eq!(count(&b).await as usize, total, "the whole history arrives over a few cycles");
        a.sync_with(&b).await;
        assert_eq!(count(&b).await as usize, total, "a further cycle adds nothing");
    });
}

#[test]
fn b09_history_thread_deleted_on_one_device_is_gone_on_the_other() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let uid = a.user_id();
        crate::conversation_history::log_exchange(&a.pool, "thread-2", &uid, "q", "a", "anthropic", "guardian", true, "typed").await.unwrap();
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-2").await.0, 1);

        crate::conversation_history::delete_session(&a.pool, "thread-2", &uid).await.unwrap();
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "thread-2").await, (0, 0), "deleted thread still on the phone");
    });
}

// ---------------------------------------------------------------------------
// 10. Renaming a device shows on peers without re-pairing.
// ---------------------------------------------------------------------------
#[test]
fn b10_device_rename_reaches_peers_without_repairing() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;

        b.svc.set_device_name("Pixel");
        b.svc.rebuild_http_client().await.unwrap(); // production does this from set_device_name
        b.sync_with(&a).await;

        let name: String = sqlx::query_scalar("SELECT device_name FROM zynk_devices WHERE device_id = ?")
            .bind(b.device_id()).fetch_one(&a.pool).await.unwrap();
        assert_eq!(name, "Pixel", "desktop still shows the phone's old name");
    });
}

// ---------------------------------------------------------------------------
// 14. Contract: a client that has not pinned the peer's certificate is refused.
// ---------------------------------------------------------------------------
#[test]
fn b14_unpinned_client_cannot_talk_to_a_peer() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(3)).build().unwrap();
        let r = client.get(format!("https://127.0.0.1:{}/api/zynksync/info", a.port)).send().await;
        assert!(r.is_err(), "a client without the pinned certificate must fail the TLS handshake");
    });
}

// ---------------------------------------------------------------------------
// 15. Chat between a user's own two devices: a message sent on the desktop to the
//     phone arrives on the phone, over the sync pairing alone (no ZynkLink).
// ---------------------------------------------------------------------------
#[test]
fn b15_chat_message_reaches_my_other_device_over_a_sync_pairing() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let (a_id, b_id) = (uuid::Uuid::parse_str(&a.device_id()).unwrap(), uuid::Uuid::parse_str(&b.device_id()).unwrap());
        let uid = uuid::Uuid::parse_str(&a.user_id()).unwrap();

        crate::zchat::send_message(&a.pool, a_id, b_id, "note to self: buy oil filter".into(), uid).await.expect("send");
        let delivered = crate::zchat::deliver_to_peer(&a.svc.transport, &b.device_id()).await.expect("deliver");
        assert_eq!(delivered, 1);

        let on_phone = crate::zchat::get_messages(&b.pool, b_id, a_id, None).await.expect("get");
        let texts: Vec<String> = on_phone.messages.iter().map(|m| m.message_text.clone()).collect();
        assert_eq!(texts, vec!["note to self: buy oil filter".to_string()]);
    });
}


// ===========================================================================
// The six behaviours whose numbers were reserved but never written (2026-09-29).
// Each states what two devices must end up with. The ones that fail on today's code
// are #[ignore]d with their known-issue number; they are the outbox rebuild's finish line.
// ===========================================================================

/// A data directory for the two behaviours that touch `.env` (API keys). The receiving
/// side of a key push writes to the app's data directory, which is process-wide, so
/// these hold the chat harness's environment lock and point the app at a temp dir.
struct EnvDir { dir: PathBuf }
impl EnvDir {
    fn new() -> EnvDir {
        let dir = std::env::temp_dir().join(format!("zynkbot-harness-env-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("zynkbot")).unwrap();
        std::env::set_var("XDG_DATA_HOME", &dir);
        EnvDir { dir }
    }
    fn env_file(&self) -> String {
        std::fs::read_to_string(self.dir.join("zynkbot").join(".env")).unwrap_or_default()
    }
}
impl Drop for EnvDir {
    fn drop(&mut self) {
        std::env::remove_var("OPENAI_API_KEY");
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// ---------------------------------------------------------------------------
// 2. Keys saved on the desktop before a phone pairs must reach that phone once it
//    does (KI-055). Today the push only runs when a key is saved or the button is
//    pressed, so a device that pairs afterwards gets nothing.

// ---------------------------------------------------------------------------
// 6. A contradiction resolved on one device ("keep the new fact") leaves both devices
//    holding only the new fact. Mirrors what resolve_memory_conflict_v2 does on
//    "keep_new": delete the old memory (tombstone, then peers), store the new one.
// ---------------------------------------------------------------------------
#[test]
fn b06_a_contradiction_resolved_on_one_device_leaves_both_with_only_the_new_fact() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let old = a.add_memory("My dog is named Max").await;
        a.sync_with(&b).await;
        assert_eq!(b.memory_contents().await, vec!["My dog is named Max".to_string()]);

        a.svc.record_tombstones(&[crate::sync_outbox::content_hash("My dog is named Max")]).await.unwrap();
        sqlx::query("DELETE FROM memories WHERE id = ?").bind(old).execute(&a.pool).await.unwrap();
        a.add_memory("My dog is named Wendy").await;

        a.sync_with(&b).await;
        b.sync_with(&a).await;
        assert_eq!(a.memory_contents().await, vec!["My dog is named Wendy".to_string()], "desktop");
        assert_eq!(b.memory_contents().await, vec!["My dog is named Wendy".to_string()], "phone");
    });
}

// ---------------------------------------------------------------------------
// 7. A phone that already holds memories, then pairs, keeps them and shares them
//    (KI-011). Today they stay under the phone's old user id and vanish from view.
// ---------------------------------------------------------------------------
#[test]
fn b07_memories_held_before_pairing_are_shared_after_it() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.add_memory("I have a cat named Pickles").await;   // under the phone's own user id
        b.pair_with(&a).await;                                // the phone adopts the desktop's
        a.add_memory("My car is a 2019 Outback").await;

        b.sync_with(&a).await;
        a.sync_with(&b).await;
        let want = vec!["I have a cat named Pickles".to_string(), "My car is a 2019 Outback".to_string()];
        assert_eq!(a.memory_contents().await, want, "desktop is missing the phone's earlier memory");
        assert_eq!(b.memory_contents().await, want, "phone lost its own earlier memory when it paired");
    });
}

// ---------------------------------------------------------------------------
// 11. A phone that is wiped and brought back from its backup is the same phone: the
//     desktop lists it once, under the identity it always had, and syncs with it without
//     pairing again (KI-050). The identity rides in the encrypted backup as its own
//     object; the user says which device the fresh install replaces — nothing is guessed
//     from a name or an address (decided 2026-09-12, confirmed 2026-10-01).
// ---------------------------------------------------------------------------
#[test]
fn b11_a_phone_restored_from_its_backup_is_the_same_phone() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        a.add_memory("Oil change every 5000 miles").await;
        a.sync_with(&b).await;
        let old_id = b.device_id();

        // The phone backs up its identity object, then is wiped.
        let bundle = crate::commands::backup::build_device_bundle(&b.pool, &b.user_id(), &b.device_id(), "phone", &b.dir).await.expect("bundle");
        drop(b);

        // A fresh install restores it before it ever pairs.
        let b2 = Peer::spawn_restored("phone", &bundle).await;
        assert_eq!(b2.device_id(), old_id, "the restored phone must have its old identity");
        assert_eq!(b2.user_id(), a.user_id(), "and the shared user id");

        // It syncs with the desktop straight away: the desktop knows this certificate,
        // and learns the new port from the request itself.
        b2.sync_with(&a).await;
        a.sync_with(&b2).await;
        let phones: Vec<_> = a.device_rows().await.into_iter().filter(|(_, name, _, paired)| name == "phone" && *paired == 1).collect();
        assert_eq!(phones.len(), 1, "desktop lists the phone {} times: {:?}", phones.len(), phones);
        assert_eq!(phones[0].0, old_id);
        assert_eq!(b2.memory_contents().await, vec!["Oil change every 5000 miles".to_string()], "the restored phone did not get the memories back");
    });
}

// ---------------------------------------------------------------------------
// 12. The newer memory fields travel with a memory: its tags and its "Remembered on
//     request" mark (KI-030: today the sync payload has no room for them).
// ---------------------------------------------------------------------------
#[test]
fn b12_tags_and_the_remembered_on_request_mark_travel_with_a_memory() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let id = a.add_memory("Sourdough needs a 12-hour rise").await;
        sqlx::query("UPDATE memories SET tags = ?, provenance_json = ? WHERE id = ?")
            .bind(r#"["kitchen","bread"]"#).bind(r#"{"requested":true}"#).bind(id)
            .execute(&a.pool).await.unwrap();
        a.sync_with(&b).await;

        let (tags, provenance): (Option<String>, Option<String>) =
            sqlx::query_as("SELECT tags, provenance_json FROM memories WHERE content = ?")
                .bind("Sourdough needs a 12-hour rise").fetch_one(&b.pool).await.expect("memory on phone");
        assert_eq!(tags.as_deref(), Some(r#"["kitchen","bread"]"#), "tags did not travel");
        assert!(provenance.as_deref().map_or(false, |p| p.contains("requested")),
            "the Remember mark did not travel: {:?}", provenance);
    });
}

// ---------------------------------------------------------------------------
// 13. A phone that was switched off while the desktop changed things catches up when
//     it returns: it gains what was added and loses what was deleted, and the deleted
//     memory does not come back to the desktop from the phone's stale copy.
// ---------------------------------------------------------------------------
#[test]
fn b13_a_device_that_was_offline_catches_up_when_it_returns() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let stale = a.add_memory("Old phone number 555-0100").await;
        a.sync_with(&b).await;
        assert_eq!(b.memory_contents().await.len(), 1);

        // The phone goes away: the desktop's record of it points at a port nothing listens on.
        sqlx::query("UPDATE zynk_devices SET port = 1 WHERE device_id = ?").bind(b.device_id()).execute(&a.pool).await.unwrap();
        a.svc.load_devices().await.unwrap();
        a.add_memory("New phone number 555-0199").await;
        a.svc.record_tombstones(&[crate::sync_outbox::content_hash("Old phone number 555-0100")]).await.unwrap();
        sqlx::query("DELETE FROM memories WHERE id = ?").bind(stale).execute(&a.pool).await.unwrap();

        // The phone comes back.
        sqlx::query("UPDATE zynk_devices SET port = ? WHERE device_id = ?").bind(b.port as i64).bind(b.device_id()).execute(&a.pool).await.unwrap();
        a.svc.load_devices().await.unwrap();
        b.sync_with(&a).await;
        a.sync_with(&b).await;
        assert_eq!(a.memory_contents().await, vec!["New phone number 555-0199".to_string()], "desktop: the deleted memory came back from the phone");
        assert_eq!(b.memory_contents().await, vec!["New phone number 555-0199".to_string()], "phone: did not catch up");
    });
}

// ---------------------------------------------------------------------------
// 16. Copies of one memory that predate the outbox hold different names on different
//     devices (the 0013 backfill is random and runs on each device alone). Syncing must
//     recognise them as one memory by their text and leave each device with one copy
//     under one shared name — and that stays on for good: identical text is the same
//     memory (Matt, 2026-10-01).
// ---------------------------------------------------------------------------
#[test]
fn b16_pre_outbox_copies_of_one_memory_converge_instead_of_doubling() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        // The same memory already on both, named differently, as after the upgrade.
        // Written under sync_suppress so neither device queues it: it is history, not a change.
        for p in [&a, &b] {
            sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&p.pool).await.unwrap();
            p.add_memory("The attic key is on the hook by the door").await;
            sqlx::query("DELETE FROM sync_suppress").execute(&p.pool).await.unwrap();
        }
        let name_on = |p: &Peer| {
            let pool = p.pool.clone();
            async move {
                sqlx::query_scalar::<_, String>("SELECT sync_id FROM memories WHERE content = ?")
                    .bind("The attic key is on the hook by the door").fetch_one(&pool).await.unwrap()
            }
        };
        assert_ne!(name_on(&a).await, name_on(&b).await, "the setup needs two different names");

        // The desktop edits it: that queues one update, which carries the desktop's name.
        sqlx::query("UPDATE memories SET content = content, updated_at = datetime('now') WHERE content = ?")
            .bind("The attic key is on the hook by the door").execute(&a.pool).await.unwrap();
        a.sync_with(&b).await;
        b.sync_with(&a).await;

        assert_eq!(a.memory_contents().await, vec!["The attic key is on the hook by the door".to_string()], "desktop doubled it");
        assert_eq!(b.memory_contents().await, vec!["The attic key is on the hook by the door".to_string()], "phone doubled it");
        assert_eq!(name_on(&a).await, name_on(&b).await, "the phone must adopt the desktop's name");
    });
}

// ---------------------------------------------------------------------------
// 17. Messages that predate the outbox hold different random names on each device.
//     When one side's copy arrives, the other must recognise it by the 0011 key
//     (thread, speaker, second, text) even though its own copy already has a name —
//     found on the OnePlus→Pixel echo, 2026-10-01: a 500 from the unique index.
// ---------------------------------------------------------------------------
#[test]
fn b17_pre_outbox_copies_of_a_message_converge_instead_of_failing() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let uid = a.user_id();
        let when = chrono::Utc::now() - chrono::Duration::minutes(5);
        for p in [&a, &b] {
            sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&p.pool).await.unwrap();
            sqlx::query("INSERT INTO conversation_sessions (session_id, user_id, title, started_at, last_active) VALUES ('old-thread', ?, 'Old', ?, ?)")
                .bind(&uid).bind(when).bind(when).execute(&p.pool).await.unwrap();
            sqlx::query("INSERT INTO conversation_messages (session_id, user_id, role, content, created_at) VALUES ('old-thread', ?, 'user', 'Did we lock the shed?', ?)")
                .bind(&uid).bind(when).execute(&p.pool).await.unwrap();
            sqlx::query("DELETE FROM sync_suppress").execute(&p.pool).await.unwrap();
        }
        // The desktop touches its copy, which queues an update carrying the desktop's name.
        sqlx::query("UPDATE conversation_messages SET content = content WHERE session_id = 'old-thread'").execute(&a.pool).await.unwrap();
        a.sync_with(&b).await;
        assert_eq!(thread_counts(&b, "old-thread").await, (1, 1), "the phone must still hold exactly one copy");
        let (na, nb): (String, String) = (
            sqlx::query_scalar("SELECT sync_id FROM conversation_messages WHERE session_id = 'old-thread'").fetch_one(&a.pool).await.unwrap(),
            sqlx::query_scalar("SELECT sync_id FROM conversation_messages WHERE session_id = 'old-thread'").fetch_one(&b.pool).await.unwrap());
        assert_eq!(na, nb, "the phone must adopt the desktop's name for the message");
    });
}

// ---------------------------------------------------------------------------
// 18. Rows that predate the outbox are not in any queue, so a device this one has never
//     drained to must get the live tables on first contact, or they never move (the
//     desktop's 12 extra memories on the first device pass, 2026-10-01).
// ---------------------------------------------------------------------------
#[test]
fn b18_first_contact_sends_what_predates_the_outbox() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        // Pre-outbox rows: written without queuing, after pairing so no pairing push carried them.
        sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&a.pool).await.unwrap();
        a.add_memory("The spare key is with the neighbour").await;
        a.add_memory("Bins go out on Tuesday").await;
        sqlx::query("DELETE FROM sync_suppress").execute(&a.pool).await.unwrap();
        assert_eq!(sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM sync_outbox").fetch_one(&a.pool).await.unwrap(), 0);

        a.sync_with(&b).await;
        assert_eq!(b.memory_contents().await, vec!["Bins go out on Tuesday".to_string(), "The spare key is with the neighbour".to_string()]);
        // Second contact is a normal drain: nothing is sent again.
        let r = a.sync_with(&b).await;
        assert_eq!(r.memories_sent, 0, "a second sync must not re-send the live tables");
    });
}


// ---------------------------------------------------------------------------
// 20. The same key on two devices with different values: the newest wins on both
//     (Matt, 2026-10-01), whichever direction syncs first.
// ---------------------------------------------------------------------------
#[test]
fn b20_the_newest_value_of_a_key_wins_on_both_devices() {
    let _env = crate::chat_harness_tests::hold_env();
    let data = EnvDir::new();
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let older = chrono::Utc::now() - chrono::Duration::hours(2);
        let newer = chrono::Utc::now() - chrono::Duration::minutes(5);
        sqlx::query("INSERT INTO sync_secrets (name, value, updated_at) VALUES ('XAI_API_KEY', 'xai-old-desktop', ?)")
            .bind(older).execute(&a.pool).await.unwrap();
        sqlx::query("INSERT INTO sync_secrets (name, value, updated_at) VALUES ('XAI_API_KEY', 'xai-new-phone', ?)")
            .bind(newer).execute(&b.pool).await.unwrap();

        // The desktop syncs first: its older value must not overwrite the phone's.
        a.sync_with(&b).await;
        for (p, who) in [(&a, "desktop"), (&b, "phone")] {
            let v: String = sqlx::query_scalar("SELECT value FROM sync_secrets WHERE name = 'XAI_API_KEY'").fetch_one(&p.pool).await.unwrap();
            assert_eq!(v, "xai-new-phone", "{} must end with the newest value", who);
        }
        assert!(data.env_file().contains("XAI_API_KEY=xai-new-phone"), ".env holds: {:?}", data.env_file());
        std::env::remove_var("XAI_API_KEY");
    });
}

// ---------------------------------------------------------------------------
// 21. A sync message is addressed to a device id, and a device that is not that device
//     refuses it — however it was reached (KI-053: an address is where a device was
//     last seen, not who it is).
// ---------------------------------------------------------------------------
#[test]
fn b21_a_message_addressed_to_another_device_is_refused() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let client = a.svc.get_http_client().await;
        let url = format!("https://127.0.0.1:{}/api/zynksync/outbox/pull", b.port);
        let wrong = client.post(&url).header("x-target-device-id", a.device_id()).json(&serde_json::json!({})).send().await.expect("request");
        assert_eq!(wrong.status().as_u16(), 421, "the phone must refuse a message addressed to the desktop");
        let right = client.post(&url).header("x-target-device-id", b.device_id()).json(&serde_json::json!({})).send().await.expect("request");
        assert!(right.status().is_success(), "a message addressed to the phone is served: {}", right.status());
    });
}

// ---------------------------------------------------------------------------
// 22. A relationship between two memories made on the desktop is on the phone, between
//     the same two memories; deleting one of the memories removes the link everywhere.
//     The old receive path carried links inside each memory; since 0016 they are rows of
//     their own, named by the two memories' sync_ids.
// ---------------------------------------------------------------------------
#[test]
fn b22_a_memory_link_travels_and_dies_with_its_memory() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let dog = a.add_memory("Max is my dog").await;
        let park = a.add_memory("Max likes the park").await;
        crate::memory::create_memory_link(&a.pool, dog, park, "elaborates", 0.9, Some("same dog"), "test").await.expect("link");
        a.sync_with(&b).await;

        let on_phone: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT s.content, t.content, l.relation_type FROM memory_links l
             JOIN memories s ON s.id = l.source_memory_id JOIN memories t ON t.id = l.target_memory_id")
            .fetch_all(&b.pool).await.unwrap();
        assert_eq!(on_phone, vec![("Max is my dog".to_string(), "Max likes the park".to_string(), "elaborates".to_string())], "the link did not arrive between the right memories");

        a.svc.record_tombstones(&[crate::sync_outbox::content_hash("Max likes the park")]).await.unwrap();
        sqlx::query("DELETE FROM memories WHERE id = ?").bind(park).execute(&a.pool).await.unwrap();
        a.sync_with(&b).await;
        let links_left: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_links").fetch_one(&b.pool).await.unwrap();
        assert_eq!(links_left, 0, "the link outlived its memory on the phone");
        assert_eq!(b.memory_contents().await, vec!["Max is my dog".to_string()]);
    });
}

// ---------------------------------------------------------------------------
// 23. A phone whose clock runs ahead recorded a key; the desktop then saves a new value
//     while its own clock still reads earlier. The desktop's save must win everywhere:
//     it followed the phone's, whatever the clocks say (KI-081).
// ---------------------------------------------------------------------------
#[test]
fn b23_a_later_save_wins_even_when_the_other_clock_runs_ahead() {
    let _env = crate::chat_harness_tests::hold_env();
    let data = EnvDir::new();
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        // The phone's clock is ten minutes ahead: its record carries a future time.
        let ahead = chrono::Utc::now() + chrono::Duration::minutes(10);
        sqlx::query("INSERT INTO sync_secrets (name, value, updated_at, revision) VALUES ('MISTRAL_API_KEY', 'mistral-from-phone', ?, 1)")
            .bind(ahead).execute(&b.pool).await.unwrap();
        b.sync_with(&a).await;
        let on_desktop: String = sqlx::query_scalar("SELECT value FROM sync_secrets WHERE name = 'MISTRAL_API_KEY'").fetch_one(&a.pool).await.unwrap();
        assert_eq!(on_desktop, "mistral-from-phone");

        // The desktop saves a new value "now" — earlier than the phone's time by the wall
        // clock. As Settings does: the environment first, then the record (the drain
        // re-records any key whose environment value differs from its row).
        crate::commands::models::apply_env_key("MISTRAL_API_KEY", "mistral-saved-on-desktop").unwrap();
        crate::sync_outbox::record_secret(&a.pool, "MISTRAL_API_KEY", "mistral-saved-on-desktop").await.unwrap();
        a.sync_with(&b).await;
        b.sync_with(&a).await;
        for (p, who) in [(&a, "desktop"), (&b, "phone")] {
            let v: String = sqlx::query_scalar("SELECT value FROM sync_secrets WHERE name = 'MISTRAL_API_KEY'").fetch_one(&p.pool).await.unwrap();
            assert_eq!(v, "mistral-saved-on-desktop", "{}: the later save lost to a clock that runs ahead", who);
        }
        std::env::remove_var("MISTRAL_API_KEY");
        let _ = data;
    });
}

#[test]
fn b24_a_change_reaches_the_peer_within_seconds_not_on_the_next_tick() {
    rt_test(async {
        // The harness interval is an hour: anything that arrives here arrives
        // because the loop woke on the change, not on the timer.
        let a = Peer::spawn("a-kick").await;
        let b = Peer::spawn("b-kick").await;
        b.pair_with(&a).await;
        let a_loop = tokio::spawn(a.svc.clone().start_auto_sync());
        let b_loop = tokio::spawn(b.svc.clone().start_auto_sync());
        // The first tick is immediate; let it pass before making the change.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        a.add_memory("the kicked memory").await;
        let started = std::time::Instant::now();
        let mut arrived = false;
        while started.elapsed() < std::time::Duration::from_secs(15) {
            if b.memory_contents().await.iter().any(|c| c == "the kicked memory") { arrived = true; break; }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        a.svc.stop_auto_sync().await;
        b.svc.stop_auto_sync().await;
        a_loop.abort(); b_loop.abort();
        assert!(arrived, "memory did not reach b within 15 s (timer is 3600 s)");
        assert!(started.elapsed() < std::time::Duration::from_secs(12), "took {:?}", started.elapsed());
    });
}

// ---------------------------------------------------------------------------
// 25. A row written while a first contact is still slicing the live tables must
//     still arrive. The slices walk sessions first, so a thread started after the
//     first slice is behind the offset and only the queue can carry it — and the
//     queue only does if the cursor set at the end points to where the queue stood
//     when the send began. Two laptop messages fell in this hole on 2026-10-06.
// ---------------------------------------------------------------------------
#[test]
fn b25_a_thread_started_during_a_first_contact_still_arrives() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let cap = crate::sync_outbox::OUTBOX_BATCH_ROWS as usize;
        // Pre-outbox memories: more than one slice, so the first contact spans two syncs.
        sqlx::query("INSERT INTO sync_suppress (flag) VALUES (1)").execute(&a.pool).await.unwrap();
        for i in 0..(cap + 20) { a.add_memory(&format!("old memory {}", i)).await; }
        sqlx::query("DELETE FROM sync_suppress").execute(&a.pool).await.unwrap();

        a.sync_with(&b).await; // slice 1: all sessions (none) + the first `cap` memories
        // Written during the send: a thread (sessions were already passed) and a memory.
        let uid = a.user_id();
        crate::conversation_history::log_exchange(&a.pool, "thread-25", &uid, "written mid-send", "noted", "anthropic", "guardian", true, "typed").await.unwrap();
        a.add_memory("memory written mid-send").await;
        a.sync_with(&b).await; // slice 2: the rest of the live tables; the full send ends here
        a.sync_with(&b).await; // the queue takes over: must carry what was written during the send

        let (sessions, messages) = thread_counts(&b, "thread-25").await;
        assert_eq!((sessions, messages), (1, 2), "the thread started during the first contact never arrived");
        let contents = b.memory_contents().await;
        assert!(contents.iter().any(|c| c == "memory written mid-send"), "the memory written during the first contact never arrived");
        assert_eq!(contents.iter().filter(|c| c.starts_with("old memory")).count(), cap + 20);
    });
}
