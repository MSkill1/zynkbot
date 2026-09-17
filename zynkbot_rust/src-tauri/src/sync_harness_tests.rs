//! ZynkSync two-peer harness (2026-09-17).
//!
//! Two (or more) real `ZynkSyncService` instances in one process, each with its own
//! SQLite file in a temp dir, its own TLS certificate and identity, listening on a
//! free loopback port. Peers talk over the real router and the real mTLS client —
//! nothing is mocked — so these tests describe what two devices must end up with,
//! not how the current code gets there. They are the acceptance tests for the sync
//! rebuild (docs/TESTING.md, "Sync behaviours the harness must cover"); the ones
//! that fail on the pre-rebuild code are `#[ignore]`d with their known-issue number.
//!
//! Run: `LD_LIBRARY_PATH=$PWD/lib/vosk cargo test --lib sync_harness -- --nocapture`
//! (add `--include-ignored` to see the current failures).

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
            if uid != self.user_id() { self.svc.set_user_id(uid); }
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
        a.svc.propagate_deletion(id).await.expect("propagate deletion");
        sqlx::query("DELETE FROM memories WHERE id = ?").bind(id).execute(&a.pool).await.unwrap();
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
#[ignore = "KI-060: an edited memory reverts after sync — until the outbox rebuild"]
fn b05_memory_edited_on_one_device_does_not_come_back_old() {
    rt_test(async {
        let a = Peer::spawn("desktop").await;
        let b = Peer::spawn("phone").await;
        b.pair_with(&a).await;
        let id = a.add_memory("My dentist is Dr Alvarez").await;
        a.sync_with(&b).await;

        sqlx::query("UPDATE memories SET content = ?, updated_at = datetime('now') WHERE id = ?")
            .bind("My dentist is Dr Alvarez on Main Street").bind(id).execute(&a.pool).await.unwrap();
        a.svc.propagate_memory_update(id, None, Some("My dentist is Dr Alvarez on Main Street".into()), None).await.expect("propagate update");

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
#[ignore = "KI-028: the receive side keys duplicates on (session, second, role), so two messages of the same role in the same second collapse into one — until the outbox rebuild keys on the message hash"]
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

#[test]
#[ignore = "KI-028 / #12: history deletions do not propagate — until the outbox rebuild"]
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

