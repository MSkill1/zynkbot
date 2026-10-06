-- Tombstones for deleted conversations (threads), like deleted_memory_hashes for
-- memories. A thread deleted on one device while still open on another came back
-- as soon as the other device saved a reply into it: saving recreates the thread
-- row, and the recreation synced out (2026-10-06, test D13). The local delete and
-- an incoming delete both write a row here; an incoming thread or message for a
-- tombstoned thread is ignored, and saving a reply into one is refused.
CREATE TABLE IF NOT EXISTS deleted_sessions (
    session_id TEXT PRIMARY KEY,
    deleted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
