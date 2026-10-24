-- 0023: the memory job queue (2026-10-24, Matt)
-- After a reply, the memory work (call 2: the facts; call 3: their links) ran as a spawned
-- task per message that stepped aside for the user's next message and gave up after three
-- interruptions. On a phone a message's work takes minutes, so four quick questions lost
-- the first one's facts outright (the dentist, 2026-10-24). Each message now becomes a row
-- here; one worker runs the rows in order when the model is idle, an interruption leaves
-- the row pending, and the rows survive an app restart.
CREATE TABLE IF NOT EXISTS memory_jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL,
    user_id           TEXT NOT NULL,
    message           TEXT NOT NULL,
    backend           TEXT NOT NULL,
    source_type       TEXT NOT NULL DEFAULT 'conversation',
    containment_mode  TEXT NOT NULL DEFAULT 'sovereign',
    explicit_remember INTEGER NOT NULL DEFAULT 0,
    remember_text     TEXT,
    user_name         TEXT,
    recent_turns      TEXT,
    query_entities    TEXT NOT NULL DEFAULT '[]',
    status            TEXT NOT NULL DEFAULT 'pending',   -- pending | running | done | failed
    attempts          INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS memory_jobs_status_idx ON memory_jobs(status, id);
