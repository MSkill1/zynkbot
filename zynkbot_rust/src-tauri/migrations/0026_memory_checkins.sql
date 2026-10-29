-- 0026: check-in questions (labs/check_in, 2026-10-29)
-- One row per memory Zynkbot has asked about ("A while back you said … Did that happen?"),
-- so a memory is asked about once per device and at most one question goes out a day.
-- The answer is kept for the record; the effect lands on the memory itself
-- (temporal_status 'completed', a later expires_at, or deletion).
CREATE TABLE IF NOT EXISTS memory_checkins (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL,
    asked_at    TEXT NOT NULL DEFAULT (datetime('now')),
    answer      TEXT,
    answered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_checkins_memory ON memory_checkins(memory_id);
