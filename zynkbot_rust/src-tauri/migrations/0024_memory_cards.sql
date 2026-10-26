-- 0024: memory cards that keep (2026-10-26, Matt)
-- A card ("Is this an update?" / "These two disagree") was only a live event to the window.
-- On the phone the pipeline raises it 40-80 s after the reply, by which time the user has
-- often left the app, and a paused WebView drops the event: the card was raised and lost
-- (Pixel, 12:28:27). Each card is now a row, shown on every app open and return until
-- answered. payload_json is the event payload as the window expects it.
CREATE TABLE IF NOT EXISTS memory_cards (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                TEXT NOT NULL,
    session_id             TEXT,
    stored_memory_id       INTEGER,
    conflicting_memory_id  INTEGER,
    payload_json           TEXT NOT NULL,
    raised_at              TEXT NOT NULL DEFAULT (datetime('now')),
    answered_at            TEXT,
    decision               TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_cards_pending ON memory_cards(user_id, answered_at);
