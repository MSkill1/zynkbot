-- App-wide key-value settings (single-user, so no user_id needed)
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Pause flag for ZynkLink pairings (local-only, does not sync to other device)
ALTER TABLE zynklink_pairings ADD COLUMN is_paused INTEGER NOT NULL DEFAULT 0;
