-- 0015: how far this device has heard from each peer (2026-10-01, KI-050)
-- The sender keeps a cursor per peer (0013). A device restored from backup onto a fresh
-- install has the old identity but an empty database, and the sender's cursor for it
-- still says "already sent". So the receiver records the highest batch it has applied
-- from each sender and reports it on every exchange; a sender that hears a lower number
-- than its own cursor starts that peer over from the live tables.
CREATE TABLE IF NOT EXISTS sync_inbox_cursor (
    peer_device_id TEXT PRIMARY KEY,
    through        INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT
);
