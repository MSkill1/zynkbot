-- Deferred cascade-remove notifications: when a peer is offline during a device
-- removal cascade, the notification is stored here and retried on next heartbeat.
CREATE TABLE IF NOT EXISTS zynk_pending_removals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    target_peer_id    TEXT NOT NULL,
    target_peer_ip    TEXT NOT NULL,
    sender_device_id  TEXT NOT NULL,
    cascade_device_id TEXT NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_pending_removals_target ON zynk_pending_removals(target_peer_id);
