-- Fix zynklink_pairings unique key: device pair instead of user pair.
-- All devices in a ZynkSync mesh share the same user_id, so UNIQUE(user1_id, user2_id)
-- allowed only one pairing per user pair — preventing a phone from ZynkLink-pairing
-- with a second device once the first pairing occupied the slot.
ALTER TABLE zynklink_pairings RENAME TO zynklink_pairings_old;

CREATE TABLE zynklink_pairings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user1_id    TEXT NOT NULL,
    user2_id    TEXT NOT NULL,
    device1_id  TEXT NOT NULL,
    device2_id  TEXT NOT NULL,
    linked_at   TEXT NOT NULL DEFAULT (datetime('now')),
    is_active   INTEGER NOT NULL DEFAULT 1,
    is_paused   INTEGER NOT NULL DEFAULT 0,
    UNIQUE(device1_id, device2_id)
);

INSERT OR IGNORE INTO zynklink_pairings
    (user1_id, user2_id, device1_id, device2_id, linked_at, is_active, is_paused)
SELECT user1_id, user2_id, device1_id, device2_id, linked_at, is_active, is_paused
FROM zynklink_pairings_old
WHERE device1_id IS NOT NULL AND device2_id IS NOT NULL;

DROP TABLE zynklink_pairings_old;

CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_user1   ON zynklink_pairings(user1_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_user2   ON zynklink_pairings(user2_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_device1 ON zynklink_pairings(device1_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_device2 ON zynklink_pairings(device2_id);
