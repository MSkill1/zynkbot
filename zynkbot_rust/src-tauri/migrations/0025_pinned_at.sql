-- 0025: pinned threads sync (2026-10-28)
-- A pin was a flag on one device's thread row and never travelled, so every device had its
-- own pinned section. pinned_at is when the pin was last changed, so the newest change
-- wins on every device, the same rule the other synced columns use. Rows that were
-- pinned before this migration get the moment of migration, so they stay pinned and win
-- over a device that has never pinned anything (its pinned_at is NULL).
ALTER TABLE conversation_sessions ADD COLUMN pinned_at TEXT;
UPDATE conversation_sessions SET pinned_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE pinned = 1 AND pinned_at IS NULL;
