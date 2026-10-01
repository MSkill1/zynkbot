-- 0014: API keys and the backup key travel through the outbox (2026-10-01, KI-055)
-- Step 3 of the outbox rebuild. Keys used to be pushed only when one was saved or the
-- button was pressed, so a device that paired later got nothing (b02). Each key is a row
-- here, named by its own name (there is one of each per user, as a session is named by
-- its session_id), and the 0013-style triggers queue every change. The newest value per
-- key wins (Matt, 2026-10-01): updated_at is when this device first saw or last changed it.
CREATE TABLE IF NOT EXISTS sync_secrets (
    name       TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    sync_id    TEXT
);

CREATE TRIGGER IF NOT EXISTS trg_sync_secrets_outbox_ai
AFTER INSERT ON sync_secrets
BEGIN
  UPDATE sync_secrets SET sync_id = NEW.name WHERE name = NEW.name AND sync_id IS NULL;
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
    SELECT 'sync_secrets', NEW.name, 'insert'
     WHERE NOT EXISTS (SELECT 1 FROM sync_suppress);
END;

CREATE TRIGGER IF NOT EXISTS trg_sync_secrets_outbox_au
AFTER UPDATE ON sync_secrets
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
  AND NOT (OLD.sync_id IS NULL AND NEW.sync_id IS NOT NULL)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op) VALUES ('sync_secrets', NEW.name, 'update');
END;

CREATE TRIGGER IF NOT EXISTS trg_sync_secrets_outbox_ad
AFTER DELETE ON sync_secrets
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op) VALUES ('sync_secrets', OLD.name, 'delete');
END;
