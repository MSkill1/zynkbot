-- 0016: memory links travel through the outbox (2026-10-02)
-- Step 5 of the sync rebuild. The relationship graph between memories was carried only
-- by the old receive path, inside each memory; with that path gone, links need their own
-- row name and triggers, like the other synced tables (0013). A link is sent naming its
-- two memories by their sync_id, so it means the same thing on every device.
ALTER TABLE memory_links ADD COLUMN sync_id TEXT;
UPDATE memory_links SET sync_id = lower(hex(randomblob(16))) WHERE sync_id IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_links_sync_id ON memory_links(sync_id);

CREATE TRIGGER IF NOT EXISTS trg_memory_links_outbox_ai
AFTER INSERT ON memory_links
BEGIN
  UPDATE memory_links SET sync_id = lower(hex(randomblob(16))) WHERE id = NEW.id AND sync_id IS NULL;
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
    SELECT 'memory_links', l.sync_id, 'insert' FROM memory_links l
     WHERE l.id = NEW.id AND NOT EXISTS (SELECT 1 FROM sync_suppress);
END;

CREATE TRIGGER IF NOT EXISTS trg_memory_links_outbox_au
AFTER UPDATE ON memory_links
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
  AND NOT (OLD.sync_id IS NULL AND NEW.sync_id IS NOT NULL)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op) VALUES ('memory_links', NEW.sync_id, 'update');
END;

CREATE TRIGGER IF NOT EXISTS trg_memory_links_outbox_ad
AFTER DELETE ON memory_links
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op) VALUES ('memory_links', OLD.sync_id, 'delete');
END;
