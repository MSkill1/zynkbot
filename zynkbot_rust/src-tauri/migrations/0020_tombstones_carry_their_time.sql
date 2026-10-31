-- 0020: a deletion carries its time, and the deletions already made go out once more (2026-10-14)
-- Test J39 of the sync pass: memories deleted on the desktop came back from a cloud
-- restore and spread to every device. One cause was that a deletion travels by the
-- row's sync_id only, and copies that predate 0013 were named independently on each
-- device, so the desktop's delete missed the OnePlus's copy of the same text. A
-- deletion now also carries sha256(content) and the moment it happened; the receiver
-- removes any copy with that text that was not changed after that moment, whatever
-- it is named. The deletions this device already holds are queued once more, by hash
-- alone (there is no row left to read the text from), so the stray copies elsewhere
-- are caught up. deleted_at: the tombstone's own time; NULL on an ordinary queued
-- delete, where the queue row's created_at is that moment.
ALTER TABLE sync_outbox ADD COLUMN deleted_at TEXT;

INSERT INTO sync_outbox (table_name, row_sync_id, op, payload, deleted_at)
  SELECT 'memories', 'hash:' || content_hash, 'delete', NULL, deleted_at
    FROM deleted_memory_hashes
   WHERE NOT EXISTS (SELECT 1 FROM sync_suppress);
