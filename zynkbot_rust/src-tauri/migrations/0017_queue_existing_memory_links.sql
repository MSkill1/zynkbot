-- 0017: links that existed before 0016 are queued once (2026-10-02)
-- 0016 named every existing memory link but queued nothing, so a peer that had already
-- had its first-contact send never received them (the OnePlus: 0 links to the desktop's
-- 998, found in section A of the manual pass). One insert row per existing link; the
-- receiver matches a link it already holds by (source, target, relation) and adopts the
-- name, so devices that both hold the old links converge instead of doubling.
INSERT INTO sync_outbox (table_name, row_sync_id, op)
  SELECT 'memory_links', sync_id, 'insert' FROM memory_links
   WHERE sync_id IS NOT NULL
     AND NOT EXISTS (SELECT 1 FROM sync_suppress);
