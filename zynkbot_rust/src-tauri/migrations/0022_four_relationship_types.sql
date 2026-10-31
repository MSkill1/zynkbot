-- 0022: four relationship types (2026-10-23, Matt)
-- A relationship type earns its place when the app does something different because of
-- it. "contradicts" asks the user; "updates" (new) replaces an older state of the same
-- thing and keeps the old memory as history (memories.temporal_status = 'superseded');
-- "related" pulls the linked memory into recall; "resolves" is the user's own explanation
-- of a resolved conflict. "supports", "elaborates" and "caused_by" behaved identically to
-- each other in the app and become "related"; "reminds_of" (the loose type, and the one
-- that did harm in the memory bench), "quotes" and "mentions" are dropped. The table's
-- CHECK names the allowed types and SQLite cannot alter a CHECK, so the table is rebuilt.
-- Sync is suppressed while it happens: every device runs this same migration.
INSERT OR IGNORE INTO sync_suppress (flag) VALUES (1);

CREATE TABLE memory_links_v2 (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id  INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id  INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    relation_type     TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 0.5,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    notes             TEXT,
    created_by        TEXT NOT NULL DEFAULT 'system',
    sync_id           TEXT,
    CONSTRAINT memory_links_no_self       CHECK (source_memory_id <> target_memory_id),
    CONSTRAINT memory_links_confidence    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT memory_links_relation_type CHECK (
        relation_type IN ('contradicts', 'updates', 'related', 'resolves')
    ),
    UNIQUE (source_memory_id, target_memory_id, relation_type)
);

-- Strongest link of a pair wins when two old types collapse into "related".
INSERT OR IGNORE INTO memory_links_v2
    (id, source_memory_id, target_memory_id, relation_type, confidence, created_at, notes, created_by, sync_id)
  SELECT id, source_memory_id, target_memory_id,
         CASE relation_type
           WHEN 'supports'   THEN 'related'
           WHEN 'elaborates' THEN 'related'
           WHEN 'caused_by'  THEN 'related'
           ELSE relation_type
         END,
         confidence, created_at, notes, created_by, sync_id
    FROM memory_links
   WHERE relation_type IN ('supports', 'elaborates', 'caused_by', 'contradicts', 'resolves')
   ORDER BY confidence DESC, id;

DROP VIEW IF EXISTS memory_relationships;
DROP TABLE memory_links;
ALTER TABLE memory_links_v2 RENAME TO memory_links;

CREATE INDEX IF NOT EXISTS idx_memory_links_source     ON memory_links(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_target     ON memory_links(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_type       ON memory_links(relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_links_confidence ON memory_links(confidence DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_links_sync_id ON memory_links(sync_id);

-- link_count upkeep (0011)
CREATE TRIGGER IF NOT EXISTS trg_memory_links_ai AFTER INSERT ON memory_links BEGIN
  UPDATE memories SET link_count = link_count + 1 WHERE id IN (NEW.source_memory_id, NEW.target_memory_id);
END;
CREATE TRIGGER IF NOT EXISTS trg_memory_links_ad AFTER DELETE ON memory_links BEGIN
  UPDATE memories SET link_count = CASE WHEN link_count > 0 THEN link_count - 1 ELSE 0 END
    WHERE id IN (OLD.source_memory_id, OLD.target_memory_id);
END;

-- outbox (0016)
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

CREATE VIEW IF NOT EXISTS memory_relationships AS
  SELECT source_memory_id AS memory_id,
         target_memory_id AS related_memory_id,
         relation_type, confidence, created_at, notes, created_by,
         'outgoing' AS direction
  FROM memory_links
  UNION ALL
  SELECT target_memory_id AS memory_id,
         source_memory_id AS related_memory_id,
         CASE relation_type
           WHEN 'contradicts' THEN 'contradicted_by'
           WHEN 'updates'     THEN 'updated_by'
           WHEN 'resolves'    THEN 'resolved_by'
           ELSE relation_type
         END AS relation_type,
         confidence, created_at, notes, created_by,
         'incoming' AS direction
  FROM memory_links;

UPDATE memories SET link_count = (
  SELECT COUNT(*) FROM memory_links l
  WHERE l.source_memory_id = memories.id OR l.target_memory_id = memories.id
);

DELETE FROM sync_suppress;
