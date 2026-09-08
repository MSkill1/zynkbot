-- 0011: memory system pass (2026-09-07 analyst findings)
--
-- 1. One timestamp format on memories (history got this in 0010).
-- 2. Exact-duplicate conversation rows removed and prevented (KI-028 symptom;
--    the sync dedupe compared timestamps in two formats and never matched).
-- 3. New fields the extractor now fills: tags (JSON array), a real event_date is
--    already a column; sentiment columns start being used.
-- 4. input_mode on messages, extraction outcome on sessions.
-- 5. link_count kept correct by triggers.
-- 6. memory_entities: one row per named thing per memory, canonical spelling.

-- 1. timestamps -------------------------------------------------------------
UPDATE memories SET created_at = replace(created_at, ' ', 'T') || '+00:00'
  WHERE created_at NOT LIKE '%T%';
UPDATE memories SET updated_at = replace(updated_at, ' ', 'T') || '+00:00'
  WHERE updated_at NOT LIKE '%T%';
UPDATE memory_links SET created_at = replace(created_at, ' ', 'T') || '+00:00'
  WHERE created_at NOT LIKE '%T%';
-- Jan-1 placeholders were never real event dates.
UPDATE memories SET event_date = NULL
  WHERE event_date IS NOT NULL AND substr(event_date, 6, 5) = '01-01' AND substr(event_date, 12) IN ('', '00:00:00', '00:00:00+00:00', 'T00:00:00+00:00');

-- 2. duplicate conversation rows --------------------------------------------
DELETE FROM conversation_messages
  WHERE id NOT IN (
    SELECT MIN(id) FROM conversation_messages
    GROUP BY session_id, role, created_at, content
  );
UPDATE conversation_sessions SET message_count = (
  SELECT COUNT(*) FROM conversation_messages m WHERE m.session_id = conversation_sessions.session_id
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_messages_unique
  ON conversation_messages(session_id, role, created_at, content);

-- 3. new memory fields --------------------------------------------------------
ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';

-- 4. provenance of messages / outcome of extraction ---------------------------
ALTER TABLE conversation_messages ADD COLUMN input_mode TEXT NOT NULL DEFAULT 'typed'
  CHECK (input_mode IN ('typed', 'dictated', 'hands_free', 'system'));
ALTER TABLE conversation_sessions ADD COLUMN last_extraction TEXT;
ALTER TABLE conversation_sessions ADD COLUMN last_extraction_at TEXT;

-- 5. link_count maintained by the database ----------------------------------
UPDATE memories SET link_count = (
  SELECT COUNT(*) FROM memory_links l
  WHERE l.source_memory_id = memories.id OR l.target_memory_id = memories.id
);
CREATE TRIGGER IF NOT EXISTS trg_memory_links_ai AFTER INSERT ON memory_links BEGIN
  UPDATE memories SET link_count = link_count + 1 WHERE id IN (NEW.source_memory_id, NEW.target_memory_id);
END;
CREATE TRIGGER IF NOT EXISTS trg_memory_links_ad AFTER DELETE ON memory_links BEGIN
  UPDATE memories SET link_count = CASE WHEN link_count > 0 THEN link_count - 1 ELSE 0 END
    WHERE id IN (OLD.source_memory_id, OLD.target_memory_id);
END;

-- 6. entities -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_entities (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id  INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  name       TEXT NOT NULL,          -- as written in the memory
  canonical  TEXT NOT NULL,          -- lower-cased, trimmed; the grouping key
  kind       TEXT NOT NULL DEFAULT 'thing',  -- person | place | org | thing
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')),
  UNIQUE (memory_id, canonical)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_canonical ON memory_entities(canonical);
CREATE INDEX IF NOT EXISTS idx_memory_entities_memory    ON memory_entities(memory_id);

-- No backfill from the old name-finder blob: on real data it produced "use" as a
-- person and split "United States" into two places. Entities start with the
-- memories extracted after this migration, where the decision call names them.
