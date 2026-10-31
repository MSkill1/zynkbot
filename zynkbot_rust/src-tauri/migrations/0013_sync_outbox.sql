-- 0013: the ZynkSync outbox (2026-09-29)
-- Step 1 of the outbox rebuild (docs/ROADMAP.md, "ZynkSync outbox rebuild — the plan").
-- Every write to memories, conversation_sessions and conversation_messages queues one row
-- in sync_outbox, so a feature syncs by writing to the database and nothing else. Replaces
-- the immediate push (misses offline devices) and the whole-table compare (slow, misses
-- edits and deletions): KI-028, KI-030, KI-055, KI-060 and the ghost rows.

-- 1. sync_id: what names a row on every device -------------------------------
-- The old key for a message was (session, second, role), which collapsed two messages
-- written in the same second (KI-028). ALTER TABLE cannot take a non-constant default,
-- so the column is added bare, backfilled here, and set on later inserts by the
-- trg_*_sync_id_ai triggers below.
ALTER TABLE memories              ADD COLUMN sync_id TEXT;
ALTER TABLE conversation_sessions ADD COLUMN sync_id TEXT;
ALTER TABLE conversation_messages ADD COLUMN sync_id TEXT;

UPDATE memories              SET sync_id = lower(hex(randomblob(16))) WHERE sync_id IS NULL;
UPDATE conversation_messages SET sync_id = lower(hex(randomblob(16))) WHERE sync_id IS NULL;
-- A session already has a cross-device name: session_id, a UUID the peers exchange and
-- the messages point at. Reusing it means two devices that already hold the same session
-- arrive at the same sync_id without having to reconcile. Memories and messages have no
-- such key, so theirs is random and pre-existing copies of one row will hold different
-- sync_ids on different devices; the step 2 receiver reconciles those once, by content
-- for a memory and by the 0011 key for a message, adopting the incoming sync_id.
UPDATE conversation_sessions SET sync_id = session_id WHERE sync_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_sync_id      ON memories(sync_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_sessions_sync_id ON conversation_sessions(sync_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_messages_sync_id ON conversation_messages(sync_id);

-- 2. the queue ---------------------------------------------------------------
-- payload is used only by a delete, and only where the row has to be named by something
-- other than its sync_id after it is gone: a deleted memory's tombstone is
-- sha256(content) (zynksync.rs, record_tombstones) and SQLite has no sha256, so the
-- trigger keeps the content and the sender hashes it. Insert and update carry nothing —
-- the sender reads the live row at send time and serialises every column, so a column
-- added later travels without touching sync code.
CREATE TABLE IF NOT EXISTS sync_outbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name  TEXT NOT NULL,
    row_sync_id TEXT NOT NULL,
    op          TEXT NOT NULL CHECK (op IN ('insert', 'update', 'delete')),
    payload     TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_sync_outbox_created ON sync_outbox(created_at);

-- How far each peer has been drained. Advanced only once the peer acknowledges a batch,
-- so a device that is off just has a longer queue waiting. Replaces
-- zynk_conversation_push_state (0012), which covered history only.
CREATE TABLE IF NOT EXISTS sync_outbox_cursor (
    peer_device_id TEXT PRIMARY KEY,
    last_outbox_id INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT
);

-- A row here means "applying what a peer sent, do not queue it back". Set and cleared
-- inside the applying transaction; SQLite serialises writers, so it cannot leak to
-- another writer.
CREATE TABLE IF NOT EXISTS sync_suppress (
    flag INTEGER PRIMARY KEY CHECK (flag = 1)
);

-- 3. the triggers -----------------------------------------------------------
-- SQLite does not promise what order two triggers on the same event run in, and in
-- practice the outbox trigger ran first and found no sync_id to record. So naming the
-- row and queuing it are one trigger per table: statements inside a trigger body do run
-- in order. The insert reads sync_id back from the table rather than from NEW, because
-- when the writer left it NULL the value was set by the line above and NEW still shows
-- NULL. Naming happens even while suppressed — a row a peer sent must still be nameable
-- when this device later changes it — so the suppress test sits in the queuing statement,
-- not in a WHEN clause. An update ignores the naming write itself (NULL to not-NULL), so
-- an insert queues one row and not an insert and an update.

CREATE TRIGGER IF NOT EXISTS trg_memories_outbox_ai
AFTER INSERT ON memories
BEGIN
  UPDATE memories SET sync_id = lower(hex(randomblob(16)))
    WHERE id = NEW.id AND sync_id IS NULL;
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
    SELECT 'memories', m.sync_id, 'insert' FROM memories m
     WHERE m.id = NEW.id AND NOT EXISTS (SELECT 1 FROM sync_suppress);
END;

CREATE TRIGGER IF NOT EXISTS trg_memories_outbox_au
AFTER UPDATE ON memories
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
  AND NOT (OLD.sync_id IS NULL AND NEW.sync_id IS NOT NULL)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
  VALUES ('memories', NEW.sync_id, 'update');
END;

CREATE TRIGGER IF NOT EXISTS trg_memories_outbox_ad
AFTER DELETE ON memories
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op, payload)
  VALUES ('memories', OLD.sync_id, 'delete', OLD.content);
END;

-- A session already has a cross-device name, so it is its own sync_id.
CREATE TRIGGER IF NOT EXISTS trg_conv_sessions_outbox_ai
AFTER INSERT ON conversation_sessions
BEGIN
  UPDATE conversation_sessions SET sync_id = NEW.session_id
    WHERE id = NEW.id AND sync_id IS NULL;
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
    SELECT 'conversation_sessions', s.sync_id, 'insert' FROM conversation_sessions s
     WHERE s.id = NEW.id AND NOT EXISTS (SELECT 1 FROM sync_suppress);
END;

CREATE TRIGGER IF NOT EXISTS trg_conv_sessions_outbox_au
AFTER UPDATE ON conversation_sessions
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
  AND NOT (OLD.sync_id IS NULL AND NEW.sync_id IS NOT NULL)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
  VALUES ('conversation_sessions', NEW.sync_id, 'update');
END;

-- A session carries no content of its own, so a delete needs no payload; the sync_id
-- names it. Deleting a session cascades to its messages (0001), which queues a delete
-- for each one through the trigger below.
CREATE TRIGGER IF NOT EXISTS trg_conv_sessions_outbox_ad
AFTER DELETE ON conversation_sessions
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
  VALUES ('conversation_sessions', OLD.sync_id, 'delete');
END;

CREATE TRIGGER IF NOT EXISTS trg_conv_messages_outbox_ai
AFTER INSERT ON conversation_messages
BEGIN
  UPDATE conversation_messages SET sync_id = lower(hex(randomblob(16)))
    WHERE id = NEW.id AND sync_id IS NULL;
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
    SELECT 'conversation_messages', m.sync_id, 'insert' FROM conversation_messages m
     WHERE m.id = NEW.id AND NOT EXISTS (SELECT 1 FROM sync_suppress);
END;

CREATE TRIGGER IF NOT EXISTS trg_conv_messages_outbox_au
AFTER UPDATE ON conversation_messages
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
  AND NOT (OLD.sync_id IS NULL AND NEW.sync_id IS NOT NULL)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op)
  VALUES ('conversation_messages', NEW.sync_id, 'update');
END;

CREATE TRIGGER IF NOT EXISTS trg_conv_messages_outbox_ad
AFTER DELETE ON conversation_messages
WHEN NOT EXISTS (SELECT 1 FROM sync_suppress)
BEGIN
  INSERT INTO sync_outbox (table_name, row_sync_id, op, payload)
  VALUES ('conversation_messages', OLD.sync_id, 'delete', OLD.content);
END;
