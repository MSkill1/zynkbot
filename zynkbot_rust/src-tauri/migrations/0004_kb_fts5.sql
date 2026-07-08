-- FTS5 full-text search for Knowledge Base chunks
-- Enables keyword-based fallback when semantic similarity is low
-- (e.g. query "other names" finds document saying "also known as" via exact-word match)

CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
    content,
    content='kb_chunks',
    content_rowid='id'
);

-- Keep FTS5 index in sync with kb_chunks
CREATE TRIGGER IF NOT EXISTS kb_chunks_ai AFTER INSERT ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS kb_chunks_ad AFTER DELETE ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS kb_chunks_au AFTER UPDATE ON kb_chunks BEGIN
    INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO kb_chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Backfill any chunks that were indexed before this migration
INSERT INTO kb_chunks_fts(rowid, content) SELECT id, content FROM kb_chunks;
