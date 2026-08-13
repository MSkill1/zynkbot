ALTER TABLE memories ADD COLUMN collection_id TEXT;
ALTER TABLE memories ADD COLUMN memory_placement TEXT NOT NULL DEFAULT 'retrieved'
    CHECK (memory_placement IN ('retrieved', 'pinned'));
ALTER TABLE memories ADD COLUMN external_id TEXT;
ALTER TABLE memories ADD COLUMN temporal_status TEXT;
ALTER TABLE memories ADD COLUMN provenance_json TEXT;

CREATE INDEX IF NOT EXISTS idx_memories_collection
    ON memories(user_id, collection_id);
CREATE INDEX IF NOT EXISTS idx_memories_pinned_collection
    ON memories(user_id, collection_id, memory_placement)
    WHERE memory_placement = 'pinned';
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_collection_external
    ON memories(user_id, collection_id, external_id)
    WHERE collection_id IS NOT NULL AND external_id IS NOT NULL;
