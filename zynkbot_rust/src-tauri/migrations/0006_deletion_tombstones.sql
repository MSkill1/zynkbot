-- Tombstones for deleted memories: a hash recorded here means this memory
-- was explicitly deleted and should never be resurrected by sync.
CREATE TABLE IF NOT EXISTS deleted_memory_hashes (
    content_hash TEXT PRIMARY KEY,
    deleted_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
