-- Zynkbot SQLite schema
-- Translated from PostgreSQL/pgvector schema
-- Types: SERIAL→INTEGER, TIMESTAMPTZ→TEXT, VECTOR→BLOB, JSONB→TEXT, UUID→TEXT, BOOLEAN→INTEGER
-- Triggers for link_count maintenance are handled in Rust (Phase 3)
-- IVFFlat/GIN/FTS indexes replaced in Phase 3 (sqlite-vec / FTS5)

-- ============================================================
-- CORE MEMORY
-- ============================================================

CREATE TABLE IF NOT EXISTS memories (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    title             TEXT,
    content           TEXT NOT NULL,
    source_type       TEXT,
    session_id        TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    parent_scroll_id  INTEGER,
    chunk_index       INTEGER,
    user_id           TEXT,
    namespace         TEXT NOT NULL DEFAULT 'personal',
    is_syncable       INTEGER NOT NULL DEFAULT 1,
    is_shareable      INTEGER NOT NULL DEFAULT 0,
    embedding         BLOB,
    link_count        INTEGER NOT NULL DEFAULT 0,
    is_ephemeral      INTEGER NOT NULL DEFAULT 0,
    expires_at        TEXT,
    sentiment_score   REAL NOT NULL DEFAULT 0.0,
    sentiment_label   TEXT NOT NULL DEFAULT 'neutral',
    event_type        TEXT,
    event_date        TEXT,
    entities_detected TEXT NOT NULL DEFAULT '[]',
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    original_text     TEXT
);

CREATE INDEX IF NOT EXISTS memories_namespace_idx       ON memories(namespace);
CREATE INDEX IF NOT EXISTS memories_user_namespace_idx  ON memories(user_id, namespace);
CREATE INDEX IF NOT EXISTS idx_memories_ephemeral        ON memories(is_ephemeral) WHERE is_ephemeral = 1;
CREATE INDEX IF NOT EXISTS idx_memories_expires_at       ON memories(expires_at)   WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_event_date       ON memories(event_date)   WHERE event_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_event_type       ON memories(event_type)   WHERE event_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_sentiment        ON memories(sentiment_label);
CREATE INDEX IF NOT EXISTS memories_shareable_idx        ON memories(is_shareable) WHERE is_shareable = 1;

-- Vector index on memories.embedding is created by sqlite-vec in Phase 3

CREATE TABLE IF NOT EXISTS memory_links (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_memory_id  INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id  INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    relation_type     TEXT NOT NULL,
    confidence        REAL NOT NULL DEFAULT 0.5,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    notes             TEXT,
    created_by        TEXT NOT NULL DEFAULT 'system',
    CONSTRAINT memory_links_no_self       CHECK (source_memory_id <> target_memory_id),
    CONSTRAINT memory_links_confidence    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT memory_links_relation_type CHECK (
        relation_type IN ('supports', 'contradicts', 'elaborates',
                          'reminds_of', 'caused_by', 'quotes', 'resolves')
    ),
    UNIQUE (source_memory_id, target_memory_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_memory_links_source     ON memory_links(source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_target     ON memory_links(target_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_type       ON memory_links(relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_links_confidence ON memory_links(confidence DESC);

-- Bidirectional view of relationships (same logic as PostgreSQL)
CREATE VIEW IF NOT EXISTS memory_relationships AS
  SELECT source_memory_id AS memory_id,
         target_memory_id AS related_memory_id,
         relation_type,
         confidence,
         created_at,
         notes,
         created_by,
         'outgoing' AS direction
  FROM memory_links
  UNION ALL
  SELECT target_memory_id AS memory_id,
         source_memory_id AS related_memory_id,
         CASE relation_type
           WHEN 'supports'    THEN 'supported_by'
           WHEN 'contradicts' THEN 'contradicted_by'
           WHEN 'elaborates'  THEN 'elaborated_by'
           WHEN 'caused_by'   THEN 'causes'
           WHEN 'quotes'      THEN 'quoted_by'
           WHEN 'resolves'    THEN 'resolved_by'
           ELSE relation_type
         END AS relation_type,
         confidence,
         created_at,
         notes,
         created_by,
         'incoming' AS direction
  FROM memory_links;

-- ============================================================
-- KNOWLEDGE BASE
-- ============================================================

CREATE TABLE IF NOT EXISTS kb_documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL,
    file_path      TEXT NOT NULL,
    file_name      TEXT NOT NULL,
    file_size      INTEGER NOT NULL,
    last_modified  TEXT NOT NULL,
    indexed_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    chunk_count    INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'indexing',
    error_message  TEXT,
    CONSTRAINT kb_documents_status CHECK (
        status IN ('indexed', 'indexing', 'needs_reindex', 'error')
    ),
    UNIQUE(user_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_kb_documents_user   ON kb_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_kb_documents_status ON kb_documents(status);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   INTEGER REFERENCES kb_documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    embedding     BLOB,
    token_count   INTEGER NOT NULL,
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_document ON kb_chunks(document_id);

-- Vector index on kb_chunks.embedding is created by sqlite-vec in Phase 3

-- ============================================================
-- ZYNKSYNC — Cross-device memory sync
-- ============================================================

CREATE TABLE IF NOT EXISTS zynk_devices (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id                TEXT NOT NULL UNIQUE,
    device_name              TEXT NOT NULL,
    device_ip                TEXT,
    device_platform          TEXT,
    pairing_code             TEXT,
    pairing_code_expires_at  TEXT,
    is_paired                INTEGER NOT NULL DEFAULT 0,
    last_seen_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    created_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    owner_user_id            TEXT,
    port                     INTEGER NOT NULL DEFAULT 57963
);

CREATE INDEX IF NOT EXISTS zynk_devices_device_id_idx      ON zynk_devices(device_id);
CREATE INDEX IF NOT EXISTS zynk_devices_pairing_code_idx   ON zynk_devices(pairing_code) WHERE pairing_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_devices_owner_user_id        ON zynk_devices(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_zynk_devices_port            ON zynk_devices(port);

CREATE TABLE IF NOT EXISTS zynk_device_pairings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    device_a_id      TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    device_b_id      TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    paired_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_sync_a_to_b TEXT,
    last_sync_b_to_a TEXT,
    is_active        INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT zynk_device_pairings_order CHECK (device_a_id < device_b_id),
    UNIQUE(device_a_id, device_b_id)
);

CREATE INDEX IF NOT EXISTS zynk_device_pairings_device_a_idx ON zynk_device_pairings(device_a_id);
CREATE INDEX IF NOT EXISTS zynk_device_pairings_device_b_idx ON zynk_device_pairings(device_b_id);

CREATE TABLE IF NOT EXISTS zynk_sync_state (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_device_id  TEXT NOT NULL,
    target_device_id  TEXT NOT NULL,
    memory_id         INTEGER NOT NULL,
    synced_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    memory_updated_at TEXT,
    UNIQUE(source_device_id, target_device_id, memory_id)
);

CREATE INDEX IF NOT EXISTS zynk_sync_state_source_idx ON zynk_sync_state(source_device_id);
CREATE INDEX IF NOT EXISTS zynk_sync_state_target_idx ON zynk_sync_state(target_device_id);
CREATE INDEX IF NOT EXISTS zynk_sync_state_memory_idx ON zynk_sync_state(memory_id);

CREATE TABLE IF NOT EXISTS user_sync_codes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    code       TEXT NOT NULL UNIQUE,
    user_id    TEXT NOT NULL,
    device_id  TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at TEXT NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0,
    used_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_user_sync_codes_code       ON user_sync_codes(code) WHERE used = 0;
CREATE INDEX IF NOT EXISTS idx_user_sync_codes_expires_at ON user_sync_codes(expires_at);

-- ============================================================
-- ZYNKLINK — Cross-device file sharing
-- ============================================================

CREATE TABLE IF NOT EXISTS zynk_linked_directories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id    TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    local_path   TEXT NOT NULL,
    share_name   TEXT NOT NULL,
    description  TEXT,
    is_readable  INTEGER NOT NULL DEFAULT 1,
    is_writable  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(device_id, local_path)
);

CREATE INDEX IF NOT EXISTS idx_shared_dirs_device ON zynk_linked_directories(device_id);

CREATE TABLE IF NOT EXISTS zynk_file_manifest (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    shared_directory_id  INTEGER REFERENCES zynk_linked_directories(id) ON DELETE CASCADE,
    relative_path        TEXT NOT NULL,
    file_size            INTEGER NOT NULL,
    file_hash            TEXT,
    mime_type            TEXT,
    last_modified        TEXT,
    indexed_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(shared_directory_id, relative_path)
);

-- Legacy manifest table (same structure, different FK column name)
CREATE TABLE IF NOT EXISTS zynk_link_manifest (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    linked_directory_id  INTEGER REFERENCES zynk_linked_directories(id) ON DELETE CASCADE,
    relative_path        TEXT NOT NULL,
    file_size            INTEGER NOT NULL,
    file_hash            TEXT,
    mime_type            TEXT,
    last_modified        TEXT,
    indexed_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(linked_directory_id, relative_path)
);

CREATE INDEX IF NOT EXISTS idx_file_manifest_share ON zynk_link_manifest(linked_directory_id);
CREATE INDEX IF NOT EXISTS idx_file_manifest_hash  ON zynk_link_manifest(file_hash);

CREATE TABLE IF NOT EXISTS zynklink_codes (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    code                  TEXT NOT NULL UNIQUE,
    creator_user_id       TEXT NOT NULL,
    creator_device_id     TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    created_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at            TEXT,
    accepted_by_user_id   TEXT,
    accepted_by_device_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    accepted_at           TEXT,
    is_active             INTEGER NOT NULL DEFAULT 1,
    revoked_at            TEXT,
    revoked_reason        TEXT
);

CREATE INDEX IF NOT EXISTS idx_zynklink_codes_creator  ON zynklink_codes(creator_user_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_codes_accepted ON zynklink_codes(accepted_by_user_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_codes_active   ON zynklink_codes(code) WHERE is_active = 1;

CREATE TABLE IF NOT EXISTS zynklink_pairings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user1_id    TEXT NOT NULL,
    user2_id    TEXT NOT NULL,
    device1_id  TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    device2_id  TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    linked_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    is_active   INTEGER NOT NULL DEFAULT 1,
    UNIQUE(user1_id, user2_id)
);

CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_user1  ON zynklink_pairings(user1_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_user2  ON zynklink_pairings(user2_id);
CREATE INDEX IF NOT EXISTS idx_zynklink_pairings_active ON zynklink_pairings(user1_id, user2_id) WHERE is_active = 1;

-- ============================================================
-- ZCHAT — Device-to-device messaging
-- ============================================================

CREATE TABLE IF NOT EXISTS zchat_messages (
    id              TEXT PRIMARY KEY,
    from_device_id  TEXT NOT NULL,
    to_device_id    TEXT NOT NULL,
    message_text    TEXT NOT NULL,
    sent_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    delivered_at    TEXT,
    read_at         TEXT,
    user_id         TEXT NOT NULL,
    CONSTRAINT message_not_empty CHECK (length(message_text) > 0)
);

CREATE INDEX IF NOT EXISTS idx_zchat_from_device ON zchat_messages(from_device_id, sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_zchat_to_device   ON zchat_messages(to_device_id, sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_zchat_user        ON zchat_messages(user_id);

-- ============================================================
-- CONVERSATION HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS conversation_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL UNIQUE,
    user_id          TEXT NOT NULL,
    title            TEXT,
    started_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_active      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    message_count    INTEGER NOT NULL DEFAULT 0,
    model_backend    TEXT,
    containment_mode TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_sessions_user_active ON conversation_sessions(user_id, last_active DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    user_id          TEXT NOT NULL,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    model_backend    TEXT,
    containment_mode TEXT,
    entry_hash       TEXT,
    prev_hash        TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_messages_session ON conversation_messages(session_id, created_at ASC);

-- FTS5 full-text search on conversation_messages.content (replaces GIN tsvector index)
CREATE VIRTUAL TABLE IF NOT EXISTS conversation_messages_fts USING fts5(
    content,
    session_id UNINDEXED,
    content='conversation_messages',
    content_rowid='id'
);

CREATE TABLE IF NOT EXISTS message_feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id       TEXT NOT NULL,
    session_id       TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    user_id          TEXT NOT NULL,
    rating           INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    model_backend    TEXT,
    containment_mode TEXT,
    UNIQUE (message_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_message_feedback_session ON message_feedback(session_id);
CREATE INDEX IF NOT EXISTS idx_message_feedback_user    ON message_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_message_feedback_rating  ON message_feedback(rating);

-- ============================================================
-- UTILITY
-- ============================================================

-- Placeholder for future TLS certificates (not referenced by application code)
CREATE TABLE IF NOT EXISTS zynk_device_certificates (
    device_id       TEXT PRIMARY KEY,
    certificate_pem TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
