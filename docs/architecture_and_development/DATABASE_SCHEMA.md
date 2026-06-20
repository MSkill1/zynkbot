# Zynkbot Database Schema

**Database:** `zynkbot.db` (SQLite, embedded — no server process)
**Last Updated:** 2026-05-30
**Schema:** Applied automatically via SQLx migrations in `zynkbot_rust/src-tauri/migrations/`

---

## Overview

Zynkbot uses SQLite for all persistent storage. The database contains **19 tables and 1 view** across five functional areas: core memory, knowledge base, ZynkSync, ZynkLink, ZChat, and conversation history. Vector similarity search is performed in-process by the Rust backend using Candle ML — no database extension required.

---

## Core Tables

### 1. memories

**Purpose:** Core memory storage with vector embeddings for semantic search

**Schema:**
```sql
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT NOT NULL,
    source_type TEXT,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    parent_scroll_id INTEGER,
    chunk_index INTEGER,
    user_id TEXT,
    namespace TEXT DEFAULT 'personal' NOT NULL,
    is_syncable BOOLEAN DEFAULT TRUE,
    is_shareable BOOLEAN DEFAULT FALSE,
    embedding VECTOR(384),
    link_count INTEGER DEFAULT 0,
    is_ephemeral BOOLEAN DEFAULT FALSE,
    expires_at TIMESTAMPTZ,
    sentiment_score REAL DEFAULT 0.0,
    sentiment_label TEXT DEFAULT 'neutral',
    event_type TEXT,
    event_date TIMESTAMPTZ,
    entities_detected JSONB DEFAULT '[]',
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    original_text TEXT
);
```

**Key Columns:**
- `id` - Primary key
- `content` - Memory text content
- `embedding` - 384-dimensional vector (all-MiniLM-L6-v2)
- `namespace` - Organization category (personal/work/family/_zynkbot)
- `is_syncable` - Whether memory syncs across devices
- `is_shareable` - Whether memory can be shared with other users
- `entities_detected` - JSONB array of named entities from BERT NER
- `is_ephemeral` - Auto-delete after expiration (used in HIPAA mode)
- `expires_at` - When memory should be deleted (NULL = never)
- `original_text` - Pre-extraction user text (preserved for audit)

**Indexes:**
- `memories_embedding_idx` - IVFFlat vector index for similarity search
- `idx_memories_entities` - GIN index on entities_detected
- `memories_namespace_idx` - B-tree on namespace
- `memories_user_namespace_idx` - B-tree on (user_id, namespace)
- `idx_memories_ephemeral` - Partial B-tree on is_ephemeral WHERE is_ephemeral = true
- `idx_memories_expires_at` - Partial B-tree on expires_at WHERE expires_at IS NOT NULL
- `idx_memories_event_date` - Partial B-tree on event_date WHERE event_date IS NOT NULL
- `idx_memories_event_type` - Partial B-tree on event_type WHERE event_type IS NOT NULL
- `idx_memories_sentiment` - B-tree on sentiment_label
- `memories_shareable_idx` - Partial B-tree on is_shareable WHERE is_shareable = true

**Triggers:**
- `update_memories_updated_at` - Fires before UPDATE; calls `update_updated_at_column()` to keep updated_at current

---

### 2. memory_links

**Purpose:** Semantic relationships between memories for knowledge graph

**Schema:**
```sql
CREATE TABLE memory_links (
    id SERIAL PRIMARY KEY,
    source_memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) NOT NULL,
    confidence REAL DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    created_by VARCHAR(255) DEFAULT 'system',
    CONSTRAINT memory_links_check CHECK (source_memory_id <> target_memory_id),
    CONSTRAINT memory_links_confidence_check CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT memory_links_relation_type_check CHECK (
        relation_type IN ('supports', 'contradicts', 'elaborates',
                          'reminds_of', 'caused_by', 'quotes', 'resolves')
    ),
    UNIQUE (source_memory_id, target_memory_id, relation_type)
);
```

**Relation Types:**
- `supports` / `supported_by` - One memory supports another
- `contradicts` / `contradicted_by` - Conflicting information
- `elaborates` / `elaborated_by` - Adds detail to another memory
- `reminds_of` - Associative connection
- `caused_by` / `causes` - Causal relationship
- `quotes` / `quoted_by` - One memory quotes another
- `resolves` / `resolved_by` - Conflict resolution

**Indexes:**
- `idx_memory_links_source` - B-tree on source_memory_id
- `idx_memory_links_target` - B-tree on target_memory_id
- `idx_memory_links_type` - B-tree on relation_type
- `idx_memory_links_confidence` - B-tree on confidence DESC

**Triggers:**
- `trigger_update_memory_link_count` - Automatically updates link_count in memories table

---

### 3. memory_relationships (View)

**Note: This is a SQL VIEW, not a table.**

**Purpose:** Bidirectional view of memory relationships — presents each link as both an outgoing and incoming relationship so queries can be written from either memory's perspective without self-joins.

**Schema:**
```sql
CREATE VIEW memory_relationships AS
  -- Outgoing relationships
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
  -- Incoming relationships (inverted)
  SELECT target_memory_id AS memory_id,
         source_memory_id AS related_memory_id,
         CASE relation_type
           WHEN 'supports' THEN 'supported_by'
           WHEN 'contradicts' THEN 'contradicted_by'
           WHEN 'elaborates' THEN 'elaborated_by'
           WHEN 'caused_by' THEN 'causes'
           WHEN 'quotes' THEN 'quoted_by'
           WHEN 'resolves' THEN 'resolved_by'
           ELSE relation_type
         END AS relation_type,
         confidence,
         created_at,
         notes,
         created_by,
         'incoming' AS direction
  FROM memory_links;
```

---

## Knowledge Base Tables

### 4. kb_documents

**Purpose:** Tracks user-uploaded documents for RAG knowledge base

**Schema:**
```sql
CREATE TABLE kb_documents (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    last_modified TIMESTAMPTZ NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'indexing',
    error_message TEXT,
    CONSTRAINT kb_documents_status_check CHECK (
        status IN ('indexed', 'indexing', 'needs_reindex', 'error')
    ),
    UNIQUE(user_id, file_path)
);
```

**Indexes:**
- `idx_kb_documents_user` - B-tree on user_id
- `idx_kb_documents_status` - B-tree on status

---

### 5. kb_chunks

**Purpose:** Document chunks with embeddings for semantic search

**Schema:**
```sql
CREATE TABLE kb_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES kb_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    token_count INTEGER NOT NULL,
    UNIQUE(document_id, chunk_index)
);
```

**Indexes:**
- `idx_kb_chunks_document` - B-tree on document_id
- `kb_chunks_embedding_idx` - IVFFlat vector index for similarity search

---

## ZynkSync Tables (Cross-Device Memory Sync)

### 6. zynk_devices

**Purpose:** Device registry for ZynkSync cross-device pairing

**Schema:**
```sql
CREATE TABLE zynk_devices (
    id SERIAL PRIMARY KEY,
    device_id TEXT UNIQUE NOT NULL,
    device_name TEXT NOT NULL,
    device_ip TEXT,
    device_platform TEXT,
    pairing_code TEXT,
    pairing_code_expires_at TIMESTAMPTZ,
    is_paired BOOLEAN DEFAULT FALSE,
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    owner_user_id TEXT,
    port INTEGER DEFAULT 57963 NOT NULL
);
```

**Key Features:**
- Default port 57963 for all ZynkSync/ZynkLink/ZChat HTTP communication
- `owner_user_id` - Memories sync only between devices with matching owner; file sharing works cross-user
- `pairing_code` - 6-digit code for secure device pairing (10-minute expiry)

**Indexes:**
- `zynk_devices_device_id_idx` - B-tree on device_id
- `zynk_devices_pairing_code_idx` - Partial B-tree on pairing_code WHERE pairing_code IS NOT NULL
- `idx_devices_owner_user_id` - B-tree on owner_user_id
- `idx_zynk_devices_port` - B-tree on port

---

### 7. zynk_device_pairings

**Purpose:** Bi-directional device relationships for ZynkSync authorization

**Schema:**
```sql
CREATE TABLE zynk_device_pairings (
    id SERIAL PRIMARY KEY,
    device_a_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    device_b_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    paired_at TIMESTAMPTZ DEFAULT NOW(),
    last_sync_a_to_b TIMESTAMPTZ,
    last_sync_b_to_a TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    CONSTRAINT zynk_device_pairings_check CHECK (device_a_id < device_b_id),
    UNIQUE(device_a_id, device_b_id)
);
```

**Indexes:**
- `zynk_device_pairings_device_a_idx` - B-tree on device_a_id
- `zynk_device_pairings_device_b_idx` - B-tree on device_b_id

---

### 8. zynk_sync_state

**Purpose:** Track which memories have been synced between devices to avoid redundant transfers

**Schema:**
```sql
CREATE TABLE zynk_sync_state (
    id SERIAL PRIMARY KEY,
    source_device_id TEXT NOT NULL,
    target_device_id TEXT NOT NULL,
    memory_id INTEGER NOT NULL,
    synced_at TIMESTAMPTZ DEFAULT NOW(),
    memory_updated_at TIMESTAMPTZ,
    UNIQUE(source_device_id, target_device_id, memory_id)
);
```

**Indexes:**
- `zynk_sync_state_source_idx` - B-tree on source_device_id
- `zynk_sync_state_target_idx` - B-tree on target_device_id
- `zynk_sync_state_memory_idx` - B-tree on memory_id

---

### 9. user_sync_codes

**Purpose:** Temporary one-time codes for device pairing

**Schema:**
```sql
CREATE TABLE user_sync_codes (
    id SERIAL PRIMARY KEY,
    code VARCHAR(6) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    device_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    used BOOLEAN DEFAULT FALSE,
    used_at TIMESTAMPTZ
);
```

**Features:**
- 6-digit codes (100000-999999)
- 10-minute expiry (default)
- One-time use only
- Automatic cleanup via `cleanup_expired_sync_codes()` function

**Indexes:**
- `idx_user_sync_codes_code` - Partial B-tree on code WHERE NOT used
- `idx_user_sync_codes_expires_at` - B-tree on expires_at

---

## ZynkLink Tables (Cross-Device File Sharing)

### 10. zynk_linked_directories

**Purpose:** Directories shared by each device for cross-device file access. Records are local configuration — they persist when a device is unlinked and survive re-pairing.

**Schema:**
```sql
CREATE TABLE zynk_linked_directories (
    id SERIAL PRIMARY KEY,
    device_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    local_path TEXT NOT NULL,
    share_name TEXT NOT NULL,
    description TEXT,
    is_readable BOOLEAN DEFAULT TRUE,
    is_writable BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(device_id, local_path)
);
```

**Indexes:**
- `idx_shared_dirs_device` - B-tree on device_id

---

### 11. zynk_file_manifest

**Purpose:** Indexed files available in shared directories. This is the table actively queried by the application code (`zynklink.rs`). Uses `shared_directory_id` as the foreign key column name.

**Schema:**
```sql
CREATE TABLE zynk_file_manifest (
    id SERIAL PRIMARY KEY,
    shared_directory_id INTEGER REFERENCES zynk_linked_directories(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    file_hash TEXT,  -- SHA256 for integrity verification
    mime_type TEXT,
    last_modified TIMESTAMPTZ,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(shared_directory_id, relative_path)
);
```

---

### 12. zynk_link_manifest

**Purpose:** Legacy file manifest table from an earlier naming convention. Structurally identical to `zynk_file_manifest` but uses `linked_directory_id` as the foreign key column name. The named indexes for file manifests reside on this table. Application code currently queries `zynk_file_manifest` (table 11); this table is a candidate for cleanup in a future migration.

**Schema:**
```sql
CREATE TABLE zynk_link_manifest (
    id SERIAL PRIMARY KEY,
    linked_directory_id INTEGER REFERENCES zynk_linked_directories(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    file_hash TEXT,  -- SHA256 for integrity verification
    mime_type TEXT,
    last_modified TIMESTAMPTZ,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(linked_directory_id, relative_path)
);
```

**Indexes:**
- `idx_file_manifest_share` - B-tree on linked_directory_id
- `idx_file_manifest_hash` - B-tree on file_hash

---

### 13. zynklink_codes

**Purpose:** Authorization codes for establishing file sharing links between users

**Schema:**
```sql
CREATE TABLE zynklink_codes (
    id SERIAL PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    creator_user_id TEXT NOT NULL,
    creator_device_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    accepted_by_user_id TEXT,
    accepted_by_device_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    accepted_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMPTZ,
    revoked_reason TEXT
);
```

**Features:**
- Code becomes inactive after acceptance (one-time use)
- Supports cross-user file sharing
- Revocation tracked with timestamp and reason

**Indexes:**
- `idx_zynklink_codes_creator` - B-tree on creator_user_id
- `idx_zynklink_codes_accepted` - B-tree on accepted_by_user_id
- `idx_zynklink_codes_active` - Partial B-tree on code WHERE is_active = TRUE

---

### 14. zynklink_pairings

**Purpose:** Active file sharing relationships between users (independent from ZynkSync memory pairing)

**Schema:**
```sql
CREATE TABLE zynklink_pairings (
    id SERIAL PRIMARY KEY,
    user1_id TEXT NOT NULL,
    user2_id TEXT NOT NULL,
    device1_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    device2_id TEXT REFERENCES zynk_devices(device_id) ON DELETE CASCADE,
    linked_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(user1_id, user2_id)
);
```

**Indexes:**
- `idx_zynklink_pairings_user1` - B-tree on user1_id
- `idx_zynklink_pairings_user2` - B-tree on user2_id
- `idx_zynklink_pairings_active` - Partial B-tree on (user1_id, user2_id) WHERE is_active = TRUE

---

## ZChat Tables (Device-to-Device Messaging)

### 15. zchat_messages

**Purpose:** Direct messages between paired devices without cloud storage

**Schema:**
```sql
CREATE TABLE zchat_messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    from_device_id UUID NOT NULL,
    to_device_id UUID NOT NULL,
    message_text TEXT NOT NULL,
    sent_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
    delivered_at TIMESTAMPTZ,
    read_at TIMESTAMPTZ,
    user_id UUID NOT NULL,
    CONSTRAINT message_not_empty CHECK (char_length(message_text) > 0)
);
```

**Features:**
- UUID-based message IDs
- Delivery and read receipts (delivered_at, read_at)
- Local storage only — no cloud sync, no server

**Indexes:**
- `idx_zchat_from_device` - B-tree on (from_device_id, sent_at DESC)
- `idx_zchat_to_device` - B-tree on (to_device_id, sent_at DESC)
- `idx_zchat_conversation` - B-tree on (LEAST(from_device_id, to_device_id), GREATEST(from_device_id, to_device_id), sent_at DESC)
- `idx_zchat_user` - B-tree on user_id

---

## Conversation History Tables

### 16. conversation_sessions

**Purpose:** One row per chat session — tracks metadata, model used, and containment mode

**Schema:**
```sql
CREATE TABLE conversation_sessions (
    id               SERIAL PRIMARY KEY,
    session_id       TEXT NOT NULL UNIQUE,
    user_id          TEXT NOT NULL,
    title            TEXT,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message_count    INTEGER NOT NULL DEFAULT 0,
    model_backend    TEXT,
    containment_mode TEXT
);
```

**Key Columns:**
- `session_id` - UUID identifying the conversation, passed with every message
- `title` - Auto-generated from first 60 characters of the first user message
- `model_backend` - Which LLM backend was used (anthropic, openai, xai, or local model path)
- `containment_mode` - Active containment mode when session was last updated

**Notes:**
- Disabled entirely in HIPAA mode — no records created when HIPAA containment is active
- Sessions are grouped by date in the UI (Today / Yesterday / This Week / Month)
- Full-text search and date-range filtering available via `search_conversations` command

**Indexes:**
- `idx_conv_sessions_user_active` - B-tree on (user_id, last_active DESC)

---

### 17. conversation_messages

**Purpose:** Every user and assistant message in every session, with hash chain columns reserved for tamper evidence

**Schema:**
```sql
CREATE TABLE conversation_messages (
    id               BIGSERIAL PRIMARY KEY,
    session_id       TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    user_id          TEXT NOT NULL,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_backend    TEXT,
    containment_mode TEXT,
    entry_hash       TEXT,
    prev_hash        TEXT
);
```

**Key Columns:**
- `role` - Either `user` or `assistant`
- `entry_hash` / `prev_hash` - Reserved for MemoryVault tamper-evidence chain (v2.0); not populated in v1.0

**Indexes:**
- `idx_conv_messages_session` - B-tree on (session_id, created_at ASC)
- `idx_conv_messages_fts` - GIN full-text search index on content (English tsvector). Note: conversation search currently uses `ILIKE` for partial matching; this index is dormant but retained for a potential future FTS upgrade.

---

### 18. message_feedback

**Purpose:** Thumbs up / down ratings on individual assistant responses, linked to conversation messages

**Schema:**
```sql
CREATE TABLE message_feedback (
    id               BIGSERIAL PRIMARY KEY,
    message_id       TEXT NOT NULL,
    session_id       TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    user_id          TEXT NOT NULL,
    rating           SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_backend    TEXT,
    containment_mode TEXT,
    UNIQUE (message_id, user_id)
);
```

**Key Columns:**
- `rating` - `1` = thumbs up, `-1` = thumbs down
- `message_id` - TEXT storing the stringified `conversation_messages.id` (BIGSERIAL). This is a soft reference by convention, not a database FK — the join is `conversation_messages cm ON cm.id::TEXT = mf.message_id`
- `model_backend` - Denormalized from the message for efficient filtering without a join
- `containment_mode` - Denormalized; allows analysis of rating patterns by mode

**One rating per message per user** enforced by the unique constraint on (message_id, user_id).

**Indexes:**
- `idx_message_feedback_session` - B-tree on session_id
- `idx_message_feedback_user` - B-tree on user_id
- `idx_message_feedback_rating` - B-tree on rating

---

## Utility Tables

### 19. schema_migrations

**Purpose:** Track which database migrations have been applied

**Schema:**
```sql
CREATE TABLE schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### 20. zynk_device_certificates

**Purpose:** Placeholder for TLS certificates for secure device communication. Exists in the schema but is not referenced by any application code in the current version. Reserved for a planned future upgrade to encrypted device-to-device communication (currently plain HTTP over local network).

**Schema:**
```sql
CREATE TABLE zynk_device_certificates (
    device_id UUID PRIMARY KEY,
    certificate_pem TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Database Functions

### cleanup_expired_sync_codes()

**Purpose:** Deletes expired pairing codes from user_sync_codes

```sql
CREATE FUNCTION cleanup_expired_sync_codes() RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM user_sync_codes
    WHERE expires_at < NOW();
END;
$$;
```

### update_memory_link_count()

**Purpose:** Trigger function to maintain link_count in memories table

```sql
CREATE FUNCTION update_memory_link_count() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE memories SET link_count = link_count + 1 WHERE id = NEW.source_memory_id;
    UPDATE memories SET link_count = link_count + 1 WHERE id = NEW.target_memory_id;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE memories SET link_count = link_count - 1 WHERE id = OLD.source_memory_id;
    UPDATE memories SET link_count = link_count - 1 WHERE id = OLD.target_memory_id;
  END IF;
  RETURN NULL;
END;
$$;
```

### update_updated_at_column()

**Purpose:** Trigger function to automatically update updated_at timestamp on row update

```sql
CREATE FUNCTION update_updated_at_column() RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;
```

---

## Performance Considerations

### Vector Search

Zynkbot uses **IVFFlat indexes** for vector similarity search:

```sql
-- memories embedding index
CREATE INDEX memories_embedding_idx
ON memories
USING ivfflat (embedding vector_cosine_ops)
WITH (lists='100');

-- kb_chunks embedding index
CREATE INDEX kb_chunks_embedding_idx
ON kb_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists='100');
```

**IVFFlat Parameters:**
- `lists='100'` - Partitions the vector space into 100 clusters
- Trade-off: Faster search vs. recall accuracy
- Optimal for databases with 10K+ vectors

**Query Performance:**
- Vector similarity search: O(log N) with IVFFlat
- Hybrid search (vector + text): Uses both IVFFlat and GIN indexes

### Index Strategy

**Indexed Columns:**
- All foreign keys (for joins)
- High-cardinality columns (user_id, device_id, namespace)
- Frequently queried columns (created_at, status, is_active)
- JSONB columns (entities_detected) with GIN indexes
- Vector columns with IVFFlat indexes

**Partial Indexes:**
- `idx_memories_ephemeral` - Only indexes ephemeral memories
- `idx_user_sync_codes_code` - Only indexes unused codes
- `memories_shareable_idx` - Only indexes shareable memories
- `idx_memories_expires_at` - Only indexes memories with an expiry set
- `idx_memories_event_date` / `idx_memories_event_type` - Only indexes event-typed memories
- These reduce index size for sparse data

---

## Backup Recommendations

**Critical Tables** (backup regularly):
- `memories` - Core user data
- `memory_links` - Relationship graph
- `kb_documents`, `kb_chunks` - Knowledge base
- `zynk_devices` - Device registry
- `zynk_device_pairings` - Sync configuration
- `conversation_sessions`, `conversation_messages` - Conversation history

**Ephemeral Tables** (can rebuild):
- `zynk_sync_state` - Rebuilds on next sync
- `user_sync_codes` - Temporary codes (expire in 10 minutes)
- `zchat_messages` - Local only; re-delivery not possible after loss

**Backup Command:**
```bash
# Linux
cp ~/.local/share/zynkbot/zynkbot.db ~/zynkbot_backup_$(date +%Y%m%d).db

# Windows
copy "%LOCALAPPDATA%\zynkbot\zynkbot.db" "%USERPROFILE%\zynkbot_backup.db"
```

**Restore Command:**
```bash
# Stop Zynkbot, then replace the database file with the backup copy
cp ~/zynkbot_backup_20260530.db ~/.local/share/zynkbot/zynkbot.db
```

---

## Schema Version

**Database:** SQLite (embedded via sqlx — no extensions required)

**Schema management:** SQLx migrations (`zynkbot_rust/src-tauri/migrations/`) — applied automatically on startup.

---

## Future Enhancements

**Planned:**
- Compression for old memories (archiving memories beyond a configurable age threshold, relevant for HIPAA-regulated deployments)
- Automatic expiration cleanup job for ephemeral memories (currently ephemeral memories are filtered at query time; a background job would purge expired rows to keep the table lean)
- Memory graph visualization support (the relationship graph exists in `memory_links`; this refers to surfacing it as an interactive visual in the UI)
- Cleanup of `zynk_link_manifest` (legacy duplicate of `zynk_file_manifest`; requires a migration to unify on one table)
- Population of `entry_hash` / `prev_hash` in `conversation_messages` for MemoryVault tamper-evidence chain (v2.0)
- TLS certificate infrastructure using `zynk_device_certificates` for encrypted device communication

---

## See Also

- [scripts/db/complete_fresh_install_schema.sql](../scripts/db/complete_fresh_install_schema.sql) - Complete schema definition
- [Networking Features](../NETWORKING_FEATURES.md) - ZynkSync, ZynkLink, and ZChat
