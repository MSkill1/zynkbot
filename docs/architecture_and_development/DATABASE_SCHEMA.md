# Zynkbot Database Schema

*Checked against the code on 2026-11-10 (`v1`, a7599eb). The column tables below were
generated from a database built by running all 28 migrations, not written by hand.*

**Database:** `zynkbot.db`, SQLite, embedded in the app. There is no database server.
**Schema source:** `zynkbot_rust/src-tauri/migrations/` (`0001` to `0028`), applied in
order by sqlx when the app starts. sqlx records what it has applied in its own table,
`_sqlx_migrations`.

---

## Overview

| | Count |
|---|---|
| Tables created by the migrations | 31 |
| Full-text (FTS5) tables | 2 |
| Views | 1 |
| Indexes | 59 |
| Triggers | 20 |

**Where the file is** (`db::get_app_data_dir()`, `src-tauri/src/db.rs`):

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/zynkbot/zynkbot.db` |
| Windows | `%LOCALAPPDATA%\zynkbot\zynkbot.db` |
| Android | `/data/data/ai.containai.zynkbot/files/zynkbot/zynkbot.db` |

**Connection settings** (`db.rs`, set on every connection in the pool): write-ahead log
(`journal_mode=WAL`), `foreign_keys=ON`, `synchronous=NORMAL`, `busy_timeout=15000`.
SQLite is compiled with `SQLITE_SECURE_DELETE`, so a deleted row's bytes are overwritten
with zeros in the file. Because of the write-ahead log, recent rows live in
`zynkbot.db-wal` until a checkpoint; a copy of the database must include that file.

**How values are stored.** SQLite has few types, so:

- **Times** are TEXT. Column defaults write `YYYY-MM-DD HH:MM:SS` in UTC
  (`datetime('now')`); the app itself writes RFC 3339 (`2026-11-10T14:03:22+00:00`).
  Migrations 0010 and 0011 converted the older rows to RFC 3339.
- **True/false** is INTEGER, 0 or 1.
- **Lists and objects** are JSON in a TEXT column (`tags`, `entities_detected`,
  `provenance_json`, `sources_json`, `payload_json`).
- **Embeddings** are a BLOB of 384 little-endian 32-bit floats (all-MiniLM-L6-v2).
  There is no vector index: a search reads the candidate rows and computes cosine
  similarity in Rust (`memory.rs`).
- **`sync_id`** names a row on every device. A trigger fills it when a row is inserted
  (see [Sync triggers](#sync-triggers)).

In the tables below, "required" means NOT NULL, and "→ `t.c`" is a foreign key.

---

## Memory tables

### memories

One row per remembered fact. Since the 2026-10-24 pipeline a memory is a single fact in
the third person, with the user's own message kept beside it in `original_text`.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `title` | TEXT |  | Short title; from the model, or the first 60 characters of a `Remember:` text |
| `content` | TEXT | required | The fact |
| `source_type` | TEXT |  | `conversation`, `hands_free`, `onboarding`, `demo_persona` |
| `session_id` | TEXT |  | The conversation it came from |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `parent_scroll_id` | INTEGER |  | Unused; left from the first design |
| `chunk_index` | INTEGER |  | Unused; left from the first design |
| `user_id` | TEXT |  |  |
| `namespace` | TEXT | required; default `'personal'` | One of 15 topics (`memory_extras::CANONICAL_NAMESPACES`), or `_zynkbot` |
| `is_syncable` | INTEGER | required; default `1` | Not read by the outbox sync; every memory travels |
| `is_shareable` | INTEGER | required; default `0` | Reserved for sharing between users |
| `embedding` | BLOB |  | 384 floats. The declared scheme is title plus fact (`EMBEDDING_SCHEME` in `memory.rs`), used by onboarding, edits, restore, the demo and the one-time re-embed; a memory stored by the chat pipeline is embedded from the fact alone (`run_memory_background`), a known mismatch |
| `link_count` | INTEGER | required; default `0` | Kept by the `trg_memory_links_ai` / `_ad` triggers |
| `is_ephemeral` | INTEGER | required; default `0` | 1 for a passing fact (a plan, an errand) that should lapse |
| `expires_at` | TEXT |  | When a passing fact lapses: 14 days after the event, or after it was said. Recall leaves out a memory past this date |
| `sentiment_score` | REAL | required; default `0.0` | Tone as a number |
| `sentiment_label` | TEXT | required; default `'neutral'` | Tone: `positive`, `neutral`, `negative`, `frustrated`, `anxious`, `excited` |
| `event_type` | TEXT |  | From the local language pass |
| `event_date` | TEXT |  | The day the fact is about, when it has one |
| `entities_detected` | TEXT | required; default `'[]'` | JSON list of names. A copy that syncs and is backed up; `memory_entities` is the working table |
| `updated_at` | TEXT | required; default `datetime('now')` |  |
| `original_text` | TEXT |  | The user's message the fact was taken from |
| `collection_id` | TEXT |  | Persona collections (0009). Travels with sync; no screen writes it yet |
| `memory_placement` | TEXT | required; default `'retrieved'` | `retrieved` or `pinned` (0009), same status as `collection_id` |
| `external_id` | TEXT |  | An outside id within a collection (0009) |
| `temporal_status` | TEXT |  | NULL for current; `superseded` when a newer memory replaced it; `completed` when a check-in confirmed it happened. Recall leaves both out |
| `provenance_json` | TEXT |  | Which backend wrote the memory and whether the user asked for it |
| `tags` | TEXT | required; default `'[]'` | JSON list of 1 to 3 topic words; cloud backends only |
| `sync_id` | TEXT |  | The row's name on every device |
| `checkin_asked_at` | TEXT |  | When a check-in question about this memory went out (0027) |
| `checkin_answer` | TEXT |  | `done`, `later`, `never` or `other` |

**Checks:** `memory_placement IN ('retrieved', 'pinned')`

**Indexes:**
- `idx_memories_collection`: on (user_id, collection_id)
- `idx_memories_collection_external`: unique, on (user_id, collection_id, external_id) WHERE collection_id IS NOT NULL AND external_id IS NOT NULL
- `idx_memories_ephemeral`: on (is_ephemeral) WHERE is_ephemeral = 1
- `idx_memories_event_date`: on (event_date) WHERE event_date IS NOT NULL
- `idx_memories_event_type`: on (event_type) WHERE event_type IS NOT NULL
- `idx_memories_expires_at`: on (expires_at) WHERE expires_at IS NOT NULL
- `idx_memories_pinned_collection`: on (user_id, collection_id, memory_placement) WHERE memory_placement = 'pinned'
- `idx_memories_sentiment`: on (sentiment_label)
- `idx_memories_sync_id`: unique, on (sync_id)
- `memories_namespace_idx`: on (namespace)
- `memories_shareable_idx`: on (is_shareable) WHERE is_shareable = 1
- `memories_user_namespace_idx`: on (user_id, namespace)

**Triggers:** `trg_memories_outbox_ad`, `trg_memories_outbox_ai`, `trg_memories_outbox_au`. The first three queue changes for sync.

---

### memory_links

A relationship between two memories. Migration 0022 (2026-10-23) rebuilt the table to
allow four types only.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `source_memory_id` | INTEGER | → `memories.id`, deleted with it | The newer memory, the one making the statement |
| `target_memory_id` | INTEGER | → `memories.id`, deleted with it | The memory it is about |
| `relation_type` | TEXT | required | `related`, `contradicts`, `updates` or `resolves` |
| `confidence` | REAL | required; default `0.5` | The model's own figure, 0.75 when it gives none |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `notes` | TEXT |  | The model's reason, or a resolution note |
| `created_by` | TEXT | required; default `'system'` | `llm` (cloud model), `local_llm` (on-device model), `same_message` (facts from one message, no model call), `user` (a card answer) |
| `sync_id` | TEXT |  | The row's name on every device (0016) |

**Unique:** `(source_memory_id, target_memory_id, relation_type)`

**Checks:** `source_memory_id <> target_memory_id`; `confidence` between 0.0 and 1.0; `relation_type IN ('contradicts', 'updates', 'related', 'resolves')`

**Indexes:**
- `idx_memory_links_confidence`: on (confidence DESC)
- `idx_memory_links_source`: on (source_memory_id)
- `idx_memory_links_sync_id`: unique, on (sync_id)
- `idx_memory_links_target`: on (target_memory_id)
- `idx_memory_links_type`: on (relation_type)

**Triggers:** `trg_memory_links_ad`, `trg_memory_links_ai`, `trg_memory_links_outbox_ad`, `trg_memory_links_outbox_ai`, `trg_memory_links_outbox_au`. The first two keep `memories.link_count`; the `outbox` three queue changes for sync.

**What each type does:**
- `related`: same specific subject; the linked memory is pulled into recall. Took over `supports`, `elaborates` and `caused_by`.
- `contradicts`: the two disagree; the user is asked on a card.
- `updates`: the source replaces the target, which gets `temporal_status = 'superseded'` and is kept as history.
- `resolves`: the user's explanation of a resolved conflict.
- `reminds_of`, `quotes` and `mentions` were dropped by 0022 and their rows deleted.

---

### memory_relationships (view)

Each link seen from both ends, so a query can start from either memory. The incoming
side renames the type: `contradicted_by`, `updated_by`, `resolved_by`; `related` stays
`related`.

```sql
CREATE VIEW memory_relationships AS
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
```

---

### memory_entities

One row per named person, place, organisation or thing in a memory (0011). The Memory
Map's hubs are built from this table. It does not sync and is not in the cloud backup,
which is why `memories.entities_detected` keeps a copy of the names.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `memory_id` | INTEGER | required; → `memories.id`, deleted with it |  |
| `name` | TEXT | required | As written in the memory |
| `canonical` | TEXT | required | Lower-cased and trimmed; the grouping key |
| `kind` | TEXT | required; default `'thing'` | `person`, `place`, `org` or `thing` |
| `created_at` | TEXT | required; default `strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')` |  |

**Unique:** `(memory_id, canonical)`

**Indexes:**
- `idx_memory_entities_canonical`: on (canonical)
- `idx_memory_entities_memory`: on (memory_id)

---

### memory_jobs

The memory pipeline's queue (0023, 2026-10-24). After a reply, the memory work for the
message is a row here. One worker per process runs the rows in order while the model is
idle, so the work survives an interruption or a restart. Code: `memory_jobs.rs`.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `session_id` | TEXT | required |  |
| `user_id` | TEXT | required |  |
| `message` | TEXT | required | The user's message the facts are taken from |
| `backend` | TEXT | required | The backend chosen for the conversation; the same one does the memory calls |
| `source_type` | TEXT | required; default `'conversation'` | `conversation` or `hands_free`, carried onto the memories |
| `containment_mode` | TEXT | required; default `'sovereign'` | The mode when the message was sent |
| `explicit_remember` | INTEGER | required; default `0` | 1 for a `Remember:` message: the text is stored as given and call 2 is skipped |
| `remember_text` | TEXT |  | The text to store for a `Remember:` message |
| `user_name` | TEXT |  |  |
| `recent_turns` | TEXT |  | The last exchange, given to call 2 so it can resolve pronouns |
| `query_entities` | TEXT | required; default `'[]'` | JSON list of names found in the message at chat time |
| `status` | TEXT | required; default `'pending'` | `pending`, `running`, `done` or `failed`. A `running` row untouched for five minutes is taken again |
| `attempts` | INTEGER | required; default `0` | A row is given up as `failed` after 4 attempts in total |
| `error` | TEXT |  |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `updated_at` | TEXT | required; default `datetime('now')` |  |

**Indexes:**
- `memory_jobs_status_idx`: on (status, id)

---

### memory_cards

A question waiting for the user: "Is this an update?" or "These two disagree" (0024,
2026-10-26). A card is a row so that it is shown on every app open until answered; on a
phone the pipeline raises it a minute after the reply, when the user may have left.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `user_id` | TEXT | required |  |
| `session_id` | TEXT |  |  |
| `stored_memory_id` | INTEGER |  | The new memory, when it was stored before asking (on-device path) |
| `conflicting_memory_id` | INTEGER |  | The older memory it disagrees with |
| `payload_json` | TEXT | required | The event payload exactly as the window expects it |
| `raised_at` | TEXT | required; default `datetime('now')` |  |
| `answered_at` | TEXT |  | NULL while the card is waiting |
| `decision` | TEXT |  | The option the user chose |

**Indexes:**
- `idx_memory_cards_pending`: on (user_id, answered_at)

---

### deleted_memory_hashes

Tombstones (0006). The SHA-256 of a deleted memory's text is kept here so that sync or a
cloud restore cannot bring the memory back.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `content_hash` | TEXT | primary key |  |
| `deleted_at` | TEXT | required; default `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` |  |

---

## Knowledge Base tables

### kb_documents

One row per indexed file. The built-in guides belong to the user id `_system`.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `user_id` | TEXT | required |  |
| `file_path` | TEXT | required |  |
| `file_name` | TEXT | required |  |
| `file_size` | INTEGER | required |  |
| `last_modified` | TEXT | required | The file's own modified time; a change marks it for re-indexing |
| `indexed_at` | TEXT | required; default `datetime('now')` |  |
| `chunk_count` | INTEGER | required; default `0` |  |
| `status` | TEXT | required; default `'indexing'` | `indexed`, `indexing`, `needs_reindex` or `error` |
| `error_message` | TEXT |  |  |

**Unique:** `(user_id, file_path)`

**Checks:** `status IN ('indexed', 'indexing', 'needs_reindex', 'error')`

**Indexes:**
- `idx_kb_documents_status`: on (status)
- `idx_kb_documents_user`: on (user_id)

---

### kb_chunks

A document cut into passages, each with an embedding.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `document_id` | INTEGER | → `kb_documents.id`, deleted with it |  |
| `chunk_index` | INTEGER | required |  |
| `content` | TEXT | required |  |
| `embedding` | BLOB |  | 384 floats, as in `memories` |
| `token_count` | INTEGER | required |  |

**Unique:** `(document_id, chunk_index)`

**Indexes:**
- `idx_kb_chunks_document`: on (document_id)

**Triggers:** `kb_chunks_ad`, `kb_chunks_ai`, `kb_chunks_au`. These keep `kb_chunks_fts` in step with the table.

---

### kb_chunks_fts (FTS5)

```sql
CREATE VIRTUAL TABLE kb_chunks_fts USING fts5(
    content,
    content='kb_chunks',
    content_rowid='id'
);
```

Word search over the passages, used beside the embedding search in `kb_rag.rs`. SQLite
creates four internal tables for it (`kb_chunks_fts_config`, `_data`, `_docsize`, `_idx`).

---

## Conversation history tables

Nothing is written to these tables in HIPAA mode.

### conversation_sessions

One row per conversation (thread).

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `session_id` | TEXT | required | A UUID; also the row's `sync_id` |
| `user_id` | TEXT | required |  |
| `title` | TEXT |  | The first 60 characters of the first message, until the user renames it (120 characters at most) |
| `started_at` | TEXT | required; default `datetime('now')` |  |
| `last_active` | TEXT | required; default `datetime('now')` |  |
| `message_count` | INTEGER | required; default `0` |  |
| `model_backend` | TEXT |  | The backend last used |
| `containment_mode` | TEXT |  | The mode when the thread was last written |
| `pinned` | INTEGER | required; default `0` | 1 when pinned to the top of the list |
| `last_extraction` | TEXT |  | What the memory pipeline did with the last message, for example `stored` or `error` |
| `last_extraction_at` | TEXT |  | When that was |
| `sync_id` | TEXT |  |  |
| `pinned_at` | TEXT |  | When the pin last changed; the newest change wins between devices (0025) |

**Unique:** `session_id`

**Indexes:**
- `idx_conv_sessions_sync_id`: unique, on (sync_id)
- `idx_conv_sessions_user_active`: on (user_id, last_active DESC)

**Triggers:** `trg_conv_sessions_outbox_ad`, `trg_conv_sessions_outbox_ai`, `trg_conv_sessions_outbox_au`. They queue changes for sync.

---

### conversation_messages

Every user and assistant message.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `session_id` | TEXT | required; → `conversation_sessions.session_id`, deleted with it |  |
| `user_id` | TEXT | required |  |
| `role` | TEXT | required | `user` or `assistant` |
| `content` | TEXT | required |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `model_backend` | TEXT |  |  |
| `containment_mode` | TEXT |  |  |
| `entry_hash` | TEXT |  | Reserved for a tamper-evidence chain; carried by sync and backup, not filled in |
| `prev_hash` | TEXT |  | Reserved, as `entry_hash` |
| `input_mode` | TEXT | required; default `'typed'` | `typed`, `dictated`, `hands_free` or `system` |
| `sync_id` | TEXT |  | The row's name on every device (0013) |
| `sources_json` | TEXT |  | The web sources behind a searched answer, in the shape the page renders (0028) |

**Checks:** `role IN ('user', 'assistant')`; `input_mode IN ('typed', 'dictated', 'hands_free', 'system')`

**Indexes:**
- `idx_conv_messages_session`: on (session_id, created_at ASC)
- `idx_conv_messages_sync_id`: unique, on (sync_id)
- `idx_conv_messages_unique`: unique, on (session_id, role, created_at, content)

**Triggers:** `trg_conv_messages_outbox_ad`, `trg_conv_messages_outbox_ai`, `trg_conv_messages_outbox_au`. They queue changes for sync.

`idx_conv_messages_unique` (0011) stops the same message being stored twice in a thread.

---

### conversation_messages_fts (FTS5)

```sql
CREATE VIRTUAL TABLE conversation_messages_fts USING fts5(
    content,
    session_id UNINDEXED,
    content='conversation_messages',
    content_rowid='id'
);
```

Created by 0001 and never filled: no trigger or code writes to it. Conversation search
(`conversation_history.rs`) uses `LIKE` on the title and the message text.

---

### message_feedback

Thumbs up or down on an assistant reply.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `message_id` | TEXT | required | The message's id as text; by convention, not a foreign key |
| `session_id` | TEXT | required; → `conversation_sessions.session_id`, deleted with it |  |
| `user_id` | TEXT | required |  |
| `rating` | INTEGER | required | `1` up, `-1` down |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `model_backend` | TEXT |  | Copied from the message so ratings can be counted per backend |
| `containment_mode` | TEXT |  | Copied from the message |

**Unique:** `(message_id, user_id)`

**Checks:** `rating IN (-1, 1)`

**Indexes:**
- `idx_message_feedback_rating`: on (rating)
- `idx_message_feedback_session`: on (session_id)
- `idx_message_feedback_user`: on (user_id)

---

### deleted_sessions

Tombstones for deleted threads (0019). A thread deleted on one device used to come back
when another device saved a reply into it. An incoming thread or message for a
tombstoned thread is ignored, and saving a reply into one is refused.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `session_id` | TEXT | primary key |  |
| `deleted_at` | TEXT | required; default `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` |  |

---

## ZynkSync tables

ZynkSync keeps one user's devices the same. Since the outbox rebuild (0013 to 0021,
2026-09-29 to 2026-10-14) a feature syncs by writing to the database and nothing else:
triggers queue every change and `sync_outbox.rs` sends the queue to each paired device.

### zynk_devices

Every device this one knows: itself, its ZynkSync peers and its ZynkLink peers.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `device_id` | TEXT | required |  |
| `device_name` | TEXT | required |  |
| `device_ip` | TEXT |  | Last known address on the local network |
| `device_platform` | TEXT |  |  |
| `pairing_code` | TEXT |  | The six-digit code this device is showing, if any |
| `pairing_code_expires_at` | TEXT |  | Ten minutes after the code was made (`zynksync.rs`) |
| `is_paired` | INTEGER | required; default `0` | 1 for any pairing, ZynkSync or ZynkLink |
| `last_seen_at` | TEXT | required; default `datetime('now')` |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `owner_user_id` | TEXT |  | Memories sync only between devices of the same owner |
| `port` | INTEGER | required; default `57963` | The port the device listens on |
| `tls_cert_der` | BLOB |  | The peer's TLS certificate, pinned at pairing. Later connections must present it |
| `sync_paired` | INTEGER | required; default `0` | 1 for a ZynkSync pairing only (0005); the ZynkSync device list reads this |

**Unique:** `device_id`

**Indexes:**
- `idx_devices_owner_user_id`: on (owner_user_id)
- `idx_zynk_devices_port`: on (port)
- `idx_zynk_devices_sync_paired`: on (sync_paired)
- `zynk_devices_device_id_idx`: on (device_id)
- `zynk_devices_pairing_code_idx`: on (pairing_code) WHERE pairing_code IS NOT NULL

---

### zynk_device_pairings

One row per pair of ZynkSync devices, with the smaller id always in `device_a_id`.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `device_a_id` | TEXT | → `zynk_devices.device_id`, deleted with it |  |
| `device_b_id` | TEXT | → `zynk_devices.device_id`, deleted with it |  |
| `paired_at` | TEXT | required; default `datetime('now')` |  |
| `last_sync_a_to_b` | TEXT |  |  |
| `last_sync_b_to_a` | TEXT |  |  |
| `is_active` | INTEGER | required; default `1` |  |

**Unique:** `(device_a_id, device_b_id)`

**Checks:** `device_a_id < device_b_id`

**Indexes:**
- `zynk_device_pairings_device_a_idx`: on (device_a_id)
- `zynk_device_pairings_device_b_idx`: on (device_b_id)

---

### user_sync_codes

One-time six-digit codes (100000 to 999999) for joining a user's devices. A code lasts
five minutes (`sync_codes.rs`) and works once.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `code` | TEXT | required |  |
| `user_id` | TEXT | required |  |
| `device_id` | TEXT | required |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `expires_at` | TEXT | required |  |
| `used` | INTEGER | required; default `0` |  |
| `used_at` | TEXT |  |  |

**Unique:** `code`

**Indexes:**
- `idx_user_sync_codes_code`: on (code) WHERE used = 0
- `idx_user_sync_codes_expires_at`: on (expires_at)

---

### sync_outbox

The queue of changes to send (0013). One row per insert, update or delete on a synced
table. The row names what changed; the sender reads the current values from the live
table when it sends.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `table_name` | TEXT | required | `memories`, `memory_links`, `conversation_sessions`, `conversation_messages` or `sync_secrets` |
| `row_sync_id` | TEXT | required | The changed row's `sync_id` |
| `op` | TEXT | required |  |
| `payload` | TEXT |  | On a delete of a memory or message: its text, from which the receiver gets the content hash |
| `created_at` | TEXT | required; default `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` |  |
| `deleted_at` | TEXT |  | The tombstone's own time when a past deletion is re-sent (0020); NULL otherwise |

**Checks:** `op IN ('insert', 'update', 'delete')`

**Indexes:**
- `idx_sync_outbox_created`: on (created_at)

---

### sync_outbox_cursor

How far this device has sent to each peer: the last `sync_outbox.id` the peer accepted.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `peer_device_id` | TEXT | primary key |  |
| `last_outbox_id` | INTEGER | required; default `0` |  |
| `updated_at` | TEXT |  |  |

---

### sync_inbox_cursor

How far this device has heard from each peer (0015). A device restored onto a fresh
install reports a low number, and the sender then starts it over from the live tables.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `peer_device_id` | TEXT | primary key |  |
| `through` | INTEGER | required; default `0` |  |
| `updated_at` | TEXT |  |  |

---

### sync_suppress

A switch (0013). While it holds a row, the triggers queue nothing. The receiving code
sets it while applying a peer's changes so they are not sent straight back.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `flag` | INTEGER | primary key |  |

**Checks:** `flag = 1`, so the table holds at most one row

---

### sync_secrets

The settings that travel to every paired device through the outbox (0014): the API keys,
the backup key and the wake-word profile, one row each, named by the setting's own name.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `name` | TEXT | primary key |  |
| `value` | TEXT | required |  |
| `updated_at` | TEXT | required; default `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` |  |
| `sync_id` | TEXT |  | Same as `name` |
| `revision` | INTEGER | required; default `0` | Goes up by one on every local change; the higher revision wins, so device clocks do not decide (0018) |

**Triggers:** `trg_sync_secrets_outbox_ad`, `trg_sync_secrets_outbox_ai`, `trg_sync_secrets_outbox_au`. They queue changes for sync.

---

### zynk_pending_removals

A device-removal notice that could not be delivered because the peer was offline (0007).
It is retried on the next heartbeat.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `target_peer_id` | TEXT | required |  |
| `target_peer_ip` | TEXT | required |  |
| `sender_device_id` | TEXT | required |  |
| `cascade_device_id` | TEXT | required |  |
| `created_at` | TEXT | required; default `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` |  |

**Indexes:**
- `idx_pending_removals_target`: on (target_peer_id)

---

### Tables kept from the earlier sync design

These three are still created by the migrations. Nothing reads them for sync any more.

### zynk_sync_state

Which memories had been sent to which device, in the whole-table compare the outbox
replaced. The code now only deletes a removed device's rows from it.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `source_device_id` | TEXT | required |  |
| `target_device_id` | TEXT | required |  |
| `memory_id` | INTEGER | required |  |
| `synced_at` | TEXT | required; default `datetime('now')` |  |
| `memory_updated_at` | TEXT |  |  |

**Unique:** `(source_device_id, target_device_id, memory_id)`

**Indexes:**
- `zynk_sync_state_memory_idx`: on (memory_id)
- `zynk_sync_state_source_idx`: on (source_device_id)
- `zynk_sync_state_target_idx`: on (target_device_id)

---

### zynk_conversation_push_state

A per-peer marker for conversation pushes (0012), replaced by the outbox cursors eleven
days later. No code refers to it.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `peer_device_id` | TEXT | primary key |  |
| `pushed_through` | TEXT | required |  |

---

### zynk_device_certificates

An earlier home for peer certificates. The pinned certificate now lives in
`zynk_devices.tls_cert_der`; the code only deletes a removed device's row from here.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `device_id` | TEXT | primary key |  |
| `certificate_pem` | TEXT | required |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `updated_at` | TEXT | required; default `datetime('now')` |  |

---

## ZynkLink tables

ZynkLink shares folders between devices, including devices of different users.

### zynk_linked_directories

The folders this device shares. They are local settings and stay when a device is unlinked.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `device_id` | TEXT | → `zynk_devices.device_id`, deleted with it |  |
| `local_path` | TEXT | required |  |
| `share_name` | TEXT | required |  |
| `description` | TEXT |  |  |
| `is_readable` | INTEGER | required; default `1` |  |
| `is_writable` | INTEGER | required; default `0` |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `updated_at` | TEXT | required; default `datetime('now')` |  |

**Unique:** `(device_id, local_path)`

**Indexes:**
- `idx_shared_dirs_device`: on (device_id)

---

### zynk_file_manifest

The files in each shared folder. This is the table `zynklink.rs` writes and reads.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `shared_directory_id` | INTEGER | → `zynk_linked_directories.id`, deleted with it |  |
| `relative_path` | TEXT | required |  |
| `file_size` | INTEGER | required |  |
| `file_hash` | TEXT |  |  |
| `mime_type` | TEXT |  |  |
| `last_modified` | TEXT |  |  |
| `indexed_at` | TEXT | required; default `datetime('now')` |  |

**Unique:** `(shared_directory_id, relative_path)`

---

### zynk_link_manifest

An older copy of the same idea under a different column name. The two delete commands in
`commands/zynklink.rs` and the device-removal code still clear rows from it; nothing
inserts into it. A candidate for removal in a later migration.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `linked_directory_id` | INTEGER | → `zynk_linked_directories.id`, deleted with it |  |
| `relative_path` | TEXT | required |  |
| `file_size` | INTEGER | required |  |
| `file_hash` | TEXT |  |  |
| `mime_type` | TEXT |  |  |
| `last_modified` | TEXT |  |  |
| `indexed_at` | TEXT | required; default `datetime('now')` |  |

**Unique:** `(linked_directory_id, relative_path)`

**Indexes:**
- `idx_file_manifest_hash`: on (file_hash)
- `idx_file_manifest_share`: on (linked_directory_id)

---

### zynklink_codes

Codes for setting up a ZynkLink between two devices. A code stops working once accepted.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `code` | TEXT | required |  |
| `creator_user_id` | TEXT | required |  |
| `creator_device_id` | TEXT | → `zynk_devices.device_id`, deleted with it |  |
| `created_at` | TEXT | required; default `datetime('now')` |  |
| `expires_at` | TEXT |  |  |
| `accepted_by_user_id` | TEXT |  |  |
| `accepted_by_device_id` | TEXT | → `zynk_devices.device_id`, deleted with it |  |
| `accepted_at` | TEXT |  |  |
| `is_active` | INTEGER | required; default `1` |  |
| `revoked_at` | TEXT |  |  |
| `revoked_reason` | TEXT |  |  |

**Unique:** `code`

**Indexes:**
- `idx_zynklink_codes_accepted`: on (accepted_by_user_id)
- `idx_zynklink_codes_active`: on (code) WHERE is_active = 1
- `idx_zynklink_codes_creator`: on (creator_user_id)

---

### zynklink_pairings

One row per linked pair of devices. Migration 0008 changed the unique key from the user
pair to the device pair, so one user's phone can link to more than one device.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | INTEGER | primary key |  |
| `user1_id` | TEXT | required |  |
| `user2_id` | TEXT | required |  |
| `device1_id` | TEXT | required |  |
| `device2_id` | TEXT | required |  |
| `linked_at` | TEXT | required; default `datetime('now')` |  |
| `is_active` | INTEGER | required; default `1` |  |
| `is_paused` | INTEGER | required; default `0` | 1 while the user has paused the link |

**Unique:** `(device1_id, device2_id)`

**Indexes:**
- `idx_zynklink_pairings_device1`: on (device1_id)
- `idx_zynklink_pairings_device2`: on (device2_id)
- `idx_zynklink_pairings_user1`: on (user1_id)
- `idx_zynklink_pairings_user2`: on (user2_id)

---

## ZChat table

### zchat_messages

Messages sent directly between paired devices. They are stored on the two devices only.

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `id` | TEXT | primary key | A UUID |
| `from_device_id` | TEXT | required |  |
| `to_device_id` | TEXT | required |  |
| `message_text` | TEXT | required |  |
| `sent_at` | TEXT | required; default `datetime('now')` |  |
| `delivered_at` | TEXT |  | Delivery receipt |
| `read_at` | TEXT |  | Read receipt |
| `user_id` | TEXT | required |  |

**Checks:** `length(message_text) > 0`

**Indexes:**
- `idx_zchat_from_device`: on (from_device_id, sent_at DESC)
- `idx_zchat_to_device`: on (to_device_id, sent_at DESC)
- `idx_zchat_user`: on (user_id)

---

## Settings table

### app_settings

Small values the app keeps for itself, by key. Keys in use: `memory_embedding_scheme`
(which text the embeddings were made from, so a change re-embeds once),
`system_docs_version` (which built-in guides are indexed) and `checkin_enabled`.
The containment mode is not stored here or anywhere else; see
[Containment Architecture](CONTAINMENT_ARCHITECTURE.md).

| Column | Type | Constraints | Meaning |
|--------|------|-------------|---------|
| `key` | TEXT | primary key |  |
| `value` | TEXT |  |  |
| `updated_at` | TEXT | default `datetime('now')` |  |

---

## Sync triggers

Five tables are synced: `memories`, `memory_links`, `conversation_sessions`,
`conversation_messages` and `sync_secrets`. Each has three triggers, named
`trg_<table>_outbox_ai`, `_au` and `_ad` (`trg_conv_sessions_…` and `trg_conv_messages_…`
for the two conversation tables):

| Trigger | When | What it does |
|---------|------|--------------|
| `_ai` | after insert | Gives the row a `sync_id` when it has none (16 random bytes as hex; a session uses its `session_id`, a secret its `name`), then queues an `insert` |
| `_au` | after update | Queues an `update`. The update that only filled in `sync_id` is skipped |
| `_ad` | after delete | Queues a `delete`. For a memory or a message the row's text goes in `payload` |

None of them queues anything while `sync_suppress` holds a row.

The other five triggers: `trg_memory_links_ai` and `trg_memory_links_ad` keep
`memories.link_count`; `kb_chunks_ai`, `kb_chunks_au` and `kb_chunks_ad` keep the
Knowledge Base word index.

There is no trigger on `memories.updated_at`; the code sets it when it changes a row.

---

## Deleting and expiring

- A deleted memory leaves its content hash in `deleted_memory_hashes`, and a deleted
  thread its id in `deleted_sessions`, so neither returns through sync or restore.
- Deleting a memory deletes its links and its `memory_entities` rows (foreign keys with
  cascade). Deleting a thread deletes its messages and their feedback.
- A passing fact past its `expires_at` stays in the table and is left out of recall
  (`memory.rs`). The `cleanup_expired_memories` command would delete such rows, but
  nothing in the app calls it.

---

## Backup

**In the app:** the cloud backup (`commands/backup.rs`) encrypts memories, their links,
conversations and keys on the device (AES-256-GCM, key derived from the backup phrase
with Argon2id) and stores the result in the user's Cloudflare R2 bucket. On restore,
links are matched to memories by content hash.

**By hand:** close Zynkbot first, then copy the database with its write-ahead log files.

```bash
# Linux
cp ~/.local/share/zynkbot/zynkbot.db* ~/zynkbot_backup/

# Windows
copy "%LOCALAPPDATA%\zynkbot\zynkbot.db*" "%USERPROFILE%\zynkbot_backup\"
```

To restore, close Zynkbot and copy the files back. On Android the file is inside the
app's private storage and cannot be copied without root or a debug build.

A second device is the simplest backup: a newly paired device receives everything.

---

## Known leftovers

- `zynk_link_manifest`, `zynk_sync_state`, `zynk_conversation_push_state` and
  `zynk_device_certificates` are unused or nearly so.
- `conversation_messages_fts` is never filled.
- `memories.parent_scroll_id`, `chunk_index`, `is_syncable` and `is_shareable` are not
  used by any feature.
- `conversation_messages.entry_hash` / `prev_hash` are reserved and empty.
- Expired passing facts are hidden, not deleted.

---

## See also

- [Memory Processing Pipeline](MEMORY_PROCESSING_PIPELINE.md): how a memory row is made
- [Relationship Detection Architecture](RELATIONSHIP_DETECTION_ARCHITECTURE.md): how links are made
- [Networking Features](../NETWORKING_FEATURES.md): ZynkSync, ZynkLink and ZChat
