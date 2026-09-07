-- Pinned conversations, and one timestamp format so ordering is correct.
--
-- Sessions restored from backup carry RFC 3339 strings ("2026-09-06T19:10:05+00:00");
-- rows written by the app used SQLite's datetime('now') ("2026-09-06 23:41:34"). The
-- history list orders these as text, so within a day the restored style sorted above
-- the app's, and the newest conversation was not first (2026-09-07). Normalise the
-- existing rows here; log_exchange writes RFC 3339 from now on.
ALTER TABLE conversation_sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
UPDATE conversation_sessions SET started_at  = replace(started_at,  ' ', 'T') || '+00:00' WHERE started_at  NOT LIKE '%T%';
UPDATE conversation_sessions SET last_active = replace(last_active, ' ', 'T') || '+00:00' WHERE last_active NOT LIKE '%T%';
UPDATE conversation_messages SET created_at  = replace(created_at,  ' ', 'T') || '+00:00' WHERE created_at  NOT LIKE '%T%';
