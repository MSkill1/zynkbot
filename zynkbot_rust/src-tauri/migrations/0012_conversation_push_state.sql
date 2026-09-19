-- Per-peer marker for conversation history pushes (2026-09-19, KI-068).
-- The marker used to live in memory, so every app start re-sent the whole history in
-- one request and the receiving side rejected it as too large; the backlog was then
-- silently dropped. Advanced only after the peer has accepted a push.
CREATE TABLE IF NOT EXISTS zynk_conversation_push_state (
    peer_device_id TEXT PRIMARY KEY,
    pushed_through TEXT NOT NULL
);
