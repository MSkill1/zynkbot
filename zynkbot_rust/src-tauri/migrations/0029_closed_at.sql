-- 0029: a thread closed with New (2026-11-17, tester Mike)
-- New used to clear the Continue line on one device only; every other device, and that one
-- after a restart, offered the thread again. closed_at is when New was last pressed on the
-- thread, on any device; it syncs like pinned_at (newest wins). The thread counts as closed
-- while closed_at is at or after last_active, so a message sent in it afterwards reopens it
-- without a NULL having to travel.
ALTER TABLE conversation_sessions ADD COLUMN closed_at TEXT;
