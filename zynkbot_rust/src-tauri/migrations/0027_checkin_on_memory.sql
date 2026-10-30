-- 0027: the check-in mark travels with the memory (2026-10-30, Matt: the list of what has
-- been asked "should definitely be synced"). 0026 kept it in memory_checkins, a table that
-- never left the device, so each phone and the desktop would have asked the same question
-- once each. The mark is now two columns on memories, which sync as part of the row:
--   checkin_asked_at  when the question went out (NULL = never asked, or asked and deferred)
--   checkin_answer    done | later | never | other, for the record
ALTER TABLE memories ADD COLUMN checkin_asked_at TEXT;
ALTER TABLE memories ADD COLUMN checkin_answer TEXT;
UPDATE memories
   SET checkin_asked_at = (SELECT c.asked_at FROM memory_checkins c WHERE c.memory_id = memories.id ORDER BY c.id DESC LIMIT 1),
       checkin_answer   = (SELECT c.answer   FROM memory_checkins c WHERE c.memory_id = memories.id ORDER BY c.id DESC LIMIT 1)
 WHERE id IN (SELECT memory_id FROM memory_checkins);
DROP INDEX IF EXISTS idx_memory_checkins_memory;
DROP TABLE IF EXISTS memory_checkins;
