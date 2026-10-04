-- 0018: a revision counter on keys, so clocks do not decide which value wins (KI-081, 2026-10-03)
-- Keys travelled with the time the device saved them and the newest time won. Two
-- phones ran a minute ahead of the desktop, so a value they merely re-recorded after a
-- restart counted as newer than one the desktop had saved later by the wall clock. The
-- counter goes up by one on every local change, starting above the highest revision this
-- device has seen for the key; a later save therefore always outranks what it followed,
-- whatever the clocks say. Time and then the value break ties between changes made apart.
ALTER TABLE sync_secrets ADD COLUMN revision INTEGER NOT NULL DEFAULT 0;
