-- Add original_text column to preserve user's verbatim input
-- This allows us to:
-- 1. Embed normalized facts for clean search
-- 2. Send original first-person text to LLM for tone preservation

ALTER TABLE memories ADD COLUMN IF NOT EXISTS original_text TEXT;

COMMENT ON COLUMN memories.original_text IS
'Original first-person form of the memory. For extracted facts, this preserves the users voice (e.g., "My dog is Max") while content contains the normalized form (e.g., "User has a dog named Max"). For conversation memories, this is typically null as content already contains the original text.';
