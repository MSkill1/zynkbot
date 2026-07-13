# Known Issues

This file tracks known bugs, edge cases, and rough edges that do not block release but should be fixed in a future update. Contributions welcome — see CONTRIBUTING.md.

---

## Memory Pipeline

### KI-012 — Original text not preserved when memory is stored via contradiction resolution
**Status:** Fixed in this release  
**Affected:** All users — any memory stored after resolving a contradiction modal  
**Description:** The `original_text` field (the verbatim user input) is correctly stored for memories created through the normal path. However, when a contradiction is detected and the user resolves it via the modal, the memory was stored through `store_pending_memory`, which passed `pending.content` (the LLM-extracted fact) as `original_text` instead of the raw user message. Both the Content and Original fields in Memory Manager showed the same extracted text. Fixed by adding `original_text` to `PendingMemory` and threading `bg_message` through the contradiction event payload.  
**Impact:** None — resolved.

---

### KI-013 — Original text not preserved when memory arrives via ZynkSync
**Status:** Fixed in this release  
**Affected:** All users — any memory received from a paired device via ZynkSync  
**Description:** The `original_text` field (the verbatim user input) was not included in the ZynkSync payload. Memories created on one device and synced to another had no `original_text` on the receiving device. Fixed by adding `original_text` to the `SyncMemory` struct, all memory SELECT queries, and the receive INSERT/UPDATE paths in `zynksync.rs`.  
**Impact:** None — resolved. Note: memories synced before this release will still lack `original_text` on the receiving device; only new syncs after upgrading will carry the field.

---

### KI-001 — Double memory on contradiction resolution (edge case)
**Status:** Partially fixed  
**Affected:** All local models  
**Description:** Memory is only stored after the user resolves the contradiction modal — never before. However, in rare cases the background duplicate check may still produce a second copy if the embedding distance between the raw user message and the MEMORY_EXTRACT fact falls between the 0.65 and 0.93 similarity thresholds, causing both to pass the near-duplicate filter.  
**Workaround:** If you see duplicate memories after a contradiction resolution, delete the lower-numbered one — the MEMORY_EXTRACT version is the cleaner fact.  
**Fix target:** Improve near-duplicate search to prefer most-recently-inserted match.

### KI-002 — Contradiction false positive: intention vs. current state
**Status:** Partially mitigated (non-contradiction example added to classifier prompt)  
**Description:** Statements expressing a future intention ("I'm thinking about leaving my job") may occasionally be flagged as contradicting a stored current state ("I work at X"). The classifier prompt includes an example to discourage this, but LLM classification is not deterministic.  
**Workaround:** Select "Not a contradiction" in the modal. No data is lost.

### KI-003 — System memories appearing in user hybrid search
**Status:** Fixed in this release  
**Description:** System memories (user_id = 'system', IDs 1–12) were appearing in user hybrid search results — for example, "Model Support" appeared at ~50% similarity for queries containing common nouns. Fixed by scoping `hybrid_search`, `list_memories`, and the Memory Manager query to exclude `user_id = 'system'` entries.  
**Impact:** None — resolved.

---

## Onboarding

### KI-004 — Onboarding relationship detection skipped on fresh install (fixed)
**Status:** Fixed in this release  
**Description:** `complete_onboarding` reported "no embedding" for all onboarding memories because the `Memory` struct uses `#[sqlx(skip)]` on the embedding field. Embeddings were present in the database but not read by the struct. Fixed by fetching embeddings separately via a raw query.

---

## Local Models

### KI-007 — Uncensored and fine-tuned models may produce lower-quality memory extraction
**Status:** Open / by design  
**Affected:** Uncensored fine-tunes (confirmed: Llama 3.1 8B Lexi Uncensored V2)  
**Description:** Zynkbot's memory pipeline relies on each model following structured extraction instructions precisely — returning only the single new fact introduced in a message. Models fine-tuned for creative or unfiltered output (rather than instruction-following) tend to extract broad context summaries instead of the specific new fact. This causes two downstream problems:

1. **Redundant memories** — the extracted "fact" repeats information already stored from onboarding rather than capturing what's new.
2. **Misdirected contradiction links** — because the extracted fact is a summary of background context rather than the specific claim being corrected, the hybrid search may not surface the most relevant existing memory, causing contradiction relationships to link to the wrong entry.

**Example (Lexi, Q3 test):** User says *"Actually, I've been at Brightline for 4 years, not 3."* Expected extraction: something about tenure correction. Actual extraction: *"Jordan is 31 years old, married to Sarah, and has a 3-year-old daughter named Emma..."* — a family summary unrelated to the correction. The contradiction modal still fired and the correction was stored, but it was linked to the onboarding question memory (Memory 104) rather than the actual tenure fact (Memory 114).

**Impact:** Memory entries may be less precise over time; contradiction links may reference the wrong prior memory. Conversations still function correctly.  
**Workaround:** Use Qwen3 or DeepSeek R1 if memory accuracy is important. Lexi is best suited for creative conversations where long-term memory precision is less critical.  
**Fix target:** No code fix planned — this is a characteristic of the model, not the pipeline.

### KI-008 — Web search trigger is model-dependent on local GGUF models
**Status:** Open / by design  
**Affected:** All local models (varies by model)  
**Description:** Web search requires the model to emit a `WEB_SEARCH_NEEDED:` marker in its response. API models (Claude, GPT-4) do this reliably. Small local models (7B) vary: Qwen3 triggers it consistently; DeepSeek R1 triggers it when the query clearly requires current information; Lexi rarely triggers it and instead gracefully tells the user to search manually.  
**Impact:** Web search may not fire automatically on some local models. The user can ask explicitly, but results depend on the model.  
**Workaround:** Use an API model or Qwen3 if web search reliability matters.  
**Fix target:** No reliable fix without a separate intent-classification model.

---

### KI-005 — Untested models may require prompt format tuning
**Status:** Open / by design  
**Description:** Zynkbot ships with verified optimizations for Qwen3, DeepSeek R1 Distill Llama 8B, and Llama 3.1 Lexi Uncensored V2. Other GGUF models should work but have not been tested. Models using non-standard prompt formats or tokenizer types may produce incomplete or malformed responses.  
**Workaround:** Check `local_models.rs` → `build_prompt_for_model` to add a detection path for your model family.

---

## Networking

### KI-009 — Unsyncing a device also removes the ZynkLink pairing
**Status:** Fixed in this release  
**Affected:** Users who have both ZynkSync and ZynkLink active between the same two devices  
**Description:** ZynkSync and ZynkLink now maintain independent trust relationships via the `sync_paired` column. Unsyncing only clears the ZynkSync pairing; the ZynkLink pairing remains active. Unlinking only clears the ZynkLink pairing; the ZynkSync pairing remains active. Each can be revoked independently without affecting the other.

---

### KI-010 — ZynkLink pairing appeared in the ZynkSync device list
**Status:** Fixed in this release  
**Affected:** Users who established a ZynkLink pairing without a ZynkSync pairing  
**Description:** Establishing a ZynkLink pairing would register the remote device in `zynk_devices` with `is_paired = 1`, causing it to appear in the ZynkSync panel as a paired sync device even though no sync pairing had been established. The `sync_paired` column now tracks sync pairings separately — ZynkLink-only devices no longer appear in the ZynkSync panel.

---

### KI-011 — Pre-existing memories are orphaned after first sync
**Status:** Open  
**Affected:** Users who have existing memories on a device before performing their first ZynkSync with a new partner device  
**Description:** When two devices sync for the first time, memories that already existed on the receiving device before the sync are not automatically merged or associated with the synced identity. They remain as orphaned records in the local database — accessible locally but not part of the synced memory set. New memories created after the first sync are handled correctly.  
**Workaround:** No workaround currently. Orphaned memories remain visible and usable in local conversation but will not propagate to other devices.  
**Fix target:** v1.0 — requires an identity merge step during the first sync handshake to adopt pre-existing memories into the synced namespace.

---

## Debug Logging

### KI-006 — Verbose debug output in development builds
**Status:** Fixed  
**Description:** Several `println!` statements in `lib.rs` and `zynksync.rs` dumped full LLM responses and raw HTTP payloads to the terminal. Gated behind `#[cfg(debug_assertions)]` — silent in release builds, visible in `cargo tauri dev`.

---

*Last updated: 2026-07-12*
