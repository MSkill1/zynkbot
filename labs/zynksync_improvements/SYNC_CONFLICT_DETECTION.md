# ZynkSync: Semantic Conflict Detection During Sync

**Status:** Design Phase (v1.0 Target)
**Related:** `src-tauri/src/zynksync.rs`, `src-tauri/src/lib.rs` (contradiction detection logic)

---

## Current Behavior

ZynkSync uses an "active device wins" reconciliation model:

- **First sync:** Device with more memories is the source of truth
- **Subsequent syncs:** Device with the most recent activity timestamp wins
- **ID collisions:** Last-write-wins by `created_at` timestamp
- **Deletions:** Propagated from the active device (subsequent syncs only)
- **Relationships:** Synced alongside memories, inserted with `ON CONFLICT DO NOTHING`

Conflict resolution is purely mechanical — no semantic analysis. If device A has "I live in Boston" and device B (active) has "I live in Chicago", device B wins silently. The user never knows a conflict occurred or that device A's version was discarded.

This is a known limitation. See `zynksync.rs` line 8: "Conflict resolution (last-write-wins by timestamp)".

---

## Problem

The current model can silently discard meaningful user data. Examples:

- User updates a memory on their laptop while their desktop is offline. Later they use the desktop heavily. On next sync, the desktop wins and the laptop edit is lost.
- Two devices accumulate diverging memories about the same fact (e.g., updated health information, changed job, moved city). Sync resolves it by timestamp, not by asking the user.
- Relationship graph edges (contradicts, elaborates, etc.) may become inconsistent if the memories they link are resolved differently on each device.

This is particularly a problem for memories that were intentionally edited — the user took an action, and sync silently undoes it.

---

## Proposed Solution: Non-Blocking Flagged Conflicts

The key design constraint is that sync must remain **silent and non-interactive**. It runs on a background timer and should not interrupt the user or require UI context to complete. However, it should not silently discard data either.

### Approach

1. **Sync completes as normal** — active device wins for the bulk transfer
2. **Before overwriting an existing memory**, run semantic contradiction detection against the incoming version
3. **If a conflict is detected**, do not overwrite — instead:
   - Store the incoming version in a `sync_conflicts` staging table
   - Mark both memories as "has pending sync conflict"
4. **After sync completes**, emit a frontend event notifying the user: *"N memories from [device name] may conflict with existing memories — tap to review"*
5. **User resolves at their leisure** using the existing `ConflictResolutionModal` UI — no new UI needed

### What Counts as a Conflict

Not every incoming memory that differs from an existing one is a conflict. The semantic contradiction detection already handles this — it uses embedding similarity plus NLP to determine whether two memories are actually about the same subject and contradict each other, versus simply being different facts.

The existing contradiction detection pipeline (in `lib.rs`) should be called as-is. No new detection logic is needed.

---

## Implementation Plan

### 1. New database table: `sync_conflicts`

```sql
CREATE TABLE sync_conflicts (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    local_memory_id INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    incoming_content TEXT NOT NULL,          -- The incoming version's content
    incoming_device_id TEXT NOT NULL,        -- Which device it came from
    incoming_created_at TIMESTAMPTZ NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE
);
```

### 2. Modify `store_received_memories()` in `zynksync.rs`

Before the final overwrite step (currently around line 1030-1068), add a call to the contradiction detection pipeline:

```
if incoming would overwrite existing:
    run contradiction_check(existing_content, incoming_content)
    if contradiction detected:
        insert into sync_conflicts (don't overwrite)
        mark local memory with pending_sync_conflict = true
        continue to next memory
    else:
        proceed with normal timestamp-based overwrite
```

### 3. Emit frontend notification after sync

In `sync_bidirectional()`, after the sync loop completes, check if any conflicts were staged:

```
if sync_conflicts count > 0 for this user:
    emit "sync_conflicts_pending" event to frontend
    include: count, source device name
```

### 4. Frontend: conflict review flow

The existing `ConflictResolutionModal` handles the resolution UI. The new flow:

- Settings panel shows a badge/indicator when pending sync conflicts exist
- User opens the review flow, sees pairs: [existing memory] vs [incoming from device X]
- Resolves using existing options: keep local / keep incoming / keep both / keep both with explanation
- On resolution, removes the entry from `sync_conflicts` and clears the `pending_sync_conflict` flag

---

## What Does NOT Change

- The sync transport layer (HTTP, inventory comparison, content hashing)
- The "active device wins" model for non-conflicting memories
- The existing `ConflictResolutionModal` UI
- The contradiction detection logic in `lib.rs`
- First-sync additive behavior (no deletions on first sync)

---

## Edge Cases to Consider

**Many conflicts at once:** If a user hasn't synced in a long time and devices have diverged heavily, there could be many conflicts. The review flow should handle batches gracefully, not require one-by-one resolution for every single memory.

**Conflict about a memory that was also deleted:** If device A deleted a memory that device B modified, the deletion should probably win — a delete is an intentional act. Flag this separately.

**Relationship graph consistency:** If memory A on device B has a "contradicts" link to memory C, but memory A was flagged as a sync conflict and not inserted, the relationship cannot be inserted either. Relationships for staged conflict memories should be held in staging until the conflict is resolved.

**Performance:** Running the full contradiction detection pipeline against every incoming memory would be expensive for large syncs. Consider a two-pass approach: fast embedding distance check first (cheap), only run full detection if distance is below a threshold.

---

## Open Questions

- Should the `pending_sync_conflict` flag be visible in the Memory Manager UI?
- Should conflicts automatically expire (resolve in favor of local) after N days if the user never reviews them?
- Should the user be able to opt out of semantic sync conflict detection and revert to pure last-write-wins?
