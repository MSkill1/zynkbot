# Relationship Detection Architecture

**Last Updated:** 2026-10-25 (call 3 of the rewritten memory pipeline)
**Status:** Production-ready

---

## Overview

Zynkbot uses a **two-tier relationship detection system**:

1. **Primary:** LLM-based relationship classifier (API or local model)
2. **Fallback:** Local LLM if primary API fails
3. **Safety:** Duplicate detection runs independently before the LLM call

---

## Complete Flow

```
User sends message; the reply ends with a hidden MEMORY: yes|no (call 1, cloud)
    ↓
Call 2 (job worker): the facts in the message, up to three (memory_prompts.rs)
    ↓
For each fact:
    Generate embedding of the fact
    ↓
    Hybrid search for similar memories (>35% similarity)
    ↓
    Duplicate Check (pre-LLM)
      • Pure cosine similarity >93% OR hybrid score >98% → skip this fact
    ↓
    Candidates: minus the message's other facts; top 10 (cloud) / top 4 (local)
    No candidates and call 2 gave a title → store, no call 3
    ↓
┌──────────────────────────────────────────────┐
│ CALL 3: LLM RELATIONSHIP CLASSIFIER          │
│ Cloud/custom: ask_llm_for_relationships      │
│   (title + extras + links)                   │
│ Local GGUF: ask_llm_for_links_only           │
│   (links only, LINKS_SCHEMA grammar)         │
│ Classifies: updates, contradicts,            │
│            related, none                     │
│            (four types since 0022, 2026-10-23)│
│ Local answers pass filter_local_links:       │
│   "updates" → "contradicts" (the card asks)  │
└──────────────────────────────────────────────┘
    ↓
Contradiction detected?
    ├─ YES, cloud  → Emit "contradiction-detected" event
    │                Show ConflictResolutionModal to user
    │                Wait for resolve_memory_conflict_v2() decision
    ├─ YES, local  → Store first, raise the card after (resolve_stored_conflict)
    └─ NO          → Store memory + relationships
    ↓
After the last fact: the message's facts linked "related" to each other
("said in the same message", created_by same_message; no model call)
```

---

## Components

### 1. LLM Relationship Classifier (Primary)

**File:** `lib.rs`
**Functions:** `ask_llm_for_relationships()` (cloud and custom backends) and `ask_llm_for_links_only()` (local GGUF models). Both are call 3 of the memory pipeline since 2026-10-24; the earlier single call `ask_llm_about_memory_with_relationships()`, which also decided whether to store, is gone. Whether to store is decided by call 1 (the MEMORY line) and call 2 (the facts); call 3 only classifies.

**Input:**
- One fact from call 2 (or the words after `Remember:`)
- Similar memories (id, content, title, similarity), the message's other facts excluded, at most 10 on the cloud and 4 on a local model
- Backend (anthropic/openai/xai/mistral/custom/local)

**Output (cloud):**
```json
{
  "title": "Memory title",
  "relationships": [
    {
      "memory_id": 123,
      "relationship_type": "contradicts",
      "reason": "Main claims oppose each other",
      "confidence": 0.95
    }
  ]
}
```

**Relationship Types** (reduced to these by migration 0022, 2026-10-23; each one makes the app do something different):
- `updates`: The new fact replaces an older state of the same thing ("moved", "now", "no longer"). The old memory is marked `superseded`, leaves recall and stays in Memory Manager as history with a "Undo replacement" button.
- `contradicts`: Same attribute, different values, with no sign of which came later. The user is asked (conflict card).
- `related`: Same specific subject, adding to it or confirming it. Pulled into recall through one-hop graph traversal. Replaces the old `supports`, `elaborates` and `caused_by`, which behaved identically.
- `none`: Different things. The old `reminds_of` (loosely connected) is gone: it was the one type that did harm in the memory bench, and semantic search already finds loosely related memories at recall time.
- `resolves` is stored too, but only written by the conflict card when the user explains how both memories are true.

**On the cloud** the answer also carries the extras (event date, namespace, tags, tone, entities), used only for fields call 2 left empty. **Local models** answer links only (`LINKS_SCHEMA`: `memory_id`, `relationship_type`, `confidence`), since call 2 already gave the title.

**On-device model:** its `updates` answers are turned into `contradicts` before storage (`filter_local_links`), so the conflict card asks the user instead of the phone's model replacing a memory on its own; the card then reads "Is this an update?" with yes / no / not related / keep both (keep both links as `related`). On the bench Qwen3-4B called four `related` pairs `updates`; the cloud models got all ten real updates right and keep the automatic path. The card's "Keep new memory" has done the `updates` thing since 2026-10-23: the old memory is marked `superseded` and kept as history, never deleted.

**Contradiction Detection Rules:**
1. Focus on MAIN CLAIM, not descriptive phrases
2. High similarity ≠ agreement (opposite claims = contradiction)
3. Ignore negations in descriptive clauses
4. Check for pattern-based contradictions (location, name, age, etc.)

---

### 2. Local LLM Fallback (removed)

The earlier design re-ran the classifier on the local GGUF model when the cloud call returned no relationships. That path went with the single decision call: when call 3 fails on the cloud, the fact is stored without links and with call 2's title (or `generate_title_from_content` when there is none). A local model is never asked to classify a cloud backend's memories.

### 3. Duplicate Detection

**File:** `commands/chat.rs` (`run_memory_background`; the duplicate check against existing memories)
**Trigger:** Runs before call 3 — the fact is skipped if a duplicate is found

**Also, within one message (2026-10-24):** before any of this, call 2's parser drops a restatement of a fact already taken from the same message and `drop_near_duplicates` drops a rewording whose embedding is within cosine 0.86 of an earlier fact.

**Algorithm:**
1. Generate embedding for the new fact
2. Hybrid search for similar memories (>35% similarity)
3. Calculate **pure cosine similarity** (not hybrid score)
4. If cosine similarity > 93% OR hybrid score > 98% → **DUPLICATE**
   - Abort — memory is not stored, no LLM call made
5. If no duplicate → proceed to LLM classification

**Why Pure Cosine in Addition to Hybrid?**
- Hybrid search combines entity (60%) + semantic (40%)
- Identical memories can score ~0.60 with hybrid alone
- Pure cosine gives 0.95–1.0 for true duplicates, making detection reliable

---

### 4. Contradiction Resolution

**File:** `lib.rs`
**Function:** `resolve_memory_conflict_v2()`

When call 3 returns a `contradicts` relationship on a cloud backend, the background task pauses and emits a `contradiction-detected` event to the frontend. On a local model the memory is stored first and the card is raised afterwards about the stored pair (`resolve_stored_conflict`, 2026-10-23), because the phone's background pass can take minutes and a memory held for an unanswered card was lost. The `ConflictResolutionModal` is shown to the user with both memories displayed side by side. The user chooses one of:

- **Keep old** — new memory is discarded
- **Keep new** — old memory is deleted, new memory is stored
- **Keep both** — both stored with a `contradicts` link and optional explanation
- **Keep both (marked contradictory)** — same as above with explicit contradiction flag

`resolve_memory_conflict_v2()` receives the decision and executes accordingly.

---

## Deprecated Components

### Rule-Based Relationship Detector

**File:** `src-tauri/src/relationship_detector.rs`
**Status:** DEPRECATED

The original rules-based relationship detector was replaced by the LLM classifier. The file is preserved for reference but no longer executes in the main pipeline.

---

## Key Design Decisions

### Why three calls?
1. **Call 1 (in the reply):** is there anything to store? One hidden yes/no line, no extra request. The on-device model is not asked (it stops answering when asked for a marker) and goes straight to call 2.
2. **Call 2:** list the facts, with titles and dates. Separate so a message with several facts becomes several focused memories, and so a small model has one job at a time.
3. **Call 3:** classify the links. Separate because the embedding of each fact must exist before the similar memories can be found, and the candidate list depends on what is already stored. Skipped when nothing similar exists.

Matt's design, 2026-10-24; the cloud backends were offered a merged 2+3 and the chart kept them apart.

### Why Trust LLM for Contradictions?
- LLM understands semantic nuance
- Can distinguish main claim from surrounding description
- Pattern matching alone is insufficient for cases like "I believe in Spinoza's God" vs "I don't believe in Spinoza's God"

### Why Keep Duplicate Check Before the LLM?
- Saves an API call on exact or near-exact duplicates
- >93% cosine similarity is deterministic — no LLM judgment needed
- Prevents spam/double-submit scenarios immediately

---

## Performance

| Operation | Speed | Cost | Accuracy |
|---|---|---|---|
| LLM Classifier (API) | Slow (network round-trip) | ~$0.001/call | 95%+ |
| LLM Classifier (Local) | Slow (LLM inference) | Free | 90%+ |
| Duplicate Check | Fast (in-process cosine) | Free | 99%+ |
| ~~Rule-Based (deprecated)~~ | Fast | Free | 70% |

---

## Test Cases

**Contradiction Detection — Spinoza's God**

```
Existing memory: "I believe in 'Spinoza's God' - a pantheistic view..."
New statement:   "I do not believe in Spinoza's God"
```

Expected:
```json
{
  "relationship_type": "contradicts",
  "memory_id": 660,
  "reason": "Main claims oppose each other (belief vs disbelief)",
  "confidence": 0.95
}
```

**Duplicate Detection**

```
Existing memory: "I believe in Spinoza's God"
New statement:   "I believe in Spinoza's God"
```

Expected: New memory aborted before LLM call (>93% cosine similarity).

---

## Code Locations

| Component | File | Function |
|---|---|---|
| LLM Classifier (cloud) | `lib.rs` | `ask_llm_for_relationships()` |
| LLM Classifier (local) | `lib.rs` | `ask_llm_for_links_only()`, `LINKS_SCHEMA`, `filter_local_links()` |
| Schema-constrained call | `lib.rs` | `call_backend_with_schema()`, `call_local_for_memory_decision()` |
| The facts (call 2) | `memory_prompts.rs` | `extraction_prompt_with()`, `parse_extraction()` |
| Job queue | `memory_jobs.rs` | `enqueue()`, `process()` |
| Vector Search | `memory.rs` | `vector_search()` |
| Duplicate Check | `lib.rs` | `pre_check_memory()` |
| Contradiction Resolution | `lib.rs` | `resolve_memory_conflict_v2()` |
| Rule-Based (deprecated) | `relationship_detector.rs` | (unused) |
