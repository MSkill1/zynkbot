# Relationship Detection Architecture

**Last Updated:** April 2026
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
User sends message
    ↓
Extract factual statements
    ↓
Generate embedding
    ↓
Vector search for similar memories (>35% similarity, top 15)
    ↓
Duplicate Check (pre-LLM)
    • Pure cosine similarity >93% OR hybrid score >98% → abort, don't store
    ↓
┌──────────────────────────────────────────────┐
│ LLM RELATIONSHIP CLASSIFIER (Primary)        │
│ Backend: anthropic/openai/xai/local         │
│ Classifies: contradicts, supports,          │
│            elaborates, caused_by,           │
│            reminds_of, none                 │
└──────────────────────────────────────────────┘
    ↓
Did LLM return relationships?
    ├─ YES → Check for contradictions
    └─ NO  → ┌──────────────────────────────────────┐
             │ LOCAL LLM FALLBACK                   │
             │ Backend: local GGUF model            │
             │ Same classification as above         │
             └──────────────────────────────────────┘
    ↓
Contradiction detected?
    ├─ YES → Emit "contradiction-detected" event
    │        Show ConflictResolutionModal to user
    │        Wait for resolve_memory_conflict_v2() decision
    └─ NO  → Store memory + relationships
```

---

## Components

### 1. LLM Relationship Classifier (Primary)

**File:** `lib.rs`
**Function:** `ask_llm_about_memory_with_relationships()`

**Input:**
- User message
- Conversation history
- Similar memories (id, content, title, similarity)
- Backend (anthropic/openai/xai/local)

**Output:**
```json
{
  "should_remember": true,
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

**Relationship Types:**
- `contradicts`: Direct contradiction (e.g., "I believe X" vs "I don't believe X")
- `supports`: Reinforces existing memory
- `elaborates`: Adds detail to existing memory
- `caused_by`: Causal relationship
- `reminds_of`: Loosely related
- `none`: No meaningful relationship

**Contradiction Detection Rules:**
1. Focus on MAIN CLAIM, not descriptive phrases
2. High similarity ≠ agreement (opposite claims = contradiction)
3. Ignore negations in descriptive clauses
4. Check for pattern-based contradictions (location, name, age, etc.)

---

### 2. Local LLM Fallback

**File:** `src-tauri/src/lib.rs` (inline in memory pipeline)
**Trigger:** When primary LLM returns no relationships (`llm_relationships.is_empty()`)

**Why?**
- API might be down
- Network issues
- Rate limiting
- User prefers local-only mode

**How?**
- Calls `ask_llm_about_memory_with_relationships()` with `backend: "local"`
- Routes through `call_local_for_memory_decision()` (`lib.rs`)
- Uses installed GGUF model (e.g., Llama-3.2-3B-Instruct)
- Marks relationships with `created_by: "local-llm"`
- Slightly lower default confidence (0.70 vs 0.75)

---

### 3. Duplicate Detection

**File:** `lib.rs`
**Function:** `pre_check_memory()`
**Trigger:** Runs before the LLM call — aborts early if duplicate found

**Algorithm:**
1. Generate embedding for new memory
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

When the LLM detects a `contradicts` relationship, the background task pauses and emits a `contradiction-detected` event to the frontend. The `ConflictResolutionModal` is shown to the user with both memories displayed side by side. The user chooses one of:

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

### Why Two API Calls?
1. **First call:** Decide if memory is worth keeping + get title
2. **Second call:** Classify relationships with similar memories

Why not one call? The embedding must be generated before searching for similar memories, and the similar memories list depends on what's already stored.

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
| LLM Classifier | `lib.rs` | `ask_llm_about_memory_with_relationships()` |
| Local LLM Router | `lib.rs` | `call_local_for_memory_decision()` |
| Local Fallback (inline) | `lib.rs` | (in memory pipeline) |
| Vector Search | `memory.rs` | `vector_search()` |
| Duplicate Check | `lib.rs` | `pre_check_memory()` |
| Contradiction Resolution | `lib.rs` | `resolve_memory_conflict_v2()` |
| Rule-Based (deprecated) | `relationship_detector.rs` | (unused) |
