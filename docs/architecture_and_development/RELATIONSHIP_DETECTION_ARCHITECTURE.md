# Relationship Detection Architecture

*Checked against the code on 2026-11-10 (`v1`, a7599eb).*

How Zynkbot decides how a new memory relates to the memories already stored. File names are relative to `zynkbot_rust/src-tauri/src/` unless a longer path is given. For the tables (`memories`, `memory_links`, `memory_entities`, `memory_cards`) see [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md).

---

## Overview

- **One classifier per backend.** Links are made by call 3 of the memory pipeline, a single LLM call on the backend the user is chatting with. There is no second model as a fallback. When the call fails, the fact is stored without links.
- **A duplicate check runs before call 3.** It uses embeddings only, no model.
- **Facts from the same message are linked `related` to each other** without a model call.
- **Four link types are stored:** `related`, `contradicts`, `updates`, `resolves` (migration `0022_four_relationship_types.sql`).

---

## Complete Flow

```
User sends a message
    ↓
Call 1: is there anything to store?
    Cloud and custom backends: the reply ends with a hidden "MEMORY: yes" or "MEMORY: no" line
    On-device model: no marker is asked for; call 2 decides
    "Remember: ..." skips calls 1 and 2; the words are stored as said
    ↓
A row is queued in memory_jobs; one worker runs the rows in order
(memory_jobs.rs: enqueue(), process())
    ↓
Call 2: the facts in the message, at most three
(commands/chat.rs extract_facts(); prompt and parser in memory_prompts.rs)
    ↓
For each fact (commands/chat.rs run_memory_background()):
    1. Embed the fact text
    2. Hybrid search: up to 15 memories; keep those at 35% similarity or more
    3. Sort by similarity, highest first
    4. Drop the facts already stored from this same message
    5. Name correction only: memories that hold the wrong name go to the front
    6. Keep the top 10 (cloud backend) or top 4 (on-device model, custom endpoint)
    7. Duplicate check on the candidates that are left:
       hybrid score above 98% or pure cosine above 93% → skip the fact
    8. Call 3, by case:
       a. Call 2 gave a title and no candidates are left → store, no call 3
       b. On-device model, title, candidates → store first, then the links-only call
       c. Any other case → the full call 3 runs before the fact is stored
    9. A "contradicts" link with confidence 0.65 or more raises the conflict card:
       cloud or custom → the fact is NOT stored; the card asks first
       on-device       → the fact is already stored or is stored now; the card asks after
    10. Store the memory (if not held for a card) and its links
        "updates" → the older memory is marked superseded
    ↓
After the last fact: the facts stored from this message are linked "related"
to each other (created_by same_message; no model call)
```

---

## Components

### 1. Call 3: the classifier

**File:** `lib.rs`
**Functions:** `ask_llm_for_relationships()` and `ask_llm_for_links_only()`
**Called from:** `run_memory_background()` in `commands/chat.rs`

Whether to store is decided by call 1 (the MEMORY line) and call 2 (the facts). Call 3 only classifies.

**Which function runs**

| Case | What runs |
|---|---|
| Call 2 gave a title and no candidate memories are left | No call 3. The fact is stored with call 2's title and fields. |
| On-device model, call 2 gave a title, candidates exist | The fact is stored first. Then `ask_llm_for_links_only()` runs on the stored memory. |
| Any other case: a cloud or custom backend with candidates, or any fact without a title (in practice a "Remember:" fact) | `ask_llm_for_relationships()` runs before the fact is stored. |

"On-device" here means the backend name ends in `.gguf` or contains `local`. A custom endpoint is not treated as on-device: it uses the full call and the ask-first card, and its links are tagged `llm`. It does get the smaller candidate list (see section 2).

**`ask_llm_for_relationships()`, the full call**

Input:
- One fact from call 2, or the words after `Remember:`
- The candidate memories (id, content, title, similarity)
- The backend name (anthropic, openai, xai, mistral, custom, or a local model)

The answer carries a title, the relationships, and the extra fields (event date, namespace, tags, tone, entities):

```json
{
  "title": "Memory title",
  "relationships": [
    {
      "memory_id": 123,
      "relationship_type": "contradicts",
      "reason": "Brief explanation",
      "confidence": 0.95
    }
  ],
  "event_date": null,
  "namespace": "personal",
  "tags": ["tag1"],
  "tone": "neutral",
  "entities": [{"name": "Vermont", "kind": "place"}]
}
```

The title and the extra fields are used only where call 2 left them empty (`merge_extras()` in `commands/chat.rs`). When a local model runs this call, its output is held to `RELATIONSHIP_SCHEMA` by a grammar.

**`ask_llm_for_links_only()`, the on-device call after storage**

The prompt comes from `links_only_prompt()`. The answer is held to `LINKS_SCHEMA`: `memory_id`, `relationship_type`, and an optional `confidence`. There is no title, no reason and no extra fields, because call 2 already gave them. In the app this function is called only on the store-first path. The memory bench (`memory_bench.rs`) can also call it when `MEMORY_BENCH_SLIM` is set.

**When call 3 fails**

- Full call: the fact is stored without links. The title is call 2's, or `generate_title_from_content()` when there is none.
- Links-only call: the memory is already stored and stays as it is, without links.

**Relationship types**

The prompt tells the model to pick the first type that fits, in this order:

- `updates`: the new fact replaces an older state of the same thing ("moved", "now", "no longer"), or both are short-lived facts of the same kind that cannot both hold now (where the car is parked).
- `contradicts`: same attribute, different values, with no sign of which came later. Or opposite claims on the same topic.
- `related`: the same specific person, pet, place, thing or habit, adding to it or confirming it.
- `none`: different things. Not stored.

The prompt gives concrete examples for each type. It adds one warning: high semantic similarity does not mean agreement, and opposite claims are a contradiction. It has no other contradiction rules.

`resolves` is the fourth stored type. The classifier never returns it. It is written only by the conflict card when the user explains how both memories are true.

Migration 0022 folded the earlier `supports`, `elaborates` and `caused_by` into `related`, and dropped `reminds_of`, `quotes` and `mentions`.

**Lenient parsing**

- A markdown code fence around the JSON is removed. Otherwise the first complete JSON object in the answer is taken (`first_json_object()`), so text after it does not break the parse.
- A `memory_id` given as a string becomes -1 and the entry is dropped (`deserialize_memory_id()`).
- `reason` is optional.
- Entries with type `none` or a `memory_id` of 0 or less are dropped.
- `confidence` is optional. A link without one is stored at 0.75. The conflict card needs a stated confidence of 0.65 or more, so a `contradicts` answer with no confidence is stored as a link but raises no card.

**On-device answers: `filter_local_links()`**

Every on-device answer passes through `filter_local_links()` in `lib.rs` before storage:

- `updates` is turned into `contradicts`, and the reason gets the prefix "Possibly an update: ". The conflict card then asks "Is this an update?" instead of the on-device model replacing a memory on its own.
- A confidence floor (`LOCAL_LINK_CONFIDENCE_FLOOR`) and a dropped-types list (`LOCAL_DROPPED_LINK_TYPES`) are applied. Both are inert today: the floor is 0.0 and the list is empty.

The code comment on `filter_local_links()` gives the reason: on the memory bench Qwen3-4B called four `related` pairs `updates`, and the cloud models got the real updates right. The bench cases are in `labs/memory_bench/cases.json`. The result files are not in the repo, so these figures are not verified against the code.

---

### 2. Candidate memories

**File:** `commands/chat.rs`, `run_memory_background()`

1. `hybrid_search()` in `memory.rs` returns up to 15 of the user's memories, across all sessions and namespaces. It leaves out superseded, completed and expired memories.
2. Results under 35% similarity are dropped.
3. The rest are sorted by similarity, highest first.
4. Facts already stored from the same message are removed, so call 3 does not compare a message with itself.
5. **Name correction.** When the message corrects a name ("it's Marta, not Maria"; `detect_name_correction()` in `memory_prompts.rs`), up to 10 current memories whose text contains the wrong name are put at the front of the list, with a similarity of 1.0. `process()` in `memory_jobs.rs` also adds a fact of its own for the correction if call 2 did not produce one.
6. The list is cut to 10 on a cloud backend and 4 otherwise (`ConversationEngine::is_api_model()` in `conversation_engine.rs`). "Otherwise" covers an on-device model and a custom endpoint.

---

### 3. Duplicate detection

**File:** `commands/chat.rs`, inline in `run_memory_background()`
**When:** after the candidate list is cut, before call 3

**Algorithm**

For each candidate that is left:
1. If its hybrid score is above 0.98, the new fact is a duplicate.
2. Otherwise the new fact's embedding is compared with the candidate's stored embedding. If the cosine similarity is above 0.93, the new fact is a duplicate.

A duplicate is not stored and no call 3 is made.

Only the candidates that reach call 3 are checked (10 or 4), not all 15 search results.

**Why pure cosine as well as the hybrid score?**

`hybrid_search()` scores a memory as 60% name overlap plus 40% semantic similarity when names match, and as semantic similarity alone otherwise. So the hybrid score is not a direct measure of how alike two texts are. The cosine check compares the two embeddings directly.

**Within one message**

Two earlier steps remove repeats among the facts of a single message:
- `parse_extraction()` in `memory_prompts.rs` drops a fact that restates one already taken (`restates()`).
- `drop_near_duplicates()` in `commands/chat.rs` drops a rewording whose embedding is within cosine 0.86 of an earlier fact.

**Known issue, from reading the code (not confirmed by running the app)**

A name-correction candidate is given a similarity of 1.0 (section 2, step 5). The duplicate check that follows treats any score above 0.98 as a duplicate. As written, a fact from a name-correction message is skipped as a duplicate whenever a memory holding the wrong name is in the list, and call 3 never sees it.

**Unused command**

`pre_check_memory()` in `commands/memory.rs` is a registered command that runs a similar check. Nothing in the pipeline or the frontend calls it.

---

### 4. Storing links

**Files:** `commands/chat.rs` (`run_memory_background()`), `memory.rs` (`create_memory_link()`), `memory_jobs.rs` (`process()`)

- A link points from the new memory to the older one. `create_memory_link()` rejects any type outside the four stored types.
- **`updates`:** the older memory is marked `superseded` (`set_temporal_status()`) and the event `memory-superseded` is emitted. This happens automatically when a cloud or custom backend answers `updates`. The older memory leaves recall and stays in Memory Manager as history, with an "Undo replacement" button (`restore_memory()` in `commands/memory.rs`).
- **Same message:** after the last fact, `process()` links every pair of facts stored from the message as `related`, confidence 1.0, note "said in the same message". A fact held for an ask-first card is not stored yet, so it gets no such link.

**Who made a link (`created_by`)**

| Value | Written by |
|---|---|
| `llm` | Call 3 on a cloud or custom backend |
| `local_llm` | Call 3 on an on-device model |
| `same_message` | The same-message links |
| `user` | A choice on the conflict card, a `resolves` explanation, or the `create_memory_link` command |
| `system` | The column default; also the Einstein demo set (`einstein_demo.rs`) |

**Where links go next**

- At recall, `generate_reply()` in `commands/chat.rs` follows links one hop from each recalled memory, whatever the type, strongest first, up to 8 linked memories. Superseded memories are skipped.
- ZynkSync sends links to paired devices.
- The encrypted cloud backup (`commands/backup.rs`) carries links. On restore they are matched to memories by a hash of the memory text (`restore_links_into()`).

---

### 5. The conflict card

**Files:** `commands/chat.rs` (raising the card), `commands/memory.rs` (`resolve_memory_conflict_v2()`, `resolve_stored_conflict()`), `zynkbot_rust/src/components/ConflictResolutionModal.jsx`, `zynkbot_rust/src/App.jsx`

A card is raised when call 3 returns a `contradicts` link with a confidence of 0.65 or more. There are two paths.

**Ask first (cloud and custom backends).** The new fact is not stored. The card's payload carries the pending memory, both texts and the links call 3 returned. The background task ends there. When the user answers, `resolve_memory_conflict_v2()` stores the memory or drops it.

**Store first (on-device model).** The memory is stored like any other, with its `contradicts` link. The card is raised afterwards about the stored pair. When the user answers, `resolve_stored_conflict()` changes the stored rows. On the main on-device path the card follows the links-only call, which itself runs after the memory is stored.

**"Remember:" is never held.** When the user said "Remember: ...", the fact is stored even if call 3 finds a contradiction. The `contradicts` link is stored and no card is raised.

**The card is a row.** Each card is written to `memory_cards` (migration `0024_memory_cards.sql`, `insert_memory_card()` in `memory.rs`) before the `contradiction-detected` event is sent. Unanswered cards are shown again when the app opens (`list_pending_memory_cards()`), and `answer_memory_card()` records the decision.

**The choices**

The card offers five choices.

| Choice | Ask-first card | Store-first card |
|---|---|---|
| Keep old | The new fact is not stored. | The new memory is deleted. |
| Keep new | The new fact is stored. The old memory is marked `superseded` and linked `updates`. | The old memory is marked `superseded`. The link becomes `updates`. |
| Not a contradiction | The new fact is stored with no link to the old memory. | The link between the two is removed. |
| Keep both ("Accept the contradiction") | The new fact is stored and linked `contradicts`. | Nothing changes; the `contradicts` link stays. |
| Resolve with explanation | The new fact is stored and linked `contradicts`. The explanation is stored as a memory and linked `resolves` to both. | The explanation is stored as a memory and linked `resolves` to both. |

"Keep new" never deletes the old memory. It is kept as history and can be brought back with "Undo replacement".

On the ask-first card, the other links call 3 returned for the new fact are created when the fact is stored.

**The "Is this an update?" form.** When the reason starts with "Possibly an update" (an on-device `updates` answer, see section 1), the card is worded as a question about an update: yes, no, not related, keep both, or explain. "Keep both" on this form links the two memories as `related`, not `contradicts`.

---

## Removed

- **Local LLM fallback.** An earlier design re-ran the classifier on a local model when the cloud call returned nothing. That path is gone. A local model is never asked to classify for a cloud backend.
- **The single decision call.** `ask_llm_about_memory_with_relationships()`, which decided whether to store and classified in one call, no longer exists.
- **Rule-based detector.** `relationship_detector.rs` no longer exists.

---

## Key Design Decisions

### Why three calls?
1. **Call 1 (in the reply):** is there anything to store? One hidden yes/no line, no extra request. The on-device model is not asked for the line and goes straight to call 2.
2. **Call 2:** list the facts, with titles and dates. Separate so a message with several facts becomes several memories, and so a small model has one job at a time.
3. **Call 3:** classify the links. Separate because the embedding of each fact must exist before similar memories can be found, and the candidate list depends on what is already stored. Skipped when nothing similar exists and call 2 gave a title.

### Why store first on the device?
The on-device calls are slow. Storing the fact before the links call means a question asked soon after the statement can find it. Asking after storage also means a memory is not lost when a card goes unanswered. Both reasons are given in the comments in `run_memory_background()`.

### Why an LLM for contradictions?
Two statements can be very similar as text and still disagree. Embedding similarity cannot tell agreement from disagreement, so a model reads both.

### Why keep the duplicate check before the LLM?
- It saves a model call on exact or near-exact repeats.
- It needs no model judgment: it is two threshold comparisons.

---

## Performance and accuracy

This document used to carry a table of speed, cost and accuracy figures. They could not be verified against the code and have been removed. Measured results are in [BENCHMARKS.md](../BENCHMARKS.md).

---

## Tests

The earlier "Spinoza's God" example cases have been removed: no test in the repo matches them. Tests that do exist:

| What | Where |
|---|---|
| First JSON object is taken, a second is ignored | `lib.rs`, `first_json_object_tests` |
| On-device `updates` becomes a question; floor and dropped types | `lib.rs`, `link_floor_tests` |
| Restatements and name corrections in call 2's parser | `memory_prompts.rs`, `tests` |
| Two facts from one message are linked `related` | `chat_harness_tests.rs`, `c09_two_facts_from_one_message_are_linked_as_related` |
| Migration 0022 collapses the old types | `sync_outbox_tests.rs`, `migration_0022_collapses_relationship_types` |
| Links come back from a cloud backup | `commands/backup.rs`, `restore_puts_links_back_between_the_restored_memories` |
| Classifier quality per backend (run by hand, not in the normal test run) | `memory_bench.rs` with `labs/memory_bench/cases.json` |

---

## Code Locations

| Component | File | Function |
|---|---|---|
| Per-fact pipeline | `commands/chat.rs` | `run_memory_background()` |
| Call 3, full | `lib.rs` | `ask_llm_for_relationships()`, `RELATIONSHIP_SCHEMA` |
| Call 3, links only | `lib.rs` | `ask_llm_for_links_only()`, `links_only_prompt()`, `LINKS_SCHEMA` |
| On-device filter | `lib.rs` | `filter_local_links()`, `apply_link_floor()` |
| Lenient parsing | `lib.rs` | `first_json_object()`, `deserialize_memory_id()` |
| Schema-constrained call | `lib.rs` | `call_backend_with_schema()`, `call_local_for_memory_decision()` |
| The facts (call 2) | `commands/chat.rs`, `memory_prompts.rs` | `extract_facts()`, `extraction_prompt_with()`, `parse_extraction()` |
| Name correction | `memory_prompts.rs` | `detect_name_correction()` |
| Job queue, same-message links | `memory_jobs.rs` | `enqueue()`, `process()` |
| Candidate search | `memory.rs` | `hybrid_search()` |
| Candidate limit | `conversation_engine.rs` | `ConversationEngine::is_api_model()` |
| Duplicate check | `commands/chat.rs` | inline in `run_memory_background()`; `drop_near_duplicates()` |
| Link storage | `memory.rs` | `create_memory_link()`, `set_temporal_status()` |
| Card storage | `memory.rs`, `commands/memory.rs` | `insert_memory_card()`, `list_pending_memory_cards()`, `answer_memory_card()` |
| Card answers | `commands/memory.rs` | `resolve_memory_conflict_v2()`, `resolve_stored_conflict()` |
| Undo a replacement | `commands/memory.rs` | `restore_memory()` |
| Recall through links | `commands/chat.rs` | `generate_reply()` |
