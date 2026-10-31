# Zynkbot Prompt Construction Pipeline

**File:** `src-tauri/src/conversation_engine.rs`

*Last Updated: 2026-10-25*

---

## Overview

Before each LLM call, Zynkbot assembles a structured prompt from these sources:
1. A largely fixed system prompt (with two dynamic insertions: today's date, and the user's display name when known)
2. Recent conversation history (adaptive length)
3. Retrieved memories (semantic + entity + graph)
4. The user's current message

KB context (if the user clicked Search Knowledge Base) is prepended to this prompt as an additional block.

---

## System Prompt

The system prompt has **two variants** chosen by `is_api_model`:
- **Full** — sent to API backends (Claude/GPT/Grok). ~1,100–1,200 tokens.
- **Slim** — sent to local GGUF backends. ~350 tokens. Preserves every behavior but condenses the voice paragraph and removes redundant examples. Detail in the Token Budget section below.

The structure described below is the **full** version. Slim differs only in verbosity, not in what it asks the LLM to do.

The system prompt is mostly static, with two dynamic insertions. It establishes six things in this order:

**1. Date stamp (dynamic):**
The first line is always `Today's date is <Month Day, Year>.` This anchors the LLM's temporal reasoning so that phrases like "next Tuesday" and "in two weeks" can be interpreted correctly regardless of when the model was trained.

**2. Companion voice (static):**
A single consolidated section of behavioral guidance covering identity (long-term companion, not human, doesn't replace people), honesty (no automatic flattery, correct factual errors gently), emotional presence (acknowledge feelings before solving, no faked emotions), boundaries (no dependency cultivation, reserve professional-help suggestions for clinical/legal stakes), response proportion (answer what was asked), and data ownership (the user owns their data).

This section was consolidated in May 2026 from two earlier sections (COMPANION PRINCIPLES and COMPANION VOICE) that had significant overlap.

**3. Memory access framing (static):**
The LLM is told that the recalled memories that will be appended later were filtered for relevance to the current question — this prevents over-referencing of incidentally-retrieved memories.

**4. Web search signaling (static):**
The LLM is instructed to include a `WEB_SEARCH_NEEDED: [suggested query]` marker in its response if current information is needed (today's date, weather, stock prices, current events, etc.). The backend detects this marker, strips it from the displayed response, and returns the suggested query to the frontend. The frontend displays the query in an **editable input box** — the user can edit it, then click to run the search. The actual search results are only injected into the next round if the user approves.

**5. User name (dynamic, conditional):**
If the user has completed onboarding and BERT NER successfully extracted a PERSON entity from their first onboarding response, the prompt includes a line telling the LLM the user's name and instructing it to use the name naturally in conversation (not as the first word of a reply). The same name is the subject of call 2's extraction prompt (`memory_prompts.rs`). If onboarding hasn't happened or no PERSON entity was extracted, this section is absent entirely and the LLM falls back to the literal word "User" in the examples that follow.

Lookup happens at prompt-build time via `memory::get_user_display_name()`, which queries the user's earliest onboarding-namespace memory and parses its `entities_detected` JSON for PERSON entities.

**6. The MEMORY line (static template, dynamic subject; cloud and custom backends only):**
Since 2026-10-24 the reply no longer carries the fact. The prompt ends with a `MEMORY LINE` block asking the model to end every reply with one final line, exactly `MEMORY: yes` or `MEMORY: no`: yes when the message states a personal fact about the user (a fact stated inside a question counts; the raspberry-pie example is in the prompt), no for a question alone, chit-chat, general knowledge, a request about stored memories, quoted or pasted text, or a message about Zynkbot itself. The line is hidden from the displayed reply and from speech. On yes, a separate background call (`memory_prompts.rs::extraction_prompt_with`) lists the facts, up to three, each with a title and an event date; see `MEMORY_PROCESSING_PIPELINE.md` section 3.3. The third-person subject is the user's name when known, else "User".

A local GGUF model (`on_device` in `conversation_engine.rs`) is not given this block: a 4B model asked for the marker wrote the marker and nothing else. For it, every message that passes the cheap worthiness gate goes to the extraction call, which answers `NONE` when there is nothing.

The older `MEMORY_EXTRACT:` fact line (one compound line per message, May 2026 policy) is gone from the prompt; the parser still accepts one from a model that has the habit and treats it as yes.

---

## Conversation History

**Function:** `build_conversation_context()`

**Adaptive limits:**
- API models (Claude, GPT, Grok): up to 40 messages (20 turns)
- Local GGUF models (desktop or phone): up to 8 messages, each cut to 400 characters (`ON_DEVICE_MESSAGE_CHARS`). The window moves in blocks of four messages (`ON_DEVICE_HISTORY_BLOCK`), not one at a time, so the start of the history stays identical between turns and the engine keeps the prompt prefix it already read; a window that slid by one message every turn threw the whole conversation away each time (2026-10-22, blocks of four since 2026-10-24).

Most recent messages are taken. Format:
```
RECENT CONVERSATION:
USER: [message]
ASSISTANT: [response]
USER: [message]
ASSISTANT: [response]
...
```

The limit difference exists because API models have large context windows and local models are slow and memory-constrained. On Android, local GGUF inference is not available in Phase 1 — the app always uses API-model limits. If your Android device is on the same local network as a desktop running Ollama and paired via ZynkSync, you can use that desktop as an Ollama proxy; when on mobile data or away from the desktop, only cloud API models are available.

---

## Memory Context

**Function:** `build_memory_context()`

**Adaptive limits:**
- API models: up to 20 memories
- Local GGUF models: up to 4 memories (7 until 2026-10-22, 3 for an afternoon on 2026-10-24), the fact only, each cut to 300 characters (`ON_DEVICE_MEMORY_CHARS`): the recalled memories are the part of the prompt a local model reads anew every turn

Memories that are identical to the current user input are filtered out before the limit is applied (prevents the LLM from citing the current message back as a "memory").

**Text per memory (cloud):** the stored fact first, then `(you said: "…")` with the user's own words when they differ from the fact, cut to 200 characters (`memory_line`). The fact goes first because the content may come from a photo, an attachment or an inference, not only from words. **Local models:** the fact alone.

**Grouping by source message (2026-10-24):** since call 2 makes one memory per fact, several recalled memories can come from one user message (same `original_text`). Two or more such facts are written as one block, the message once and the facts as bullets, instead of the same quote repeated per fact:
```
USER'S STORED MEMORIES:
1. (October 24, 2026) You said: "I'm flying to Denver Thursday for my nephew's wedding, and the cat stays with Dave"
   - Matt is flying to Denver on Thursday for his nephew's wedding.
   - Matt's cat stays with Dave while he is away.
2. (October 20, 2026) Matt started a keto diet. (you said: "started keto today, wish me luck")
...
```
The quote is cut to 400 characters (300 on a local model).

Memories are passed in the order returned by hybrid search (entity + semantic weighted score, highest first), with graph-traversal linked memories appended at the end. Entity-matched memories fetched for contradiction detection are intentionally excluded from the prompt — they did not score high enough in hybrid search to be relevant context.

---

## KB Context (Optional)

If the user clicked the KB button, the search runs over their indexed documents (10 chunks, system docs excluded) and `kb_prompt.rs` builds a block that is prepended *before* the rest of the prompt. The wording depends on what the search found, one of three outcomes:

1. **Found** (at least one chunk above the 15% similarity threshold): the matching chunks are listed as `Document N: filename (similarity: xx%)`. The model is told to answer from these passages, to quote the exact row or line it relied on, and, if the passages do not actually contain the answer, to say plainly that the knowledge base has no record of it rather than filling the gap from general knowledge. It is also told not to suggest a web search.
2. **Weak only** (chunks returned, none above the threshold): the five best are listed as `Weak match N`. The model is told that nothing clearly matched, that the weak matches are shown only in case one contains the answer, to answer from one only if it does, and otherwise to say the knowledge base has no record of this. It must not answer from general knowledge as if it came from the documents.
3. **Nothing** (the search returned no chunks): no documents are listed. The model is told to say plainly that the knowledge base has no record of this, not to invent an answer, and to keep any general-knowledge addition short and clearly labelled as not from the documents.

Every block opens with `=== KNOWLEDGE BASE SEARCH (user pressed the KB button) ===` and closes with `=== END OF KNOWLEDGE BASE SEARCH ===`. Only the first outcome counts as grounded: for the other two the chat command returns a `kb_note` that the UI shows in the reply header, and memory extraction is skipped for that reply so an ungrounded answer cannot be stored as a fact (an explicit "Remember:" still stores). <!-- added by Claude 2026-09-09, review -->

---

## Final Assembly

```
[SYSTEM PROMPT]

[RECENT CONVERSATION (if any)]

[USER'S STORED MEMORIES (if any)]

[KB CONTEXT (if requested — prepended before system prompt)]

USER'S QUESTION: [user input]

YOUR RESPONSE:
```

The `YOUR RESPONSE:` suffix is a direct instruction that anchors where the LLM should begin writing, reducing preamble.

---

## Token Budget

| Model Type | History | Memories | System prompt baseline | Total est. tokens |
|---|---|---|---|---|
| API (Claude/GPT/Grok) | 40 messages | 20 memories | ~1,100–1,200 (full) | ~6,500–13,000 |
| Local GGUF | 8 messages (400 chars each) | 4 memories (300 chars each) | ~85 words (on-device) | smaller; the phone reads at roughly 40 tokens/s |

The "system prompt baseline" is the fixed portion that precedes conversation history, memory recall, and KB context. Two variants exist:

- **Full system prompt** (~1,100–1,200 tokens) — sent to API models. Includes the complete COMPANION VOICE section (11 bullets), full WEB SEARCH explanation with two examples, and the MEMORY LINE block (the yes/no marker; the facts themselves are listed by a separate call).

- **On-device prompt** (about 85 words, 2026-10-24) — sent to local GGUF models (`self.on_device`): the companion voice in four sentences, the memories line and the web-search rule, no MEMORY LINE block, no "ask for missing details first" line (a 4B model would question back instead of answering), and the date placed after the conversation so the prompt prefix stays identical between turns. A middle "slim" variant (~350 tokens, with the voice condensed to a paragraph) remains in `build_prompt` for a backend that is neither an API model nor on-device. The behavior contract is identical — same marker formats, same MEMORY_EXTRACT rules, same voice principles — just compressed prose.

Slim mode is selected automatically: `build_prompt()` branches on the `is_api_model` flag. Why this matters: Q4-quantized 3B–7B local models typically expose a 4K-token context window. With the full prompt (1.2k) plus KB context (up to ~1.4k for 10 chunks) plus memory recall (~350) plus conversation history (~400), the prompt alone reaches ~3.3k tokens, leaving under 800 for the LLM's response. Slim mode reclaims ~800 tokens for the response and gives the model substantially more room to reason. Cloud API models have 100K+ context and don't need this trimming.

---

## Relationship to Memory Recall

Prompt construction happens *after* all memory retrieval is complete. The three memory pools (semantic hybrid search results, entity-matched memories, and one-hop graph-traversal memories) are merged before being passed to `build_memory_context()`. The prompt builder does not differentiate between pool sources — it just applies the adaptive limit and filters duplicates.

See [MEMORY_PROCESSING_PIPELINE.md](MEMORY_PROCESSING_PIPELINE.md) for how memories are retrieved before this step.
