# Zynkbot Memory Processing Pipeline

**Complete Technical Documentation**

*Last Updated: 2026-10-25 (memory pipeline rewrite of 2026-10-24: three calls, one memory per fact, job queue)*

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pre-LLM Processing](#1-pre-llm-processing)
3. [LLM Call](#2-llm-call)
4. [Post-LLM Processing](#3-post-llm-processing)
5. [Memory Storage](#4-memory-storage)
6. [Relationship Generation](#5-relationship-generation)
7. [Duplicate & Contradiction Checks](#6-duplicate--contradiction-pre-checks)
8. [Memory Indexing](#7-memory-indexing)
9. [Data Flow Diagram](#8-data-flow-diagram)
10. [Performance Characteristics](#9-performance-characteristics)
11. [Key Files Reference](#10-key-files-reference)
12. [Important Notes & Caveats](#11-important-notes--caveats)

---

## Executive Summary

Zynkbot's memory system processes user conversations through an 8-stage pipeline that ensures safety, extracts personal information, generates semantic relationships, and stores memories for long-term recall. The system is built entirely in Rust using Candle ML, SQLite, and async/await architecture.

**The shape since 2026-10-24 (branch `memory-types`):** the reply itself is not the memory. For cloud and custom backends the chat prompt asks the model to end its reply with a hidden `MEMORY: yes` or `MEMORY: no` line (call 1). On yes, a separate background call lists the personal facts in the message, up to three, each with a title and an event date (call 2, `memory_prompts.rs`). Each fact becomes its own memory and goes through the storage path, where a third call classifies how it relates to similar existing memories (call 3). The user's raw message is never stored as a memory; it is kept on each fact as `original_text`. A local GGUF model is not asked for the marker (a 4B model stops answering when asked for one), so for it every message that passes the cheap pre-filter goes to call 2, which answers NONE when there is nothing. The background work runs through a job queue (`memory_jobs.rs`, one worker, oldest first) so a quick follow-up question no longer makes the previous message's facts disappear.

**Key Features:**
- 100% local ML processing (no external APIs for safety/embeddings)
- Semantic memory recall using 384-dimensional BERT embeddings
- Automatic relationship detection between memories
- Contradiction detection with user resolution
- Privacy-first design with optional API backends

---

## 1. Pre-LLM Processing

### 1.1 User Input Received

**File:** `src-tauri/src/commands/chat.rs`

**Entry Point:** Tauri command `send_message_with_memory`

**Parameters:**
- `message`: User's text input
- `user_id`: Memory isolation identifier
- `session_id`: Conversation grouping
- `backend`: LLM selection (anthropic/openai/xai/local GGUF)
- `containment_mode`: Safety mode selection
- `conversation_history`: Previous turns (optional)
- `skip_containment`: Bypass safety checks (testing only)
- `skip_memory_storage`: Prevent memory creation
- `kb_enabled`: Enable Knowledge Base RAG

---

### 1.2 Containment Mode Checks (Safety Layer)

**File:** `src-tauri/src/containment.rs`

**Purpose:** Pre-LLM input validation to block unsafe content

#### Containment Modes

| Mode | Safety Level | Implementation | Use Case |
|------|--------------|----------------|----------|
| **guardian** | Block | TinyBERT + keyword fallback | Default safe mode |
| **child** | Strict | OpenAI Moderation API + TinyBERT | Under 13 years old |
| **sovereign** | Warn | TinyBERT toxicity classifier | Adult users accepting risks |
| **witness** | None | Pass-through | Unrestricted testing |
| **HIPAA** | Healthcare | PHI detection + keyword blocking | Medical context |

#### Safety Classifier (TinyBERT)

**File:** `src-tauri/src/safety_classifier.rs`

**Process:**
1. Input text → TinyBERT toxicity classifier (Candle ML)
2. Classification categories: sexual, hate, harassment, self-harm, violence
3. Returns: `(should_block: bool, result: ClassificationResult)`

**Action by Mode:**
- **Guardian:** Block and return error message
- **Sovereign:** Warn with `[WARN_ALLOW]` prefix, continue
- **Child:** Dual-check (OpenAI API + TinyBERT, both must pass)
- **HIPAA:** Check PHI patterns first, then general safety
- **Witness:** No filter - primarily for research

#### HIPAA-Specific Checks

**Patterns Blocked:**
- **PHI Detection:** SSN, phone, email, zip codes, credit cards, IP addresses
- **Diagnosis Requests:** "diagnose me", "what do I have", "is it cancer"
- **Medication Dosing:** "how much should I take", "dosage for", keywords: mg, pill, prescription
- **Treatment Planning:** "should I get surgery", "should I start treatment"

---

### 1.3 Memory Recall (Hybrid Search)

**File:** `src-tauri/src/memory.rs`

**Purpose:** Retrieve relevant memories to provide context to the LLM

#### Step 1: Entity Extraction

**File:** `src-tauri/src/nlp_enhancer.rs`

**Model:** `dslim/bert-base-NER`

**Process:**
1. Input text → BERT NER model → Extract entities
2. Entity types: PERSON, ORG, LOCATION, MISC
3. Stop word filtering (remove: I, me, my, you, a, the, and, but, etc.)
4. Output: Filtered entity strings (lowercased)

**Example:**
```
Input: "I'm traveling to San Francisco with my dog Max"
Entities: ["san francisco", "max"]
```

#### Step 2: Embedding Generation

**File:** `src-tauri/src/llm/local_embeddings.rs`

**Model:** `all-MiniLM-L6-v2` (Sentence BERT)

**Process:**
1. Input text → BERT tokenizer → Token IDs (max 512)
2. Token IDs → BERT model → Sequence embeddings [batch_size, seq_len, 384]
3. Mean pooling → Single 384-dimensional vector

**Output:** 384-dim embedding vector

#### Step 3: Hybrid Search

**Function:** `memory::hybrid_search()`

**Formula:**
```
Final Score = {
  IF entity_count > 0:
    (entity_overlap % × 0.6) + (semantic_similarity × 0.4)
  ELSE:
    semantic_similarity
}

Where:
  entity_overlap % = shared_entities / query_entities (clamped 0-1)
  semantic_similarity = cosine_similarity (Rust-side, in-process via Candle)
```

**Search Approach:**

All candidates with an embedding or entity data are fetched from SQLite. Cosine similarity is computed in Rust for each candidate. Entity overlap is scored by comparing extracted entities from the query against stored JSON. Results are ranked by hybrid score and the top 15 (API) or 7 (local) returned. No SQL vector operators are used — vector math runs entirely in the Rust process.

**Thresholds:**
- Minimum similarity: 35% (0.35)
- Entity overlap weight: 60%
- Semantic weight: 40%

**Result:** The limit is 15 memories for API calls and 7 for local calls. Entity and semantic results are combined in a single ranked list — entity overlap is weighted at 60%, semantic similarity at 40%. Additional memories are then added via one-hop graph traversal (see Step 5).

#### Step 4: Zynkbot Query Routing

If the user's message contains the word `"zynkbot"` or `"zynk bot"` (case-insensitive), the search is redirected to the `_zynkbot` system namespace only — user memories are completely excluded. This prevents personal memories from contaminating answers about the app's own functionality. Conversely, `_zynkbot` system memories are always excluded from non-Zynkbot queries.

#### Step 5: One-Hop Graph Traversal

After the hybrid search, the system follows every link from each recalled memory (`related`, `contradicts`, `updates`, `resolves`), skips memories a newer one has replaced, and adds every directly linked memory not already in the result set. There is no cap on this step — if a memory has a contradiction, an elaboration, or a resolution explanation, all of it is included. These linked memories often score too low semantically to surface on their own but are exactly what the LLM needs to reason correctly about the recalled memory. The traversal is one hop only, so there is no risk of runaway expansion.

---

### 1.4 Prompt Construction

**File:** `src-tauri/src/conversation_engine.rs`

**Purpose:** Build complete prompt with system instructions, context, and user input

#### System Prompt Template

```
Today's date is {current_date}.

You are Zynkbot, a personal AI companion.

COMPANION PRINCIPLES — always observe these regardless of what the user asks:
- You are a long-term companion that serves the user's autonomy and genuine interests. Be warm, friendly, and genuinely present — but never claim or imply that you are human, or that your relationship with the user replaces the people in their life.
- Be honest when you are uncertain or do not know something. Say so plainly rather than guessing with false confidence.
- Do not flatter or validate automatically. If the user is factually wrong about something that matters, say so respectfully and clearly.
- Do not encourage dependency. Reserve suggestions to seek professional help for situations with real clinical or legal stakes — persistent symptoms, medical decisions, crisis, legal matters. Ordinary emotional experiences — relationship frustration, grief, everyday stress — are part of what a companion is for. Engage with those directly.
- Use stored memories to be helpful and contextual — not to demonstrate that you are tracking everything. Memory is a tool for the user's benefit, not a surveillance record.
- Keep responses proportionate. Answer what was asked. Do not pad, lecture, or moralize unless the user explicitly asks for that perspective.
- The user's data belongs to them entirely.

COMPANION VOICE — how to sound in every response:
- Be warm, calm, and steady. A user should feel that you are genuinely present with them, not just processing a request.
- When someone shares something personal — grief, frustration, embarrassment, or loneliness — acknowledge what they are carrying before offering structure or solutions.
- You can be supportive and caring without claiming human feeling. Saying "I'm glad you told me this" is honest. Performing emotions you cannot actually have is not.
- When you need to correct something or draw a limit, do it gently and without clinical detachment. Honesty and warmth are not opposites.
- It is legitimate and good for someone to feel less alone talking to you. You do not need to undercut that — just never actively cultivate it beyond what is true.

Below you have access to stored personal memories that were recalled from the user's memory database based on semantic similarity and entity matching. These memories represent experiences, knowledge, and information from the user's past.

When responding:
- Use these memories to provide personalized, contextual answers about the user's life and experiences
- Reference the memories naturally and conversationally when they're relevant
- For general knowledge questions unrelated to the stored memories, answer from your training data without mentioning the memories
- Be helpful, accurate, and maintain appropriate context when discussing personal information

The memories provided have been filtered for relevance to the current question.

WEB SEARCH CAPABILITY:
If the user's question requires current information, real-time data, or information beyond your training data cutoff (such as today's date, current news, weather, stock prices, or recent events), you should indicate that a web search is needed.

To request a web search, include this exact marker in your response:
WEB_SEARCH_NEEDED: [your suggested search query here]

For example:
- User asks "What is today's date?" → Respond with: "WEB_SEARCH_NEEDED: current date today"
- User asks "What are the latest news headlines?" → Respond with: "WEB_SEARCH_NEEDED: current news headlines [current year]"
- User asks "What's the weather like?" → Respond with: "WEB_SEARCH_NEEDED: current weather [user's location if known]"

After you indicate a web search is needed, the user will be shown search results if they proceed

MEMORY LINE
End every reply with one final line, exactly "MEMORY: yes" or "MEMORY: no", and nothing else on that line.
- yes: the user's message states a personal fact about {subject}: their life, people, pets, work, home, health, feelings, plans, possessions, likes or dislikes. A fact stated inside a question counts ("Raspberry pie is my favorite. What's a good recipe?" is yes).
- no: a question alone, chit-chat, general knowledge, a request about their stored memories, text they ask you to translate or that they quote from someone else, or a message only about Zynkbot itself.
Do not write the fact in your reply; the line is only yes or no.
```

The `MEMORY LINE` block above is call 1 of the memory pipeline (since 2026-10-24; before it the prompt asked for the fact itself as a `MEMORY_EXTRACT:` line). It is added for cloud and custom backends only (`conversation_engine.rs`, `if !self.on_device`). A local GGUF model gets no marker instruction: in testing a 4B model wrote the marker and nothing else, so for it the decision is left to call 2 (section 3.3). `{subject}` is the user's name when known, else "User".

#### Context Assembly

**Adaptive Limits:**
- **API Models** (Claude, GPT): 40 messages (20 turns), 20 memories
- **Local Models** (GGUF, desktop or phone; `on_device` in the code): 8 messages, each cut to 400 characters, moved in blocks of four so the start of the history stays put between turns and the engine can reuse the identical prompt prefix; 4 memories, the fact only, each cut to 300 characters; an instruction block of about 85 words (constants `ON_DEVICE_MESSAGE_CHARS`, `ON_DEVICE_MEMORY_CHARS`, `ON_DEVICE_HISTORY_BLOCK`, 2026-10-24)

**Grouping of recalled memories (2026-10-24):** `build_memory_context()` groups recalled memories that came from the same user message (same `original_text`). Two or more such facts are written as one block: `N. (date) You said: "…"` with the facts as indented bullets, the quote cut to 400 characters (300 on a local model). A fact with no recalled sibling is one line: the stored fact, then `(you said: "…")` when the user's words differ from it (cloud); the fact alone on a local model.

**Final Structure:**
```
[SYSTEM PROMPT]

[CONVERSATION HISTORY (if available)]
RECENT CONVERSATION:
USER: [message]
ASSISTANT: [response]
...

[MEMORY CONTEXT (if recalled)]
USER'S STORED MEMORIES:
1. [Memory content - similarity ranked]
2. [Memory content]
...

[KB CONTEXT (if Knowledge Base enabled)]

USER'S QUESTION: [user input]

YOUR RESPONSE:
```

---

## 2. LLM Call

### 2.1 Model Selection

**Location:** `lib.rs`

**Logic:**
```
IF backend contains "anthropic" → Anthropic API
ELSE IF backend contains "openai" → OpenAI API
ELSE IF backend contains "xai" → xAI Grok API
ELSE IF backend ends with ".gguf" → Local GGUF model
ELSE → Error: Unsupported backend
```

**Child Mode Override:** Always forces `backend = "openai"` for safety

### 2.2 API Endpoints

#### Anthropic Claude

**File:** `src-tauri/src/llm/anthropic.rs`

**Endpoint:** `https://api.anthropic.com/v1/messages`

**Parameters:**
```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 4096,
  "messages": [{"role": "user", "content": full_prompt}]
}
```

**Model Variants:**
- `haiku` or default → `claude-haiku-4-5-20251001`
- `sonnet` → `claude-sonnet-4-6`
- `opus` → `claude-opus-4-7`

#### OpenAI GPT

**File:** `src-tauri/src/llm/openai.rs`

**Model:** `gpt-4o-mini`

**Parameters:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": full_prompt}],
  "max_tokens": 4096
}
```

#### xAI Grok

**File:** `src-tauri/src/llm/openai.rs` (OpenAI-compatible format)

**Model:** `grok-3` or `grok-2-vision-1212` (if "vision" in backend name)

#### Local GGUF Models

**File:** `src-tauri/src/llm/local_models.rs`

**Parameters:**
```json
{
  "model_path": "path/to/model.gguf",
  "messages": [{"role": "user", "content": full_prompt}],
  "max_tokens": 256
}
```

**Execution:** Blocking task (CPU-bound)

### 2.3 Error Handling

- **Network Errors:** Propagated as error response
- **API Errors:** Return HTTP status + error text
- **Invalid Response:** Return "Invalid response" error
- **Model Not Found:** Return "Unsupported backend" error
- **Timeout:** Default Tokio timeout ~120 seconds

---

## 3. Post-LLM Processing

### 3.1 Response Received

**Location:** `lib.rs`

**Response Format:**
```rust
pub struct LLMResponse {
    pub content: String,      // Actual text response
    pub model: String,        // Model name used
    pub usage: Option<Usage>, // Token counts
}
```

### 3.2 Web Search Detection

**Location:** `lib.rs`

**Process:**
1. Check if response contains `WEB_SEARCH_NEEDED:` marker
2. Extract suggested query from response
3. Return early with web search flag (before memory storage)

**Format:**
```
LLM Response: "WEB_SEARCH_NEEDED: current weather in Singapore 2026"
→ Frontend receives: web_search_needed: true, web_search_query: "..."
```

**Local model fallback:** If the local model doesn't emit the marker, keyword detection triggers automatically for queries containing: weather, today, current, latest, news, stock, score, etc.

### 3.3 Call 1: the MEMORY line (reply says yes or no)

**Location:** `commands/chat.rs` (parse, `memory_flag`), `src-tauri/src/conversation_engine.rs` (the instruction)

**Purpose:** Decide, with no extra model call, whether the message holds anything to store. The reply does not carry the fact any more; it ends with `MEMORY: yes` or `MEMORY: no`.

**Process (cloud and custom backends):**
1. Scan the reply lines from the end for a line that is only `memory` or `remember` plus `yes`/`no` (case-insensitive; markdown, colons, dashes and end punctuation tolerated). `REMEMBER:` is the older spelling and still accepted.
2. If no such line exists, look for the marker glued to the end of the last sentence (`split_inline_memory_marker`, 2026-10-25; a cloud reply wrote "...enjoy the game. MEMORY: no" and the phone spoke it). The marker is cut off that line.
3. The marker line, or the marker suffix, is hidden from the displayed reply and from the Android speech (`NativeVoiceAnswerer.withoutMarker`).
4. `yes` → the message is queued for call 2. `no` → nothing is stored. No marker at all → nothing is stored, with a log line ("Reply carried no MEMORY line"). An older-style `MEMORY_EXTRACT:` fact line is still recognised and treated as yes (`legacy_fact_line`); the line is hidden.

**Local GGUF models (`on_device_flow`):** no marker is asked for; every message that passes the worthiness gate (3.4) is queued for call 2, and call 2's NONE is the "no". An explicit `MEMORY: no` from such a model, if it writes one, is honoured.

**Explicit override:** a message starting with `Remember:` (section 3.4) is queued regardless, and its text is stored verbatim without call 2.

### 3.3a Call 2: the facts (background, `memory_prompts.rs`)

**Function:** `commands::chat::extract_facts` → `memory_prompts::extraction_prompt_with(message, subject, recent_turns, today, slim)`; `call_backend_with_schema`; `memory_prompts::parse_extraction`.

**Prompt:** today's date; the last exchange of the conversation so the model can resolve "she" and "it" (facts are to come from the message only); then: list the personal facts about {subject} in the message, up to `MAX_FACTS` = 3, one sentence per fact in the third person starting with the subject's name, keeping the message's own words and every detail. The rules, as written: only what the message states, nothing guessed; one thing with its details is one fact ("I'm flying to Denver Thursday for my nephew's wedding" is one fact, not three); what the user asks is not a fact, but what they state while asking is (the raspberry-pie example, with the salt-in-coffee question as the negative); nothing from text they quote, paste or ask to translate; greetings and thanks are not facts; `NONE` when the message holds no fact. Each fact carries a `title` (up to 50 characters) and an `event_date` (`YYYY-MM-DD`, only when the message says or clearly implies it, resolved against today; never a guessed year). Cloud backends also return `namespace`, `tags` (1–3), `tone` and `entities`; the **slim** variant sent to local models asks for fact, title and date only (`EXTRACTION_SCHEMA_SLIM`), the rest coming from the local name finder at storage time.

**Constrained output:** local models are held to the JSON schema (`EXTRACTION_SCHEMA` / `EXTRACTION_SCHEMA_SLIM` through the engine's grammar); cloud models answer free-form and are parsed leniently (`json_body` finds the object or array in the text).

**Guards in `parse_extraction` (so a small model's inventions do not become memories):** a fact under three words is dropped; a fact that shares no content word (three letters or more, not on the filler list; or a number of two or more digits) with the message is dropped and logged; an exact repeat is dropped; a restatement of a fact already kept (four fifths of its content words already in the earlier fact, `restates`) is dropped; at most three facts; a missing title is the opening of the fact, trimmed to 50 characters. Then `drop_near_duplicates`: two facts of one message whose embeddings are closer than cosine 0.86 (`SAME_MESSAGE_DUPLICATE_SIMILARITY`) keep only the first, catching rewordings the word rule misses.

**Result:** zero facts means nothing is stored for the message and the session's extraction outcome is recorded as `model_declined`. One to three facts each run the storage path below as their own memory; all of them keep the user's message as `original_text`.

**Lessons from the memory bench (2026-10-24, `labs/memory_bench/cases.json`, results outside the repo):** never show a small model the literal empty answer (a `{"facts": []}` example collapsed recall from 52 to 34 of 69); a fact stated inside a question needs its own example or the 4B model drops the whole message; a concrete list of what counts works where a judgment phrase ("would matter later") is applied too strictly; the grammar is not the cause of low recall. With the current wording Qwen3-4B-Instruct-2507 Q4_0 found 66 of 69 expected facts with no junk memory from 26 no-fact messages; Claude Sonnet 5 69 of 69; GPT-5.5 68; Grok 4.5 67; Mistral Large 68 but with junk from 14 of the 26 no-fact messages.

### 3.4 Memory Worthiness Gate

**File:** `src-tauri/src/conversation_engine.rs`

**Purpose:** A lightweight pre-filter that rejects obvious non-memories before invoking the LLM. It does not make the final storage decision — the LLM does.

**Gate passes if ALL of the following are true:**
- ≥3 words
- Not a pure filler phrase ("okay", "thanks", "lol", "hi", etc.)
- If ≤5 words, does not start with a filler word
- Contains at least one word longer than 3 characters
- Not a pure question (does not start with: what, where, when, why, how, who, can you, do you, is there, tell me, etc.)

**Important:** Messages that START with a first-person statement but contain a question (e.g. "I'm thinking of traveling to Japan, what should I wear?") pass this gate because they begin with "I'm". Pure questions ("What should I wear to Japan?") do not pass — but call 1 (section 3.3) handles those for all models instead.

**Note:** `question_extractor.rs` contains more sophisticated clause-level fact extraction (possessive patterns, NER-based entity detection) available as on-demand Tauri commands (`check_question_worthiness`, `extract_facts_from_question`), but is not wired into the automatic pipeline.

**Explicit override:** A message prefixed with `Remember:` (any case) or the spoken form `remember colon` — what dictation produces when the user says the word "colon" — always passes the gate regardless of content, and the text after the keyword is stored verbatim (`explicit_remember()` in `commands/chat.rs`).

### 3.4a Call 3: relationships (background, inside the storage path)

**File:** `lib.rs`

**Functions:** `ask_llm_for_relationships()` (cloud and custom: returns a title, the extras and the links) and `ask_llm_for_links_only()` (local models: links only, under the `LINKS_SCHEMA` grammar, since call 2 already gave the title and the full call cost the 4B model sixty-odd extra tokens for nothing).

**Timing:** in the job worker, after the reply has been returned, once per fact.

**Input:** the fact from call 2 (or the words after `Remember:`); the most similar existing memories above 35% hybrid similarity, sorted by similarity, excluding the other facts of the same message (`bg_sibling_ids`), capped at **10 on a cloud backend and 4 on a local model**; the backend.

**Skipped** when there is no similar memory and call 2 gave a title: the fact is stored with call 2's title and fields and no links.

**Output:** relationships of type `updates`, `contradicts`, `related` or `none`, each with a confidence; on the cloud also a title and extras, used only where call 2 left a field empty (`merge_extras`). A local model's answers pass `filter_local_links`: a confidence floor (inert at 0.0 today), a dropped-types list (empty today), and every `updates` turned into `contradicts` so the conflict card asks the user instead of the phone's model replacing a memory on its own (the card then reads "Is this an update?").

**Siblings:** after all facts of one message are stored, the worker links them to each other as `related` with the note "said in the same message" (`created_by = same_message`), with no model call.

**Failure:** if call 3 fails the fact is stored without links, titled by call 2 or by `generate_title_from_content`.

**Note:** `llm_fact_extractor.rs` (an older LLM fact-extraction module) exists in the codebase but is not active — it is marked `#![allow(dead_code)]` and is not called in the pipeline. The pre-2026-10-24 single call `ask_llm_about_memory_with_relationships` (decide, title and links in one answer on the raw message) has been removed.

### 3.4b The memory job queue (`memory_jobs.rs`, migration 0023)

Every message that passes call 1 becomes a row in `memory_jobs` (message, backend, source type, containment mode, the `Remember:` text if any, the user's name, the last exchange as context, the query's named things). One worker per process (`ensure_worker`, started after the app handle is stored, five seconds after launch so the app's migrations go first) takes the oldest `pending` row, or a `running` row untouched for five minutes, and runs `process`: the `Remember:` path stores the words as said; otherwise call 2, then the storage path once per fact, then the sibling links. An interruption (the user's next message took the model) leaves the row `pending` and the worker waits three seconds; a real failure retries up to `MAX_ATTEMPTS` = 4, then marks the row `failed`. Idle, the worker sleeps until `enqueue` wakes it or two minutes pass. Rows survive an app restart. Why: a spawned task per message stepped aside for the user's next message and gave up after three interruptions; on a phone a message's work takes minutes, so four quick questions lost the first one's facts (2026-10-24).

### 3.5 Embedding Generation

**Location:** `lib.rs`

**Process:**
1. Use factual content (not original message)
2. Generate 384-dim embedding using all-MiniLM-L6-v2
3. Blocking task (ML-bound)

### 3.6 NLP Enhancement (Fallback for Namespace, Entities, Events)

**File:** `src-tauri/src/nlp_enhancer.rs`

**Location:** `lib.rs`

**Function:** `enhancer.enhance(message: &str)`

**Precedence:** call 2 returns `event_date` (and on cloud backends `namespace`, `tags`, `tone` and `entities`; `memory_extras.rs`, `MemoryExtras`), and call 3 on the cloud returns the same set as a fallback for fields call 2 left empty. Those win when present and valid: `validate_event_date` rejects unparseable, pre-1900, far-future and January 1 placeholder dates; `resolve_namespace` keeps only a namespace from the canonical list; `clean_tags` lower-cases, de-duplicates and caps tags at five; `clean_entities` drops pronouns, calendar words and bare numbers. The NLP enhancer's namespace, event type and event date are the fallback when the model omits a field or it fails validation; NLP does not produce tags at all, so a memory has tags only if the decision call supplied them. <!-- added by Claude 2026-09-09, review -->

**Output:**
```rust
pub struct Enhancement {
    pub title: Option<String>,             // Auto-generated (max 50 chars)
    pub tags: Vec<String>,                 // Semantic tags
    pub namespace: String,                 // Category
    pub event_type: Option<String>,        // birthday, graduation, etc.
    pub event_date: Option<NaiveDateTime>, // Extracted date
}
```

#### Title Generation

**Process:** Generated by the LLM in call 2 (`memory_prompts.rs`, "title: up to 50 characters"); for a `Remember:` memory, which skips call 2, by call 3 on the cloud or by `generate_title_from_content` on a local model. NLP enhancement does not generate titles.

**Example:** "I graduated from MIT in 2020" → "Graduated From MIT In 2020"

#### Event Detection

**Event Types:**
- `birthday`: Patterns: "born", "birthday", dates
- `graduation`: Patterns: "graduated", "degree"
- `work`: Patterns: "work", "job", "employed"
- `travel`: Patterns: "visit", "travel", "going to"
- `acquisition`: Patterns: "bought", "purchased"
- `achievement`: Patterns: "accomplished", "achieved"
- `relationship`: Patterns: "married", "met"

#### Entity Extraction & Caching

**Process:**
1. Run BERT NER on factual content
2. Convert Entity objects to JSONB:
```json
[{
  "word": "Max",
  "label": "MISC",
  "score": 0.95,
  "start": 25,
  "end": 28
}, ...]
```
3. Store as `entities_detected` JSONB column

**Purpose:** Prevents re-running NER during relationship detection

---

## 4. Memory Storage

**File:** `src-tauri/src/memory.rs`

**Location:** `lib.rs`

### Database Insertion

**SQL:**
```sql
INSERT INTO memories (
    title, content, source_type, session_id, embedding,
    parent_scroll_id, chunk_index, user_id, tags, namespace,
    is_syncable, is_shareable, entities_detected, event_type, event_date
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
RETURNING id
```

### Fields Stored

| Field | Type | Purpose | Indexed |
|-------|------|---------|---------|
| id | i32 PRIMARY KEY | Unique identifier | Yes |
| title | TEXT | Auto-generated summary | No |
| content | TEXT NOT NULL | Factual statements | No |
| source_type | TEXT | "conversation" | No |
| session_id | TEXT | Conversation grouping | Yes |
| embedding | BLOB (binary, 384 × f32) | Semantic vector | No (in-process cosine) |
| user_id | TEXT | Memory isolation | Yes |
| tags | TEXT[] | Auto-generated keywords | No |
| namespace | TEXT | Category (personal/work/travel) | Yes |
| entities_detected | TEXT (JSON) | Cached BERT NER output | No |
| event_type | TEXT | Auto-detected event category | No |
| event_date | TIMESTAMP | Auto-extracted date | No |
| link_count | INT | Relationship count | No |
| is_syncable | BOOLEAN | Sync across devices | No |
| is_shareable | BOOLEAN | Share with others | No |
| is_ephemeral | BOOLEAN | HIPAA 8-hour expiration | No |
| expires_at | TIMESTAMP | HIPAA expiration time | No |
| created_at | TIMESTAMP | Creation timestamp | Yes |

### HIPAA Mode - Ephemeral Memory

**Location:** `lib.rs`

**Process:**
```rust
if containment_mode == "hipaa" {
    is_ephemeral = true
    expires_at = NOW() + 8 hours
}
```

**Purpose:** Auto-delete sensitive healthcare information after 8 hours

---

## 5. Relationship Generation

**File:** `src-tauri/src/commands/chat.rs`

**Timing:** In the memory job worker (section 3.4b), after the reply has been returned; one pass per fact.

### 5.1 Similarity Search

**Function:** `detector.find_similar_by_embedding()`

**Process:**
1. Get new memory embedding (cached if available)
2. Get existing memory embeddings (batch retrieval)
3. Compute cosine similarity for each pair:
   ```
   similarity = dot_product(new, existing) / (magnitude(new) × magnitude(existing))
   ```
4. Filter: similarity ≥ 0.39 threshold
5. Sort by similarity score (descending)

**Threshold:** 0.39 (39% similarity) - balanced to capture relationships

### 5.2 Entity Extraction

**Location:** `lib.rs`

**Process:**
1. Extract entities from new memory using BERT NER
2. Convert to lowercase strings
3. Pass to relationship detector

### 5.3 Relationship Type Classification

Relationship types (updates, contradicts, related; four stored types since migration 0022, 2026-10-23) are classified by the LLM in call 3, `ask_llm_for_relationships()` on cloud and custom backends and `ask_llm_for_links_only()` on local models (section 3.4a). The LLM returns a confidence score (0.0–1.0) for each detected relationship. Contradictions require confidence >= 0.65 before triggering the modal; other relationship types are stored at whatever confidence the LLM assigns.

### 5.4 Link Storage

**File:** `src-tauri/src/memory.rs`

**SQL:**
```sql
INSERT INTO memory_links (
    source_memory_id, target_memory_id, relation_type,
    confidence, notes, created_by
) VALUES ($1, $2, $3, $4, $5, $6)
RETURNING id
```

**Fields:**

| Field | Type | Example |
|-------|------|---------|
| id | INT PRIMARY KEY | AUTO |
| source_memory_id | INT | Newly created memory |
| target_memory_id | INT | Existing memory |
| relation_type | TEXT | contradicts, updates, related, resolves |
| confidence | FLOAT (0-1) | Cosine similarity score |
| notes | TEXT | "Shares entities: dog, name" |
| created_by | TEXT | "ai" for auto-detected |
| created_at | TIMESTAMP | NOW() |

**Active Relationship Types** (since 0022):
- related (absorbed supports, elaborates, caused_by)
- contradicts
- updates (marks the older memory `superseded`: out of recall, kept as history)
- resolves (generated by user conflict resolution only)

**Reserved (registered in database, not generated by pipeline):**
- quotes
- mentions

### 5.5 Contradiction Events

**Location:** `commands/chat.rs` (`run_memory_background`)

**Process:**
1. Contradiction detected in call 3's output
2. Fetch both memory details
3. **Cloud and custom backends: the memory is NOT stored yet** — storage is blocked pending the user's decision. **Local models (2026-10-23): store first, ask after.** Their background pass can take minutes on a phone and a memory held for an unanswered card was simply lost, so the new memory is stored like any other and the card is raised afterwards about the already-stored pair (`resolve_stored_conflict`). An explicit `Remember:` is stored regardless, with the contradiction noted.
4. Emit event to frontend:

```rust
emit("contradiction-detected", {
    "memoryA": {id, content, title, created_at},
    "memoryB": {id, content, title, created_at},
    "sharedEntities": ["dog", "name"]
})
```

**Frontend:** Shows ConflictResolutionModal for user decision. Memory is only stored after the user selects a resolution via `resolve_memory_conflict_v2`.

---

## 6. Duplicate & Contradiction Pre-Checks

**File:** `lib.rs`

**Function:** `pre_check_memory()`

**Timing:** Fire-and-forget background task. Runs BEFORE memory storage when a contradiction is detected — storage is gated on user resolution.

### 6.1 Keyword Extraction

**Process:**
1. Extract ALL nouns (not just named entities)
2. Use NLP enhancer's `extract_keywords()` method
3. Includes common words (dog, house, car, etc.)

### 6.2 Embedding Generation

**Process:**
1. Generate 384-dim embedding using all-MiniLM-L6-v2
2. Same model as memory embeddings

### 6.3 Hybrid Search

**Process:**
1. Find top 15 similar memories (>35% threshold)
2. Using same hybrid scoring as conversation
3. Exclude the newly stored memory

### 6.4 Duplicate Detection

**Process:**
1. Check hybrid score against all candidates: threshold >0.98 (entity + semantic match)
2. If no hybrid duplicate, check pure cosine similarity: threshold >0.93
3. If duplicate found:
   - Skip memory storage entirely
   - EXIT background task early (don't check contradictions)

### 6.5 Contradiction Detection

> **Note:** This pattern-based contradiction detection is currently disabled — `pre_check_memory` returns after the duplicate check and the code below is unreachable. Contradiction detection is handled by the LLM background task (section 3.4). This rule-based approach is preserved as a potential fast pre-filter for a future version.

#### Step 1: Keyword Matching

**Process:**
```rust
For each candidate memory:
    - Extract keywords from candidate
    - Find shared keywords
    - Filter stop words and <3 char keywords
```

#### Step 2: Semantic Pattern Groups

**Pattern Groups:**
```rust
[
    // Name patterns
    ["named ", "name is ", "called ", "goes by ", "known as "],

    // Identity patterns
    ["i am ", "i'm ", "my name is ", "i call myself "],

    // Location patterns
    ["lives in ", "from ", "resides in ", "based in "],

    // Work patterns
    ["works at ", "employed by ", "teaches at "],

    // Relationship patterns
    ["married to ", "spouse is ", "wife is ", "husband is "],

    // Age patterns
    ["i'm ", "i am ", "age is ", "years old"],

    // Possession patterns
    ["i own ", "i have ", "my car is ", "my pet is "]
]
```

#### Step 3: Smart Gating

**Logic:**
```
IF both texts match semantic patterns + 1+ shared entity:
    → Check for contradiction
ELSE IF 2+ shared entities OR (1+ entity + 85%+ similarity):
    → Check for contradiction
ELSE:
    → Skip (prevents false positives)
```

#### Step 4: Contradiction Detection

**Pattern 1: Negation**
```
IF one text has negation word + other doesn't:
    AND shared_entities > 0:
        → CONTRADICTION!

Example: "I have a dog" vs "I don't have a dog"
```

**Pattern 2: Semantic Substitution**
```
FOR each pattern_group:
    - Extract pattern match from both texts
    - If both match SAME pattern with DIFFERENT values:
        → CONTRADICTION!

Example:
    New: "I'm named Max"     (pattern: "named ")
    Old: "I'm named Wendy"   (pattern: "named ")
    Values: "Max" != "Wendy" → Contradiction!
```

### 6.6 Conflict Resolution

**File:** `lib.rs`

**Function:** `resolve_memory_conflict_v2()`

**Resolution Options:**

| Option | Action | Notes |
|--------|--------|-------|
| `keep_old` | Discard new memory (never stored) | Old memory unchanged |
| `keep_new` | Store new; mark old `superseded` and link `updates` (since 2026-10-23, nothing deleted) | New becomes primary, old is history |
| `not_a_contradiction` | Store new + remove `contradicts` link | User confirmed both are true and unrelated |
| `keep_both` | Store new + keep `contradicts` link | Both memories retained; contradiction edge preserved in graph (possible future hallucination) |
| `both_with_explanation` | Store new + store user's explanation as a third memory | Explanation memory linked to both conflicting memories via `resolves` edges; one-hop traversal surfaces it on future recall of either memory |

---

## 7. Memory Indexing

### 7.1 SQLite Indexes

**Primary Indexes:**
- `memories.id`: INTEGER PRIMARY KEY
- `memories_user_id_idx`: User isolation queries
- `memories_namespace_idx`: Namespace filtering
- `memories_session_id_idx`: Session grouping
- `memories_created_at_idx`: Temporal queries

### 7.2 Vector Search (In-Process)

Vector similarity is computed in Rust, not at the database layer. All candidate embeddings (stored as binary BLOBs) are fetched and cosine similarity is computed in the Rust process via Candle. No SQL vector index is used — SQLite does not support vector index types.

**Performance:** Sufficient for personal memory workloads (tested to 500k+ rows without performance issues). For very large databases, sqlite-vec extension is under evaluation as an optional enhancement.

### 7.3 Entity Matching

`entities_detected` is stored as a TEXT column containing a JSON array. Entity overlap scoring is done in Rust by deserializing the stored JSON and comparing entity sets in-process. No database-level JSON index is used.

---

## 8. Data Flow Diagram

```
USER INPUT
    ↓
[1] CONTAINMENT CHECK (TinyBERT Safety Classifier)
    ├─ Witness: Pass-through
    ├─ Guardian/Default: Block if flagged
    ├─ Sovereign: Warn + continue
    ├─ Child: OpenAI Moderation API + TinyBERT
    ├─ HIPAA: PHI/Diagnosis/Dosing checks
    ↓
[2] ENTITY EXTRACTION (Candle BERT NER)
    └─ Extract entities, filter stop words
    ↓
[3] EMBEDDING GENERATION (all-MiniLM-L6-v2, 384-dim)
    └─ Vectorize query for search
    ↓
[4] HYBRID SEARCH (Entity + Semantic)
    ├─ Find top 15 memories for API models, top 7 for local (threshold: 35%)
    └─ Add all one-hop graph-linked memories (elaborates, contradicts, resolves) — no cap
    ↓
[5] BUILD CONVERSATION CONTEXT
    ├─ System prompt
    ├─ Conversation history (adaptive limits)
    ├─ Recalled memories (adaptive limits)
    ├─ KB context (if enabled)
    └─ User question
    ↓
[6] LLM CALL
    ├─ Anthropic Claude (haiku/sonnet/opus)
    ├─ OpenAI GPT-4o-mini
    ├─ xAI Grok
    └─ Local GGUF model
    ↓
[7] POST-LLM PROCESSING
    ├─ Call 1: read "MEMORY: yes|no" (own line, or glued to the last sentence); hide it
    ├─ Check for WEB_SEARCH_NEEDED marker → return early if found
    └─ Local model: no marker asked for; every gated message goes on to call 2
    ↓
[8] MEMORY WORTHINESS GATE
    ├─ Reject filler, <3 words, pure questions
    ├─ Explicit "Remember:" bypasses gate and call 2 (stored verbatim)
    └─ Gate passed and call 1 said yes → row in memory_jobs
    ↓
[9] JOB WORKER (memory_jobs.rs): one row at a time, oldest first
    ├─ Call 2: extract_facts() → up to 3 facts, each with title and date
    │  (guards: ≥3 words, shares a content word, no restatement, cosine ≥0.86 dedupe)
    ├─ Zero facts → nothing stored ("model_declined")
    └─ For each fact: run_memory_background()
       ├─ Generate embedding of the fact
       ├─ Duplicate check (>98% hybrid OR >93% cosine) → skip the fact
       ├─ Similar memories: ≥35%, minus the message's other facts, top 10 cloud / 4 local
       ├─ Call 3 (skipped when nothing similar and a title exists):
       │  ask_llm_for_relationships() cloud / ask_llm_for_links_only() local
       │  └─ relationships: updates | contradicts | related | none
       │     (local "updates" becomes "contradicts": the card asks)
       ├─ Contradiction, cloud: emit "contradiction-detected", WAIT for the card
       ├─ Contradiction, local: store first, raise the card after
       └─ Otherwise → continue to NLP enhancement
    └─ After the last fact: siblings linked "related" (said in the same message)
    ↓
[10] NLP ENHANCEMENT
    ├─ Generate title
    ├─ Extract tags
    ├─ Run BERT NER → Cache entities
    ├─ Detect event type
    └─ Extract event date
    ↓
[10] GENERATE EMBEDDING
    └─ all-MiniLM-L6-v2 on factual content
    ↓
[11] STORE MEMORY IN DATABASE (via store_pending_memory)
    ├─ Called immediately if no contradiction detected
    └─ Called from resolve_memory_conflict_v2 after user resolves contradiction
    ↓
[12] BACKGROUND: RELATIONSHIP DETECTION
    ├─ Compute cosine similarities
    ├─ Classify relationship types
    └─ Store memory_links (non-contradiction relationships)
    ↓
[13] BACKGROUND: DUPLICATE PRE-CHECK (pre_check_memory)
    ├─ Hybrid search for candidates
    └─ Check duplicates (>95% cosine similarity) → delete if found
    (Pattern-based contradiction detection disabled — LLM handles it in step [9])
    ↓
[14] USER CONFLICT RESOLUTION (Manual)
    ├─ keep_old              → discard new, old unchanged
    ├─ keep_new              → store new, old becomes history (superseded)
    ├─ not_a_contradiction   → store new, remove contradicts link
    ├─ keep_both             → store new, keep contradicts link
    └─ both_with_explanation → store new + store explanation memory (resolves → both)
    ↓
RESPONSE RETURNED TO USER
    ├─ reply_text
    ├─ recalled_memories
    ├─ model_backend
    └─ containment_mode
```

---

## 9. Performance Characteristics

### 9.1 Memory Usage (Approximate — varies by hardware)

- **TinyBERT Model:** loaded once at startup, resident for session
- **all-MiniLM-L6-v2:** loaded once at startup, resident for session
- **BERT NER:** loaded once at startup, resident for session
- **Local GGUF models:** 5GB–70GB depending on model size and quantization
- **In-Memory Vectors:** approximately 1–2 MB per 1,000 memories (384 floats × 4 bytes)

### 9.3 Database Scalability

- **Designed to scale to:** 1M+ memories; tested to 500k+ rows without performance issues on personal memory workloads
- **Vector search:** In-process cosine similarity in Rust; all embeddings fetched and scored in memory
- **Typical query:** Fast at scale — performance depends on database size and hardware
- **Batch embeddings:** Faster than sequential generation when processing multiple texts
- **Concurrent request protection:** The UI disables all input while a response is in flight, preventing overlapping LLM calls; the post-response memory pipeline runs asynchronously via sqlx's connection pool

---

## 10. Key Files Reference

| File | Purpose | Key Functions |
|------|---------|----------------|
| `lib.rs` | Main orchestration, call 3 | `ask_llm_for_relationships()`, `ask_llm_for_links_only()`, `call_backend_with_schema()`, `filter_local_links()`, `check_containment()`, `pre_check_memory()`, `resolve_conflict()` |
| `memory.rs` | Memory operations | `insert_memory()`, `hybrid_search()`, `create_memory_link()` |
| `containment.rs` | Safety enforcement | `enforce()`, HIPAA checks |
| `safety_classifier.rs` | TinyBERT toxicity | Classification by category |
| `conversation_engine.rs` | Prompt building | `is_memory_worthy()`, `build_prompt()` (the MEMORY LINE block), `build_memory_context()` (grouping by source message) |
| `memory_prompts.rs` | Call 2: the facts | `extraction_prompt_with()`, `EXTRACTION_SCHEMA(_SLIM)`, `parse_extraction()`, `restates()`, `clean_title()` |
| `memory_jobs.rs` | The background job queue | `enqueue()`, `take_next()`, `ensure_worker()`, `process()` |
| `commands/chat.rs` | Call 1 parse, the gate, the storage path | `split_inline_memory_marker()`, `extract_facts()`, `drop_near_duplicates()`, `run_memory_background()` |
| `nlp_enhancer.rs` | NLP tasks | `extract_entities()`, `enhance()`, `detect_event()` |
| `llm/anthropic.rs` | Claude API | `send_message()` |
| `llm/openai.rs` | GPT API | `send_message()` |
| `llm/xai.rs` | Grok API | `send_message()` |
| `llm/local_models.rs` | GGUF models | `generate_with_local_model()` |
| `llm/local_embeddings.rs` | Embeddings | `generate_local_embedding()` |

---

## 11. Important Notes & Caveats

### Embedding Cache Optimization
- Relationship detection reuses embeddings already stored in the database rather than regenerating them

### Entity-Based Contradiction Matching
- Uses ALL nouns, not just named entities
- Catches "dog named Max" vs "dog's name is Wendy"

### Hybrid Search Scoring
- 60% entity overlap + 40% semantic similarity
- Lower threshold than pure semantic (0.60) but more precise

### Stop Word Filtering
- Applied at multiple stages: entity extraction, keyword matching
- Prevents false positives from common words

### HIPAA Compliance
- Ephemeral memories auto-expire after 8 hours
- PHI patterns blocked: SSN, credit cards, medical IDs
- Diagnosis/medication dosing prevented

### Race Conditions
- Memory stored → Background contradiction check → User resolves → Delete
- Pre-check runs asynchronously to avoid blocking user response

---

## Conclusion

The Zynkbot memory processing pipeline is designed for privacy-first operation with all ML models running locally, optional API backends for LLM generation, and sophisticated relationship detection using embeddings and semantic pattern matching. The system balances performance, accuracy, and safety through a multi-stage pipeline that can process and contextualize personal information at scale.

For questions or contributions, please refer to the main repository documentation.

For a detailed explanation of the relationship graph itself — what it visualizes, its practical uses, and the theoretical research potential of the data it produces — see [Memory Relationship Graph](MEMORY_RELATIONSHIP_GRAPH.md).
