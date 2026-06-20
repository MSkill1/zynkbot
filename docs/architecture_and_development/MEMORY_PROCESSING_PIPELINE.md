# Zynkbot Memory Processing Pipeline

**Complete Technical Documentation**

*Last Updated: June 17, 2026*

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

**Key Features:**
- 100% local ML processing (no external APIs for safety/embeddings)
- Semantic memory recall using 384-dimensional BERT embeddings
- Automatic relationship detection between memories
- Contradiction detection with user resolution
- Privacy-first design with optional API backends

---

## 1. Pre-LLM Processing

### 1.1 User Input Received

**File:** `src-tauri/src/lib.rs`

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

After the hybrid search, the system follows `elaborates`, `contradicts`, and `resolves` links from each recalled memory and adds every directly linked memory not already in the result set. There is no cap on this step — if a memory has a contradiction, an elaboration, or a resolution explanation, all of it is included. These linked memories often score too low semantically to surface on their own but are exactly what the LLM needs to reason correctly about the recalled memory. The traversal is one hop only, so there is no risk of runaway expansion.

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

PERSONAL FACT EXTRACTION:
If the user's message — even a question — contains personal facts about them (possessions, family, plans, travel, preferences, or experiences), extract those facts in addition to answering.

To save a fact, include a line in your response:
MEMORY_EXTRACT: [all personal facts combined into one third-person statement, e.g., "User has a 3-year-old golden retriever named Max and is planning to visit Japan in March"]

Include at most ONE MEMORY_EXTRACT line per message, combining all personal facts into a single compound statement. If MEMORY_EXTRACT is present, it must be the very first line of the response. Omit entirely if the message contains no personal facts.
```

#### Context Assembly

**Adaptive Limits:**
- **API Models** (Claude, GPT): 40 messages (20 turns), 20 memories
- **Local Models** (GGUF): 8 messages (4 turns), 7 memories

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

### 3.3 MEMORY_EXTRACT: Inline Fact Extraction from Questions

**Location:** `lib.rs`, `src-tauri/src/conversation_engine.rs`

**Purpose:** Capture personal facts embedded in questions that the Memory Worthiness Gate would otherwise reject. Example: "What should I pack for Japan?" → stores *"User is planning a trip to Japan"*.

**Scope:** All models. Local models may occasionally miss the marker, but this causes a fact to be silently skipped — it does not cause errors.

**Process:**
1. Scan each line of the LLM response for `MEMORY_EXTRACT:` prefix
2. Extract fact text, collect into `extracted_facts: Vec<String>`
3. Strip all `MEMORY_EXTRACT:` lines from the response shown to the user
4. If any facts were extracted, **skip the regular memory pipeline** — the extracted statements are a better representation than the raw question
5. For each fact: spawn background task → generate embedding → NLP enhance (entities, namespace) → store directly

**Note:** Relationship detection (contradicts/elaborates) is not yet applied to extracted facts. See Roadmap.

### 3.4 Memory Worthiness Gate

**File:** `src-tauri/src/conversation_engine.rs`

**Purpose:** A lightweight pre-filter that rejects obvious non-memories before invoking the LLM. It does not make the final storage decision — the LLM does.

**Gate passes if ALL of the following are true:**
- ≥3 words
- Not a pure filler phrase ("okay", "thanks", "lol", "hi", etc.)
- If ≤5 words, does not start with a filler word
- Contains at least one word longer than 3 characters
- Not a pure question (does not start with: what, where, when, why, how, who, can you, do you, is there, tell me, etc.)

**Important:** Messages that START with a first-person statement but contain a question (e.g. "I'm thinking of traveling to Japan, what should I wear?") pass this gate because they begin with "I'm". Pure questions ("What should I wear to Japan?") do not pass — but `MEMORY_EXTRACT` (section 3.3) handles those for all models instead.

**Note:** `question_extractor.rs` contains more sophisticated clause-level fact extraction (possessive patterns, NER-based entity detection) available as on-demand Tauri commands (`check_question_worthiness`, `extract_facts_from_question`), but is not wired into the automatic pipeline.

**Explicit override:** A message prefixed with `Remember:` always passes the gate regardless of content.

### 3.4 LLM Memory Decision (Background Task)

**File:** `lib.rs`

**Function:** `ask_llm_about_memory_with_relationships()`

**Timing:** Runs in a background task after the response has already been returned to the user.

**Purpose:** The LLM decides in a single call whether the message is worth storing, generates a title if so, and classifies relationships with any similar memories already found.

**Input:**
- Full original message (no pre-filtering or clause splitting)
- Conversation context
- Up to 6 similar existing memories (by embedding similarity ≥45%)
- Current LLM backend

**Output:** `(should_remember: bool, title: Option<String>, relationships: Vec<RelationshipClassification>)`

**Fallback:** If the API call fails and the current backend is not already local, retries automatically using the local GGUF model.

**Note:** `llm_fact_extractor.rs` (a separate LLM-based fact extraction module) exists in the codebase but is not active — it is marked `#![allow(dead_code)]` and is not called in the pipeline.

### 3.5 Embedding Generation

**Location:** `lib.rs`

**Process:**
1. Use factual content (not original message)
2. Generate 384-dim embedding using all-MiniLM-L6-v2
3. Blocking task (ML-bound)

### 3.6 NLP Enhancement (Title, Tags, Entities, Events)

**File:** `src-tauri/src/nlp_enhancer.rs`

**Location:** `lib.rs`

**Function:** `enhancer.enhance(message: &str)`

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

**Process:** Generated by the LLM in the memory decision call (`ask_llm_about_memory_with_relationships`). NLP enhancement does not generate titles.

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

**File:** `src-tauri/src/lib.rs`

**Timing:** Fire-and-forget async background task (doesn't block user response)

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

Relationship types (supports, contradicts, elaborates, caused_by, reminds_of) are classified by the LLM during `ask_llm_about_memory_with_relationships()`. The LLM returns a confidence score (0.0–1.0) for each detected relationship. Contradictions require confidence >= 0.65 before triggering the modal; other relationship types are stored at whatever confidence the LLM assigns.

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
| relation_type | TEXT | supports, contradicts, elaborates, reminds_of, caused_by |
| confidence | FLOAT (0-1) | Cosine similarity score |
| notes | TEXT | "Shares entities: dog, name" |
| created_by | TEXT | "ai" for auto-detected |
| created_at | TIMESTAMP | NOW() |

**Active Relationship Types:**
- supports
- contradicts
- elaborates
- reminds_of
- caused_by
- resolves (generated by user conflict resolution only)

**Reserved (registered in database, not generated by pipeline):**
- quotes
- mentions

### 5.5 Contradiction Events

**Location:** `lib.rs`

**Process:**
1. Contradiction detected in LLM relationship output (confidence >= 0.65)
2. Fetch both memory details
3. **Memory is NOT stored yet** — storage is blocked pending user decision
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
| `keep_new` | Delete old memory, then store new | New becomes primary |
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
    ├─ Parse MEMORY_EXTRACT markers → extracted_facts
    ├─ Strip MEMORY_EXTRACT lines from displayed response
    ├─ Check for WEB_SEARCH_NEEDED marker → return early if found
    └─ Store extracted facts in background (skip regular pipeline)
    ↓
[8] MEMORY WORTHINESS GATE (regular pipeline — skipped if extracted_facts non-empty)
    ├─ Reject filler, <3 words, pure questions
    ├─ Explicit "Remember:" bypasses gate
    └─ Gate passed → spawn background memory task
    ↓
[9] BACKGROUND: LLM MEMORY DECISION
    ├─ Generate embedding
    ├─ Duplicate check (>98% hybrid OR >93% cosine)
    ├─ ask_llm_about_memory_with_relationships()
    │  ├─ should_remember: bool
    │  ├─ title: Option<String>
    │  └─ relationships: Vec<RelationshipClassification>
    ├─ If contradiction detected (confidence >= 0.65):
    │  ├─ Emit "contradiction-detected" event to frontend
    │  ├─ Show ConflictResolutionModal to user
    │  └─ WAIT — memory is NOT stored until user resolves
    └─ If no contradiction → continue to NLP enhancement
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
    ├─ keep_new              → delete old, store new
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
| `lib.rs` | Main orchestration | `send_message_with_memory()`, `check_containment()`, `pre_check_memory()`, `resolve_conflict()` |
| `memory.rs` | Memory operations | `insert_memory()`, `hybrid_search()`, `create_memory_link()` |
| `containment.rs` | Safety enforcement | `enforce()`, HIPAA checks |
| `safety_classifier.rs` | TinyBERT toxicity | Classification by category |
| `conversation_engine.rs` | Prompt building | `is_memory_worthy()`, `build_prompt()` |
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
