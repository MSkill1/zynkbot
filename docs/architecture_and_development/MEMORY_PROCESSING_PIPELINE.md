# Zynkbot Memory Processing Pipeline

*Checked against the code on 2026-11-10 (`v1`, a7599eb).*

This document follows one chat message from the moment it arrives to the memories and links it leaves behind. Paths are relative to `zynkbot_rust/src-tauri/src/` unless shown otherwise. Code is named by file and function.

Related documents, which this one links to instead of repeating:

- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md): backends, model names, context limits.
- [PROMPT_CONSTRUCTION_PIPELINE.md](PROMPT_CONSTRUCTION_PIPELINE.md): the full prompt text.
- [CONTAINMENT_ARCHITECTURE.md](CONTAINMENT_ARCHITECTURE.md): the safety modes.
- [RELATIONSHIP_DETECTION_ARCHITECTURE.md](RELATIONSHIP_DETECTION_ARCHITECTURE.md): call 3 in detail.
- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md): table definitions.
- [MEMORY_RELATIONSHIP_GRAPH.md](MEMORY_RELATIONSHIP_GRAPH.md): the graph the links form.
- [../BENCHMARKS.md](../BENCHMARKS.md): the memory bench and the recall bench.

---

## Contents

1. [Overview](#1-overview)
2. [Before the reply](#2-before-the-reply)
3. [The reply (call 1)](#3-the-reply-call-1)
4. [Deciding whether to queue the message](#4-deciding-whether-to-queue-the-message)
5. [The memory job queue](#5-the-memory-job-queue)
6. [Call 2: the facts](#6-call-2-the-facts)
7. [Storing one fact](#7-storing-one-fact)
8. [Conflict cards](#8-conflict-cards)
9. [What is written to the database](#9-what-is-written-to-the-database)
10. [Data flow diagram](#10-data-flow-diagram)
11. [Key files](#11-key-files)
12. [Not verified against the code](#12-not-verified-against-the-code)

---

## 1. Overview

A message can cause up to three kinds of model call.

- **Call 1 is the reply itself.** For cloud and custom backends the prompt asks the model to end its reply with a hidden line, `MEMORY: yes` or `MEMORY: no`. An on-device model is not asked for the line.
- **Call 2 lists the facts.** It runs in the background after the reply has been returned. It returns up to three personal facts from the message, each with a title (`commands/chat.rs` `extract_facts`, prompt in `memory_prompts.rs`).
- **Call 3 classifies relationships.** For each fact it says how the fact relates to similar memories already stored (`lib.rs` `ask_llm_for_relationships` or `ask_llm_for_links_only`). It is skipped when nothing similar is found.

Each fact becomes its own memory. The user's message is not stored as a memory. It is kept on each fact as `original_text`.

Calls 2 and 3 run from a queue in SQLite (`memory_jobs.rs`), one job per message, one worker per process.

Safety classification, name finding and embeddings run locally. Child mode is the exception: it calls the OpenAI Moderation API.

### Three kinds of backend

The code treats backends in three groups. `ConversationEngine::is_api_model` is false for `custom`, `local` and any name ending in `.gguf`. "On device" in the pipeline means the backend name ends in `.gguf` or contains "local".

| | Cloud (anthropic, openai, xai, mistral) | Custom endpoint (`custom`) | On device (`.gguf`, `local`) |
|---|---|---|---|
| `MEMORY:` line asked for | yes | yes | no |
| Memories searched for the reply | 30 | 10 | 10 |
| Memories placed in the prompt | 30 | 7 | 4 |
| Call 2 output | full field set, free-form JSON | full field set, free-form JSON | fact, title, event_date, passing; held to a JSON schema |
| Candidates shown to call 3 | 10 | 4 | 4 |
| Call 3 | `ask_llm_for_relationships` | `ask_llm_for_relationships` | store first, then `ask_llm_for_links_only` (see 7.4) |
| `updates` from the model | stored as `updates` | stored as `updates` | turned into `contradicts`, so a card asks the user |
| Conflict card | asked before storing | asked before storing | memory stored first, then asked |

---

## 2. Before the reply

### 2.1 Entry point

The Tauri command is `send_message_with_memory` in `commands/chat.rs`. Its parameters are `message`, `user_id`, `session_id`, `backend`, `containment_mode`, `conversation_history`, `skip_containment`, `skip_memory_storage`, `kb_enabled`, `user_query`, `image_data`, `mode` and `web_sources`. It passes them to `generate_reply`, which does the work and also serves the hands-free (wake word) path.

Child mode forces the backend to `openai`. A cloud backend with no API key falls back to `get_best_available_backend()`.

### 2.2 Weather shortcut

A weather question is answered from Open-Meteo before containment and before any model call (`weather::detect`, `weather::answer`). The function returns at that point, so no memory work runs for it. If the lookup fails the message goes on to the model as usual. The shortcut is skipped when the message has a picture, starts with `Remember:`, or `skip_containment` is set.

### 2.3 Containment

Child mode calls `ContainmentLayer::check_openai_moderation` directly. An API error blocks the message. Every other mode goes through `commands/safety.rs` `check_containment`, which calls `ContainmentLayer::enforce` in `containment.rs`.

| Mode | What `enforce` does |
|---|---|
| witness | No check. |
| guardian, default | Blocks when the pattern check for harmful instructions matches, or when the classifier's top category is severe_toxic or threat with confidence above 0.9. |
| sovereign | Does not block. Returns a `[WARN_ALLOW]` warning when the pattern check matches or a harmful category scores above 0.85. The reply is still generated. |
| hipaa | Runs `enforce_hipaa` first (PHI, diagnosis requests, medication dosing), then the guardian rule. |
| child | OpenAI Moderation first. A missing key or an API error blocks. Then the local classifier blocks any harmful category above 0.7. |

The local classifier is `system/toxic-bert` with six outputs: toxic, severe_toxic, obscene, threat, insult, identity_hate (`safety_classifier.rs`, `should_block`). If the classifier fails, `fallback_keyword_check` is used. Details are in [CONTAINMENT_ARCHITECTURE.md](CONTAINMENT_ARCHITECTURE.md).

### 2.4 Recall

Recall finds stored memories to place in the prompt.

**Names in the message.** `NLPEnhancer::extract_entities` (`nlp_enhancer.rs`) runs the `dslim/bert-base-NER` model plus a rule-based pass. The words are lower-cased and a stop-word list is removed. The result, `query_entities`, is used for this search and is saved with the memory job for later (7.2).

**Embedding.** `llm/local_embeddings.rs` `generate_local_embedding` embeds the message with `all-MiniLM-L6-v2` into a 384-number vector.

**Hybrid search.** `memory.rs` `hybrid_search` fetches the user's candidate memories and scores each one in Rust (`hybrid_score`):

```
base  = entity_overlap * 0.6 + cosine * 0.4     when at least one name matches
      = cosine                                   otherwise
score = base
      + 0.05 if the memory's namespace is one the question points at
      + 0.25 * tag_overlap

entity_overlap = shared names / names in the question
tag_overlap    = share of the question's words found in the memory's tags
```

The namespace and tag hints come from `memory::query_hints(question)`. They nudge the score and never filter. The weights were set on the recall bench; see [../BENCHMARKS.md](../BENCHMARKS.md).

A memory is kept when its score is at least 0.35, or when any name matches, whatever the score. The search returns the top 30 for a cloud backend and the top 10 otherwise. On the web-search answer pass a non-cloud backend gets no recalled memories at all.

Cosine similarity is plain Rust (`memory.rs` `cosine_similarity`). There is no vector index in SQLite. All candidate embeddings are read and scored in the process.

**What recall leaves out.** The search SQL excludes:

- memories whose `temporal_status` is `superseded` or `completed` (replaced by an `updates` link, or by the user's choice on a card);
- memories whose `expires_at` day is before today (passing facts that have lapsed, see 7.8);
- the `system` user's memories, unless the question is about Zynkbot.

**Questions about Zynkbot.** `lib.rs` `is_query_about_zynkbot` matches "zynkbot" and its misheard forms (zinc bot, zink bot), ZynkSync, ZynkLink, ZChat, ensemble mode, containment mode, snap-in, memory manager, knowledge base, "hey zynk" and "wake word". For such a question the search runs only on the `system` user's `_zynkbot` namespace, link traversal is skipped, and the built-in guide passages are added from the `_system` knowledge base when the KB button is off. For every other question `_zynkbot` memories are removed from the results.

**One hop along the links.** For each recalled memory the code follows its links of any type and adds the linked memories. Links are taken strongest first. At most 8 linked memories are added (`MAX_HOP_MEMORIES`). A linked memory is skipped when it is already in the list, has the same text as one in the list, or is `superseded` or `completed`. The combined list is then cut to the prompt limit in 2.5, so a linked memory can still be dropped.

### 2.5 Prompt

`conversation_engine.rs` `build_prompt` writes the prompt. The full text is described in [PROMPT_CONSTRUCTION_PIPELINE.md](PROMPT_CONSTRUCTION_PIPELINE.md). The points that matter for memory:

| | Cloud | Custom endpoint | On device |
|---|---|---|---|
| Instruction text | full | short | one short block |
| History | 40 messages | 8 messages | 8 messages, each cut to 400 characters |
| Memories (`build_memory_context`) | 30 | 7 | 4 |
| Memory heading | `USER'S STORED MEMORIES:` | `USER'S STORED MEMORIES:` | `WHAT THE USER HAS TOLD YOU BEFORE:` |

Each memory line starts with the date it was stored, plus "about" and the event day when that differs (`event_note`). Memories that came from the same user message (same `original_text`) are grouped under one quote of that message. A memory identical to the current message is left out.

Cloud and custom backends get this block, which is call 1's instruction (`{subject}` is the user's name when known, else "User"):

```
MEMORY LINE
End every reply with one final line, exactly "MEMORY: yes" or "MEMORY: no", and nothing else on that line.
- yes: the user's message states a personal fact about {subject}: their life, people, pets, work, home, health, feelings, plans, possessions, likes or dislikes. A fact stated inside a question counts ("Raspberry pie is my favorite. What's a good recipe?" is yes).
- no: a question alone, chit-chat, general knowledge, a request about their stored memories, text they ask you to translate or that they quote from someone else, or a message only about Zynkbot itself.
Do not write the fact in your reply; the line is only yes or no.
```

The on-device prompt has no such block.

Order of the parts:

- Cloud and custom: instructions, history, memories, `USER'S QUESTION:`.
- On device: instructions, memories, history, today's date, `USER'S MESSAGE:`.

`generate_reply` then puts further blocks in front of that prompt when they apply: the conversation-mode block, the knowledge base context, the Zynkbot guide passages, and the text of web pages named in the message. A note about attached pictures is added at the end.

---

## 3. The reply (call 1)

### 3.1 Sending the prompt

`generate_reply` picks the provider from the backend name: `anthropic` or `claude`, `openai` or `gpt`, `xai` or `grok`, `mistral`, `custom`, and `local` or a `.gguf` file. Every path streams, with a limit of 4096 output tokens. An empty reply is an error. xAI is called through `llm/openai.rs` `send_message_streaming` at the x.ai address. Model names and defaults are in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

### 3.2 Web search marker

The reply is checked for the web search marker before the `MEMORY` line is read. `find_web_search_marker` accepts `WEB_SEARCH_NEEDED`, "web search needed" and "web-search-needed".

- **Typed chat:** the marker line is removed from the shown reply (`strip_web_search_marker`), the suggested query is returned to the window in `web_search_needed` and `web_search_query`, and the function carries on. Memory processing still runs for the message.
- **Hands-free:** the search runs at once, its answer is the reply, and the function returns there.

There is no keyword list that triggers a search without the marker.

### 3.3 Reading the MEMORY line

In `generate_reply`:

1. The reply's lines are scanned from the end for a line that is only `memory` or `remember` plus `yes` or `no`. Case, markdown, colons, dashes and end punctuation are tolerated.
2. If there is none, `split_inline_memory_marker` looks for the marker at the end of the last sentence ("...enjoy the game. MEMORY: no") and cuts it off.
3. An older-style `MEMORY_EXTRACT:` line counts as yes (`legacy_fact_line`). Its text is not stored.
4. The marker is removed from the shown reply (`is_memory_flag_line`) and from Android speech (`NativeVoiceAnswerer.withoutMarker`). `<think>` blocks are also removed.

The result is `memory_flag`: yes, no, or absent.

### 3.4 The response is returned

The response is built and returned before any memory work. Its fields include `reply_text`, `recalled_memories`, `model_backend`, `containment_mode`, and the web search fields. The exchange is logged to conversation history, except in HIPAA mode and for a hands-free `NO_QUERY` turn.

---

## 4. Deciding whether to queue the message

This is the last step of `generate_reply`. A job is queued when all three hold:

```
not skipped   and   gate passed   and   call 1 said yes
```

**Skipped** (`effective_skip_memory`) when any of these is true:

- the caller set `skip_memory_storage`;
- the containment mode is `hipaa`. HIPAA mode stores no memories and no conversation history;
- the KB button was on and the answer could not be grounded in the knowledge base (`kb_ungrounded_reply`). A `Remember:` message is still stored.

**Gate** (`memory_gate_passed`): `ConversationEngine::is_memory_worthy(message)` and not `memory_extras::is_rewrite_request(message)`.

- `is_memory_worthy` requires at least three words, a message that is not exactly a filler phrase, no filler phrase at the start when the message has five words or fewer, and at least one word longer than three characters. It has no rule about questions.
- `is_rewrite_request` is true for a request to translate, reprint, rewrite, reword, rephrase, paraphrase or proofread text, or to fix its grammar or spelling. Such a message carries someone else's words.

**Call 1's answer** (`call1_yes`):

- Cloud and custom: yes only when the reply carried `MEMORY: yes` or an old-style fact line. No marker at all means no.
- On device: yes unless the model wrote an explicit `MEMORY: no`. Call 2 makes the real decision.

**`Remember:`** `explicit_remember` recognises a message that starts with `Remember:` (any case) or the dictated form "remember colon" with its common mis-hearings. Such a message passes the gate and call 1 regardless, skips call 2, and its text after the keyword is stored as said. It is still not stored when the message is skipped for `skip_memory_storage` or HIPAA mode.

When no job is queued, the session's extraction outcome is recorded as `skipped`, `gate_skipped` or `model_declined`.

---

## 5. The memory job queue

`memory_jobs.rs`, table `memory_jobs` (migration 0023).

A job row holds the message, the backend, the source type (`conversation` or `hands_free`), the containment mode, the `Remember:` text if any, the user's name, the last exchange (so call 2 can resolve "she" and "it"), and the names found in the message at chat time.

One worker per process (`ensure_worker`) starts five seconds after launch. `take_next` takes the oldest `pending` row, or a `running` row untouched for five minutes. `process` then does the following:

1. A `Remember:` job stores its text through the storage path (section 7) with no title, and ends.
2. Otherwise call 2 runs (section 6).
3. If the message is a name correction (6.3), a correction fact is put first.
4. Zero facts: the outcome is `model_declined` and nothing is stored.
5. Each fact runs the storage path in turn. The ids already stored for this message are passed along so a fact is not compared with its own siblings.
6. When two or more facts were stored, they are linked to each other as `related`, confidence 1.0, note "said in the same message", `created_by` `same_message`. No model call is made for this.

Failure handling:

- An interruption (the user's next message took the model) puts the row back to `pending` without counting an attempt. The worker waits three seconds.
- Any other error from call 2 waits ten seconds and retries. After `MAX_ATTEMPTS` = 4 attempts in total the row is marked `failed`.
- `process` reports an error only when call 2 fails. A fact that fails inside the storage path (for example an embedding or database error) is dropped without a retry.
- When idle, the worker sleeps until `enqueue` wakes it or two minutes pass. Rows survive an app restart.

---

## 6. Call 2: the facts

`commands/chat.rs` `extract_facts` builds the prompt with `memory_prompts::extraction_prompt_with`, calls the same backend that wrote the reply (`lib.rs` `call_backend_with_schema`), and reads the answer with `memory_prompts::parse_extraction`.

### 6.1 Prompt and output

The prompt gives today's date and the last exchange, then asks for the personal facts about the user stated in the message: at most three (`MAX_FACTS`), one sentence each in the third person, keeping the message's own words. What the user asks is not a fact. Text they quote, paste or ask to have translated is not a source of facts. The answer is `NONE` when the message holds no fact.

Fields per fact:

| Field | Cloud and custom | On device (slim) |
|---|---|---|
| `fact` | yes | yes |
| `title` (up to 50 characters) | yes | yes |
| `event_date` (`YYYY-MM-DD`, only when stated or clearly implied) | yes | yes |
| `passing` (true when the fact is over in days or weeks) | yes | yes |
| `namespace`, `tags` (1 to 3), `tone`, `entities` | yes | no |

An on-device model is held to `EXTRACTION_SCHEMA_SLIM` by the engine's grammar. Cloud and custom backends answer free-form and `json_body` finds the JSON in the text.

### 6.2 Guards in `parse_extraction`

These run in order, so that a small model's inventions do not become memories.

1. A "none" answer gives no facts.
2. A fact under three words is dropped.
3. A fact that restates a question is dropped (`restated_question`): the subject followed by "is asking", "asks", "wants to know", "is wondering", "is curious" and similar.
4. A fact that shares no content word with the message is dropped. A content word has three letters or more and is not on the filler list, or is a number of two or more digits.
5. An exact repeat of an earlier fact is dropped.
6. A fact that restates an earlier one is dropped (`restates`: four fifths of its content words, and at least two, are already in the earlier fact).
7. A missing title becomes the opening of the fact (`clean_title`, 50 characters).
8. Only the first three facts are kept.

Then `drop_near_duplicates` in `commands/chat.rs` embeds the facts and keeps only the first of any two closer than cosine 0.86 (`SAME_MESSAGE_DUPLICATE_SIMILARITY`).

### 6.3 Name correction

`memory_prompts::detect_name_correction` looks for a cue ("spelling", "should be", "correct name", "same person", "typo", "not ..." and others) plus two capitalised names in a pattern such as "X, not Y" or "X should be Y". It returns the wrong name and the right name.

When it finds one, two things happen:

- In `memory_jobs::process`, unless call 2 already produced a fact containing both names, this fact is put first: "{wrong} and {right} are the same person; the correct name is {right}. Earlier memories that say {wrong} mean {right}." Its title is "Name correction: {wrong} is {right}".
- In the storage path, live memories whose text contains the wrong name are put first among the candidates (7.2).

### 6.4 How the wording was chosen

The extraction prompt was scored on the memory bench (`labs/memory_bench/cases.json`, runner `memory_bench.rs`). The results per model, and the wording rounds, are in [../BENCHMARKS.md](../BENCHMARKS.md), section 1. No bench figures are repeated here.

---

## 7. Storing one fact

`commands/chat.rs` `run_memory_background` runs once per fact. The steps below are in the order the code runs them.

### 7.1 Embedding

The fact text alone is embedded (`generate_local_embedding(fact)`). The title is not part of the embedded text here. This vector is used for the candidate search, for the duplicate check, and is the embedding stored with the memory.

Other writers build the embedded text with `memory::embedding_text(title, content)`, which joins the title and the fact: the onboarding insert (`lib.rs` `insert_memory`), the memory edit command, backup restore, and the one-time `memory::reembed_if_scheme_changed` that runs at startup when the stored scheme name differs from `EMBEDDING_SCHEME` (`title_fact_v1`). The chat pipeline does not call `embedding_text`.

### 7.2 Candidates

1. `memory::hybrid_search` runs with the fact's embedding, the names found in the user's message at chat time (not names from the fact), no namespace or tag hints, across all sessions and namespaces, limit 15.
2. Results with a score below 0.35 are dropped. The rest are sorted by score.
3. The other facts of the same message are removed (`bg_sibling_ids`).
4. For a name correction, up to ten live memories whose text contains the wrong name are put first, each with a score of 1.0.
5. The list is cut to 10 for a cloud backend and 4 otherwise (custom endpoint and on device).

### 7.3 Duplicate check

Each remaining candidate is checked. The fact is a duplicate when the candidate's score is above 0.98, or when the cosine between the fact's embedding and the candidate's stored embedding is above 0.93. A duplicate is not stored and the outcome is `duplicate`.

The check sees only the candidates left after 7.2. A candidate added for a name correction carries a score of 1.0, which is above the 0.98 line. As the code is written, a fact in a name-correction message is therefore treated as a duplicate whenever a live memory holds the wrong name, and is not stored. This is read from the code and has not been confirmed by running it.

### 7.4 Call 3

Which call runs depends on the backend and on what call 2 gave:

| Situation | What happens |
|---|---|
| No candidates, and the fact has a title | Call 3 is skipped. The fact is stored with call 2's title and fields and no links. |
| On device, the fact has a title, there are candidates | The fact is stored first. `ask_llm_for_links_only` runs afterwards (7.9). |
| Anything else (cloud and custom with candidates; any `Remember:` text, which has no title) | `ask_llm_for_relationships` runs now. It returns a title, relationships and the extra fields. |

`ask_llm_for_relationships` returns relationships of type `updates`, `contradicts`, `related` or `none`, each with a confidence and a reason. Its title is used only when call 2 gave none. Its extra fields fill only the fields call 2 left empty (`merge_extras`). On a `.gguf` backend it is held to `RELATIONSHIP_SCHEMA`.

Answers from an on-device model pass `lib.rs` `filter_local_links`. Every `updates` becomes `contradicts` with the reason prefixed "Possibly an update: ", so the card asks the user instead of the model replacing a memory by itself. The function also has a confidence floor (0.0 today) and a list of dropped types (empty today).

If call 3 fails, the fact is stored without links. Its title is call 2's, or `generate_title_from_content` when there is none.

More detail is in [RELATIONSHIP_DETECTION_ARCHITECTURE.md](RELATIONSHIP_DETECTION_ARCHITECTURE.md).

### 7.5 Title of a `Remember:` memory

A `Remember:` memory is titled with the first 60 characters of its own text, with "…" added when the text is longer. Call 3's title is not used for it.

### 7.6 Contradiction

A contradiction is any `contradicts` relationship with confidence 0.65 or higher.

- **`Remember:`** The memory is stored regardless. The `contradicts` link is stored with it and no card is raised.
- **Cloud and custom:** a card is raised and the function returns without storing the fact. See 8.1.
- **On device:** the memory is stored, and the card is raised afterwards about the stored pair. See 8.2.

### 7.7 Fields and the insert

`NLPEnhancer::enhance(fact)` gives a namespace, an event type and an event date by keyword rules. `NLPEnhancer::extract_entities(fact)` gives the name list stored as JSON in `entities_detected`.

The model's fields win when present and valid (`memory_extras.rs`):

| Field | Rule | Fallback |
|---|---|---|
| namespace | `resolve_namespace`: kept only if it is on the canonical list | the NLP namespace |
| event date | `validate_event_date`: must parse, not be before 1900, not be after today, and not be January 1 unless the message mentions it | the NLP event date |
| tags | `clean_tags`: lower-cased, no repeats, none over 30 characters, at most five | none. NLP tags are not used |
| tone | `tone_to_sentiment`: mapped to a label and a score | neutral, 0.0 |
| entities | `clean_entities`: pronouns, calendar words and bare numbers removed, at most twelve | names from the name finder |
| event type | not supplied by the model | the NLP event type |

Because the slim call 2 has no namespace, tags, tone or entities, an on-device memory stored through the store-first path has the NLP namespace, no tags and neutral tone.

`memory::insert_memory` then writes the row: title, the fact as `content`, source type, session, embedding, user, namespace, `is_syncable` true, `is_shareable` false, `entities_detected`, event type, event date, and the user's message as `original_text`.

After the insert:

- `provenance_json` is set to `{"annotated": "at write", "by": <backend>, "requested": <true for Remember:>}`.
- `memory::set_memory_extras` writes `tags`, `sentiment_label` and `sentiment_score`.
- `memory::insert_memory_entities` writes one `memory_entities` row per entity. The model's list is used when it gave one. Otherwise the name finder's words are used when they have three letters or more and a score of at least 0.6: person, place or org by label, and "thing" for any other capitalised word.

### 7.8 Passing facts

When the model marked the fact `passing`, the row gets `is_ephemeral = 1` and `expires_at` set by `memory_extras::passing_expiry`: 14 days (`PASSING_DAYS`) after the event date when that is in the future, otherwise 14 days from now. After that day recall leaves the memory out (2.4). The row is not deleted by this path. `commands/memory.rs` `keep_memory` clears the expiry when the user chooses to keep it.

HIPAA mode does not use these columns. It stores nothing (section 4).

### 7.9 Links

On device with a titled fact and candidates, `ask_llm_for_links_only` runs now that the memory is on disk, under the `LINKS_SCHEMA` grammar. Its answer passes `filter_local_links`. If the call fails the memory stays, without links.

Each relationship other than `none` is stored with `memory::create_memory_link`:

| Column | Value |
|---|---|
| `source_memory_id` | the new memory |
| `target_memory_id` | the existing memory |
| `relation_type` | `related`, `contradicts` or `updates` |
| `confidence` | the model's value, 0.75 when it gave none |
| `notes` | the model's reason |
| `created_by` | `local_llm` on device, `llm` otherwise |

`create_memory_link` accepts only the four types in `RELATION_TYPES` (`contradicts`, `updates`, `related`, `resolves`; migration 0022). `resolves` is written only when a card is answered with an explanation.

An `updates` link has a side effect: the target memory's `temporal_status` is set to `superseded` (`memory::set_temporal_status`) and the event `memory-superseded` is sent. The old memory leaves recall and stays in the database. `commands/memory.rs` `restore_memory` undoes this.

`updated_at` is then bumped on the new memory and on every linked memory so the change syncs.

### 7.10 Events to the window

`memory-processing-started` is sent when the fact's work begins. `memory-processing-complete` is sent with status `stored`, `skipped`, `contradiction` or `error`. The session's extraction outcome is recorded as `stored`, `duplicate`, `contradiction`, `model_declined` or `error`.

---

## 8. Conflict cards

A card asks the user about two memories that disagree, or about a possible update. The window shows it with `ConflictResolutionModal.jsx`.

Every card is saved as a row in `memory_cards` (migration 0024) before the event is sent, through `memory::insert_memory_card`. The window may be closed or paused when the card is raised. On open and on every return, `App.jsx` calls `list_pending_memory_cards` and shows the oldest unanswered card. `answer_memory_card` marks a card answered. On Android, `card_notice` also posts a notification: "Zynkbot has a question about a memory".

The card is about the first `contradicts` relationship in the model's answer.

### 8.1 Cloud and custom: ask first

The fact is not stored. The card's payload holds the old memory (`memoryA`), the new fact (`memoryB`), the reason, the confidence, all of call 3's relationships, and a `pending_memory` with the content, title, embedding and original message. The event is `contradiction-detected`.

The answer goes to `commands/memory.rs` `resolve_memory_conflict_v2`:

| Decision | Effect |
|---|---|
| `keep_old` | Nothing is stored. |
| `keep_new` | The new memory is stored. The old one is marked `superseded` and linked from the new one with `updates`. Nothing is deleted. |
| `not_a_contradiction` | The new memory is stored. No link is made to the old one. |
| `keep_both` | The new memory is stored and linked `contradicts`, note "user_acknowledged". If the card was the "Is this an update?" form (reason starts "Possibly an update"), the link is `related` instead. |
| `both_with_explanation` | The new memory is stored and linked `contradicts`, note "resolved:explained". The explanation is stored as a third memory (source type `user_explanation`) and linked `resolves` to both. |

In every storing case the other relationships from call 3 are also written as links. Links made by the user's choice have `created_by` `user` and confidence 0.95.

**What a memory stored this way lacks.** It is written by `lib.rs` `store_pending_memory`, not by the path in section 7. It gets the content, title, embedding, original message, the NLP namespace, event type and event date, and `entities_detected`. It does not get call 2's namespace, tags, tone, `memory_entities` rows, the passing flag, or `provenance_json`. Its source type is always `conversation`, also for a hands-free message.

### 8.2 On device: store first, ask after

The memory is already stored, with its `contradicts` link. The payload holds `stored_memory_id` and both memories, and has no `pending_memory`.

The answer goes to `commands/memory.rs` `resolve_stored_conflict`:

| Decision | Effect |
|---|---|
| `keep_new` | The link between the pair is removed. The old memory is marked `superseded` and linked from the new one with `updates`. |
| `keep_old` | The link is removed and the new memory is deleted. |
| `not_a_contradiction` | The link is removed. Both memories stay. |
| `keep_both` | Nothing changes. The `contradicts` link stays. |
| `keep_both_related` | The link's type is changed to `related`. The window sends this for "keep both" on the "Is this an update?" card. |
| `both_with_explanation` | The explanation is stored as a memory and linked `resolves` to both. The `contradicts` link stays. |

---

## 9. What is written to the database

Column definitions are in [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) and in `zynkbot_rust/src-tauri/migrations/`. This is what the pipeline writes.

| Table | Written by | What |
|---|---|---|
| `memory_jobs` | `memory_jobs::enqueue` | one row per queued message |
| `memories` | `memory::insert_memory`, then updates | the fact and its fields (7.7); `is_ephemeral` and `expires_at` for passing facts; `temporal_status` when replaced; `provenance_json`; `tags`, `sentiment_label`, `sentiment_score` |
| `memory_entities` | `memory::insert_memory_entities` | name, canonical (lower-case) name, kind |
| `memory_links` | `memory::create_memory_link` | relationships from call 3, sibling links, links from card answers |
| `memory_cards` | `memory::insert_memory_card` | one row per card, with the payload as JSON |

In SQLite, `embedding` is a BLOB, `tags` and `entities_detected` are TEXT holding JSON, and dates are TEXT.

Indexes on `memories` (migrations 0001, 0009, 0013): `memories_namespace_idx`, `memories_user_namespace_idx`, `idx_memories_ephemeral`, `idx_memories_expires_at`, `idx_memories_event_date`, `idx_memories_event_type`, `idx_memories_sentiment`, `memories_shareable_idx`, `idx_memories_sync_id`, and the collection indexes. `memory_links` has indexes on source, target, type, confidence and `sync_id`. `memory_entities` has indexes on `canonical` and `memory_id`. There is no index on `user_id` alone, `session_id` or `created_at`.

---

## 10. Data flow diagram

```
MESSAGE  (send_message_with_memory -> generate_reply)
   |
   |- Weather question? -> answered from Open-Meteo, return. No memory work.
   v
[1] CONTAINMENT
   |   child: OpenAI Moderation, then local classifier
   |   others: check_containment -> enforce (block, warn, or pass)
   v
[2] RECALL
   |   names in the message (bert-base-NER) + embedding (all-MiniLM-L6-v2)
   |   hybrid_search: top 30 cloud / 10 otherwise, floor 0.35
   |     leaves out superseded, completed and expired memories
   |   Zynkbot question: _zynkbot namespace only, no link traversal
   |   one hop along links: up to 8 linked memories, strongest first
   v
[3] PROMPT  (build_prompt)
   |   memories in the prompt: 30 cloud / 7 custom / 4 on device
   |   cloud and custom: the MEMORY LINE instruction is included
   v
[4] REPLY = CALL 1  (streamed)
   v
[5] READ THE REPLY
   |   web search marker: typed -> strip, return the query, carry on
   |                      hands-free -> search now, return
   |   MEMORY line -> yes / no / absent; hidden from the user
   v
[6] RESPONSE RETURNED TO THE USER
   v
[7] QUEUE DECISION
   |   skipped: skip_memory_storage, HIPAA mode, ungrounded KB answer
   |   gate: is_memory_worthy and not a translate/rewrite request
   |   call 1: cloud/custom need "yes"; on device anything but "no"
   |   "Remember:" passes the gate and call 1
   |   -> row in memory_jobs, or outcome skipped / gate_skipped / model_declined
   v
============ background worker (memory_jobs.rs), one job at a time ============
   v
[8] "Remember:" job -> its text goes straight to [10], no call 2
   v
[9] CALL 2  (extract_facts)
   |   up to 3 facts: fact, title, event_date, passing (+ namespace, tags,
   |   tone, entities on cloud and custom)
   |   guards: < 3 words, restated question, no shared word, repeats;
   |           cosine >= 0.86 between two facts keeps the first
   |   name correction detected -> a correction fact is put first
   |   zero facts -> outcome model_declined, stop
   v
[10] FOR EACH FACT  (run_memory_background)
   |
   |  a. embed the fact text
   |  b. candidates: hybrid_search (15) -> score >= 0.35 -> sort
   |       -> remove sibling facts
   |       -> name correction: memories holding the wrong name go first
   |       -> keep 10 cloud / 4 otherwise
   |  c. duplicate check on those candidates
   |       score > 0.98 or cosine > 0.93 -> not stored (outcome duplicate)
   |  d. call 3
   |       no candidates + title        -> skipped
   |       on device + title            -> deferred to step h
   |       otherwise                    -> ask_llm_for_relationships
   |       (on device: "updates" becomes "contradicts")
   |       call fails                   -> store without links
   |  e. contradiction (contradicts, confidence >= 0.65), not "Remember:"
   |       cloud/custom -> save card, send contradiction-detected,
   |                       STOP without storing            -> [12]
   |       on device    -> carry on, card raised in step j
   |  f. fields: model's values when valid, NLP as fallback
   |  g. insert_memory; then passing expiry, provenance_json,
   |     tags and tone, memory_entities rows
   |  h. on device with title and candidates: ask_llm_for_links_only
   |  i. store links; "updates" marks the old memory superseded
   |  j. on device contradiction: save card, send contradiction-detected -> [12]
   v
[11] AFTER THE LAST FACT
   |   two or more stored -> linked "related" (said in the same message)
   v
[12] CARD ANSWERED BY THE USER  (any time later)
       cloud/custom: resolve_memory_conflict_v2 (stores the pending memory
                     through store_pending_memory, unless keep_old)
       on device:    resolve_stored_conflict (both memories already stored)
       then answer_memory_card
```

---

## 11. Key files

| File | Role in the pipeline | Functions |
|---|---|---|
| `commands/chat.rs` | Entry point, recall, reply, call 1 parse, queue decision, call 2, storage path | `send_message_with_memory`, `generate_reply`, `find_web_search_marker`, `split_inline_memory_marker`, `explicit_remember`, `extract_facts`, `drop_near_duplicates`, `merge_extras`, `run_memory_background` |
| `memory_jobs.rs` | The job queue | `enqueue`, `take_next`, `ensure_worker`, `process` |
| `memory_prompts.rs` | Call 2 prompt, schemas and guards | `extraction_prompt_with`, `EXTRACTION_SCHEMA`, `EXTRACTION_SCHEMA_SLIM`, `parse_extraction`, `restates`, `clean_title`, `detect_name_correction` |
| `memory_extras.rs` | Field validation, passing expiry, rewrite check | `validate_event_date`, `resolve_namespace`, `clean_tags`, `tone_to_sentiment`, `clean_entities`, `passing_expiry`, `is_rewrite_request` |
| `lib.rs` | Call 3, backend calls, pending-memory storage | `ask_llm_for_relationships`, `ask_llm_for_links_only`, `call_backend_with_schema`, `filter_local_links`, `is_query_about_zynkbot`, `store_pending_memory`, `generate_title_from_content` |
| `memory.rs` | Search and writes | `hybrid_search`, `hybrid_score`, `query_hints`, `insert_memory`, `set_memory_extras`, `insert_memory_entities`, `create_memory_link`, `set_temporal_status`, `insert_memory_card`, `embedding_text` |
| `commands/memory.rs` | Card answers and card list | `resolve_memory_conflict_v2`, `resolve_stored_conflict`, `list_pending_memory_cards`, `answer_memory_card`, `restore_memory`, `keep_memory` |
| `conversation_engine.rs` | Prompt and gate | `build_prompt`, `build_memory_context`, `is_memory_worthy`, `is_api_model` |
| `commands/safety.rs`, `containment.rs`, `safety_classifier.rs` | Containment | `check_containment`, `ContainmentLayer::enforce`, `check_openai_moderation`, `should_block` |
| `nlp_enhancer.rs` | Name finder and keyword rules | `extract_entities`, `enhance`, `detect_event`, `detect_namespace` |
| `llm/local_embeddings.rs` | Embeddings | `generate_local_embedding`, `cosine_similarity` |
| `weather.rs` | Weather shortcut | `detect`, `answer` |

Code that exists but is not part of this pipeline:

- `commands/memory.rs` `pre_check_memory` is a registered Tauri command for a duplicate check. Nothing in the pipeline or in the window's code calls it. The duplicate check that runs is the one in 7.3.
- `llm_fact_extractor.rs` is marked unused (`#![allow(dead_code)]`).
- `question_extractor.rs` is reached only through on-demand commands (`check_question_worthiness` and those in `commands/nlp.rs`), not from the chat path.
- `lib.rs` `ask_llm_for_extras` is called only by `commands/memory_report.rs` `enrich_memory_backlog`, which fills in the fields of older memories.

---

## 12. Not verified against the code

These points were in earlier versions of this document. The code does not settle them, so they are listed here and not stated as fact.

- How many memories the in-process search has been tested with. An earlier version said 500,000 rows. No test or measurement in the repository shows this.
- Memory use of the loaded models and of the vectors.
- Request timeouts for the cloud calls.
- Whether the window disables input while a reply is in flight.
