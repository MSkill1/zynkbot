# Zynkbot Prompt Construction Pipeline

*Checked against the code on 2026-11-10 (`v1`, a7599eb).*

Code: `zynkbot_rust/src-tauri/src/conversation_engine.rs` builds the prompt. `zynkbot_rust/src-tauri/src/commands/chat.rs`, `generate_reply`, calls it and adds the blocks described under [Final Assembly](#final-assembly).

---

## Overview

Before each reply, Zynkbot builds one prompt text from these parts:

1. Instructions (the "system prompt"), with the date and, when known, the user's name
2. Recent conversation history
3. Recalled memories
4. The user's current message

`generate_reply` can then put more blocks in front of that text: a conversation mode, Knowledge Base passages, passages from the built-in guides, and the text of web pages named in the message. When pictures are sent, a short block about them goes at the end.

---

## Three Variants

`ConversationEngine::build_prompt` writes one of three versions of the instructions. It branches first on the `is_api_model` argument, then on `self.on_device`.

| Variant | Used for | How it is chosen |
|---|---|---|
| **Full** | Anthropic, OpenAI, xAI, Mistral | `ConversationEngine::is_api_model(backend)` is true: the backend is not `local`, does not end in `.gguf`, and is not `custom` |
| **On-device** | A local GGUF model, on desktop or on the phone | `generate_reply` calls `.with_on_device(...)` with true when the backend name ends in `.gguf` or contains `local` |
| **Slim** | The custom OpenAI-compatible endpoint (for example Ollama) | Neither of the above |

On Android a local GGUF model runs on the phone and gets the on-device variant. A phone can also use a paired desktop's Ollama through the desktop's proxy route (`zynksync.rs`, `handle_ollama_proxy`). That is the custom backend, so it gets the slim variant.

### Full (cloud)

In this order:

**1. Date.** The first line is `Today's date is <weekday>, <month> <day>, <year> at <time> <zone>.` (format `%A, %B %-d, %Y at %-I:%M %p %Z`). The weekday and time are included so the model can answer "what day is it" and "what time is it" without a search.

**2. Companion voice.** One line of identity ("You are Zynkbot, a personal AI companion.") and a `COMPANION VOICE` section of 12 bullets. They cover: a long-term companion that never claims to be human; honesty about uncertainty; no automatic flattery; no encouraging dependency, with professional-help suggestions kept for real clinical or legal stakes; acknowledging feelings before solutions; no performed emotions; gentle correction; not undercutting that someone feels less alone; memory as a tool and not a surveillance record; proportionate replies; asking for one or two missing details when they would change the answer; and that the user's data belongs to them.

**3. Memory framing.** The model is told that the memories below were recalled by similarity and entity matching, how to use them, to give the date of each plan it lists, and that they were filtered for relevance to the current question.

**4. Web search.** The model is told to answer date and time questions from the prompt, and to write `WEB_SEARCH_NEEDED: [query]` when the question needs current information (news, weather, prices, sports results, recent events). Two examples are given. It must not use the marker for the user's own history.

What happens to the marker: `generate_reply` finds it (`find_web_search_marker`) and returns the suggested query to the frontend. The frontend shows the query in an editable box (`ChatMessage.jsx`) and runs the search when the user confirms. A hands-free voice turn runs the search without asking.

**5. User name (only when known).** `The user's name is <name>.` plus a line telling the model to use the name now and then, never as the first word. The name comes from `memory::get_user_display_name`, which reads `user_profile.json`: `preferred_name`, else `full_name`. When there is no name, this part is absent and the word "User" is used as the subject in the MEMORY LINE block.

**6. The MEMORY line.** A `MEMORY LINE` block asks the model to end every reply with one final line, exactly `MEMORY: yes` or `MEMORY: no`. Yes means the message states a personal fact about the user; a fact stated inside a question counts, and the prompt gives the raspberry-pie example. No means a question alone, chit-chat, general knowledge, a request about stored memories, text to translate or text quoted from someone else, or a message only about Zynkbot. The model is told not to write the fact itself. The line is removed from the displayed reply. On yes, a separate background call lists the facts (`memory_prompts.rs`, `extraction_prompt_with`), up to three. See the parts on call 1 and call 2 in [MEMORY_PROCESSING_PIPELINE.md](MEMORY_PROCESSING_PIPELINE.md).

No variant asks the model for a `MEMORY_EXTRACT:` line. The reply parser still accepts one from a model that writes it, and treats it as yes.

### On-device (local GGUF)

A single short block, rewritten on 2026-11-09. It has four paragraphs:

1. Voice: a warm, honest, direct companion; never claims to be human; no flattery, no therapy language; acknowledge grief or frustration before solutions.
2. Memories: they are facts from the user's own life with dates; use them only when they bear on the message; if the user asks for something the memories do not contain, say so and never substitute another figure or date.
3. Statements: when the user tells you something, reply briefly as a friend would, take it as true and current even where a memory says otherwise, and do not discuss whether it is stored. It is saved separately.
4. Web search: `WEB_SEARCH_NEEDED: [query]` for news, weather, prices or recent events; never for the date, the time, or the user's own history.

Then, when the name is known: `The user's name is <name>. Use it occasionally, never as the first word.` Then: `Keep it short: a few sentences, unless asked for more.`

Differences from the full variant:

- No date at the top. The date goes after the memories and the conversation, so the start of the prompt stays the same from turn to turn and the engine can reuse what it already read.
- No MEMORY LINE block. For a local model, every message that passes the worthiness gate goes to the extraction call, unless the model wrote an explicit no.
- No "ask for missing details first" instruction.

`labs/reply_bench` measures a change to this block against the model. A test in `conversation_engine.rs` keeps the instructions under 215 words.

### Slim (custom endpoint)

The date line first, then three condensed paragraphs: the voice, the memories (including that the model cannot change a memory itself and should point the user to Memory Manager), and the web-search rule. Then the name line and the same MEMORY LINE block as the full variant.

---

## Conversation History

**Function:** `build_conversation_context`

- Full: up to 40 messages.
- On-device: up to 8 messages, each cut to 400 characters (`ON_DEVICE_MESSAGE_CHARS`). The window moves in blocks of four messages (`ON_DEVICE_HISTORY_BLOCK`), not one at a time, so the start of the history stays identical between turns.
- Slim: up to 8 messages, not cut. The window slides one message at a time.

The most recent messages are taken. Format:
```
RECENT CONVERSATION:
USER: [message]
ASSISTANT: [response]
USER: [message]
ASSISTANT: [response]
...
```

---

## Memory Context

**Function:** `build_memory_context`

- Full: up to 30 memories.
- On-device: up to 4 memories, the fact only, each cut to 300 characters (`ON_DEVICE_MEMORY_CHARS`).
- Slim: up to 7 memories.

A memory whose text is identical to the current message is removed before the limit is applied.

**Heading.** `USER'S STORED MEMORIES:` for full and slim. `WHAT THE USER HAS TOLD YOU BEFORE:` on-device.

**Text per memory (full and slim).** The stored fact first, then `(you said: "...")` with the user's own words when they differ from the fact, cut to 200 characters (`memory_line`). **On-device:** the fact alone.

**Event day.** When the day the fact is about differs from the day it was stored, the line says "about <date>" (`event_note`).

**Grouping by source message.** One user message can produce several facts (same `original_text`). Two or more recalled facts from one message are written as one block, the message once and the facts as bullets:
```
USER'S STORED MEMORIES:
1. (October 24, 2026) You said: "I'm flying to Denver Thursday for my nephew's wedding, and the cat stays with Dave"
   - Matt is flying to Denver on Thursday for his nephew's wedding.
   - Matt's cat stays with Dave while he is away.
2. (October 20, 2026) Matt started a keto diet. (you said: "started keto today, wish me luck")
...
```
The quote is cut to 400 characters (300 on-device).

**Which memories arrive here.** `generate_reply` passes the hybrid search results, highest score first, followed by the memories reached through one link from them (at most 8). The search returns up to 30 memories for a cloud backend and 10 otherwise. See the part on memory recall in [MEMORY_PROCESSING_PIPELINE.md](MEMORY_PROCESSING_PIPELINE.md).

---

## KB Context (Optional)

If the KB button is on, `generate_reply` searches the user's indexed documents and `kb_prompt.rs`, `build`, writes a block that goes in front of the prompt.

- Cloud and custom backends: 10 chunks; when nothing clears the threshold, the 5 best are shown as weak matches.
- Local model: 4 chunks; 2 weak matches. A comment in `chat.rs` records why: ten chunks did not fit the phone's 4K window.
- Similarity threshold: 15%.

The wording depends on what the search found:

1. **Found** (at least one chunk above the threshold): the matching chunks are listed as `Document N`. The model is told to answer from these passages, to quote the exact row or line it relied on, and, if the passages do not contain the answer, to say plainly that the knowledge base has no record of it. It is told not to suggest a web search.
2. **Weak only** (chunks returned, none above the threshold): the best few are listed as `Weak match N`. The model is told that nothing clearly matched, to answer from a weak match only if it contains the answer, and otherwise to say the knowledge base has no record of this. It must not answer from general knowledge as if it came from the documents.
3. **Nothing** (no chunks returned): the model is told to say plainly that the knowledge base has no record of this, not to invent an answer, and to keep any general-knowledge addition short and labelled as not from the documents.

Every block opens with `=== KNOWLEDGE BASE SEARCH (user pressed the KB button) ===` and closes with `=== END OF KNOWLEDGE BASE SEARCH ===`. Only the first outcome counts as grounded. For the other two the reply carries a `kb_note` that the UI shows, and no memory is extracted from that reply (an explicit "Remember:" still stores).

**Built-in guides.** When the KB button is off and the question is about Zynkbot itself, `generate_reply` searches the built-in user guides and `kb_prompt.rs`, `build_guides`, writes a `=== ZYNKBOT'S OWN GUIDES ===` block: up to 5 passages for a cloud backend, 3 otherwise, with a 30% threshold. If nothing clears the threshold, no block is added.

---

## Final Assembly

What `build_prompt` returns for the full and slim variants:

```
[instructions, starting with the date; MEMORY LINE block last]

RECENT CONVERSATION: ...            (if any)

USER'S STORED MEMORIES: ...         (if any)

USER'S QUESTION: [user input]

YOUR RESPONSE:
```

What it returns for the on-device variant:

```
[instructions, no date]

WHAT THE USER HAS TOLD YOU BEFORE: ...   (if any)

RECENT CONVERSATION: ...                 (if any)

Today's date is <date and time>.

USER'S MESSAGE: [user input]

YOUR RESPONSE:
```

On-device, memories come before the conversation, so the thread is the last thing the model reads before the message. The label is `USER'S MESSAGE`, not `USER'S QUESTION`.

`generate_reply` then adds blocks around that text. The final order is:

```
[web page text]              when the message contains a web address (web_read.rs, up to 2 pages)
[ZYNKBOT'S OWN GUIDES]       when found
[KNOWLEDGE BASE SEARCH]      when the KB button is on
[Mode: ...]                  when a conversation mode is set (chat_modes.rs: tool, debate, premise, editing)
[the build_prompt text above, ending with YOUR RESPONSE:]
[Pictures attached, in order: ...]   when pictures are sent (cloud backends only)
```

The pictures block lists each picture and asks the model to end its reply with a `PICTURE:` line describing the picture sent with this message.

**Hands-free turns.** On Android, `android_jni.rs` appends a fixed note (`HANDS_FREE_NOTE`) to the transcript before calling `generate_reply`. The note tells the model that the text came from a wake word, to reply `NO_QUERY` unless it is plainly a request to an assistant, and to write a reply fit for reading aloud.

**Local model chat formats.** The text above is not sent to a local model as is. `llm/local_models.rs`, `build_prompt_for_model`, splits it at the `USER'S MESSAGE:` or `USER'S QUESTION:` label (`split_engine_prompt`) and wraps the two halves in the chat format of the model family (Qwen, Gemma, Gemma 4, Llama 3, Mistral and others), chosen by file name and then by the GGUF architecture. The table of formats is in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## Size

| Variant | History | Memories | Instructions (words in the source) |
|---|---|---|---|
| Full | 40 messages | 30 memories | about 685, plus about 110 for the MEMORY LINE block |
| On-device | 8 messages (400 characters each) | 4 memories (300 characters each) | about 185, about 200 with the name line and "Keep it short" |
| Slim | 8 messages | 7 memories | about 225, plus about 110 for the MEMORY LINE block |

The word counts were taken from the source text at this commit. Token counts depend on the model's tokenizer and were not measured for this document; earlier token estimates in this file have been removed because they were not verified against the code.

Why the on-device variant is small: a local model's context window is 4,096 tokens on Android and 8,192 on desktop (`llm/local_models.rs`, `n_ctx`), and a local model reads the prompt slowly. Engine settings are in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## Relationship to Memory Recall

The prompt is built after memory retrieval is complete. `build_memory_context` does not treat search results and linked memories differently. It removes a memory identical to the message, applies the limit for the variant, and groups facts that came from one message.

See [MEMORY_PROCESSING_PIPELINE.md](MEMORY_PROCESSING_PIPELINE.md) for how memories are retrieved before this step.
