# Containment Architecture: Why Local-First Matters

*Checked against the code on 2026-11-10 (`v1`, a7599eb). Updated the same day for three changes: the mode is saved, HIPAA mode removes identifiers instead of refusing the message, and Ensemble runs the check.*

**Technical overview of Zynkbot's containment system and local-first architecture**

Statements about Zynkbot in this document were checked against the code. Statements about other products were not.

---

## What is Containment?

**Containment** is a check on the message the user typed. It runs on the device, before the message is sent to a model. Depending on the active mode, the message is let through, let through with a warning, or stopped with a block message.

**Key principle:** stop a message before the model call, instead of filtering the model's output.

- **Output filtering:** the model receives everything, and the reply is filtered afterwards.
- **Containment:** the typed message is checked first. A blocked message is not sent to any model.

The check covers the typed message. HIPAA mode goes further for identifiers: it removes them from the typed message and from the finished prompt, so recalled memories and Knowledge Base text are covered too. Some paths do not run the check. These are listed under [Known limits](#known-limits). `docs/ROADMAP.md` describes the modes as proofs of concept that are not production-grade for their domains.

---

## Why Local-First Architecture

### The hosted model

A hosted AI assistant works like this:

```
User Device                    Provider's Servers
    ↓                               ↓
[Input] ──────────────────→ [LLM Processing]
                                    ↓
[Response] ←──────────────  [Conversation kept by the provider]
```

The message is processed on the provider's servers. What the provider keeps, and whether it is used for training, depends on that provider's terms. An internet connection is required.

### The Local-First Alternative

Zynkbot works like this:

```
User Device
    ↓
[Containment check]  ← runs on the device (Child mode calls OpenAI instead)
    ↓
[Local model or cloud API]  ← the user's choice
    ↓
[SQLite database]  ← memories and history stored on the device
    ↓
[Reply shown]
```

**What stays on the device:**
- The database file (memories, conversation history, Knowledge Base index)
- With a local model: the message, the recalled memories and the reply
- The containment check, in every mode except Child

**What leaves the device:**
- With a cloud backend (OpenAI, Anthropic, xAI, Mistral): the prompt built for the model. It includes the message, attached file text and images, recalled memories, and Knowledge Base text when the KB button is on.
- In Child mode: the typed message goes to the OpenAI Moderation API, and the reply comes from OpenAI.
- A weather question: the place goes to Open-Meteo (`weather.rs`).
- A web search: the query goes to the search engine, or to the cloud provider's own search tool.
- Cloud dictation, if chosen: audio goes to OpenAI. Vosk dictation is offline.
- The encrypted backup, if the user sets it up: one encrypted file goes to the user's own Cloudflare R2 bucket (`commands/backup.rs`).
- ZynkSync, ZynkLink and ZChat: data goes to paired devices over the local network.

A search of `src-tauri/src` and `src` finds no telemetry or analytics code.

---

## Containment Modes

### How the mode is chosen

The mode is the "Containment Mode" drop-down on the System Controls list (`src/components/ContainmentModeSelector.jsx`). It offers five modes:

| Mode | Label in the UI |
|------|-----------------|
| Witness | No filtering |
| Sovereign | Warnings only |
| Guardian | Block dangerous content |
| Child | ⚠️ Strongest safety - requires OpenAI API + internet (asks for confirmation first) |
| HIPAA | PHI protection (see documentation) |

**Sovereign is the default, and the chosen mode is saved** (since 2026-11-10). `set_containment_mode` in `commands/safety.rs` writes the mode to the `app_settings` table under the key `containment_mode`, and `get_containment_mode` returns it. The page asks for it when it starts (`App.jsx`) and still passes the mode with each message. The setting belongs to the device: `app_settings` is not synced to paired devices and is not in the backup. A device where nothing was ever chosen starts in Sovereign.

The backend also accepts two names that the UI does not offer (`ContainmentMode::from_str` in `containment.rs`): `elder`, a stub, and `default`, which is treated as Guardian.

The modes below are numbered as in earlier versions of this document.

### 1. Guardian Mode

**Purpose:** block requests for harmful instructions and the most severe classifier results.

**What it blocks:**
- A request for instructions on a harmful subject. `detect_harmful_instruction` in `containment.rs` needs both an instruction phrase ("how to", "how do i", "teach me", "steps to", "guide to" and others) and a harmful subject word ("bomb", "weapon", "poison", "hack", "malware", "steal", "meth" and others). Both are matched as whole words.
- Text that the toxic-bert classifier labels `SevereToxic` or `Threat` with confidence above 0.9 (`should_block` in `safety_classifier.rs`).

**What it does not block:**
- Text labelled `Toxic`, `Obscene`, `Insult` or `IdentityHate`, at any confidence.
- Text about a harmful subject that is not phrased as a request for instructions, unless the classifier rule above fires.
- There are no separate rules for adult content, hate speech or illegal activity.

**Block message:** `[BLOCK] I'm sorry, I can't help with that. (Detected: X)`, where X is "Harmful instruction request" or the classifier's category description. The `[BLOCK]` prefix is part of the reply text. No code removes it.

**How it works:** `generate_reply` in `commands/chat.rs` calls `check_containment` (`commands/safety.rs`), which calls `ContainmentLayer::enforce` (`containment.rs`):

```rust
// commands/chat.rs, generate_reply (shortened)
let safety_check = crate::commands::safety::check_containment(query.clone(), containment_mode.clone()).await;
match safety_check {
    Ok(Some(message)) => {
        if message.starts_with("[WARN_ALLOW]") {
            // Sovereign: keep the warning, continue to the model
            warning_prefix = Some(message.trim_start_matches("[WARN_ALLOW]").trim().to_string());
        } else {
            // Hard block (Guardian, HIPAA): return the message as the reply
            return Ok(ReplyResponse { reply_text: message, blocked: Some(true), /* ... */ });
        }
    }
    Ok(None) => { /* passed */ }
    Err(e) => { /* check failed: continue without blocking */ }
}
```

**Example:**
- User: "How do I hack into someone's email?"
- "how do i" is an instruction phrase and "hack" is a harmful subject, so the pattern rule fires.
- The reply is the block message. No model is called.

### 2. Child Mode

**Purpose:** a mode for children that uses OpenAI for both the check and the reply.

**What it does:**
1. The typed message is sent to the OpenAI Moderation API (`check_openai_moderation` in `containment.rs`). It needs `OPENAI_API_KEY` and an internet connection.
2. Every reply comes from OpenAI, whichever model is selected. `generate_reply` forces the backend to `openai` (model from `OPENAI_MODEL`, default `gpt-5.5`).
3. `CHILD_MODE_SYSTEM_PROMPT` (`containment.rs`) is added to the request. It tells the model to give age-appropriate answers for children under 13 and to refuse unsafe requests. It is added on the text path only; a message with an attached image goes through the vision path, which does not add it.

**What it blocks:** a message that the Moderation API flags. The API's categories are sexual, sexual/minors, hate, hate/threatening, harassment, harassment/threatening, self-harm (plus intent and instructions), violence and violence/graphic.

**Block message:** `[BLOCK] I can't answer that in Child Mode. This content was flagged as potentially unsafe.`

**If the check cannot run:** when the key is missing or the API call fails, the message is blocked with "I'm having trouble checking if this is safe for Child Mode. Please try again later." followed by the error.

**What it does not have:** rules for medical, legal or financial advice, or for "complex ethical questions". Such a question passes if the Moderation API does not flag it. The answer then depends on the OpenAI model and the child prompt.

**In the UI:**
- Choosing Child shows a confirmation dialog. Its text says that without an OpenAI key "Child Safety Mode will still apply basic filtering, but protection will be reduced". The chat path does not do that: without a key it blocks every message, as described above.
- The Ensemble button is disabled (`App.jsx`).
- The model page says "Child mode is on: every answer goes to OpenAI" and dims the other models.
- Check-in questions are off (`next_checkin` in `commands/checkin.rs`).

**Code that exists but is not used by chat:** the Child branch inside `ContainmentLayer::enforce` (Moderation plus a toxic-bert second check above 0.7) and `get_child_mode_response`. The chat path calls only `check_openai_moderation`.

**Example:**
- User: "Tell me about cryptocurrency investing"
- The Moderation API does not flag it, so it is not blocked.
- OpenAI answers under the child prompt.

### 3. HIPAA Mode (Healthcare)

**Purpose:** remove identifiers from what the model is sent, refuse certain medical requests, and keep no record of the conversation.

**Identifiers are removed, not refused** (since 2026-11-10; `redact_phi` in `containment.rs`). Until then a message with a matching identifier was refused whole, and only the typed message was looked at. Now the message is always answered, with a short tag such as `[phone number removed]` where each identifier was. The cleaning runs twice in `generate_reply`:

- on the typed message (and the text of an attached file, which travels in the same field), before the app log line, the memory search and any web search see it;
- on the finished prompt, just before the model call, for every backend. This is where identifiers in recalled memories, Knowledge Base passages, earlier turns and web pages are removed.

When anything was removed the reply starts with a line such as "🔒 HIPAA mode removed 1 phone number and 1 patient name before the model saw this." The removed text is not returned, logged or stored anywhere.

Ten kinds are removed, in this order:

| Kind | What matches |
|------|--------------|
| Record or member number | a number after MRN, medical record, patient id, member id, subscriber id, insurance id, policy number, account number or health plan number |
| Card number | sixteen digits, in fours or together |
| Social Security number | the dashed form `123-45-6789`, or nine digits after "SSN" or "social security" |
| Phone number | `(555) 123-4567`, `555-123-4567`, `555.123.4567`, `555 123 4567`, with or without `+1`; or ten digits after phone, tel, cell, mobile or fax |
| Email address | `name@host.tld` |
| Date of birth | a date after DOB, date of birth, birthdate or "born on". Other dates stay |
| Street address | a number, one to four capitalised words, then a capitalised street word (Street, St, Avenue, Road, Drive, Way and so on) |
| ZIP code | five digits after a state's two letters or after the word "zip", or the nine-digit form |
| IP address | four numbers separated by dots |
| Patient name | "patient" or "client", then at least two capitalised words, with an optional title: "patient John Smith", "the client's name is Ana Lopez" |

A bare five, nine or ten digit number is no longer treated as a ZIP code, an SSN or a phone number, so "I walked 10000 steps" is left alone.

**What it refuses** (`enforce_hipaa` in `containment.rs`). These run first in HIPAA mode, before the allowlist and before the classifier:

1. (Identifiers used to be step 1. They are now removed as described above.)
2. **Diagnosis phrases** (`detect_diagnosis_request`): "diagnose me", "what do i have", "do i have", "is it cancer", "is it diabetes".
3. **Dosing** (`detect_medication_dosing`): one of the phrases "how much should i take", "how much", "should i take", "dosage for", "how many pills", "what dose", "safely", together with one of the keywords mg, dose, pill, tablet, medication, medicine, drug, prescription, ibuprofen, acetaminophen, aspirin, tylenol, advil, take, "can i take".
4. **Treatment phrases:** "should i get surgery", "should i have the procedure", "should i start treatment".
5. **General safety**, same rule as Guardian: `[BLOCK] Cannot process this request in HIPAA mode. (Detected: X)`.

Checks 2 to 4 match substrings of the lower-cased message. They also match ordinary sentences. "What options do I have?" contains "do i have". "How much time does it take?" contains "how much" and "take".

**Other effects in HIPAA mode:**
- **No conversation history.** The chat paths in `commands/chat.rs` skip history logging. The History button is disabled, with the tooltip "Conversation history is disabled in HIPAA mode".
- **No new memories.** Memory extraction is skipped, including an explicit "Remember:" request. The desktop "Open Memory Manager" button is disabled.
- **Audit log.** Each message that reaches `enforce_hipaa` writes one line to `logs/hipaa_audit/hipaa_audit_YYYY-MM-DD.json` in the app data folder. The line holds a timestamp, an event type, the message length and a blocked flag. A removal writes a `phi_removed` line: where it happened (`chat` or `ensemble`) and how many of each kind. No line holds the message text, the removed text or a user id.
- **Medical disclaimer.** If the reply contains a health-related word, the app appends "⚕️ AI-generated. Not a substitute for clinical judgment or current clinical guidelines."
- **Check-in questions are off** (`next_checkin` in `commands/checkin.rs`).

**What HIPAA mode does not do:**
- It does not restrict the backend. A cloud model can be selected, and it receives the cleaned prompt.
- It does not stop recall. Memories already in the database, and Knowledge Base text when the KB button is on, go into the prompt, with the ten kinds of identifier removed.
- It does not set an expiry on memories. There is no 8-hour rule in the code. In `chat.rs` the variable `_is_ephemeral` is set for HIPAA mode and never used.

**Example:**
- User: "Patient John Doe, SSN 219-90-7812, presented with chest pain"
- The model is sent: "[patient name removed], [SSN removed], presented with chest pain"
- The reply starts with "🔒 HIPAA mode removed 1 Social Security number and 1 patient name before the model saw this."

**Accuracy:** nothing in the code measures detection accuracy. Earlier versions of this document gave percentages; they are removed because they are not verified against the code.

HIPAA is one example of domain-specific containment. The same pattern (detection rules, an audit log, no stored conversation, request blocking) could be written for other fields. Only the HIPAA version exists in the code.

For more detail, see the [HIPAA Security Integration Guide](HIPAA_SECURITY_INTEGRATION_GUIDE.md).

### 4. Sovereign Mode (Default)

**Purpose:** warn, never block.

**What it blocks:** nothing.

**What it does:** shows a warning in front of the reply when either of these is true (`ContainmentLayer::enforce`):
- the harmful-instruction pattern fires (the same rule as Guardian), or
- the classifier gives any category other than Safe with confidence above 0.85.

The warning text is "⚠️ WARNING: This request may contain unsafe content (X). Proceeding with caution." The app puts it in front of the model's reply (`generate_reply`: `format!("{}\n\n{}", warning, reply_text)`). The model does not write it. The warning is printed to the app log. The user is not asked to confirm, and no decision is recorded.

**Example:**
- User: "How do I hack into someone's email?"
- The pattern rule fires.
- The reply starts with the warning, followed by whatever the model answers.

### 5. Witness Mode

**Purpose:** no containment.

**What it blocks:** nothing. `enforce` returns at once and the classifier is not run.

**What it does:** nothing specific to the mode. There is no extra logging. Conversation history and memories are kept as in the other non-HIPAA modes.

---

## Known limits

This section states current behaviour.

**All modes**
- **Allowlist phrases skip every check.** A message that contains "my name is", "what is my name", "what's my name", "do you remember my name" or "i am called" is allowed without any further check (`ContainmentLayer::enforce`). In HIPAA mode the medical refusals run before the allowlist and the identifiers are removed whatever the message says, so only the general-safety rule is skipped. Child mode is not affected, because it does not go through `enforce`.
- **Weather questions are answered before the check.** `generate_reply` answers them from Open-Meteo before containment, in every mode.
- **A caller can skip the check.** `generate_reply` takes `skip_containment`. The app sets it for the model call that writes up web search results (`App.jsx`) and for the second pass of a hands-free search.
- **Ensemble runs the check on the question only** (since 2026-11-10). `run_ensemble` in `commands/chat.rs` calls `check_containment` with the mode in force: a Guardian or HIPAA refusal is shown in the Ensemble window and no model is called, and a Sovereign warning is put in front of the combined answer. In HIPAA mode the question and the shared context (memories and Knowledge Base text) are cleaned of identifiers. The Ensemble button is disabled in Child mode. If the check itself fails, the question goes through, as in the chat path.
- **Only the typed question is checked for refusal.** Attached file text and images are not.
- **Recalled memories and Knowledge Base text are not checked for refusal.** They go into the prompt in every mode. HIPAA mode removes identifiers from them; nothing reads a picture.
- **Android hands-free answers always use Sovereign.** `NativeVoiceAnswerer.kt` calls `nativeSendMessage` with the mode fixed to "sovereign", whatever mode is selected in the app.
- **The typed message is written to the app log.** `generate_reply` prints the message (up to 1,500 characters) before the check, and `app_log.rs` writes all such output to `logs/zynkbot.log` in the app data folder. This happens in every mode, for blocked messages too. In HIPAA mode the line shows the cleaned message, with tags in place of the identifiers.
- **An unknown mode name does not block.** If the check returns an error, the message continues to the model.
- **The `[BLOCK]` prefix is shown.** Block messages that carry it are returned with it.

**The classifier**
- It reads at most 512 tokens of the message.
- If it returns an error (for example its files are missing), `enforce` falls back to a keyword list (hack, exploit, illegal, malware, virus, crack, pirate, steal, weapon, bomb, drug, suicide). In HIPAA mode the medical refusals have already run by then (since 2026-11-10 they come first), so only the general-safety rule depends on the fallback.
- Guardian and HIPAA act on `SevereToxic` and `Threat` only, above 0.9.
- `docs/ROADMAP.md` ("Real safety classifier") records that it gives false positives on clinical and grief language.

**HIPAA mode**
- Identifiers are found by ten patterns, not by understanding. What they miss: a name with no "patient" or "client" in front of it ("John Smith, a 45-year-old male, presents with..."), a single first name, an address written in lower case, a date not marked as a birth date, an age, and anything inside a picture.
- The patterns can also remove text that is not an identifier: any four numbers joined by dots (a software version such as 10.2.0.1), sixteen digits that are not a card, two capitalised words after "patient" or "client" that are not a name ("the patient Monday Morning group").
- A web search the page runs itself (the Web button) sends the typed query to the search engine before `generate_reply` sees it, so that query is not cleaned.
- The medical refusals match substrings, so they also refuse ordinary sentences: "What options do I have?" contains "do i have".
- The backend is not restricted, and stored memories and Knowledge Base text go into the prompt, including to a cloud model, after the cleaning above.
- Conversation rows can still be written by paths that do not check the mode: onboarding (`store_onboarding_response` in `commands/onboarding.rs`), rows arriving from a paired device (`sync_outbox.rs`) and a backup restore (`commands/backup.rs`).
- Memories can still be written by `insert_memory` in `lib.rs`, which does not check the mode.
- The mobile Memory Manager button is not disabled in HIPAA mode. The desktop button is.
- If the audit log folder cannot be created, the entry is skipped.
- `hipaa_conversation_allowed` is written when the three medical checks pass, even if the general-safety check then blocks the message.

**Child mode**
- It works only with an OpenAI key and an internet connection.
- The child system prompt is not added when an image is attached.
- Weather answers and the web search write-up do not go through the Moderation API.

---

## Local-First Privacy Benefits

### 1. Data Ownership

**The SQLite database file is on the device:**
```
Windows: %LOCALAPPDATA%\zynkbot\zynkbot.db
Linux:   ~/.local/share/zynkbot/zynkbot.db
```

**What this means:**
- The file can be copied for backup. The database uses write-ahead logging (`db.rs`), so copy the `-wal` file with it or recent rows can be missing.
- The file can be opened with `sqlite3` or any SQLite browser.
- Deleting the file removes the data. The bundled SQLite is built with `SQLITE_SECURE_DELETE` (`zynkbot_rust/.cargo/config.toml`), so deleted rows are overwritten in the file.
- One conversation thread can be exported with a signed seal (`seal.rs`).
- Memories, links, history and API keys can be backed up, encrypted, to the user's own Cloudflare R2 bucket (`commands/backup.rs`).
- There is no command that exports everything to CSV or JSON.

### 2. Offline Functionality

**Works without internet:**
- Local model inference through llama.cpp. The desktop download list (`SetupWizard.jsx`) has Sophro 4B, Gemma 4 12B, Gemma 4 E4B, Qwen3.5 9B and Ministral 3 8B.
- Memory search (hybrid entity and semantic)
- Knowledge Base search over indexed documents
- Entity extraction (bert-base-NER)
- Safety classification (toxic-bert)
- Embeddings (all-MiniLM-L6-v2)
- Dictation with Vosk
- ZynkSync, ZChat and ZynkLink between devices on the same local network

**Needs internet:**
- Cloud models (OpenAI, Anthropic, xAI, Mistral), optional
- Child mode (the check and every reply go to OpenAI)
- Cloud dictation through OpenAI, optional
- Web search, optional. A cloud backend uses its provider's own search tool. A local model uses Brave (with a key), a SearXNG server (with a URL) or DuckDuckGo (`web_search.rs`).
- Weather answers (Open-Meteo)
- The R2 backup, optional

See [emergency_resilience.md](../case_studies/emergency_resilience.md) for the offline use case.

### 3. Transparent Memory

**The Memory Manager shows what is stored:**
- View all memories
- Edit a memory
- Delete a memory
- Filter by namespace
- Each reply carries the memories that were recalled for it (`recalled_memories`), and the chat shows them

### 4. No Training on Your Data

- The app does not train a chat model on conversations. Local chat models are downloaded files, and conversations do not change them.
- The Android wake-word verifier is trained on the device from the user's recorded wake phrases (`WakeVerifierTrainer.kt`).
- With a cloud backend, the prompt goes to that provider. What the provider does with it depends on the provider's terms, which this document does not check.
- The source is public (see `LICENSE`), so these statements can be checked.

### 5. Containment at the Edge

**Hosted approach:**
```
Message → sent to the provider → model processes it → output filtered
The provider has received the message.
```

**Local approach:**
```
Message → containment check on the device → blocked
A blocked message is not sent to a model.
```

**Example (HIPAA mode):** in a typed message with "SSN 219-90-7812" the number is replaced by `[SSN removed]` before any model call, and a request for a diagnosis is refused before any model call. The cases where this does not hold are in [Known limits](#known-limits).

---

## Comparison: Hosted AI vs. Local-First

The hosted column is general and was not checked against any provider.

| Feature | Hosted AI assistant | Zynkbot |
|---------|---------------------|---------|
| **Data location** | Provider's servers | Database on the device |
| **Training on conversations** | Depends on the provider's terms | The app does not train a chat model on conversations |
| **Offline use** | No | Yes, with a local model |
| **Memory visibility** | Varies by product | Memory Manager: view, edit, delete |
| **Containment** | Varies by product | Check on the typed message before the model call (see Known limits) |
| **Data ownership** | Governed by terms of service | The user holds the database file |
| **Audit of privacy claims** | Provider's statements | Public source |
| **Cost** | Subscription or API fees | Free with local models. A cloud API key is billed by its provider. |
| **Latency with a cloud API** | Network round trip | The same, when a cloud backend is used |

**Where a hosted assistant is the better fit:**
- The largest models
- No setup
- Works on any device at once

**Where local-first is the better fit:**
- Privacy-sensitive use with a local model
- No or unreliable internet
- The data must stay on a device the user controls
- No subscription

---

## Use Cases for Containment + Local-First

### Medical (HIPAA Mode)
- General reference questions, with a check for structured identifiers in the typed text
- Nothing from the conversation is stored: no history and no new memories
- Read [Known limits](#known-limits) first. With a cloud backend, stored memories and Knowledge Base text are sent to the provider.
- See [hipaa_compliance.md](../case_studies/hipaa_compliance.md)

### Personal Privacy (Guardian/Sovereign)
- Journaling that stays on the device, with a local model
- Sensitive personal questions (financial, relationship, health)
- Creative writing
- See [conversational_memory.md](../case_studies/conversational_memory.md)

### Field Work (Offline-First)
- A technician reads manuals without internet (manuals added to the Knowledge Base)
- Emergency response with disrupted infrastructure
- Areas with unreliable connectivity
- See [hvac_field_service.md](../case_studies/hvac_field_service.md)

### Research (Data Sovereignty)
- Research notes kept on the device
- Proprietary R&D conversations, with a local model
- A journalist's notes, with a local model

---

## Implementation Details

### Containment Pipeline

**Code location:** `generate_reply` in `commands/chat.rs`. The Tauri command `send_message_with_memory` calls it.

**Flow:**
```
1. Message received. HIPAA mode: identifiers removed from it. It is printed to the app log.
2. Child mode: the backend is forced to OpenAI.
3. Weather question: answered from Open-Meteo. Stop.
   (If the lookup fails, the message continues as normal.)
4. If skip_containment is set: go to step 7.
5. Child mode:
   - OpenAI Moderation API called on the typed message
   - Flagged, missing key or API error: return a block message. Stop.
6. Other modes: check_containment → ContainmentLayer::enforce
   a. HIPAA: diagnosis, dosing and treatment refusals
   a2. Allowlist phrase present: allow
   b. Witness: allow
   c. Harmful-instruction pattern (whole-word match)
   d. toxic-bert classifier (on error: keyword fallback, then return)
   e. Action by mode:
      - Guardian: block if (c) fired, or SevereToxic/Threat above 0.9
      - Sovereign: warn if (c) fired, or any harmful category above 0.85
      - HIPAA: the Guardian rule
7. Memory recall and, if the KB button is on, Knowledge Base search
8. HIPAA mode: identifiers removed from the finished prompt.
   Model call. A Sovereign warning, or the HIPAA "removed" line, is put in front of the reply.
9. HIPAA mode: disclaimer appended to health-related replies; no history, no memory extraction
```

The commands `set_containment_mode` and `get_containment_mode` in `commands/safety.rs` save and return the mode (`app_settings`, key `containment_mode`). The frontend still passes the mode with each message; the saved value is what the page starts with.

### Local ML Models

**Storage:** in a development build, `zynkbot_rust/src-tauri/models/system/`. In an installed app, `models/system` under the app data folder (`get_models_dir` in `db.rs`).

| Model | Used by | Purpose | File size |
|-------|---------|---------|-----------|
| toxic-bert | `safety_classifier.rs` | Toxicity labels (Toxic, SevereToxic, Obscene, Threat, Insult, IdentityHate) | about 440 MB |
| bert-base-NER | `nlp_enhancer.rs` | Entity extraction | about 430 MB |
| all-MiniLM-L6-v2 | `llm/local_embeddings.rs` | Embeddings for semantic search | about 90 MB |

The code comments call the classifier "TinyBERT". The model file is toxic-bert, a 12-layer BERT.

**Loading:** the classifier loads on first use. The NER model is warmed at startup.

**Framework:** Candle (Rust, no Python). Local chat models run through llama.cpp.

**Inference:** CPU for these three models.

### Database Architecture

**SQLite, embedded, no server process.**

**Memories table (selected columns from `migrations/0001_initial.sql`; later migrations add more):**
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT,
    namespace TEXT NOT NULL DEFAULT 'personal',
    embedding BLOB,                              -- vector, stored as binary
    is_ephemeral INTEGER NOT NULL DEFAULT 0,     -- passing facts, any mode
    expires_at TEXT,                             -- when a passing fact leaves recall
    entities_detected TEXT NOT NULL DEFAULT '[]' -- JSON
);
```

`is_ephemeral` and `expires_at` are not tied to HIPAA mode. They mark passing facts (an errand, this week's trip) in any mode. A memory past its `expires_at` date is left out of recall.

See [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for the full schema.

**Editable:**
- Standard SQL UPDATE and DELETE
- The Memory Manager can view, edit and delete memories
- Rows are not hashed or signed in the database. A single thread can be exported with a signed seal (SHA-256 and Ed25519, `seal.rs`).

**Why SQLite:**
- Embedded: no separate server or installation
- One database file, plus its `-wal` file while the app runs
- Standard SQL
- Can be inspected with `sqlite3` or any SQLite browser
- Vector search runs in the Rust process; no SQLite extension is needed

---

## Limitations and Trade-offs

The containment limits are in [Known limits](#known-limits). The points below are about local-first use in general.

**1. Local model quality**
- The local models on the download list are much smaller than cloud models. `docs/BENCHMARKS.md` has the measured results.
- A cloud backend can be used instead. The prompt then goes to that provider.

**2. Setup**
- The three system models are about 1 GB together. A chat model is extra; the smallest on the desktop list, Sophro 4B, is listed at 2.4 GB.
- A hosted assistant needs no download.

**3. Dictation**
- Vosk works offline. OpenAI transcription is optional and needs internet.

**4. Platforms**
- Release builds are made for Linux, Windows and Android (`.github/workflows/release.yml`). There is no iOS target in `src-tauri/gen/`.

**Zynkbot is not the better choice for:**
- Users who want no setup
- Users who need the largest models and do not want to use a cloud API key
- Users who want sync through a hosted service

**Zynkbot is the better choice for:**
- Users who want their data on their own device
- Users with no or unreliable internet
- Users who want to see, edit and delete what is remembered
- Users who do not want a subscription

---

## Future Enhancements

This document does not set a roadmap. [ROADMAP.md](../ROADMAP.md) is the source. At the time of this check it lists, among other items:
- "Real safety classifier": a replacement for toxic-bert
- "AI-Based PHI Detection", under HIPAA Mode
- Elder Mode, for v1.1. `ContainmentMode::Elder` is a stub today.

---

## Technical References

**Related Documentation:**
- [ARCHITECTURE_COMPREHENSIVE.md](ARCHITECTURE_COMPREHENSIVE.md) - System architecture
- [FEATURES.md](../FEATURES.md) - Feature list
- [HIPAA Security Integration Guide](HIPAA_SECURITY_INTEGRATION_GUIDE.md) - HIPAA mode in detail
- [HIPAA Compliance Case Study](../case_studies/hipaa_compliance.md)
- [Digital Resilience](../DIGITAL_RESILIENCE.md) - Offline use

**Source Code** (paths under `zynkbot_rust/`):
- `src-tauri/src/commands/chat.rs` - `generate_reply`: conversation flow and the containment step; `run_ensemble`
- `src-tauri/src/commands/safety.rs` - `check_containment` and the mode commands
- `src-tauri/src/containment.rs` - `ContainmentLayer::enforce`, `enforce_hipaa`, `check_openai_moderation`, `CHILD_MODE_SYSTEM_PROMPT`
- `src-tauri/src/safety_classifier.rs` - toxic-bert classifier, `should_block`
- `src-tauri/src/memory.rs` - Hybrid search and storage
- `src/components/ContainmentModeSelector.jsx` - The mode drop-down
- `src/App.jsx` - Holds the active mode and passes it with each message

---

**ContainAI**
