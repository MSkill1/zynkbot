# HIPAA Security Integration Guide

*Checked against the code on 2026-11-10 (`v1`). Updated the same day: a message with an identifier in it is not sent and the user is told what was found; identifiers in recalled memories, Knowledge Base text and web pages are removed before the model call; the mode is saved; Ensemble follows it.*

**Document Version:** 4.1
**Purpose:** What Zynkbot's HIPAA mode does, what it does not do, and what a healthcare deployment still has to provide

## ⚠️ Important: Read Before Using in a Healthcare Environment

**THIS SOFTWARE IS NOT HIPAA-CERTIFIED. IT IS A PRIVACY FRAMEWORK, NOT A COMPLIANCE SOLUTION.**

Zynkbot has features that can support a HIPAA compliance effort. It is not a certified medical product, is not a complete HIPAA solution, and does not replace a Business Associate Agreement (BAA). PHI removal is ten text patterns, with the limits listed in [Known limits](#known-limits). Developers assume no liability for compliance failures, data breaches, or regulatory penalties.

Statements about Zynkbot's behaviour in this document were checked against the code. Industry background, figures about healthcare, and descriptions of other products were not checked. The description of the HIPAA regulatory framework (BAA requirements, risk assessment obligations, Security Rule structure) is high-level only. Whether a configuration satisfies a HIPAA audit depends on the covered entity's own risk assessment. This document is not a substitute for qualified HIPAA compliance counsel.

**Current state:** in HIPAA mode the app looks for ten kinds of identifier. A typed message or attached file that holds one is not sent, and the user is told what was found. Identifiers in text the user did not type (recalled memories, Knowledge Base passages, pages read from the web) are removed before the model call. The app also refuses certain medical requests, writes an audit line, keeps no conversation history, and does not extract memories. It does not restrict which model is used. The mode is saved on the device and is still on after a restart. A shared-server deployment (Scenario B below) does not exist. `docs/ROADMAP.md` describes the containment modes as proofs of concept that are not production-grade for their domains.

---

## AI in Healthcare: Why This Matters Now

*Background. Not checked against the code.*

Healthcare has a high documentation and administrative burden. Physicians in the US are often reported to spend a large share of their time on EHR documentation and paperwork, and administrative load is a known driver of burnout in nursing, primary care and specialty medicine.

AI is already being deployed to address this. Products such as Microsoft's Nuance DAX, Abridge and Suki help physicians document patient encounters, and EHR vendors are adding AI to scheduling, documentation and clinical decision support.

Beyond documentation, clinical and administrative staff use AI for:
- **Reference and lookup:** drug interactions, clinical guidelines, protocol checks, literature summaries
- **Administrative work:** prior authorization drafts, billing code suggestions, discharge instruction templates, patient communication drafts, compliance policy questions
- **Training and onboarding:** procedure questions, institutional policies, coding standards

**The practical reality:** healthcare workers already use AI informally, through personal ChatGPT and Claude accounts, often against policy. An organisation's choice is not only "allow AI or not". It is also whether staff get a tool with guardrails the organisation can inspect, or keep using personal consumer accounts with no oversight.

**What Zynkbot records about use.** In HIPAA mode each checked message writes one audit line: a timestamp, an event type, the message length and a blocked flag. A message refused for an identifier, or a removal from recalled text, writes a line with the number of identifiers of each kind. No line has a user id, the message text or the removed text, and conversation history is off in HIPAA mode. The audit log therefore shows how many messages were checked, how many were refused and how many identifiers were removed. It does not show who asked what.

---

## The Problem With Current Healthcare AI

*Background. Not checked against the code.*

### "HIPAA Compliant" Often Means a Contract

When a cloud provider offers "HIPAA-compliant AI", the offer usually rests on a **Business Associate Agreement**: a contract that says how the provider will handle the data. The architecture is unchanged. The query, including any PHI in it, travels to the provider's data centers and is processed by its model. The BAA governs what happens afterwards.

```
Cloud AI with a BAA:

User types: "My patient John Smith, DOB 01/15/1985, MRN 4829201, presents with..."
                                    ↓
                    Sent over the internet to the cloud provider
                                    ↓
                    The model processes the full query, including the PHI
                                    ↓
                    Response returned
```

Once PHI is in a cloud prompt, it has been sent to and processed by infrastructure outside the organisation's control. A BAA sets contractual accountability. It does not change the fact that the data left the building.

HIPAA's **Minimum Necessary standard** (§164.514) requires covered entities to limit PHI disclosure to what is needed for the purpose. Sending every staff query, in full, to a cloud model is hard to reconcile with that principle.

### The Other Side: Avoiding AI Does Not Work Either

Staff already use AI informally, whatever the policy says. The choices are: ban it and accept that personal accounts will be used anyway, sign BAAs with cloud providers and accept the exposure described above, or run a local system and keep the data on the organisation's own devices.

### What Zynkbot Does Differently

In HIPAA mode Zynkbot checks the message on the device, before any model call. A message with an identifier in it is not sent:

```
Zynkbot HIPAA mode:

User types: "My patient John Smith, DOB 01/15/1985, MRN 4829201, presents with..."
                                    ↓
                    checked on the device: three identifiers found
                                    ↓
Nothing is sent.   The user sees: "🔒 Not sent. HIPAA mode found 1 record or
                   member number, 1 date of birth and 1 patient name in your
                   message. Remove what was found and send it again. Nothing
                   went to a model and nothing was stored."
```

The message is refused rather than cleaned because a matching identifier is a sign of other patient detail the patterns cannot see: the age, the complaint, a name with nothing to mark it. Removing the one thing found and sending the rest would give false comfort.

Text the user did not type is treated differently. Memories recalled for the question, Knowledge Base passages and pages read from the web join the prompt before the model call, and any identifier in them is removed there, with a short tag in its place. The reply then starts with a line such as "🔒 HIPAA mode removed 1 phone number from the recalled text before the model saw this." Refusing a question over text the user cannot see would be baffling, and a memory with an old phone number in it would otherwise make every question that recalled it fail.

Until 2026-11-10 only the typed message was looked at, against seven patterns, and this very example passed untouched: there was no pattern for a name, a date of birth or a record number.

**With a local model**, the message, the recalled memories and the reply stay on the device.

**With a cloud model** (OpenAI, Anthropic, xAI, Mistral), the prompt goes to the provider. HIPAA mode does not prevent a cloud model from being selected. The prompt includes the message, any stored memories that were recalled for it, and Knowledge Base text when the KB button is on. The recalled text is cleaned, as one text, just before the call.

The check is pattern matching, not understanding. What it misses is listed in [Known limits](#known-limits). An earlier version of this guide said that no configuration can bypass the check. That is not the case.

**What this means in practice:** the check is aimed at accidental disclosure of a structured identifier: a typed or pasted SSN, phone number, email address, record number, birth date, or a name introduced as a patient's. Nothing in the code measures how often it succeeds, so this guide gives no accuracy figure.

### Where the PHI Check Runs

Zynkbot is a desktop and Android application (Tauri and Rust). The code is `redact_phi` in `src-tauri/src/containment.rs`. It runs in the app's own process on the device the user is typing on.

- With a **local model**: the check runs and the prompt is answered on the device.
- With a **cloud model**: the check runs on the device first; a message that passes is sent, with its recalled text cleaned.

In a typed message a found identifier stops the message. In recalled text each identifier is replaced by a short tag such as `[phone number removed]`. The found text is not returned, logged or stored.

### How This Looks in a Healthcare Setting

**Scenario A: Per-workstation (what exists today)**

Each workstation runs Zynkbot as a desktop application. Staff use it from their own computer for reference questions, protocol review, administrative drafts and policy questions.

- HIPAA mode is selected once on each workstation. The choice is saved on that device and is not copied to paired devices.
- A message with a matching identifier (record or member number, card number, SSN, phone number, email address, labelled birth date, street address, ZIP code, IP address, a name introduced as a patient's) is not sent. The same kinds are removed from recalled text before any model call.
- BERT NER runs in the app for memory entity extraction. The HIPAA removal does not use it. A name with no "patient" or "client" in front of it is not removed.

**Knowledge base in Scenario A:** each user keeps an indexed document library on the local machine. Relevant sections are added to the prompt when the KB button is on. With a local model that text stays on the device. With a cloud model it is sent to the provider as part of the prompt.

**Staff communication in Scenario A:** ZChat sends messages device to device between paired Zynkbot instances on the same local network. ZynkLink sends files the same way. The traffic uses TLS (see Encryption in Transit).

**Scenario B: Central server with shared access (not built)**

A shared Zynkbot server for a department or facility does not exist. Nothing in this subsection is implemented.

The idea: a hospital IT department runs Zynkbot on a local server, and staff reach it from tablets, workstations or shared terminals. The query path would be staff device → server → PHI check → model. PHI would cross the local network to the server and would be checked there. Earlier versions of this guide described further detection layers for such a server (person names from BERT NER, and a local model asked whether a message links an identifiable person to health information), a shared Knowledge Base and server-routed ZChat. None of these exist. `docs/ROADMAP.md` is the source for what is planned. ContainAI is interested in building a server deployment with a healthcare partner.

**Branding:** a build under a different name and icon would be a custom build from source. See `COMMERCIAL_LICENSE.md` for the terms. Not verified against the code.

**What Zynkbot is not designed for:**
- Storing patient records or EMR data
- Replacing clinical decision support systems

The intended use is a general-purpose assistant for healthcare staff, for reference, drafting and administrative work, with a step that stops a message holding a structured identifier before the model sees it.

---

## Implemented Features

All of the following apply only while HIPAA mode is selected.

**1. PHI Check**
- Ten patterns (`redact_phi`): record or member number, card number, Social Security number, phone number, email address, date of birth, street address, ZIP code, IP address, patient name. What each one matches is in [Containment Architecture](CONTAINMENT_ARCHITECTURE.md), HIPAA Mode.
- The typed message, with the text of an attached file, is checked as soon as it arrives, before the app log line, the memory search and any model call. A match stops the message: the reply says what was found, and nothing is logged, stored or sent. In Ensemble the question is checked the same way and the refusal is shown in the Ensemble window.
- The finished prompt is cleaned just before the model call, for every backend: identifiers in recalled memories, Knowledge Base passages, earlier turns and web pages are replaced by tags, and the reply starts with a line saying how many of each kind were removed. In Ensemble the shared context is cleaned the same way.
- Source: `src-tauri/src/containment.rs`, `redact_phi` and `phi_refusal`; `generate_reply` and `run_ensemble` in `commands/chat.rs`

**2. Medical Request Blocking**

These run after the identifier check and refuse the whole message. They are aimed at someone asking the AI to diagnose or treat them. They match substrings of the lower-cased message, so they also match some ordinary sentences (see Known limits).

- Diagnosis phrases (`detect_diagnosis_request`): "diagnose me", "what do i have", "do i have", "is it cancer", "is it diabetes"
- Dosing (`detect_medication_dosing`): one of the phrases "how much should i take", "how much", "should i take", "dosage for", "how many pills", "what dose", "safely", together with one of the keywords mg, dose, pill, tablet, medication, medicine, drug, prescription, ibuprofen, acetaminophen, aspirin, tylenol, advil, take, "can i take"
- Treatment phrases: "should i get surgery", "should i have the procedure", "should i start treatment"
- Source: `src-tauri/src/containment.rs`, `enforce_hipaa`

**3. General Safety Block**
- After the medical checks, the same rule as Guardian mode applies: a request for instructions on a harmful subject, or a toxic-bert result of `SevereToxic` or `Threat` above 0.9, is blocked.
- Message: `[BLOCK] Cannot process this request in HIPAA mode. (Detected: X)`
- Source: `ContainmentLayer::enforce` in `containment.rs`, `should_block` in `safety_classifier.rs`

**4. Audit Logging**
- One JSON line per checked message, in a daily file `logs/hipaa_audit/hipaa_audit_YYYY-MM-DD.json` in the app data folder
- Events: message refused for an identifier, identifiers removed from recalled text, diagnosis blocked, dosing blocked, treatment blocked, conversation allowed
- Details and limits are under Security Requirements, Audit Logging
- Source: `log_hipaa_audit` in `containment.rs`

**5. No Memory Extraction**
- The memory extraction pipeline does not run. No memory job is queued, including for an explicit "Remember:" request.
- There is no automatic expiry fallback. No 8-hour value exists in the code. In `chat.rs` the variable `_is_ephemeral` is set for HIPAA mode and never used.
- The desktop "Open Memory Manager" button is disabled, with the tooltip "Memory storage is disabled in HIPAA mode".
- Source: `generate_reply` in `src-tauri/src/commands/chat.rs`, `src/components/MemoryManager.jsx`

**6. No Conversation History**
- The chat paths do not write to `conversation_sessions` or `conversation_messages`. Weather answers and web search answers are not logged either (`log_web_search_answer` in `commands/conversation.rs`).
- The History button is disabled, with the tooltip "Conversation history is disabled in HIPAA mode".
- Three paths outside chat do not check the mode. They are listed in Known limits.
- Source: `src-tauri/src/commands/chat.rs`, `src/App.jsx`

**7. Medical Disclaimer**
- Appended when the reply contains one of: symptom, treatment, medication, diagnosis, disease, condition, health, medical, doctor, patient, therapy
- Text: *"⚕️ AI-generated. Not a substitute for clinical judgment or current clinical guidelines."*
- Source: `generate_reply` in `src-tauri/src/commands/chat.rs`

**8. No Check-In Questions**
- The app's check-in questions are off in HIPAA mode.
- Source: `next_checkin` in `src-tauri/src/commands/checkin.rs`

---

## Known limits

This section states current behaviour.

**The mode**
- **The mode is saved per device.** It is kept in that device's settings and is not synced. Each device has to be set once.
- **The backend is not restricted.** Any model can be selected in HIPAA mode, including a cloud model.
- **Android hands-free answers always use Sovereign.** `NativeVoiceAnswerer.kt` calls `nativeSendMessage` with the mode fixed to "sovereign". A hands-free voice question is handled as Sovereign whatever mode is selected: nothing is removed or refused, and history and memory behave as in Sovereign.

**What reaches the model uncleaned**
- **Pictures.** Nothing reads the text in an image.
- **A web search started with the Web button.** The page sends the typed query to the search engine before the check sees it.
- **Whatever the patterns miss** (next list).

**What is not refused**
- **Messages with an allowlist phrase** ("my name is", "what is my name", "what's my name", "do you remember my name", "i am called") skip the general-safety rule. Since 2026-11-10 they no longer skip the medical refusals or the identifier check.
- **Weather questions** are answered from Open-Meteo before the refusal check.
- **Calls that set `skip_containment`** (the model call that writes up web search results, and the second pass of a hands-free search) skip the refusals, the identifier check included: the text they carry is search results, not something the user typed. Identifiers in it are removed instead.
- **Ensemble** checks the question only. Its shared context is cleaned, not checked for refusal.

**What the check finds and misses**
- It finds ten kinds by pattern. It misses a name with no "patient" or "client" in front of it: "John Smith, a 45-year-old male, presents with..." goes through unchanged. It misses a single first name, an address written in lower case, a date not marked as a birth date, and an age.
- It can match text that is not an identifier: four numbers joined by dots (a software version such as 10.2.0.1), sixteen digits that are not a card, two capitalised words after "patient" or "client" that are not a name. A typed message with such text is refused; such text in a memory is removed.
- The medical request rules are substring matches. "What options do I have?" is refused as a diagnosis request. "How much time does it take?" is refused as a dosing request.

**What is still written to disk**
- **The app log.** `app_log.rs` writes the app's output to `logs/zynkbot.log` in the app data folder. The chat and Ensemble request lines print the typed message in the other modes (up to 1,500 characters); in HIPAA mode they print its length only (since 2026-11-12). Other lines about the request, such as the model's name and timings, are written as in every mode.
- **Conversation rows from paths that do not check the mode:** onboarding (`store_onboarding_response` in `commands/onboarding.rs`), rows arriving from a paired device over ZynkSync (`sync_outbox.rs`), and a backup restore (`commands/backup.rs`).
- **Memories from `insert_memory` in `lib.rs`**, which does not check the mode.
- Memories and history stored before HIPAA mode was selected stay in the database. The recent-memories list on the desktop still shows them.

**The audit log**
- It has no user id and no message text.
- Its folder is `logs/hipaa_audit` in the app data folder (until 2026-11-10 it was relative to wherever the app was started from). If the folder cannot be created, the entry is skipped and an error line is printed.
- `hipaa_conversation_allowed` is written when the three medical checks pass, even if the general-safety check then blocks the message.
- Entries are not hashed or signed.

**Backup**
- The encrypted R2 backup (`backup_memories_to_r2` in `commands/backup.rs`) does not check the mode. It is started by hand from the Memory Manager window. The desktop button that opens that window is disabled in HIPAA mode. The mobile button is not.

---

## Not Implemented

- **Encryption at rest inside the app.** The database is a plain SQLite file. Use disk encryption (see Security Requirements).
- **Authentication and access control** (SSO, RBAC, MFA, user accounts with roles)
- **A shared server deployment** (Scenario B)
- **User attribution in the audit log**
- **Incident response procedures**
- **Business Associate Agreements** with third-party vendors

Two items that earlier versions listed here do exist:
- **Encryption in transit.** LAN traffic for ZynkSync, ZynkLink and ZChat uses TLS (see Security Requirements).
- **Backup.** An encrypted backup to the user's own Cloudflare R2 bucket exists (`commands/backup.rs`): AES-256-GCM, with an Argon2id passphrase option. It holds memories, links, API keys and conversation history in one object, `backup.enc`. It runs only when the user starts it. Whether it meets an organisation's backup and disaster recovery requirements is for that organisation to decide.

---

## Comparison With Cloud AI

The cloud column is general and was not checked against any provider.

| Feature | Cloud AI with a BAA | Zynkbot in HIPAA mode |
|---------|---------------------|-----------------------|
| Before the model sees the message | Nothing on the user's device | A message with one of ten kinds of identifier is stopped, the same kinds are removed from recalled text, and three medical request rules apply, all on the device |
| Data leaving the device | The full query goes to the provider | Local model: nothing is sent to a provider. Cloud model: a message that passed the check goes to the provider, with recalled memories and Knowledge Base text cleaned of identifiers. |
| Memory | Varies by product | No extraction in HIPAA mode. Stored memories are still recalled, and cleaned with the rest of the prompt. |
| Conversation history | Varies by product | Not written by the chat paths in HIPAA mode |
| Audit trail | Provider-managed | Local JSON lines: event type, length, blocked flag, counts of what was found or removed. No user id. |
| Self-hostable | No | Runs on the user's device |
| Source | Closed | Public (see `LICENSE`) |
| BAA | Required | With a cloud model the prompt goes to that provider. Whether a BAA is needed is for the covered entity to decide. |

### Processing Flow

```
┌──────────────────────────────────────┐
│           YOUR SECURITY LAYER        │
│  (disk encryption, device access,    │
│   network controls)                  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│         ZYNKBOT, HIPAA MODE          │
│  • Identifier check on the message   │
│    (one found: not sent)             │
│  • Medical request rules (refuse)    │
│  • General safety rule               │
│  • Audit line (local JSON)           │
│  • No memory extraction              │
│  • No conversation history           │
└──────────────────┬───────────────────┘
                   │ message not refused
                   ▼
┌──────────────────────────────────────┐
│     PROMPT BUILT FOR THE MODEL       │
│  message + recalled memories         │
│  + Knowledge Base text (KB button)   │
│  then identifiers removed from these │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│   LOCAL MODEL  or  CLOUD PROVIDER    │
│        (the user's selection)        │
└──────────────────────────────────────┘
```

---

## PHI Detection in Action

**Try it yourself:** open System Controls and set the "Containment Mode" row to **HIPAA**. Then type any of the messages below. The mode stays on until you change it.

All names, numbers, and identifiers below are fictional. The "User sees", "joins the prompt as" and "reply starts with" lines are checked by a test (`the_hipaa_guides_examples_read_as_printed` in `containment.rs`).

---

**Not sent: name and SSN**
> *"My patient is Mary Johnson. Her social is 234-56-7890 and she has been having chest pains."*

**User sees:** `🔒 Not sent. HIPAA mode found 1 Social Security number and 1 patient name in your message. Remove what was found and send it again. Nothing went to a model and nothing was stored.`

An audit line with those two counts is written. The message is not printed to the app log.

---

**Not sent: phone number**
> *"I've tried reaching the patient twice at 555-867-5309 with no answer. What should I document in the chart?"*

**User sees:** `🔒 Not sent. HIPAA mode found 1 phone number in your message. Remove what was found and send it again. Nothing went to a model and nothing was stored.`

---

**Not sent: name, date of birth and record number**
> *"My patient John Smith, DOB 01/15/1985, MRN 4829201, presents with chest pain."*

**User sees:** `🔒 Not sent. HIPAA mode found 1 record or member number, 1 date of birth and 1 patient name in your message. Remove what was found and send it again. Nothing went to a model and nothing was stored.`

---

**Removed from recalled text**
> *"What did I note about the clinic's fax line?"*, with a stored memory *"The clinic's fax is 555-867-5309."*

The question holds no identifier, so it goes through. The memory is recalled and **joins the prompt as:** `The clinic's fax is [phone number removed].`

**The reply starts with:** `🔒 HIPAA mode removed 1 phone number from the recalled text before the model saw this.`

---

**Blocked: diagnosis request**
> *"Based on the symptoms I just described, what do I have?"*

**User sees:** `🏥 I cannot provide medical diagnoses. Please consult with a licensed healthcare provider for diagnostic evaluations.`

---

**Blocked: medication dosing**
> *"How much acetaminophen should I take after the procedure?"*

**User sees:** `🏥 I cannot provide medication dosing advice. Please consult your doctor or pharmacist for accurate dosing information.`

---

**Blocked: treatment decision**
> *"Should I get surgery for this?"*

**User sees:** `🏥 I cannot advise on treatment decisions. Please discuss treatment options with your healthcare provider.`

---

**Allowed: general clinical question**
> *"What are the common side effects of beta blockers?"*

No pattern or phrase matches. The model answers. If the reply contains one of the trigger words, the disclaimer is appended: *"⚕️ AI-generated. Not a substitute for clinical judgment or current clinical guidelines."*

---

**Not stopped: a name with nothing to mark it**
> *"John Smith, a 45-year-old male, presents with substernal chest pain radiating to the left arm."*

Nothing says "patient" or "client" before the name, and there is no structured identifier. The message goes to the model unchanged.

---

**Not sent even with an allowlist phrase**
> *"My name is Mary Johnson and my social is 234-56-7890."*

**User sees:** `🔒 Not sent. HIPAA mode found 1 Social Security number in your message. Remove what was found and send it again. Nothing went to a model and nothing was stored.` The speaker's own name is not one of the ten kinds; the number is.

---

## Security Requirements

The requirements below vary by deployment and organisation size. A solo practitioner with one laptop has different obligations from a hospital IT department. The items here are the categories that usually need to be addressed. How to address them depends on existing infrastructure, the compliance officer's guidance and the organisation's own risk assessment.

**European deployments / GDPR:** not verified against the code as a legal matter. What the code provides: processing on the device when a local model is used; deletion of memories in the Memory Manager, with SQLite built with `SQLITE_SECURE_DELETE` so deleted rows are overwritten in the file; export of a single thread with a signed seal (`seal.rs`); and the encrypted R2 backup. There is no command that exports all data to CSV or JSON, and no breach notification workflow. GDPR compliance is the deploying organisation's responsibility.

**A note on scope:** ContainAI cannot configure Zynkbot for every security environment, EHR integration or regulatory scenario. HIPAA requirements interact with state privacy laws, accreditation requirements, payer contracts and organisation-specific risk. Each provider is responsible for deciding how Zynkbot fits into its compliance program.

### Encryption at Rest

**Requirement:** all data stored by Zynkbot must be encrypted at rest.

**What the app does:** it does not encrypt its files. The database is a plain SQLite file (`~/.local/share/zynkbot/zynkbot.db` on Linux, `%LOCALAPPDATA%\zynkbot\zynkbot.db` on Windows), with a `-wal` file beside it while the app runs. The app log (`logs/zynkbot.log` in the same data folder) and the HIPAA audit log are plain text. SQLite is built with `SQLITE_SECURE_DELETE` (`zynkbot_rust/.cargo/config.toml`), so deleted rows are overwritten in the database file.

**How to satisfy it:** OS-level disk encryption (BitLocker, LUKS) covers these files with no change to Zynkbot. SQLCipher would be an alternative for the database file and would need a custom build; that is not verified against the code.

---

### Encryption in Transit

**Requirement:** all network communication carrying user data must use TLS 1.2 or higher.

**Current status:**
- Database: not applicable. SQLite is a local file. There is no network database connection.
- ZynkSync, ZynkLink and ZChat LAN traffic uses TLS through `rustls` (`transport.rs`). ZynkSync protected routes also require a pinned client certificate. ZynkLink and ZChat rely on their pairing records and per-request authorization instead of a client certificate. The ZynkLink chat client accepts any server certificate (`zynklink.rs`, trust on first use). Treat this difference as a deployment-review item; coverage is not uniform mTLS.
- Calls to cloud providers go to their HTTPS endpoints.

**For enterprise deployments:** review this against your network policy with your network team.

---

### Authentication & Access Control

**Standalone deployment:** the app has no user login. Access to the app is access to the device account it runs under.

**Network listener:** the app listens on `0.0.0.0:57963` with TLS from launch (`auto_start_http_server` in `lib.rs`, `serve` in `transport.rs`). The listener serves ZynkSync, ZynkLink and ZChat. ZynkSync protected routes require a pinned client certificate. ZynkLink and ZChat requests are checked against pairing records (`check_zynklink_authorized` in `zynklink.rs`). Apply firewall rules as your policy requires.

**Multi-user server deployment:** not built. The chat commands take a `user_id` and a `session_id`, but there is no server product and no authentication layer.

---

### Audit Logging

**Where:** `logs/hipaa_audit/hipaa_audit_YYYY-MM-DD.json` in the app data folder (`~/.local/share/zynkbot/` on Linux, `%LOCALAPPDATA%\zynkbot\` on Windows). The date is the UTC date.

**Format:** one JSON object per line. Each line has three fields (key order may differ):

```json
{"timestamp": "2026-11-10T14:32:10.123456789+00:00", "event_type": "diagnosis_request_blocked", "metadata": {"blocked": true, "query_length": 87}}
{"timestamp": "2026-11-10T14:33:02.519203611+00:00", "event_type": "phi_detection_blocked", "metadata": {"where": "chat", "blocked": true, "found": {"phone number": 1, "patient name": 1}}}
{"timestamp": "2026-11-10T14:35:41.002117455+00:00", "event_type": "phi_removed", "metadata": {"where": "chat", "removed": {"phone number": 1}}}
```

- `timestamp`: UTC, RFC 3339, with a `+00:00` offset
- `event_type`: `phi_detection_blocked`, `phi_removed`, `diagnosis_request_blocked`, `dosing_request_blocked`, `treatment_request_blocked` or `hipaa_conversation_allowed`
- `metadata.query_length`: length of the message in bytes (the medical rules and the allowed event)
- `metadata.blocked`: true or false
- For `phi_detection_blocked` (a typed message or Ensemble question not sent): `metadata.where` is `chat` or `ensemble`, and `metadata.found` gives the number of each kind
- For `phi_removed` (recalled text cleaned): `metadata.where` is `chat` or `ensemble`, and `metadata.removed` gives the number of each kind

**Not in the log:** the message text, a user id, a device id, the model used, and blocks made by the general-safety rule.

**For enterprise compliance, an organisation may want to add** (none of these exist in the app):
- Hash-chaining of log entries for tamper detection
- Forwarding to a SIEM (Splunk, ELK and similar)
- User identity

---

## Deployment Checklist

**Technical configuration** covers what IT sets up in Zynkbot and around it. **Organizational obligations** covers what the compliance officer, legal team and administration must handle. No software can satisfy those.

### Technical Configuration

- [ ] HIPAA mode selected once on every device (the mode is saved per device and is not synced)
- [ ] Model choice reviewed: with a cloud model, a message that passed the check goes to the provider, with recalled memories and Knowledge Base text cleaned of identifiers
- [ ] Existing memories and Knowledge Base documents reviewed: they are used in HIPAA mode
- [ ] Encryption at rest: OS disk encryption enabled
- [ ] Encryption in transit: LAN traffic is TLS; the ZynkSync and ZynkLink/ZChat differences above reviewed
- [ ] Network listener on port 57963 reviewed against firewall policy
- [ ] Audit logs: `logs/hipaa_audit` in the app data folder included in backup rotation
- [ ] App log (`logs/zynkbot.log`) included in the retention policy: it holds the text of every message that passed the identifier check
- [ ] PHI check tested with representative examples (see above), including examples that are expected to go through unchanged
- [ ] Medical disclaimer confirmed on health-related replies
- [ ] No memory extraction verified: no new memories after a test session (the desktop Memory Manager button is greyed out)
- [ ] No conversation history verified: the History button is greyed out and no new rows appear in `conversation_sessions` after a test exchange
- [ ] Android hands-free use reviewed: it always runs in Sovereign mode

### Organizational Obligations

- [ ] Business Associate Agreements signed with all third-party vendors
- [ ] HIPAA Security Rule risk assessment completed and documented (§164.308(a)(1))
- [ ] Written security policies and workforce training in place
- [ ] Incident response and breach notification plan documented
- [ ] Access provisioning and deprovisioning process defined
- [ ] Audit logs reviewed on a regular schedule
- [ ] Penetration testing conducted
- [ ] Cyber liability insurance obtained

---

**Language for describing Zynkbot in healthcare contexts**

Matches the code:
- "Stops a message that holds one of ten kinds of structured identifier, and removes the same kinds from what is recalled, on the device, before any model call"
- "Keeps no conversation history and extracts no memories in HIPAA mode"
- "Has features that can support a HIPAA compliance effort"

Does not match the code:
- "Catches all PHI before it reaches any AI model": ten patterns are checked; a name with nothing to mark it, free-text detail and pictures are not
- "PHI is never transmitted to an LLM provider": whatever the patterns miss is sent, and hands-free answers on the phone do not use HIPAA mode
- "HIPAA-compliant": requires certification this software does not have
- "Guarantees HIPAA compliance": no software can guarantee this

---

## Contact

**HIPAA module / provider partnerships:** matt@containai.ai

**Technical questions:** Open an issue on GitHub or see the `/docs` directory.

**What developers do not provide:** Legal advice, compliance consulting, BAAs, warranties, or indemnification. Engage qualified HIPAA counsel before production deployment.

**External resources:**
- HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
