# Zynkbot Features

## Memory Manager

A full interface for viewing and managing everything Zynkbot knows about you.

- Browse memories in chronological or relevance order
- Search by keyword or semantic meaning
- Filter by namespace (personal, work, health, travel, etc.)
- Edit memory content and titles inline
- Delete individual memories or clear all
- View relationship links between memories
- See extracted entities and event metadata

**Access:** Click the Memory Manager button in the main interface.

---

## Knowledge Base

Index your own documents for semantic search. Zynkbot splits files into overlapping chunks, generates local embeddings, and stores them for retrieval. When you click the KB button before sending a message, Zynkbot searches your indexed documents and brings the most relevant sections into the conversation.

**Supported file types:** `.txt`, `.md`, `.csv`, `.json`, `.log`, `.rs`, `.js`, `.jsx`, `.ts`, `.tsx`, `.py`, `.java`, `.cpp`, `.c`, `.h`, `.html`, `.css`, `.xml`, `.yaml`, `.yml`, `.toml`

**How to use:**
1. Open Settings → Knowledge Base
2. Click Open Folder to place files in your KB folder
3. Click Knowledge Base Manager and index the files you want searchable
4. Click the 📚 KB button before your message to search

If you mention a filename in your message, Zynkbot prioritizes results from that file.

PDF and DOCX support is planned.

---

## Conversation History

Every completed conversation exchange is automatically saved to your local database.

- Sessions grouped by date (Today, Yesterday, This Week, This Month)
- Full-text search across all past conversations
- Date range filter
- Resume a past session — reloads messages into the active chat
- Delete individual sessions
- Disabled entirely in HIPAA mode

**Access:** System Controls (⚙️ bottom left) → View History

---

## Ensemble Mode

Runs multiple AI models simultaneously on the same question and synthesizes their responses.

**How it works:**
1. A coordinator model first checks if live web search is needed
2. Each selected model answers independently with your memory context (and search results if applicable)
3. The coordinator synthesizes a final answer, identifying agreements and disagreements

**Notes:**
- Ensemble responses are not stored as memories
- Requires at least 2 models selected
- Can mix local and API models
- Displays individual responses alongside the synthesized result

---

## Relationship Graph

A visual, interactive map of how your memories connect.

- Nodes = memories, edges = relationships
- Color-coded by relationship type (contradicts, supports, elaborates, etc.)
- Click any node to view the full memory
- Zoom, pan, and filter by relationship strength
- Useful for finding contradictions, memory clusters, and context chains

---

## ZynkSync — Device Pairing

Sync your memory database across your own devices over your local network. No cloud relay, no third-party servers.

**How to pair:**
1. Generate a 6-digit code on Device A
2. Enter the code on Device B within the time limit
3. Devices exchange certificates and begin syncing

Memory sync uses certificate-based authentication and runs automatically in the background once paired.

---

## ZynkLink — File Sharing

Share files directly with other Zynkbot users over your local network.

- Share directories from your machine with linked users
- Browse files shared by linked devices
- **→ KB**: Download a file directly into your Knowledge Base and index it immediately
- **Save...**: Download to any location on your file system

Both devices must be online and running Zynkbot. No internet connection required.

---

## ZChat — Device Messaging

Direct device-to-device messaging between paired Zynkbot instances. Messages are delivered over your local network without any cloud relay.

---

## Safety and Containment Modes

Zynkbot includes a layered content safety system. The active mode is selected in Settings.

| Mode | Description |
|---|---|
| **Guardian** (default) | Basic client-side content filtering using the local toxicity classifier. Balanced for everyday use. |
| **Child** | Strict filtering using OpenAI Moderation API. Forces cloud backend for safety checks. Designed for minors. |
| **Sovereign** | User-defined rules take precedence. Maximum personal autonomy. |
| **Witness** | Observes and logs but does not enforce. For monitoring and research. |
| **HIPAA** | Disables memory extraction and conversation history entirely. No data is stored. For healthcare and sensitive professional contexts. |

**Safety stack:** Input filtering → TinyBERT toxicity classifier (toxic-bert) → output filtering. Child mode adds OpenAI Moderation API as an additional layer.

---

## Onboarding

A guided first-run questionnaire that builds your initial memory profile.

- Covers name, age, family, relationships, interests, goals
- Each response stored as a memory
- Can be skipped and revisited later
- All responses editable in Memory Manager

---

## Snap-ins

Specialized behavioral modules that adapt Zynkbot for specific use cases. The Therapist snap-in is the first implemented example — it organizes session notes separately and is accessible only within the snap-in context.

---

## Web Search

On-demand web search integrated into conversation. When a query requires current information (news, recent events, live data), Zynkbot can search the web and incorporate results into its response. Used automatically in Ensemble Mode's coordinator phase.

---

## Voice Input

Voice transcription via local Whisper models is implemented but temporarily disabled due to a library conflict between whisper.cpp and llama.cpp (GGML symbol collision). This will be re-enabled when the upstream conflict is resolved.

---

## Model Management

**Local Models (GGUF format):**

Recommended models (offered by the installer):
- **DeepSeek R1 Distill Llama 8B** (4.7GB, Q4_K_M) — reasoning model with chain-of-thought
- **Llama 3.1 8B Lexi Uncensored V2** (4.9GB, Q4_K_M) — creative, unfiltered responses
- **Qwen3 8B** (5.0GB, Q4_K_M) — best coding and instruction-following in the 8B class

Most other GGUF models from HuggingFace also work. Place any `.gguf` file in the models folder and it appears in the model dropdown automatically.

**Cloud API Models:**
- Anthropic Claude (Haiku, Sonnet, Opus)
- OpenAI GPT-4 series
- xAI Grok

API keys are stored locally in your `.env` file and configured via Settings → API Keys.
