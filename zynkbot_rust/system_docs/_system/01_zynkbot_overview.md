# Zynkbot — Local-First AI Companion

## What is Zynkbot?

Zynkbot is a local-first AI companion that remembers everything you share and builds a persistent understanding of your life, goals, relationships, and interests. Unlike cloud-based AI assistants, Zynkbot stores all data locally on your device. Your memories, conversations, and personal information never leave your machine unless you explicitly sync with devices you control.

## Core Philosophy

**Privacy First**: Everything runs on your device. No telemetry, no surveillance, no third-party data collection.

**User Control**: You own your data. Every memory can be viewed, edited, or deleted at any time. Zynkbot only knows what you choose to share.

**Transparency**: All memories are visible in the Memory Manager. You can see exactly what Zynkbot knows about you and how memories are connected to each other.

## Key Features

1. **Persistent Memory** — Zynkbot builds a long-term memory graph from your conversations. Facts, preferences, relationships, and context are stored locally and retrieved semantically when relevant.

2. **Contradiction Detection** — When new information conflicts with something already stored, Zynkbot surfaces the conflict and asks you to resolve it before storing anything.

3. **Relationship Graph** — Memories are linked by semantic relationships (contradicts, supports, elaborates, caused_by, reminds_of). A visual graph lets you explore how your memories connect.

4. **Knowledge Base** — Index your own documents (text files, markdown, code, etc.) for semantic search. Ask questions about your documents and Zynkbot retrieves the most relevant sections.

5. **Conversation History** — Every conversation is automatically saved. Browse by date, search by keyword, and resume past sessions.

6. **Multi-Model Support** — Use local GGUF models (Llama, Mistral, Qwen) for complete privacy, or connect to cloud APIs (Anthropic Claude, OpenAI GPT, xAI Grok) when you want more capability.

7. **Ensemble Mode** — Query multiple AI models simultaneously and get a synthesized answer. Useful for research, fact-checking, and complex questions.

8. **ZynkSync** — Pair your own devices and sync your memory database across them, entirely over your local network with no cloud relay.

9. **ZynkLink** — Share files and documents directly with other Zynkbot users over your local network. Download files from a linked device straight into your Knowledge Base.

10. **ZChat** — Device-to-device messaging between paired Zynkbot instances, no cloud relay required.

11. **Safety Modes** — Containment modes (Guardian, Child, HIPAA, Sovereign, Witness) enforce different levels of content filtering and data handling appropriate to the context.

12. **Snap-ins** — Specialized behavioral modules that adapt Zynkbot for specific use cases (e.g., the Therapist snap-in).

## Technology Stack

- **Frontend**: React (JavaScript/JSX) with Tauri
- **Backend**: Rust — memory-safe, high-performance native code
- **Database**: SQLite (embedded, local file — no server process required)
- **Embeddings**: all-MiniLM-L6-v2, runs entirely on-device (384-dim vectors)
- **NLP**: BERT-based Named Entity Recognition (dslim/bert-base-NER)
- **Local Models**: GGUF format via llama.cpp integration
- **Cloud APIs**: Anthropic, OpenAI, xAI

## Supported Platforms

- Windows 10/11
- Linux (Ubuntu 20.04+, and most modern distributions)
- macOS support planned
- Mobile (Android, iOS) planned via Tauri Mobile
