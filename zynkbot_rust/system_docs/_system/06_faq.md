# Zynkbot — Frequently Asked Questions

## General

**What is Zynkbot?**
A local-first AI companion that builds a persistent memory of your conversations and stores everything on your own device. Unlike cloud AI assistants, your data never leaves your machine unless you explicitly sync with devices you control.

**How is Zynkbot different from ChatGPT or Claude?**
- Persistent memory across conversations
- Runs on your device — no cloud required
- You can view, edit, or delete any memory at any time
- Supports local models for complete offline use
- Contradiction detection surfaces conflicting memories and asks you to resolve them
- Cross-device sync without a cloud relay

**Do I need an internet connection?**
No, if you use local GGUF models. Internet is only needed for cloud API models (Anthropic, OpenAI, xAI) and web search in Ensemble Mode. ZynkSync and ZynkLink work entirely over your local network.

**Is Zynkbot free?**
The software is free and open-source (AGPL v3 for non-commercial use; commercial license available). Local models are free after the hardware investment. Cloud API usage costs what the respective provider charges.

**What license is Zynkbot under?**
Dual-licensed: AGPL v3 with anti-surveillance provisions for personal, educational, and non-profit use; a separate commercial license for business use. See the LICENSE and COMMERCIAL_LICENSE.md files in the repository.

---

## Privacy and Data

**Where is my data stored?**
Locally in a SQLite database file on your machine. Nothing is sent to external servers unless you explicitly use a cloud API model (in which case only the message goes to that provider — memories stay local).

**What happens when I use Claude or GPT?**
Your message and relevant memory context are sent to the API provider. Your memory database and personal data remain on your device. Using local models keeps everything private.

**Is Zynkbot HIPAA compliant?**
HIPAA mode disables memory extraction and conversation history entirely — no records of any kind are stored. Use local models for fully air-gapped operation. Store the database on an encrypted drive for full compliance.

**Can I delete my data?**
Yes. Delete individual memories in Memory Manager, clear all memories in Settings, or delete the database file entirely: `rm ~/.local/share/zynkbot/zynkbot.db` on Linux, or delete `%LOCALAPPDATA%\zynkbot\zynkbot.db` on Windows.

**How do I back up my memories?**
```bash
# Linux
cp ~/.local/share/zynkbot/zynkbot.db ~/zynkbot_backup.db
```
Restore by copying the backup file back to the same path. On Windows, copy `%LOCALAPPDATA%\zynkbot\zynkbot.db` to a safe location.

---

## Memory System

**How does memory extraction work?**
After every conversation turn, a background pipeline runs:
1. A heuristic check filters out obvious filler (very short messages, "ok", "thanks", etc.)
2. The LLM decides if the message contains something worth remembering
3. A duplicate check prevents storing near-identical memories
4. If a contradiction is detected, you're asked to resolve it before anything is stored
5. Memories that pass all checks are stored with entity extraction, event detection, and namespace classification

**What does Zynkbot remember vs. ignore?**
Zynkbot stores personal facts, plans, preferences, relationships, and emotional context. It ignores conversational filler, questions without informational content, and acknowledgments. The LLM makes the final judgment.

**What is contradiction detection?**
When new information conflicts with something already stored, Zynkbot shows both memories side by side and asks you to choose from five options: Keep Old (discard the new information), Keep New (replace the old memory), Not a Contradiction (store the new memory normally without linking), Keep Both (store both with a `contradicts` link), or Resolve with Explanation (store both plus a user-written explanation). Nothing is stored until you make a choice.

**Can I add memories manually?**
Yes. Open Memory Manager → Add Memory → enter title and content.

**How do I organize memories?**
Using namespaces. The default is `personal`. Common namespaces include `work`, `health`, `travel`. You can create custom ones. Searches can be filtered by namespace.

---

## Knowledge Base

**What is the Knowledge Base?**
A RAG (Retrieval-Augmented Generation) system for your own documents. Index text files, markdown, code, logs, and other supported formats. When you click the 📚 KB button before sending a message, Zynkbot searches your indexed documents semantically and brings the most relevant sections into the conversation.

**What file types are supported?**
`.txt`, `.md`, `.csv`, `.json`, `.log`, `.rs`, `.js`, `.jsx`, `.ts`, `.tsx`, `.py`, `.java`, `.cpp`, `.c`, `.h`, `.html`, `.css`, `.xml`, `.yaml`, `.yml`, `.toml`

PDF and DOCX support is planned.

**Does Zynkbot search the KB on every message?**
No. KB search only runs when you click the 📚 KB button. Once results are in the conversation, Zynkbot can discuss them across multiple turns without re-searching.

---

## Models

**What models does Zynkbot support?**

Local (GGUF format): Llama 3.2/3.3, Mistral 7B, Dolphin-Mistral variants, Qwen 2.5 series, and most GGUF models from HuggingFace.

Cloud APIs: Anthropic Claude (Haiku, Sonnet, Opus), OpenAI GPT-4 series, xAI Grok.

**How do I add local models?**
Download a `.gguf` file from HuggingFace and place it in the `models/` folder. It appears in the model dropdown automatically after restart.

**How much RAM do I need?**
- 3B model: ~4GB RAM (fast, good for most tasks)
- 7B model: ~8GB RAM (better quality)
- 13B model: ~16GB RAM
- GPU VRAM requirements are roughly half the above if using CUDA

Quantized models (Q4_K_M) use approximately half the memory of full precision.

**What model should I use?**
Zynkbot's installer offers three recommended local models:
- **DeepSeek R1 Distill Llama 8B** (4.7GB) — reasoning model with chain-of-thought; best for complex analysis
- **Llama 3.1 8B Lexi Uncensored V2** (4.9GB) — creative, unfiltered responses; best for open-ended conversations
- **Qwen3 8B** (5.0GB) — best instruction-following and coding in the 8B class; recommended starting point

For maximum quality, use Anthropic Claude Sonnet (API key required).

---

## Sync and Sharing

**How does ZynkSync work?**
Pairs your own devices over your local network. Generate a 6-digit code on one device, enter it on the other. Devices exchange certificates and sync memory databases. No internet required. No cloud relay.

**What is ZynkLink?**
File sharing between Zynkbot users on the same local network. You can browse shared directories on a linked device and download files — either to your Knowledge Base (indexed immediately) or anywhere on your file system.

**Can I share memories with other people?**
Memory sync (ZynkSync) is for your own devices only. ZynkLink enables cross-user file sharing. Selective memory sharing with other users is on the roadmap.

---

## Ensemble Mode

**What is Ensemble Mode?**
Runs multiple AI models simultaneously on the same question. A coordinator model first checks if web search is needed, then each selected model answers independently with your memory context, then the coordinator synthesizes the best answer.

Ensemble responses are not stored as memories — it's a research and fact-checking tool.

---

## Containment and Safety

**What containment modes are available?**
Guardian (default), Child, Sovereign, Witness, and HIPAA. Each applies different levels of content filtering and data handling. Guardian uses the local toxic-bert classifier for balanced everyday filtering. Child adds the OpenAI Moderation API for robust filtering designed for minors. HIPAA disables all memory extraction and conversation history entirely.

---

## Development

**Is Zynkbot open source?**
Yes. Licensed under AGPL v3 for non-commercial use. Commercial license available separately. See the repository for details.

**How do I report a bug?**
Open an issue on GitHub with steps to reproduce, expected vs. actual behavior, and relevant logs.

**What's on the roadmap?**
Mobile apps (Android, iOS), macOS support, streaming ZynkLink file transfers, PDF/DOCX support in the Knowledge Base, voice input re-enablement, snap-in marketplace, and memory sharing between users.
