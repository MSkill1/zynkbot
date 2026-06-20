# Case Study — HVAC Technician in the Field

*Offline recall, hands-free notes, and technical documentation that follows you to the job site*

---

## The Situation

Jake is a licensed HVAC technician working in rural Pennsylvania. Most days, he's driving from site to site — installing, diagnosing, and repairing residential and commercial systems. His phone is his lifeline, but internet access is unreliable in the areas he works, and his workday is too fast-paced for typing out detailed notes or digging through email chains.

He's not looking for a personal companion chatbot. He wants a quiet assistant that can keep track of things he forgets, surface useful information on the fly, and stay out of the way when he doesn't need it.

> **Note:** This case study describes the planned mobile version of Zynkbot (Android/iOS via Tauri Mobile or similar, expected 2026-2027). The desktop version (Windows, Linux) is production-ready now (v0.9) with all core features: offline operation, Knowledge Base RAG, ZynkLink device sync, and local storage.

---

## Remembering What Happened at Every Site

After finishing a job at the Thompson property, Jake dictates a quick note:

> *"Remember: Thompson site, replaced capacitor, 370V dual-run, Carrier part CR43160370, next maintenance due September 2026."*

Zynkbot stores this as a memory — extracting the client name, part number, and date as named entities so it can find the record later through both exact and semantic search. The memory lives in a local database on his phone. Nothing goes to a server.

Three weeks later, before heading back to the area, he asks:

> *"When's Thompson's next service?"*

Zynkbot returns: *"Thompson next maintenance: September 2026 (capacitor replacement)."*

He didn't have to remember what folder he put the note in, or scroll through photos of handwritten labels. He asked a question and got the answer.

---

## Recalling Past Solutions

The same hybrid search that finds specific facts also surfaces conceptual matches — things Jake recorded but didn't explicitly file under a category.

Standing in a basement with a low-airflow problem, he asks:

> *"What did I do for low airflow issues before?"*

Zynkbot searches his stored memories and returns three previous jobs where he recorded the fix: a clogged return vent at one residence, an undersized duct at a commercial building, a wrong fan speed setting at a third site. He didn't use those exact words when he stored those memories — the system found them because the concepts cluster together in semantic space. "Low airflow" maps to "clogged vent," "undersized duct," and "fan speed" without requiring him to remember how he phrased anything.

---

## Technical Manuals, Searchable Offline

Jake has years of service manuals on his home desktop: Carrier installation guides, Trane troubleshooting manuals, Lennox wiring diagrams, R-22/R-410A/R-32 pressure-temperature charts, ACCA Manual J/D references. He's indexed all of them into Zynkbot's Knowledge Base.

On the roof of a commercial property, he asks:

> *"What's the R-22 pressure chart for 90 degrees ambient?"*

Zynkbot searches the indexed PDFs locally and returns: *"At 90°F ambient, R-22 high-side pressure should be 225-250 PSI, low-side 68-70 PSI. Source: Carrier Service Manual page 47."*

No signal needed. The search runs on his phone using local embeddings against the indexed document chunks. The answer cites the source page so he can pull the full context if he needs it.

---

## Getting Manuals to His Phone

Jake's manual library lives on his home desktop. To get a manual onto his phone before a job, he uses ZynkLink — Zynkbot's local device-to-device file sharing.

His phone pairs with his desktop over local WiFi using a 6-digit code. He browses the shared folder, finds the manual he needs, and clicks **→ KB**. The file downloads directly into his phone's Knowledge Base and indexes automatically — one step, no separate import. The next time he's in the field, it's searchable offline.

Files transfer directly between devices. Nothing touches a cloud server.

---

## Voice Input

Voice input via Web Speech API is available now and requires an internet connection. Offline Whisper transcription is a planned feature for a future release — and will be built from Rust directly (currently only python version available).

When Jake has signal, he uses voice for hands-free operation: dictating notes while finishing an install, querying past jobs while his hands are occupied. In basements and crawlspaces with no cell signal, he types. All memory search, Knowledge Base lookup, and local LLM inference continue to work without any network connection.

---

## What Works Without Internet

Everything except the optional cloud-based LLMs and online voice transcription runs locally:

- Memory recall — hybrid entity + semantic search, local database
- Knowledge Base search — indexed PDFs, local vector search
- Entity extraction — BERT NER, local Rust/Candle
- Semantic embeddings — all-MiniLM-L6-v2, local Rust/Candle
- Local LLM inference — Llama 3.2 3B via llama.cpp
- ZynkLink file sharing — local WiFi/LAN

API-based models (OpenAI, Anthropic, xAI) are optional and require internet. Voice transcription via Web Speech API requires internet; offline Whisper is a planned feature.

---

## Client Data That Doesn't Leave the Phone

Jake's memory database lives in a local SQLite database on his device. His Knowledge Base documents are stored in a local folder. ZynkSync, if he uses it, transfers data to his home desktop over local WiFi — never through a third-party server.

If a client relationship ends, he can delete those memories through the Memory Manager. If he wants to separate work notes from personal ones, namespace filtering keeps them in different buckets. If he changes phones, he exports the database and imports it on the new device. His employer has no access to his notes. His clients' service histories are his, not a company's asset.

---

## The Architecture That Makes This Possible

| Capability | What It Does Here |
|---|---|
| **Persistent memory with hybrid search** | Part numbers, client history, and past fixes recalled from natural-language queries |
| **Knowledge Base RAG** | Service manuals indexed locally and searchable offline — no internet required after setup |
| **ZynkLink → KB** | One-click download from shared desktop folder directly into phone's Knowledge Base |
| **Offline-first operation** | Local LLM, local embeddings, local search — works in basements and rural areas |
| **Voice input** | Web Speech API (requires internet now); offline Whisper transcription planned |
| **Local SQLite storage** | All memories and documents stay on device — no cloud, no employer access |
| **Memory Manager** | Edit or delete any stored memory; namespace filtering separates work from personal |

---

*For field service professionals, tradespeople, and anyone working in environments where cloud connectivity is unreliable or unacceptable: matt@containai.ai*
