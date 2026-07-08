# ContainAI: Ethical AI Infrastructure

**Flagship Product: Zynkbot** – Your epistemic history — a structured record of how your thinking evolves

---

Zynkbot is a local-first tool for structured, personal memory. The closest points of reference aren't chat assistants like ChatGPT — they're the tools people already use to think over time: Obsidian, Roam, Logseq, journals, personal CRMs. Like those, Zynkbot is a place to store and revisit what matters to you. Unlike those, it doesn't just hold notes — it maintains a relational memory graph that records not only facts, but how those facts connected to each other at a specific point in time: what you believed, what you doubted, how an idea that began in one context became something different later.

That distinction is the whole point. Journals are linear. Notes are unstructured. Search history is behavioral, not reflective. Zynkbot's memory is relational, persistent, and local — a structure for not just what you recorded, but how you understood it as you recorded it. And because it runs entirely on your own device, that record is yours alone: no servers, no profiling, no training on your life.

**Why this matters: recursion.**

In computing, recursion is a method where a system refers back to earlier steps to solve a problem more effectively. People do something similar. We reflect on past actions, revisit old conversations, and adjust as our understanding changes — reflection naturally forms feedback loops that shape future decisions. This looping process — memory, reflection, adjustment — is how people grow.

Most digital systems disrupt that loop. Corporate algorithms redirect attention, capture behavioral data, and optimize for engagement rather than growth. Zynkbot inverts that model: it gives you the same capabilities — pattern tracking, reflection, adaptive interaction — but places them entirely under your control. The same tools used to manipulate people for profit become tools for deliberate self-understanding.

This isn't artificial intelligence in the usual sense. It's structured reflection — built to mirror the way real growth happens.
Not artificial. Aligned. And recursive.

---

## Current State: Desktop Application (v0.9)

**Feature-complete (v0.9, hardening):**
- ✅ Persistent semantic memory (local SQLite, in-process vector search)
- ✅ Persistent conversation history (searchable, date-grouped, resume support, disabled in HIPAA mode)
- ✅ Hybrid search (entity extraction + semantic similarity)
- ✅ Cross-device memory sync (ZynkSync)
- ✅ Containment modes (Guardian, Child, HIPAA, Sovereign, Witness)
- ✅ Pure Rust ML stack (Candle framework)
- ✅ Multi-model support (local .gguf + API backends)
- ✅ Knowledge base with RAG
- ✅ Transparent, editable memory

**Platform support:**
- ✅ Windows 10/11 (tested, v0.9)
- ✅ Linux (Ubuntu, Arch, Fedora - tested, v0.9)
- 🔄 macOS (untested, should work)
- 📱 Android/iOS (planned via Tauri Mobile)

**License:**
- AGPL v3 for non-commercial use (prevents surveillance capitalism)
- Commercial license available for enterprise

---

## What Comes Next

Zynkbot is the foundation of a larger ecosystem. The two planned expansions are documented separately:

- **[Zynkbot SDK](SDK_VISION.md)** — modular components (memory system, containment layer, ZynkSync, HIPAA framework) available as a privacy-first developer platform. Planned for 2027.
- **[ContainAI Foundation](FOUNDATION_VISION.md)** — non-profit providing long-term governance, independent security audits, and grants to aligned developers. Planned for 2028.

---

## Design Principles

### 1. Privacy-First, Always
- Local processing by default
- No telemetry or tracking
- User owns all data
- Transparent about what leaves device
- API calls are opt-in, not required

### 2. Consent-Bound Systems
- Explicit user consent for all operations
- Containment modes enforce safety boundaries
- Audit trails for sensitive modes (HIPAA)
- Users can disable any feature

### 3. Transparent Decision-Making
- Memory retrieval is visible
- AI reasoning is explainable
- Safety filters are user-configurable
- Source code is auditable (AGPL)

### 4. Sustainable Open Source
- Dual licensing prevents exploitation
- Commercial use funds development
- Foundation ensures long-term maintenance
- Community-driven roadmap

### 5. Anti-Surveillance Capitalism
- AGPL prevents closed-source derivatives
- No data harvesting for training
- No engagement optimization
- No dark patterns or manipulation

### 6. Digital Resilience
- Offline-first architecture (works without internet)
- No dependency on cloud infrastructure or subscriptions
- Continues functioning during network outages
- Enables deployment in resource-limited environments
- Humanitarian applications (disaster response, developing regions, educational access)

**[→ Complete Digital Resilience Documentation](DIGITAL_RESILIENCE.md)**

---

## Why This Matters

**Most AI assistants:**
- Store your data on their servers
- Use conversations to train models
- Require internet connectivity
- Black-box decision making
- No control over memory

**Zynkbot:**
- ✅ Your data stays on your device
- ✅ No training on your conversations
- ✅ Works offline (local-first)
- ✅ Transparent memory and reasoning
- ✅ Full control over all data

**The vision:**
- Prove that ethical AI is commercially viable
- Create tools that empower, not exploit
- Build a developer ecosystem around privacy
- Establish standards for consent-based AI
- Prove AI can be trustworthy to the public

---

## Success Metrics

**Year 1 (2026):**
- 1000 active desktop users
- v1.0 stable release
- Mobile beta (Android)
- Community contributions

**Year 2 (2027):**
- 10,000 active users across desktop + mobile
- SDK v1.0 released
- 10+ third-party Snap-ins
- First commercial SDK licenses

**Year 3 (2028):**
- 100,000 active users
- ContainAI Foundation established
- 50+ SDK integrations
- Self-sustaining revenue model

**Long-term:**
- SDK standard for privacy-first AI development
- Ecosystem of aligned projects and companies
- Demonstrable alternative to surveillance capitalism

---

## The Path Forward

**Immediate (2026):**
1. v1.0 desktop release
2. Open source launch (GitHub, HN, Reddit)
3. Documentation and community building
4. Mobile development begins

**Near-term (2027):**
1. Mobile app release (Android priority)
2. SDK extraction and documentation
3. Commercial licensing established
4. Snap-in marketplace prototype

**Long-term (2028+):**
1. Foundation establishment
2. Grant programs for aligned developers
3. Partnerships with privacy-focused organizations
4. Expand to new platforms and use cases

---

## Get Involved

Zynkbot is a statement that **AI should serve users, not exploit them**.

**For users:** Take control of your AI interactions. Your data, your device, your choice.

**For developers:** Build privacy-first AI applications. Use the SDK, contribute code, create Snap-ins.

**For organizations:** Deploy ethical AI that respects user privacy. Commercial licensing supports the ecosystem.

**For everyone:** Spread the word. Surveillance capitalism is not inevitable. Better alternatives exist.

---

## Brand Structure

**Organization:** ContainAI
**Tagline:** Ethical AI Infrastructure
**Flagship Product:** Zynkbot (Privacy-first AI assistant)
**Future Platform:** ContainAI SDK
**Future Entity:** ContainAI Foundation

**Websites:**
- containai.ai – Main site (company, SDK, foundation)
- containai.ai/zynkbot – Product page
- zynkbot.com – Product marketing (redirect or standalone)
- docs.containai.ai – Developer documentation
- foundation.containai.ai – Foundation information

**Contact:** matt@containai.ai
**GitHub:** https://github.com/MSkill1/zynkbot
**License:** AGPL v3 (non-commercial) + Commercial license
**Founded:** 2025
**Founder:** Matthew Skillman

---
**ContainAI** – Building the infrastructure for ethical AI

*"Memory without surveillance. Intelligence without manipulation."*

---

*Looking ahead*

If someone uses Zynkbot consistently — for years — something further may become possible. Because the memory graph preserves how understanding evolved, not just what was recorded, a long enough history could let a person reconstruct not merely what happened in their life, but how they understood it as it happened. Whether that potential is realized depends on consistent use and data longevity that no one can yet verify. But if it holds, the implications reach beyond productivity — into how people understand themselves across time.

