# Ensemble Mode: Multi-Model Intelligence

**Get a second — and third — opinion by querying multiple AI models at once, then synthesize their answers into a single best response**

Ensemble mode is not a networking feature — it works entirely on a single machine using local models, or any combination of local and API models you have configured. It is an AI quality and verification tool.

---

## What It Does

Ensemble mode sends the same question to multiple AI models simultaneously, then uses a designated coordinator model to read all the responses, identify where models agree and disagree, and synthesize a final answer.

**When it's most useful:**
- Important decisions where you want more than one AI perspective
- Fact-checking claims that might be outdated or wrong
- Complex questions where different viewpoints matter
- Catching confident-sounding errors — when one model gets something wrong, others often don't

**When to skip it:**
- Casual conversation or simple questions
- Time-sensitive requests (ensemble questions can often take 60+ seconds or more)
- Creative writing where variation is noise, not signal

---

## How It Works

First, the coordinator model checks whether your question needs current information — news, software versions, recent events. If so, it runs a quick web search and shares those results with all the other models.

Then each selected model answers your question independently. They don't see each other's answers at this stage. All models receive the same context: your relevant memories, any knowledge base documents you've enabled, and web search results if retrieved.

Finally, the coordinator reads every response and produces a structured synthesis: what all models agreed on, where they diverged and why, and a final best answer that draws on all of them.

Individual model responses are shown alongside the synthesis so you can evaluate them yourself.

---

## Key Features

- ✅ **Multi-model support**: Mix local (.gguf) and API models — minimum 2 required
- ✅ **Parallelized**: All models answer simultaneously; total time = slowest model + coordinator
- ✅ **Web search integration**: Auto-detects when current information is needed
- ✅ **Memory context**: All models receive your relevant memories
- ✅ **Knowledge base context**: Toggle KB to search your indexed documents from within ensemble
- ✅ **File attachment**: Attach a document directly in the ensemble window
- ✅ **Disagreement detection**: Coordinator explicitly flags where models conflict
- ✅ **Child mode protection**: UI blocks ensemble when Child containment is active

---

## Use Cases

### Fact-Checking and Verification

Use ensemble when you want to verify a claim — especially one that may have changed recently or involves technical specifics.

**Developer:** *"What changed in PostgreSQL 17's connection handling?"*
If your local model states something different from a cloud model, that disagreement is a signal to check the actual documentation before relying on either answer.

**Student:** *"Was the Treaty of Versailles the primary cause of World War II?"*
Different models emphasize different historical arguments. The synthesis surfaces where historians agree, where they don't, and what evidence underlies each position — more useful than one model's confident take.

---

### Comparing Perspectives on Complex Questions

Ensemble is valuable when a question doesn't have a single clean answer and you want to see the range of reasonable positions.

**Business decision:** *"Should I price my SaaS product per-seat or with usage tiers?"*
Models may have different priors on pricing strategy. Disagreement is signal — it shows you the actual tradeoffs, not just whichever model you happened to ask.

**Medical context (informational only):** *"What are the evidence-based treatments for chronic lower back pain?"*
Different models may emphasize different clinical guidelines. The synthesis can show consensus recommendations vs. contested approaches — useful context before a doctor's appointment, not a substitute for one.

---

### Arts, Humanities, and Interpretation

Ensemble isn't just for technical questions. Any field where reasonable people disagree benefits from multiple perspectives.

**Literary analysis:** *"Is Frankenstein more a story about the dangers of unchecked ambition or about the loneliness of being misunderstood?"*
Models trained on different literary commentary will lean different ways. The synthesis shows you what each argument is, not just a single interpretation.

**Historical interpretation:** *"Was Napoleon a liberator or a conqueror?"*
A genuinely contested historical question. The coordinator surfaces the real debate — which evidence each side finds compelling, and why it has never had a clean answer.

**Music theory:** *"Why does the tritone sound so unsettling?"*
Blend a local model with an API model. You might get an acoustics-focused answer from one and a cultural history answer from another. Both are true, and together they're more complete.

---

### Research

For questions requiring depth, different models may emphasize different angles. The synthesis aggregates them rather than requiring you to run multiple conversations.

**Example:** *"What does the research say about spaced repetition vs. massed practice for long-term retention?"*

Models may cite different studies or different practical implications. The synthesis gives you a cleaner overview than any single response.

---

## Example: What Disagreement Looks Like

**Question:** *"Has social media been good or bad for society overall?"*

| Model | Position |
|---|---|
| Claude | Mixed — enabled unprecedented connection and democratized information, but amplified polarization and contributed to mental health crises |
| GPT-4 | Cautiously negative — the way these platforms are designed to maximize engagement makes harmful outcomes predictable and hard to fix |
| Local (Llama) | Depends on who's using it — positive for adults and professional networking, harmful for teenagers |

**Synthesis output (approximate):**

> **DISAGREEMENT DETECTED**
> All three models acknowledge both benefits and harms, but differ on which outweighs the other. GPT-4 focuses on how the platforms are built as the root cause; Claude and the local model weigh the outcomes differently depending on use case and age group.
>
> **SYNTHESIZED ANSWER:** Social media has had genuinely mixed effects that resist a clean verdict. The benefits are real — political movements, health support communities, and long-distance relationships have all been helped by it. But the documented harms — rising anxiety and depression in teenagers, political polarization, the spread of misinformation — are also real, and largely a result of how these platforms chose to build their recommendation systems. The honest answer: net positive for adults using it deliberately, net negative for teenagers using it constantly, and largely a function of design choices that could have been made differently.

This is more useful than asking any single model — you get the actual dispute surfaced and resolved, not just one model's confident take.

---

## Setup

1. Click the **"🤝 Ensemble"** button in the chat interface
2. Select 2 or more models from your available list
3. Optionally: toggle **📚 KB** to include your knowledge base, or **📎** to attach a file
4. Enter your question
5. Wait for all models to respond (typically 1–3 minutes; local models may take longer)
6. Review individual responses and the synthesized answer
7. Optionally add the result to your active conversation

**Coordinator model:** The coordinator is automatically selected as the most capable API model in your selection (preference order: Anthropic → OpenAI → xAI → local). If only local models are selected, the first selected model acts as coordinator. User-selectable coordinator is planned for a future release.

---

## Performance

| Factor | Detail |
|---|---|
| Execution time | 1–3 minutes typical (depends on slowest model) |
| Parallelization | All models answer simultaneously |
| Bottleneck | Slowest model + coordinator synthesis |
| Web search | Adds approximately 1 minute or more if triggered |

**Tip:** Including large local models running on CPU significantly increases total wait time — the coordinator waits for every model before synthesizing. If speed matters, stick to API models if not using CUDA.

---

## Limitations

- Requires at least 2 configured models
- Coordinator waits for the slowest model — one slow model delays the whole response
- Web search is best-effort (DuckDuckGo, 5-second timeout); some queries may not return useful results
- Synthesis quality depends on the coordinator model — a weak coordinator produces weaker synthesis
