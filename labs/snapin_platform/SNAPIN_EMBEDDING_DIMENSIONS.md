# Snap-in Embedding Dimensions: Higher-Quality Semantic Search

**Status:** Design Reference (Future Capability)
**Related:** `labs/snapin_platform/SNAPIN_PLATFORM_DESIGN.md`, `src-tauri/src/lib.rs` (embedding pipeline)

---

## Background

Zynkbot's core memory system uses `all-MiniLM-L6-v2`, a 384-dimensional embedding model
that runs fast on CPU and is well-suited to short personal memories. This was a deliberate
choice — it's purpose-built for semantic similarity, runs locally without a GPU, uses ~90MB
of RAM, and encodes a sentence in roughly 15ms on consumer hardware.

For most snap-in use cases, this is fine. But snap-ins targeting specialized professional
domains — medical, legal, financial, academic research — may benefit from higher-dimensional
embeddings that capture domain-specific terminology and nuance more accurately.

This document explains how that works, what the tradeoffs are, and why it is not a barrier
for snap-in developers who don't want to think about it.

---

## The Short Version for Snap-in Developers

**You don't have to do anything.** If you don't specify an embedding model in your
`snapin.toml`, your snap-in uses the same 384-dimensional model as everything else.
It works. It's fast. It's good enough for most use cases.

If you are building a domain-specific professional tool and want better semantic recall
for specialized terminology, you can optionally declare a higher-dimension model. This
document explains what that means and what tradeoffs come with it.

---

## Why Dimensions Matter for Domain-Specific Content

A 384-dimensional embedding model encodes the meaning of a sentence into a list of 384
numbers. More dimensions means more "space" to encode subtle distinctions between concepts.

For general personal memories — "I prefer tea over coffee", "My sister lives in Portland"
— 384 dimensions captures the meaning well. The concepts are common English, and the model
was trained on exactly this kind of text.

For domain-specific content, the gap widens:

- **Medical:** "The patient presents with idiopathic thrombocytopenic purpura" needs the
  model to understand that this is a specific blood disorder, not just words about a patient.
- **Legal:** Distinguishing between "tortious interference" and "breach of fiduciary duty"
  requires understanding precise legal distinctions that a general model compresses poorly.
- **Financial:** "Duration risk in a rising rate environment" has a specific meaning that
  differs from its surface reading.

Larger models trained on domain-specific corpora encode these distinctions more accurately,
meaning search results are more relevant and recall is higher for the terminology that
matters in that profession.

---

## The Technical Constraint: Embedding Dimensions

Zynkbot stores embeddings as binary BLOBs in SQLite and computes vector similarity in-process in Rust. The core memory system uses 384-dimensional vectors (all-MiniLM-L6-v2). All embeddings stored in the same BLOB column must use the same dimension — mixing 384-dim and 768-dim vectors in the same column would produce incorrect similarity scores.

This means a snap-in using a different embedding model cannot share the same database column
as the core memory system. It needs its own table with its own embedding column at the correct
dimension.

This is not a problem. Snap-ins already have isolated storage by design. A snap-in with a
domain model simply declares its knowledge base tables at its chosen dimension. The isolation
that already exists for privacy and data separation also cleanly accommodates different
embedding dimensions.

---

## The Cross-Search Problem

Here is the real tradeoff to understand.

When the main memory search runs, it encodes the user's query as a 384-dimensional vector
and searches all memories in the core 384-dimensional space. If a snap-in's memories live
in a 768-dimensional space, they cannot be included in that search — cosine similarity
between a 384-dim and a 768-dim vector is mathematically undefined.

**In practice, this means:**

- Within the snap-in: full search quality at 768 dims. A therapist snap-in searching its
  own session notes, a legal snap-in searching its own case files — works perfectly.
- Cross-namespace recall: the main Zynkbot memory search will not surface snap-in memories
  in its results unless a bridge strategy is used (described below).

For most professional snap-ins this is actually fine. A therapist snap-in's patient session
notes should not appear in general Zynkbot conversation recall — that's the right behavior
for privacy and focus. The snap-in is a separate workspace.

Where it matters is if you want the user's snap-in knowledge to influence their general
Zynkbot conversations. A research snap-in that has indexed hundreds of papers might want
those to be reachable from general queries, not just from within the snap-in itself.

---

## Workarounds and Solutions

### Option 1: Accept the isolation (simplest, usually correct)

Most professional snap-ins benefit from strict isolation. Don't try to bridge the spaces.
The snap-in has its own search, its own UI, and its own context. The user explicitly opens
it when they need it. This is the default behavior and requires no extra work.

### Option 2: Dual embedding — bridge to the core space (moderate complexity)

When indexing content into a snap-in's knowledge base, generate two embeddings:

1. A 768-dimensional embedding stored in the snap-in's own table (for high-quality
   within-snap-in search)
2. A 384-dimensional embedding stored in the core memories table (for cross-namespace
   search from general Zynkbot conversation)

The 384-dim version is lower quality but still useful for surfacing snap-in content in
general recall. The snap-in then handles the full-quality search itself when opened.

This is straightforward to implement in the `snapin_kb_index_document` API call — the
platform handles generating both embeddings automatically when a snap-in declares both
a domain model and cross-namespace indexing in its manifest.

### Option 3: Matryoshka Representation Learning (MRL) models (elegant, future-facing)

MRL is a training technique where the first N dimensions of a larger embedding are
themselves a valid, useful embedding. OpenAI's `text-embedding-3-small` and several
open models support this.

A snap-in using an MRL-capable model at 768 dims can:
- Store the full 768-dimensional vector for within-snap-in search
- Truncate to the first 384 dimensions for cross-namespace search

One model, one embedding pass, compatible with both spaces. This requires the platform
to support MRL-aware truncation at index time, which is a straightforward addition to
the embedding pipeline.

This is the cleanest long-term approach and worth targeting when the snap-in platform
matures.

---

## Manifest Declaration (Proposed)

A snap-in that wants to use a domain-specific embedding model declares it in
`snapin.toml`:

```toml
[snapin]
name = "Legal Research Assistant"
id = "com.example.legal_research"
version = "1.0.0"

[embeddings]
model = "BAAI/bge-large-en-v1.5"   # 1024-dimensional domain model
dimensions = 1024
cross_namespace = false             # Don't bridge to core memory space
# cross_namespace = "dual"         # Generate both 1024-dim and 384-dim embeddings
# cross_namespace = "mrl"          # Use MRL truncation to 384 dims
```

If the `[embeddings]` section is omitted, the snap-in uses the default 384-dimensional
core model. No action required.

---

## Model Reference

Models suitable for snap-in use, in order of increasing quality and resource cost:

| Model | Dims | CPU speed | RAM | Best for |
|-------|------|-----------|-----|----------|
| all-MiniLM-L6-v2 (default) | 384 | ~15ms | 90MB | General personal memories |
| all-MiniLM-L12-v2 | 384 | ~25ms | 120MB | Marginal improvement, same dims |
| all-mpnet-base-v2 | 768 | ~80ms | 420MB | General text, better quality |
| BAAI/bge-large-en-v1.5 | 1024 | ~200ms | 1.3GB | High-quality general + domain |
| medical-text-embedding (various) | 768 | varies | varies | Healthcare snap-ins |
| legal-bert embeddings (various) | 768 | varies | varies | Legal snap-ins |

All of the above run locally. No API call required. GPU acceleration is supported where
available and falls back to CPU automatically.

---

## Summary for Platform Implementors

When implementing the snap-in platform (see `SNAPIN_PLATFORM_DESIGN.md`):

1. The `snapin_kb_index_document` command should read the snap-in's declared embedding
   model from its manifest and use that model for indexing, falling back to the default
   if none is declared.

2. Each snap-in's knowledge base tables should be created with the correct vector
   dimension for that snap-in's declared model.

3. The `snapin_kb_search` command searches only within the snap-in's own dimensional
   space — it never queries the core 384-dim memory table directly.

4. Cross-namespace bridging (dual embedding or MRL truncation) is an optional
   enhancement, not a requirement for the initial platform implementation.

5. Loading multiple embedding models simultaneously has a RAM cost. The platform should
   load a snap-in's model on-demand when the snap-in is opened and unload it when closed,
   rather than holding all models in memory at once.

---

## Open Questions

- Should the platform enforce a maximum dimension to prevent unreasonable memory usage
  on constrained hardware?
- Should MRL truncation be automatic for any MRL-capable model, or require explicit
  opt-in in the manifest?
- For the dual-embedding approach, should the 384-dim bridge embedding be generated by
  the snap-in's model (truncated) or by re-encoding with the default model?
