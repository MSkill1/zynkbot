# Zynkbot Memory System

## Overview

Zynkbot's memory system is the core of its intelligence. Rather than processing each conversation in isolation, Zynkbot builds a persistent, interconnected memory graph that grows over time. Memories are extracted from conversations, stored with semantic embeddings, and retrieved by relevance when you send a new message.

## Memory Types

### Personal Memories
- Facts, experiences, preferences, and context extracted from your conversations
- Stored in the default `personal` namespace unless categorized otherwise
- Fully editable and deletable
- Retrieved via hybrid search during conversations

### Namespaced Memories
- Memories can be organized by context: work, health, travel, relationships, etc.
- Namespace is assigned automatically based on content or set manually
- Searches can be filtered to specific namespaces

### Ephemeral Memories (HIPAA Mode)
- Auto-expire after a configured duration (default: 8 hours in HIPAA mode)
- Used for sensitive information that should not persist
- Marked with expiration timestamps and cleaned up automatically

### System Memories (hidden)
- Core identity and feature knowledge about Zynkbot itself
- Stored in the `_zynkbot` namespace
- Never shown in Memory Manager
- Cannot be edited or deleted by users
- Retrieved only when you ask about Zynkbot

## Memory Structure

Each memory contains:
- **ID**: Unique integer identifier
- **Title**: Brief summary (LLM-generated or manual)
- **Content**: The full memory text
- **Namespace**: Organizational category
- **Embedding**: 384-dimensional vector for semantic search
- **Entities**: Extracted names, places, organizations (JSONB)
- **Link Count**: Number of relationships to other memories
- **Sentiment**: Positive/negative/neutral label and score
- **Event Type / Event Date**: If the memory describes an event
- **Ephemeral Flag**: Whether the memory auto-expires
- **Timestamps**: Created and last updated (timezone-aware)

## How Memories Are Created

### The Memory Pipeline

Every conversation goes through a background pipeline after Zynkbot responds. The pipeline never blocks the conversation — you receive your reply immediately, and memory processing happens asynchronously.

**Step 1 — Heuristic Gate**: A fast rule checks if the message is worth processing at all. Very short messages, pure filler ("ok", "thanks", "lol"), and messages with no substantive words are discarded immediately without an LLM call.

**Step 2 — LLM Decision**: If the message passes the heuristic gate, the LLM evaluates whether the message contains something worth remembering — personal facts, plans, emotional states, relationship context. It also generates a memory title and classifies relationships with similar existing memories.

**Step 3 — Duplicate Check**: Before storing, the embedding is compared against existing memories. If cosine similarity exceeds 93% (or hybrid score exceeds 98%), the memory is discarded as a duplicate.

**Step 4 — Contradiction Check**: If the LLM detected a `contradicts` relationship, the memory is held and a conflict modal is shown to you. Nothing is stored until you resolve it.

**Step 5 — Storage**: If no duplicate or contradiction, the memory is stored with NLP enhancement (entity extraction, event detection, namespace classification).

## Hybrid Search

When retrieving memories for context, Zynkbot uses a hybrid search combining:

1. **Semantic Search**: In-process cosine similarity (Rust/Candle) — finds conceptually related memories even when exact words don't match
2. **Entity Matching**: Named entity extraction — boosts memories that share specific people, places, or organizations with the query
3. **SQLite fallback**: Standard SQL filtering for exact term matches and recency as secondary signals
4. **Recency Boost**: More recent memories are weighted slightly higher

## Memory Relationships

The LLM classifies relationships between memories automatically:

- **contradicts**: Two memories that directly conflict
- **supports**: One memory reinforces another
- **elaborates**: One memory adds detail to another
- **reminds_of**: Thematically related but not directly connected
- **caused_by**: A causal relationship between two memories
- **none**: No meaningful relationship

Relationships are stored with confidence scores (0–1) and can be visualized in the Relationship Graph.

## Contradiction Resolution

When a contradiction is detected, a modal appears showing both the existing memory and the new conflicting information. You choose:

- **Keep Old** — discard the new information
- **Keep New** — replace the old memory with the new one
- **Keep Both** — store both with a `contradicts` link (and optionally an explanation)

Nothing is stored until you make a choice. This prevents contradictory data from silently accumulating.

## Privacy Controls

- View all memories in Memory Manager
- Edit any memory's content or title
- Delete individual memories
- Clear all memories (full reset) via Settings
- Back up all memories by copying the SQLite database file: `cp ~/.local/share/zynkbot/zynkbot.db ~/zynkbot_backup.db` (Linux) or copying `%LOCALAPPDATA%\zynkbot\zynkbot.db` on Windows
- Filter memories by namespace, date range, or search term
