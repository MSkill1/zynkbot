# Snap-in Platform: Architecture & Implementation Plan

**Status:** Design Phase (v1.0 Target)
**Purpose:** Define the scaffolding needed to turn the current proof-of-concept snap-in demo into a real third-party developer platform

---

## Background

The current therapist snap-in (v0.9) demonstrates that the concept works: domain-specific workspaces can organize data using the existing RAG/knowledge base infrastructure, maintain privacy isolation via namespaced file paths, and surface a custom UI to the user. However, the current implementation is baked directly into the main codebase — the backend logic lives in lib.rs, and the frontend is rendered inside the main app window. A third-party developer cannot build a snap-in without modifying core files.

The goal for v1.0 is a platform where a developer can build and install a snap-in without touching the core codebase at all.

---

## Proposed Architecture

### User Interface

The only change to the main Zynkbot UI is a **Snap-ins button** in the settings panel (already present in v0.9). Clicking it shows a list of installed snap-ins with load/unload controls. No other changes to the existing interface.

When a user launches a snap-in, it opens in a **separate Tauri window**. This window is:
- Fully isolated from the main chat UI
- Owned by the snap-in (its own HTML/CSS/JS)
- Able to invoke backend commands via the standard Tauri `invoke()` interface
- Closeable without affecting the main app state

This approach keeps the main interface clean and makes it architecturally clear that snap-ins are a distinct layer.

### Snap-in Structure

Each snap-in is a self-contained directory installed under `snap_ins/`:

```
snap_ins/
  my_snapin/
    snapin.toml        # Manifest — required
    index.html         # Entry point UI — required
    assets/            # Optional: CSS, JS, images
    data/              # Optional: snap-in's private data store
```

### Manifest Format (`snapin.toml`)

```toml
[snapin]
name = "My Snap-in"
id = "com.developer.my_snapin"     # Reverse-domain unique ID
version = "1.0.0"
author = "Developer Name"
description = "What this snap-in does"
entry = "index.html"

[permissions]
memory_namespaces = ["my_snapin"]  # Which memory namespaces it can read/write
knowledge_base_paths = ["snap_ins/my_snapin/"]  # Which KB paths it can access
network = false                    # Whether it can make external network calls
```

### The Snap-in API

Snap-ins call the Tauri backend via `invoke()`, the same mechanism the main app uses. To create a clean public contract without exposing all of lib.rs, all snap-in-accessible backend commands follow a naming convention:

```
snapin_*
```

For example:
- `snapin_kb_index_document` — index a document into the snap-in's knowledge base namespace
- `snapin_kb_search` — RAG search within the snap-in's permitted paths
- `snapin_memory_store` — store a memory tagged to the snap-in's namespace
- `snapin_memory_search` — search memories within the snap-in's namespace
- `snapin_get_context` — retrieve current user context (subject to permissions)

These are wrappers around existing lib.rs functionality, scoped to what the manifest declares. The developer documentation lists these commands as the stable public interface. Everything else in lib.rs is internal and not guaranteed to be stable for snap-in use.

---

## Loader & Installation

### Detection

On startup, Zynkbot scans the `snap_ins/` directory for valid `snapin.toml` files and registers available snap-ins. No manual registration step required — drop the folder in and it appears.

### Installation UX (v1.0)

Simple flow:
1. User downloads a snap-in as a `.zip` file
2. Opens the Snap-ins panel → "Install Snap-in" button
3. Selects the zip — app unpacks it into `snap_ins/` and validates the manifest
4. Snap-in appears in the installed list immediately

This is intentionally minimal. A marketplace/discovery layer is a later milestone.

### Load / Unload

- **Load**: Opens the snap-in in a new Tauri window, reads the manifest, initializes its data directory
- **Unload**: Closes the window, frees resources. Data persists in `snap_ins/<id>/data/`
- **Remove**: Unloads and deletes the directory (with confirmation)

---

## What a complete snap-in contains (added 2026-10-29)

The therapist demo shows one part of a snap-in: a workspace that files notes and searches
them. A complete snap-in is more than a folder. It bundles everything a domain
needs, and it is the natural home for a conversation mode: a "legal mode" on its own is a
paragraph of text anyone can copy, but a legal mode that comes with case files, a deadline
tracker and citation checks is a product. Snap-ins are installed and switched on from the
Snap-ins page in System Controls (it exists today, waiting for snap-ins); the Mode menu
shows a snap-in's mode only while that snap-in is on. Pricing is the kits note's business
(`labs/kits/KITS_DESIGN.md`); the `[license]` block below only leaves room for it.

Seven parts. A snap-in declares the ones it uses; the manifest below is the contract.

| part | what it is | exists today? |
|---|---|---|
| **workspace** | its own page (`index.html`): forms, lists, timelines | yes (therapist demo, hard-coded) |
| **knowledge base folder** | documents it indexes and searches, scoped to its path | yes (`_system` style folder, RAG) |
| **memory namespace** | memories it may read and write, nothing outside | yes (namespaces exist; scoping does not) |
| **mode** | a prompt block and rules that shape every reply while the snap-in is active, listed in the Mode menu like Tool or Debate | no (modes are in `chat_modes.rs`, fixed list) |
| **tools** | actions the model may call or the page offers: "add to timeline", "draft a letter from this template", "check this citation", "export the file" | no |
| **check-in questions** | the one-question-a-day pattern from `labs/check_in`, with the snap-in's own rules on what to ask | no |
| **policy** | containment interplay (a therapy snap-in forces HIPAA), consent text shown at install, and the licence | partly (containment modes exist) |

### Manifest, extended

```toml
[snapin]
id = "ai.containai.legal"
name = "Legal case files"
version = "0.1.0"
entry = "index.html"

[permissions]
memory_namespaces = ["legal"]
knowledge_base_paths = ["snap_ins/legal/"]
network = false

[mode]                                  # shown in the Mode menu while the snap-in is installed
key = "legal"
name = "Legal"
blurb = "Precise, cites sources, flags deadlines; never gives advice as a lawyer would."
prompt = "mode.md"                      # the block put in front of the prompt, same mechanism as chat_modes.rs
forces_containment = "guardian"         # optional: therapy would say "hipaa"

[[tools]]                               # each tool is a snapin_* command plus a line the model sees
id = "timeline_add"
label = "Add to the case timeline"
model_visible = true                    # the model may ask for it, the user confirms (like web search today)

[[tools]]
id = "deadline_list"
label = "Deadlines in the next 30 days"
model_visible = true

[checkins]                              # optional; off unless the user turns the snap-in's questions on
patterns = ["file", "serve", "call the clerk"]
max_per_day = 1

[license]
kind = "one_time"                       # or "free"
unlock = "receipt_file"                 # a signed receipt placed next to the manifest; no server call
```

### Worked example: therapy / journal snap-in

- **Workspace:** today's demo (clients, dated session notes) plus a session shape: opening
  check-in, the body, a closing summary the user approves.
- **Mode:** "Journal". Asks one open question per reply, reflects back without diagnosing,
  never suggests dependency on itself; the same restraint rules as Normal. This is where
  the "interview mode" idea from 2026-10-29 lives; it is not a general mode.
- **Tools:** "summarise this month's sessions", "find the first mention of X", export.
- **Check-ins:** only about things the user set themselves in a session ("I said I would
  call my sister"), never about mood.
- **Policy:** forces HIPAA containment; install text says notes never leave the device.

### Worked example: legal snap-in

- **Workspace:** matters (cases), each with parties, documents, a timeline, deadlines.
- **Mode:** "Legal". Precise wording, cites the document and page it relies on, states
  plainly that it is not a lawyer and that deadlines must be verified against the court's
  rules; refuses to guess at law it has no source for in the knowledge base.
- **Tools:** add to timeline, list deadlines, draft from a template (letter, notice),
  citation check against the indexed documents, export a matter as a folder.
- **Check-ins:** "You said you would file the response by the 14th. Done?"
- **Policy:** Guardian containment at least; network off; one-time licence.

### Worked example: music player snap-in

- **Workspace:** what is playing now, the queue, and a list of the user's playlists, read
  from the music app on the device.
- **Mode:** "DJ". Short replies while music is on; talks about the music only when asked;
  never interrupts a track to comment.
- **Tools:** play, pause, skip, "play something like this", "add this to <playlist>",
  "what is this song" (from the player's own metadata, no network). On Android these are
  media-session commands to whichever player is open; on the desktop they would go to
  MPRIS on Linux (the standard desktop media interface) and the equivalent on Windows.
- **Check-ins:** none. Nothing about listening habits is a plan to ask after.
- **Memory namespace:** `music`, for likes and dislikes the user states out loud ("I love
  this one", "never play this again"); nothing is inferred from play counts.
- **Policy:** any containment; network off; the player's own account stays in the player.
  Not a music service: it controls what the user already has.

### What this means for the platform plan above

The loader and the `snapin_*` API stand. Two additions: `chat_modes::apply` reads
snap-in modes as well as the built-in list, and the Mode menu (drawn by the page since
2026-10-29) lists them with the snap-in's name as a group header. Tools reuse the web-search
confirmation path: the model writes a marker, the page asks the user, the command runs.

## Permissions & Isolation

This is the most important design decision and the one most worth getting right before writing code.

### The problem

Without a permissions model, a third-party snap-in has access to everything in the user's database — all memories, all knowledge base documents, all namespaces. For a privacy-first app, this is unacceptable. A snap-in for a lawyer should not be able to read the user's personal health memories.

### The solution (v1.0)

Permissions are declared in `snapin.toml` and enforced by the backend:

- **Memory access**: scoped to declared namespaces only. A snap-in that declares `memory_namespaces = ["legal"]` cannot read memories tagged `personal` or `health`.
- **Knowledge base access**: scoped to declared paths. A snap-in can only index/search within its own folder.
- **Network access**: off by default. Must be explicitly declared and will be shown to the user during install.
- **No access to conversation history** by default. A snap-in that needs conversation context must declare it and the user must approve.

The backend enforces these at the command level — the `snapin_*` commands check the calling snap-in's manifest before executing.

### Trust model

v1.0 snap-ins are installed manually by the user, who reads the manifest and approves permissions at install time. There is no automatic execution, no background processes. A snap-in only runs when the user explicitly opens it.

---

## Developer Experience

A developer building a snap-in needs:

1. **A template** — a working minimal snap-in (simpler than the therapist demo) that shows the structure, a working `invoke()` call, and the manifest format
2. **API documentation** — the full list of `snapin_*` commands with parameters and return types
3. **A guide** — "Build your first snap-in" walkthrough from blank directory to working window

This documentation does not exist yet and needs to be created alongside the platform code.

---

## What the Therapist Demo Needs

The existing therapist snap-in should be refactored to use this platform once it exists:
- Extract its backend logic into `snapin_*` commands
- Move its UI into its own `index.html` under `snap_ins/therapist/`
- Add a `snapin.toml` manifest
- Remove its hardcoded frontend from the main app

This makes it the reference implementation that developers can learn from.

---

## Implementation Order

1. Define and implement `snapin_*` command subset in lib.rs
2. Implement manifest parser and permission enforcement
3. Implement loader (startup scan + registration)
4. Add Snap-ins window to main UI (install, load, unload)
5. Implement zip install flow
6. Refactor therapist demo to use the new platform
7. Write developer documentation and template
8. (Later) Marketplace / discovery layer

---

## Out of Scope for v1.0

- Snap-in marketplace or discovery
- Automatic updates
- Revenue sharing infrastructure
- Inter-snap-in communication
- Snap-ins that run background processes

These are v3.0+ concerns per the roadmap.

---

## Open Questions

- Should snap-ins be able to register their own Tauri commands (more powerful, harder to sandbox) or only call the defined `snapin_*` API (safer, simpler)?
- What happens to a snap-in's data if the user removes it — offer export first?
- Should the permissions approval UI show a diff when a snap-in is updated and requests new permissions?
