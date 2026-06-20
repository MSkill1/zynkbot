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
