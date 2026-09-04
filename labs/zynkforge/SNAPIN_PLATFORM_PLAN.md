# Snap-in Platform — Build Plan

**Status:** Labs plan, 2026-09-04. Not scheduled.
**Builds on:** `labs/snapin_platform/SNAPIN_PLATFORM_DESIGN.md` (July 2026), which
defines the manifest, the `snapin_*` command convention, permissions, the loader, and the
per-snap-in window. This document turns that design into a build sequence and extends it
with what an agent snap-in (`ZYNKFORGE.md`) needs. Where the two disagree, this one is
newer; the July document is the reference for the parts it covers.

---

## Where things stand (verified in the tree, 2026-09-04)

| Piece | State |
|---|---|
| Therapist demo | Works, but baked into the core (`index_snapin_notes` in `commands/knowledge_base.rs`; notes indexed under `snap_ins/therapist/<patient>/`). |
| Manifest (`snapin.toml`), loader, install-from-zip | Designed, not built. |
| `snapin_*` command surface | Designed, not built. |
| Permission enforcement (memory namespaces, KB paths, network) | Designed, not built. |
| Per-snap-in Tauri window | Designed, not built. Open question below for Android. |
| Model tool-use loop usable by a snap-in | Not designed. Needed by ZynkForge. |
| Scoped filesystem and command capabilities | Not designed. Needed by ZynkForge. |
| Developer template and documentation | Not written. |
| Scheduling | `docs/SDK_VISION.md` says v3.0 (2027) for the SDK; the July design says "v1.0 target"; the platform is not on the v1.0 checklist in `docs/ROADMAP.md`. Needs one decision. |

Recommendation on scheduling: after v1.0. The v1.0 list still holds the ZynkSync outbox,
Windows offline dictation, the ZynkLink certificate exchange, the licensing transition, and
the pre-submission audit. The platform touches the backend's trust boundary and deserves
its own release.

---

## Principles carried through every phase

1. **The core owns every gate.** Confirmation dialogs, diff display, trust levels, and
   budgets are implemented in Zynkbot's own UI and Rust, never in a snap-in's JavaScript.
2. **Capabilities are declared, granted, and checked per call.** A manifest declares what
   a snap-in wants; the user grants it at install; every `snapin_*` command checks the
   grant on every call, with the calling snap-in's identity supplied by the core, not by
   the caller.
3. **Snap-ins never see keys or raw backends.** Model access goes through the core, which
   holds the API keys and the backend selection.
4. **Same API for first-party and third-party.** ZynkForge, the therapist demo, and an
   outside developer's snap-in all use the same commands.

---

## Phase 1 — Platform core (from the July design)

Goal: a developer can build and install a snap-in without touching the core.

1. **Manifest and loader.** Parse `snapin.toml`; scan `snap_ins/` at startup; register
   valid snap-ins; expose list / load / unload / remove commands to the main UI.
2. **Identity plumbing.** Each snap-in window is created by the core with a label the
   core controls; every `snapin_*` command receives that label from the window handle and
   looks up the manifest. This is what makes per-call permission checks trustworthy.
3. **`snapin_*` command surface, read side first.** `snapin_kb_search`,
   `snapin_memory_search`, `snapin_get_context` with namespace and path scoping enforced.
   Then the write side: `snapin_kb_index_document`, `snapin_memory_store`.
4. **Permission enforcement.** Memory namespaces, knowledge-base paths, network (off by
   default), conversation history (off by default). Denied calls return a typed error the
   snap-in can show.
5. **Install from zip** with manifest validation and a permissions summary the user
   approves before anything is unpacked into `snap_ins/`.
6. **Therapist demo refactored** onto the platform: its own `index.html` and manifest,
   backend logic moved behind `snapin_*` commands, hard-coded UI removed from the main app.
   It becomes the reference implementation.
7. **Template and documentation.** A minimal working snap-in, the command reference, and a
   "build your first snap-in" guide.

Tests: manifest parsing (valid, missing fields, bad ids), a permission matrix test per
command (declared vs undeclared namespace and path), install of a malformed zip.

## Phase 2 — Model access for snap-ins

Goal: a snap-in can run a conversation, including tool use, through the user's configured
backend, without holding keys.

1. **`snapin_chat`**: a streaming chat call through the core's existing LLM layer
   (`llm/`, `generate_reply`), scoped by the snap-in's permissions and the active
   containment mode. Streams tokens to the snap-in's window as events.
2. **Tool definitions and the loop.** The snap-in passes tool definitions (name,
   description, JSON schema) with the call. The **core runs the loop**: it sends the
   definitions to the model, receives a tool call, emits it to the snap-in as an event,
   waits for the snap-in to return the result, and continues, up to a step budget the
   core enforces. The snap-in only executes tools; it never talks to the model directly.
   Reasons: keys stay in the core; budgets and cancellation are enforced in one place;
   every backend that supports tool use is handled once.
3. **Budgets.** Maximum steps, maximum wall-clock time, and cancellation, all core-owned
   and visible in the main UI while a snap-in loop runs.
4. **Audit log.** Structured JSON lines per session under `snap_ins/<id>/data/logs/`:
   each tool call, arguments, result size, error. Written by the core.

Tests: a fake tool round-trip with a recorded model response; budget exhaustion; cancel
mid-loop; a backend without tool support returns a typed error.

## Phase 3 — Filesystem, command, and git capabilities

Goal: the capabilities ZynkForge needs, built as platform features with core-owned gates.

1. **`filesystem` capability.** Declared in the manifest; granted by the user choosing a
   root directory through the native folder picker at grant time (never a path typed by
   the snap-in). Commands: `snapin_fs_list`, `snapin_fs_read`, `snapin_fs_write`,
   `snapin_fs_patch`. The core canonicalises every path against the root on every call,
   refuses symlinks that resolve outside it, and redacts credential-shaped files (`.env`,
   key and certificate files, anything the user adds) from both results and model context.
   Writes go through a diff preview and confirmation in the core UI, with a snapshot
   taken first (see git below).
2. **`command` capability.** The manifest declares an allow-list; the user sees it at
   grant time. `snapin_run_command` executes only allow-listed commands, in the root, with
   a timeout, an output cap, no terminal, and network off unless declared. Anything not on
   the list goes through a core confirmation dialog that shows the exact command. Decision
   to record before this ships: OS-level sandbox on Linux (bubblewrap or a container) or
   documented acceptance of the risk. Not available on Android; the capability is
   desktop-only until that changes.
3. **`git` capability.** Read commands (`status`, `diff`, `log`) are ungated. Snapshot
   before every write: an `add -A` style snapshot committed to a platform-owned reference
   (`refs/zynkbot/snapshots/<snapin>/<session>`), never to the user's branch. `snapin_git_commit`
   is gated with the staged diff shown. No push, force-push, or branch deletion in the
   first version. "Undo last change" restores from the snapshot reference.
4. **Trust levels in the UI.** Reads silent; writes gated with the diff visible in the
   dialog; commands gated with the command visible. One place, one component.

Tests: path-escape suite (`..`, absolute paths, symlink out of root, case tricks on
case-insensitive filesystems), credential redaction, allow-list bypass attempts
(`sh -c`, `env`, shell metacharacters), snapshot-and-restore including an untracked file.

## Phase 4 — ZynkForge

Build it on Phases 1 to 3 following the sequencing in `ZYNKFORGE.md`: throwaway clone and
adversarial tests first, read-only tools and plan mode, then write with snapshot and
confirmation, then the command runner, then confirmed commit, then live-repository
promotion. First target audience: desktop users with a cloud backend; local-model support
after the measurement in `ZYNKFORGE.md` note 7.

## Later (out of scope, as in the July design)

Marketplace and discovery, automatic updates, revenue sharing, inter-snap-in
communication, background snap-ins.

---

## Open questions

Carried from the July design:
- Registered Tauri commands per snap-in vs the fixed `snapin_*` API. This plan assumes the
  fixed API; it is the only version the core can gate.
- Export before removal.
- Permission diff on update.

New:
- **Android windows.** The July design opens each snap-in in its own Tauri window. Whether
  Tauri v2 supports more than one window on Android needs to be verified before Phase 1;
  if not, snap-ins on Android render inside the main WebView (an isolated frame) and the
  identity plumbing in Phase 1 step 2 must work for frames as well as windows.
- **Containment modes.** Which snap-in capabilities are unavailable in Child, HIPAA, and
  Elder modes. The July design says snap-ins must respect mode constraints; the table of
  what that means per capability does not exist yet.
- **Effort.** Rough guesses, to be replaced by estimates once Phase 1 is scoped in detail:
  Phase 1 about two to three weeks, Phase 2 about one to two weeks, Phase 3 about two to
  three weeks plus the adversarial test suite, Phase 4 per the ZynkForge sequencing.
