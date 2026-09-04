# ZynkForge — Coding-Agent Snap-in for Zynkbot

**Status:** Labs proposal. Not scheduled. Recommended for after v1.0 (see "Where it sits").
**Origin:** Matt's draft (worked out with other AI assistants), reviewed by Claude Code on
2026-09-04. The draft's decisions are kept as written; the review notes are marked as such.
**Depends on:** the snap-in platform, which does not exist yet — see
`SNAPIN_PLATFORM_PLAN.md` in this directory and `labs/snapin_platform/SNAPIN_PLATFORM_DESIGN.md`.

---

## Concept

A snap-in that gives Zynkbot Claude-Code-like capability: read files, propose and apply
edits, run commands, and interact with git, scoped to a chosen project directory. It is
the flagship first-party snap-in: it demonstrates the platform to other developers, and it
is usable by the developer himself (dogfooding the tool on its own codebase, and eventually
on its own source).

## Firm decisions (from the draft)

**Build on the real public snap-in API.** Do not special-case ZynkForge internally.
Building on the same API external developers would use is itself the dogfooding: it is how
platform friction (confirmation gates, diff preview, scoped filesystem access) gets found
and fixed. If the API cannot cleanly express something ZynkForge needs, that is a platform
gap to close, not a reason to bypass the API.

**Core mechanism.**
- Tool definitions for the model: `read_file`, `write_file`, `list_directory`,
  `run_command`, `git_status`, `git_diff`, `git_log`, `git_commit`.
- Standard agentic loop: model requests a tool call, ZynkForge executes it, the result
  goes back to the model, repeat until the task is complete.
- Scoped to a single designated project directory.
- A read-only / plan mode: the agent can read, analyse, and propose a change plan without
  executing anything. Useful for running against a live repository before write access
  is trusted.
- "Task complete" is defined explicitly, with a hard maximum step count as a backstop.

**Command execution: allow-list, not deny-list.** Deny-lists are incomplete by
construction (`curl | bash`, `git clean -fdx`, a one-line Python delete all bypass a naive
deny-list). Instead:
- Allow-list a small set of known-safe commands: specific git subcommands, the project's
  build and test runner, a linter, the package manager. Everything else requires explicit
  user approval before running.
- Hard timeout and capped output size on every command. No interactive terminal.
- Network access disabled by default per project.
- Reads of `.env` and credential-shaped filenames inside the project root are blocked or
  redacted unless explicitly opted in. Agents tend to helpfully read and echo secrets into
  logs and output.

**Path scoping, as implementation decisions.**
- Resolve and validate every path against the project root on every tool call, not once
  at configuration time.
- Symlinks that point outside the project root are not followed.
- `run_command`'s internal behaviour (`cd`, subprocess spawning) is invisible to path
  checks. That is the strongest argument for the allow-list: restrict which commands are
  reachable at all, since what an allowed command does internally cannot be fully
  sandboxed.

**Undo and safety: git-based, not a custom backup system.** The existing Cloudflare R2
backup stays exactly as it is, scoped to memory data.
- The local snapshot / auto-commit ships with the first write capability, as one unit.
  There is no window, even in early development, where the agent can modify files with no
  recovery path.
- Diff preview, confirmation gate, and local snapshot ship together for the write path.
- `git_commit` requires confirmation and a shown staged diff. No push, force-push, or
  branch deletion in v1 defaults. Remote operations come later, behind an extra gate.
- A simple "undo last agent change" action (`git revert HEAD` underneath) as a convenience;
  full history stays available through normal git tooling.

**Structured session audit log.** For diagnosing ZynkForge itself when it breaks, not for
undoing edits to a target repository. Every tool call is logged: tool, arguments, result,
error, in a structured form. Recovery when ZynkForge is broken: stop the session, take the
log, open a separate healthy session (plain Claude Code, or a working ZynkForge) pointed at
ZynkForge's own source, and let it diagnose from outside. Keep the core loop simple;
logging is the safety net, not a substitute for the loop being solid.

**Confirmation UX.** Gates only work if users read them. Show diffs in the confirmation
itself. Use trust levels: reads (`read_file`, `git_status`) are not gated like writes;
writes and commands are gated hard.

**Hard loop limits.** Maximum tool calls per task, maximum files touched, maximum bytes
written, wall-clock timeout.

**Build sequencing (adversarial first).**
1. Adversarial testing on a throwaway clone before anything else: path escapes, symlink
   escapes, command-injection strings, sensitive filenames, mid-loop cancellation.
   Sandbox-vs-live is an explicit mode requiring deliberate promotion, not developer memory.
2. Read-only tools first (`list_directory`, `read_file`, `git_status`, `git_diff`,
   `git_log`), with plan mode. Already useful, zero destructive risk.
3. Write capability, diff preview, confirmation gate, and local snapshot, shipped as one
   unit.
4. Constrained (allow-listed) command runner.
5. `git_commit` only, confirmed, diff shown. No push in v1 defaults.
6. Only after deliberately trying to break the sandbox on a throwaway clone: graduate to
   the live repository.
7. Ship as the free, bundled first-party snap-in, built on the public snap-in API
   throughout.

---

## Review notes (Claude Code, 2026-09-04)

The safety design above is sound and matches how Claude Code itself is built: allow-lists,
per-call path checks, plan mode, hard budgets, adversarial testing first. The notes below
are about the premises and about three gaps.

### 1. The API the draft builds on does not exist yet

Verified in the tree on 2026-09-04:
- `docs/SDK_VISION.md` marks the SDK "Planned for v3.0 (2027)".
- `labs/snapin_platform/SNAPIN_PLATFORM_DESIGN.md` (July 2026) defines the manifest
  (`snapin.toml`), the `snapin_*` command convention, permissions, a loader, and a separate
  Tauri window per snap-in. None of it is implemented.
- The only snap-in code is the therapist demo's notes indexer
  (`index_snapin_notes` in `commands/knowledge_base.rs`), baked into the core.

"Build on the real public API" therefore means building the platform first. That is the
larger project, and ZynkForge also needs capabilities the July design does not have
(filesystem scope outside `snap_ins/`, command execution, a model tool-use loop). The plan
for closing that gap is `SNAPIN_PLATFORM_PLAN.md`.

### 2. It will not speed up v1.0

The draft's dogfooding argument assumes ZynkForge would be useful on Zynkbot's own code
while v1.0 is being finished. The quality of a coding agent is mostly the model driving it.
Zynkbot runs on the user's cloud API key or a local model through Ollama; the local models
people choose for privacy handle multi-step tool use poorly, and even the cloud path would
be a new, untested harness. Building v1.0 with ZynkForge means debugging the tool instead
of the product. Recommendation: keep building Zynkbot with the tool that works, and treat
ZynkForge as a post-v1.0 product whose real audience is people who want a private helper
for small edits on their own machine.

### 3. Git-as-undo needs two refinements

- The snapshot must include new untracked files (a `git add -A` style snapshot, honouring
  `.gitignore`), or "undo" cannot restore a file the agent created and then deleted.
- Snapshot commits on the user's own branch pollute their history. Keep them on a separate
  reference (the way an editor keeps its own local history) and leave the user's branch
  alone. `git revert HEAD` as the undo mechanism only works if the agent's change *is*
  HEAD, so define undo against the snapshot ref, not the branch.

### 4. The confirmation gate must live in the core, not in the loop

The model must never be the party deciding whether an action is low-risk. Trust levels,
diff display, and approval must be implemented by the platform's UI, outside the snap-in's
JavaScript and outside the model's reach. A snap-in declares what it wants to do; the core
shows the user and waits. This is a platform requirement, not a ZynkForge feature.

### 5. Allow-listed commands still run the repository's code

`npm test`, `make`, `cargo build` all execute code from the project, including hooks and
build scripts. The allow-list bounds *which* commands run, not what they do. Two options,
to be decided before the command runner ships: run commands inside an OS-level sandbox on
Linux (bubblewrap or a container; not available on Android), or accept the risk, say so in
the confirmation, and rely on the throwaway-clone policy. The draft leans toward the
second; it should be a written decision either way.

### 6. Secrets and cloud models

Redacting credential-shaped files from tool output protects the log. It must also protect
the model context: when the configured backend is a cloud API, no content of such a file
may be sent to it, redacted or not. The platform's filesystem capability should enforce
this rather than each snap-in.

### 7. Which models can drive it

Before designing the loop, measure. Run the same ten small tasks through each backend
Zynkbot supports and record completion rate and wasted steps. If only the strongest cloud
models complete them, ZynkForge is a cloud-only feature at first, and the product page
should say so.

## Where it sits

Not on the v1.0 checklist (`docs/ROADMAP.md`, "Remaining for v1.0"). Order of work:
finish v1.0, then the snap-in platform (`SNAPIN_PLATFORM_PLAN.md`, phases 1 to 3), then
ZynkForge following the draft's sequencing above.
