# Standing Instructions: an always-loaded memory tier

**Status:** Speculative design note. Not scheduled. Do not implement before v1.0.
**Date:** September 2026
**Origin:** Comparison between Zynkbot's memory graph and the flat, hand-curated memory
an AI coding assistant keeps about a project it works on.

---

## The observation

Zynkbot's memory system and the memory an agent keeps about *how to work with someone*
turn out to be solving different problems, and Zynkbot only implements one of them.

| | Zynkbot's memory graph | Standing instructions |
|---|---|---|
| Holds | facts about the user's life | how to behave toward this user |
| Arrives by | LLM extraction from conversation | the user writing it deliberately |
| Recalled by | hybrid search, per query | unconditionally present in every prompt |
| Curated by | the extraction pipeline | the user, directly |
| Typical content | "I went to Penn State" | "never suggest alcohol to me" |
| Failure mode | kitchen-sink summaries, duplicates (KI-016, KI-017) | going stale |

Everything in Zynkbot's memory competes for retrieval. That is correct behaviour for
biographical facts — there is no reason to spend context on someone's university unless
the conversation is about university. It is the wrong behaviour for a standing rule.

**A standing instruction that has to be retrieved is a standing instruction that can be
missed.** If a user says "never bring up my divorce unless I raise it first," that must
hold on every turn, including the turns where nothing in the message is semantically
close to divorce. Similarity search cannot guarantee that, and the cases where it fails
are exactly the cases where failing is worst.

---

## Why this is a real gap rather than a tidy abstraction

Two pieces of evidence from the project itself.

**1. Containment modes are already standing instructions — just hardcoded ones.**

Guardian, Child, HIPAA and Sovereign modes work by injecting fixed rules into every
prompt. That is precisely the mechanism described here, with two limitations: the rule
sets are authored by us rather than the user, and they are coarse (a whole mode, not a
single preference). A user cannot express "don't discuss my health with the tone you
use for everything else" without us shipping a mode for it.

This matters most for the modes that are not yet built. Parenting Mode and Elder Mode
are, in substance, *standing-instruction products*. Parenting Mode is a parent writing
rules about their child's use. Elder Mode is a family and an elder agreeing on how
memory support should be delivered. If the underlying mechanism is a hardcoded mode per
use case, every new use case is an engineering project. If the mechanism is
user-authored standing instructions, the modes become presets over a general facility.

**2. The extraction pipeline is imperfect, and standing instructions do not need it.**

KI-016 (kitchen-sink summaries) and KI-017 (duplicated already-stored facts) are
inherent to inferring structured memories from free conversation. A hand-written
instruction has no extraction step, so it has no extraction defects. It is also
inspectable: the user can read exactly what the model is being told about them, which
is the transparency claim the whole product rests on.

---

## What NOT to build

**Do not add a second store.** The temptation is a separate file- or table-backed
system alongside the memory graph. Reject it:

- Two stores means two things to sync across Windows, Linux and Android. ZynkSync is
  being rebuilt around an outbox precisely because per-feature sync wiring did not
  scale. Adding a second store adds a second thing to wire.
- Two stores means two places for a user to look when Zynkbot behaves unexpectedly.
- Markdown files on disk work for a developer with a text editor. They are not an
  interface for a companion-app user.

The flat-file design that prompted this note works because its consumer is an agent with
filesystem access and its author is a developer. Neither is true here. Borrow the
*distinction*, not the storage.

---

## Suggested shape

A retrieval class, not a new subsystem.

**Storage:** reuse `memories`. Standing instructions are memories in a dedicated
namespace (the `namespace` column already exists and already defaults to `personal`).
No schema change beyond whatever flag distinguishes them, and possibly not even that if
a namespace value suffices.

**Retrieval:** the change lives in prompt assembly, not in search. Memories in the
standing-instruction namespace bypass similarity ranking and are injected into every
prompt, in a distinct section from recalled memories, so the model can tell "this is a
rule" from "this is something I know about you."

**Curation:** a dedicated view in Memory Manager. Write, edit, reorder, delete, and
crucially *see* them — a plain list of every rule currently in force. No LLM extraction
path; these are only ever written by the user.

**Budget:** they consume context on every single request, so they need a cap. A
character or token ceiling with a visible indicator, and an ordering so that if the cap
is hit the user knows which rules survive. This is the main design risk: unbounded
always-on content silently degrades every response, and on a 4K-context local GGUF
model it would be ruinous. The slim prompt variant would likely need a tighter cap than
the API-model prompt.

**Interaction with containment modes:** modes should compose with standing instructions,
not compete. A mode supplies a rule set we author; standing instructions supply rules
the user authors. Where they conflict, the mode wins — a Child-mode restriction must not
be removable by a user-written instruction, or the mode is not a safety feature. That
precedence rule needs to be explicit before any of this is built, and it is the reason
this cannot be bolted on later without thought.

**Sync:** falls out of the outbox for free, since they are rows in `memories`. That is
the main argument for waiting until the outbox exists rather than building this first.

---

## Open questions

- Should standing instructions be conversationally editable ("stop asking me about
  work"), or strictly UI-authored? Conversational editing reintroduces the extraction
  problem and the risk of a passing remark becoming a permanent rule. Probably UI-only
  at first, with an explicit "make this a standing rule?" confirmation if it is ever
  offered from chat.
- Do they belong to the user or the device? A rule about tone is probably the user's; a
  rule about notifications may be the device's.
- How do they interact with Ensemble mode, where several models answer? Injecting into
  every model's prompt is the obvious answer but multiplies the context cost by the
  number of models.
- Should there be starter defaults, or is an empty list the honest starting state?

---

## Why not now

Twenty-seven items remain open for v1.0, and this is a new facility with a UI, a context
budget, and a precedence interaction with the safety-relevant containment modes. It also
wants the outbox to exist first so that sync is free rather than another hand-wired
path.

The order that makes sense: finish v1.0, land the outbox, then build this as the
foundation Parenting Mode is authored on top of — rather than building Parenting Mode as
another hardcoded mode and retrofitting this underneath it afterwards.
