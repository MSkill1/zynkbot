# AI Collaboration Case Study: Claude vs. GPT Codex on ZynkBot Networking Security

**Project:** ZynkBot — open-source, privacy-first AI assistant (Tauri + Rust + React)  
**Feature area:** ZynkSync peer-to-peer networking authentication  
**Date:** August 2026

---

## Background

ZynkBot devices communicate over a local network using a custom Rust TLS server (ZynkSync, port 57963). The existing security model used cert-pinning: each device pins its peer's self-signed certificate during pairing, so connections from unknown devices are rejected by the TLS client. However, the server side accepted any TLS client without verifying its identity — device identity was claimed purely via an `x-device-id` request header, which any process on the LAN could forge.

The question was: how should we harden server-side authentication?

---

## The Experiment

Two AI coding assistants were given the same codebase and asked to solve networking security issues independently:

- **GPT Codex** was given access to create a new branch (`networking-fixes`) and implement fixes
- **Claude** (Anthropic) was given the same codebase context and asked to design an approach

Neither AI was shown the other's work during the design phase.

---

## What GPT Codex Proposed

Codex created a `networking-fixes` branch and added:

1. A **shared mesh authentication token** stored in `app_settings` as `zynksync_auth_token`
2. A new migration adding `auth_token` to `zynklink_pairings`
3. Bearer token validation in request handlers

The intent was that all paired devices would share one secret token, and requests would be rejected if the token didn't match.

---

## What Claude Proposed

Claude proposed **mutual TLS (mTLS)** authentication, reusing the existing per-device self-signed certificate infrastructure:

1. **`OptionalClientCertVerifier`** — server requests (but does not require) a client certificate during the TLS handshake. Any cert is accepted at the TLS layer; verification against the DB happens in middleware.
2. **Accept loop change** — extract the client's presented cert DER from the TLS stream post-handshake and inject it as a `PeerCertDer` request extension.
3. **`inject_verified_device` middleware** — looks up the cert bytes in `zynk_devices.tls_cert_der` (exact match, same approach as the existing `PinnedCertVerifier`). If found in a paired record, injects a `VerifiedDevice { device_id, device_name }` extension.
4. **`require_verified_device` middleware** — rejects requests without a `VerifiedDevice` extension (HTTP 401). Applied to the highest-value endpoint: `push-api-key`.
5. **Client cert presentation** — `build_pinned_client_config_with_cert()` presents our own device cert on outgoing connections so the remote server can verify us.

---

## The Critical Comparison

### Codex's flaw: shared mesh token

When shown Claude's mTLS proposal, Codex self-identified its own architectural flaw without prompting:

> "The primary security gap I see in my approach: I'm using a single shared mesh token across all devices. If any device in the mesh is compromised or needs to be removed, changing the token requires re-synchronizing it to every device simultaneously... Claude's per-device certificate approach avoids this entirely."

This is a significant limitation:
- Removing one compromised device from the sync group requires rotating the token on **all** devices simultaneously
- During the rotation window, removed devices still hold a valid credential
- A compromised device can impersonate any other device in the mesh

### Claude's approach: per-device identity

With mTLS using per-device certificates:
- Each device has its own keypair generated at first launch, stored locally
- Revocation is immediate: remove a device from `zynk_devices` and its cert no longer matches the DB lookup
- A compromised device cannot impersonate another device (it doesn't have the other device's private key)
- No new credential infrastructure is needed — the certs already existed for server auth

---

## What Made the Experiment Work

The most interesting result was not that one AI was "better" — it's how the two-AI review process surfaced the architectural flaw:

1. Codex built a working implementation that addressed the surface-level problem
2. Claude designed an alternative grounded in the existing security infrastructure
3. When Codex was shown Claude's design, it independently reasoned about why the shared token was weaker
4. The flaw would have been much harder to catch via human code review alone, because the code is syntactically correct and the token flow looks secure at first glance

**Lesson:** Using two AI systems as independent reviewers of each other's architectural decisions is an effective way to stress-test security choices — especially when the second AI proposes a competing design rather than just auditing the first one.

---

## What Was Implemented

Claude's mTLS approach was implemented on a new branch (`feature/mtls-auth`) with these changes:

| File | Change |
|------|--------|
| `src-tauri/src/tls.rs` | Added `PeerCertDer`, `VerifiedDevice`, `OptionalClientCertVerifier`, `build_server_config_with_optional_client_auth()`, `build_pinned_client_config_with_cert()` |
| `src-tauri/src/zynksync.rs` | Switched server to optional client auth, extract peer cert in accept loop, added `inject_verified_device` and `require_verified_device` middleware, gated `push-api-key` behind cert requirement, updated `rebuild_http_client` to present our own cert on outgoing connections, expanded `push-api-key` allowlist to include R2 and model keys |

Backward compatibility is preserved: devices that have not yet updated will connect without a client cert and continue to work on all routes except `push-api-key`, which requires a cert match.

---

## Takeaway for Future Development

This experiment suggests a practical workflow for security-sensitive features:

1. Have one AI (or human) write an implementation
2. Have a second AI propose an alternative, independently
3. Show each implementation to the other AI and ask it to critique its own work
4. Use the divergence between the two proposals as a map of the design space

The overhead is low. The catch rate on non-obvious flaws appears high.

---

## Working notes (Matt, 2026-09-09; transcribed by Claude from a voice note)

Three observations from the beta build-up, kept here because they extend the takeaway above.

**1. The skill is reading, not writing.** The job that is forming around this work has no title yet, but its shape is clear: instead of learning to read and write one programming language, learn to *read* three or four well enough to check an AI's work in each. Writing is delegated; reading, and knowing what to ask, is not. The mTLS experiment above is an example: the flaw was caught by reading two designs side by side, not by writing either.

**2. Ensemble review of the code itself.** After the beta build is frozen, hand the repository to several independent models (Claude in the desktop app, and possibly Grok, Mistral and Codex) and ask each for a plain audit pass. This is the programming equivalent of Zynkbot's Ensemble mode: no model is expected to be right, but the disagreements are a cheap map of where to look. The findings come back to the coding agent as questions, not as patches. Rule of thumb from the mTLS case: ask for a competing design or an audit, never "fix it" in a second tool, so that there is only ever one hand on the code.

**3. Networking is where agents lose track of *where*.** While designing ZynkSync, the recurring failure was not logic but location: an agent writing "to the database" was sometimes writing to the peer's database, or to the copy it was syncing from, and reported success either way. When debugging anything that crosses a machine boundary, the first question to ask an agent is literal: *which machine, which file path, which process is doing this write?* Requiring that answer before accepting "done" caught more sync bugs than any test did.

**4. Articulation is capability.** (Matt's note, 2026-09-09; the reasoning below is Claude's, and Claude agrees.) How well you can put a thing into words sets the ceiling on what the model can do for you, because the words are the whole specification. Three reasons:

- *A precise word is a precise constraint.* "The close tone played after the reply instead of after my words" took one log search to diagnose. "The sound was weird" would have taken five questions. The model cannot see the screen or hear the phone; it has only the sentence.
- *Vocabulary is addressing.* Knowing the name of a thing (provenance, tombstone, search path, role) lets one word land on exactly the concept the model already holds, instead of a paragraph that lands near it. This is the "read three or four languages" thesis from note 1 applied to nouns: you do not need to write the code, you need to be able to name its parts.
- *It works in both directions.* The same vocabulary is what lets you catch the model overstating. "Structurally prevented" and "explicitly instructed" describe different guarantees; a reader who knows the difference sends the claim back.

The qualification: this is about precision and concreteness, not about sounding technical. A plain, specific description of what was observed ("it fired six times in four minutes while the TV was on") beats a technical term used slightly wrong, because a wrong term is followed confidently in the wrong direction. Say what happened, name what you can name, and describe the rest.

Matt's framing of the same point: rhetoric, the discipline lawyers study in order to be clear, may be something engineers now need to study for the same reason. The goal is signal clarity, not eloquence.

**5. The language penalty.** (Matt's question, 2026-09-09; answer by Claude.) If this is true, and the models are trained mostly on English and Chinese, is everyone else at a disadvantage? Yes, and it compounds. Large models are measurably weaker in languages with little training data: they follow instructions less reliably, know less, and their tokenisers split non-Latin and low-resource text into more pieces, so the same request costs more and fits less into the context window. A speaker of such a language faces a choice between writing in their own language to a weaker model, or writing in a second language and losing exactly the precision note 4 says matters. For the widely used European and Asian languages the gap has narrowed with each model generation; for languages with little written corpus it remains large. It is a fairness problem the field acknowledges and has not solved. Practical consequence for Zynkbot's Elder Mode and any non-English market: test the memory extraction and the voice pipeline in the target language before promising anything, because the offline Vosk model is English-only and the extraction prompts are written in English.
