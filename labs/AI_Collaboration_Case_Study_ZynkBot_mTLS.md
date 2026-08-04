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
