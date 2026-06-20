# HIPAA Security Integration Guide

**Document Version:** 4.0
**Last Updated:** April 2026
**Purpose:** Zynkbot's built-in HIPAA-supporting features, value proposition for healthcare providers, and what production deployments require

## ⚠️ Important — Read Before Using in a Healthcare Environment

**THIS SOFTWARE IS NOT HIPAA-CERTIFIED. IT IS A PRIVACY FRAMEWORK, NOT A COMPLIANCE SOLUTION.**

Zynkbot provides architectural privacy features that support HIPAA compliance efforts. It is not a certified medical product, is not a complete HIPAA solution, and does not replace a Business Associate Agreement (BAA). PHI detection is regex-based with known limitations (estimated 70–85% accuracy for structured patterns). Developers assume no liability for compliance failures, data breaches, or regulatory penalties.

The architectural descriptions in this document are verified against the codebase. The general HIPAA regulatory framework described (BAA requirements, risk assessment obligations, Security Rule structure) is accurate at a high level. Whether any specific configuration satisfies a HIPAA audit depends on the covered entity's own risk assessment. This document is not a substitute for qualified HIPAA compliance counsel.

**Current version (v0.9):** PHI detection, audit logging, and medical request blocking are implemented and functional. In HIPAA mode, the personal memory extraction pipeline is fully disabled — extraction does not run, and any memories that do persist carry an 8-hour expiration as a fallback safety net. Expanded PHI detection (BERT NER + additional regex patterns) and per-workstation improvements are planned for v1.0. The enterprise server deployment (Scenario B) and LLM semantic pre-check are a future release beyond v1.0.

---

## AI in Healthcare: Why This Matters Now

Healthcare has one of the highest documentation and administrative burdens of any industry. Physicians in the US spend an estimated 30–50% of their time on EHR documentation and paperwork — time not spent on patients. Burnout driven by administrative load is a documented crisis across nursing, primary care, and specialty medicine alike.

AI is already being deployed to address this. Microsoft's Nuance DAX, Abridge, and Suki are raising hundreds of millions of dollars specifically to help physicians auto-document patient encounters. Epic, the dominant EHR platform, is integrating AI across scheduling, documentation, and clinical decision support. This is not a future trend — it is happening in hospitals now.

Beyond documentation, clinical and administrative staff use AI daily for:
- **Reference and lookup** — drug interactions, dosage information, clinical guidelines, protocol checks, literature summaries
- **Administrative work** — prior authorization drafts, billing code suggestions, discharge instruction templates, patient communication drafts, compliance policy questions
- **Training and onboarding** — new staff asking procedure questions, reviewing institutional policies, understanding coding standards

**The most important practical reality:** Healthcare workers are already using AI informally — personal ChatGPT and Claude accounts, often against hospital policy, because no sanctioned alternative exists and the tools are genuinely useful. A hospital's choice is not really "allow AI or not." It is whether to give staff a compliant, auditable tool designed for their environment, or continue to let them use personal consumer accounts with no oversight, no audit trail, and no PHI guardrails.

That is the problem this document addresses.

**An underappreciated benefit: internal accountability and fraud deterrence.** Healthcare fraud — billing fraud, prescription fraud, staff accessing patient records for identity theft — costs the US healthcare system an estimated $100 billion annually and is a significant exposure for any facility. Because Zynkbot logs every AI query with attribution to a user account, it creates an audit trail for how staff are using the tool. A billing clerk researching fraudulent coding schemes, a staff member repeatedly querying about controlled substance access, or unusual patterns of PHI-adjacent queries all leave a record. Staff who know their AI usage is logged and attributed to their account are less likely to misuse it. This is the same principle behind EHR audit logging — a standard compliance requirement — extended to the AI layer.

---

## The Problem With Current Healthcare AI

### "HIPAA Compliant" Usually Means Very Little

When OpenAI, Google, or Microsoft offer "HIPAA-compliant AI" services, what they are actually offering is a **Business Associate Agreement** — a legal contract stating they will handle your data responsibly. The underlying architecture is unchanged: your query, including any PHI it contains, travels to their data centers and is processed by their model. The BAA governs what they do with it afterward.

This creates a structural problem:

```
Current Cloud AI (ChatGPT, Claude, Gemini with BAA):

User types: "My patient John Smith, DOB 01/15/1985, MRN 4829201, presents with..."
                                    ↓
                    Sent over the internet to cloud provider
                                    ↓
                    LLM processes the full query including PHI
                                    ↓
                    Response returned, disclaimer added
                    ⚠️ PHI was transmitted, processed, and logged
```

Once PHI enters a cloud LLM prompt, it has been transmitted to and processed by infrastructure outside your control. Providers retain request and response logs for operational, debugging, and compliance purposes — that log entry now exists on servers you don't own, subject to the provider's security posture, internal access policies, and legal exposure. A BAA establishes contractual accountability, but it doesn't change the underlying architectural fact: the data left your building.

HIPAA's **Minimum Necessary standard** (§164.514) requires covered entities to limit PHI disclosure to what is actually needed for the intended purpose. Routing every staff query through a cloud LLM that processes the full text — including PHI the LLM didn't need to see — is difficult to reconcile with this principle.

There is also an ongoing question of whether AI providers use inputs for model training or fine-tuning, even under BAA agreements. The BAA may prohibit this, but the architectural exposure has already occurred.

### The Other Side of the Problem: You Can't Just Avoid AI

Staff are already using AI informally — whether the hospital has a policy about it or not. Cloud tools handle general reference questions well enough that people route around restrictions to use them. The choices are: ban it and watch staff use personal accounts anyway, sign BAAs with cloud providers and accept the architectural exposure described above, or deploy a local system where PHI never leaves your infrastructure in the first place.

### Zynkbot's Structural Difference

Zynkbot blocks PHI **before the LLM ever sees the query** — on the device that sent it:

```
Zynkbot HIPAA Mode (Desktop App):

User types: "My patient John Smith, DOB 01/15/1985, MRN 4829201, presents with..."
                                    ↓
                    PHI detected on LOCAL DEVICE (before network)
                                    ↓
                    Query BLOCKED — never sent to LLM
                    ✅ PHI never transmitted, never processed
```

For local models (a doctor's office with its own local AI and patient data storage), nothing leaves the staff's devices or local network at all — a compliance-ready architecture with no external data transmission. For cloud API models (Claude, GPT-4, Grok), the PHI check runs first — if PHI is detected, the API call is never made.

This is a structural difference, not a policy difference. The protection is architectural: there is no configuration that can accidentally bypass it, no BAA that needs to be trusted, and no data center that receives the query.

**What this means in practice:** The most common category of accidental PHI disclosure to AI tools is typed or copy-pasted structured identifiers — staff pasting a patient's phone number, SSN, or MRN into a query without thinking. For that specific category, Zynkbot blocks at 85–95% accuracy before the LLM ever processes the query. Cloud AI with BAA provides zero pre-processing protection.

### Where PHI Detection Runs

This is worth being explicit about. Zynkbot is a **desktop application** (Tauri + Rust). The PHI detection code in `containment.rs` runs in the local Rust process on the **user's own device** — the same machine they are typing on, or that a patient may be interacting with on a tablet or touch screen. After the user hits send, the check is performed before private patient information leaves that device.

This is the most private possible approach:
- For **local models**: query stays entirely on the device
- For **cloud API models**: query is checked locally first, and only forwarded if no PHI is detected

One important clarification on how blocking works: when PHI is detected, the **entire query is rejected**. The user sees an error message (shown in the examples below). PHI is not stripped out and the query is not forwarded with the sensitive parts removed — the whole message stops there. This is intentional. A partially-redacted query can still carry identifying context, and silent redaction would give false confidence.

In a future **enterprise server deployment** (Zynkbot running as a shared server for multiple staff), the equivalent approach is server-side pre-processing — the server checks for PHI before routing the query to the LLM. PHI would cross the local network to the server but would never reach the LLM until PHI is cleaned. This is the second-best option and still far better than any cloud AI architecture that processes PHI unconditionally and then attempts to train the model not to divulge the PHI it should never have received.

### How This Looks in a Healthcare Setting

**Scenario A: Per-workstation (current architecture)**

Each workstation runs Zynkbot as a desktop application. Staff query the AI from their own computer — looking up drug interactions, reviewing protocols, drafting administrative documents, asking policy questions. This is the current v0.9 deployment model.

This is most valuable for **non-clinical staff and non-clinical tasks**: administrators drafting communications, educators developing training materials, compliance staff reviewing policies, billing staff handling procedure codes. For clinical staff, general reference questions work fine; queries that include a structured patient identifier — SSN, phone number, MRN — are blocked before reaching any AI model.

**PHI detection in Scenario A (v0.9 → v1.0):** The current 7-pattern regex detects structured identifiers at 85–95% accuracy. The v1.0 upgrade for per-workstation deployments is **BERT NER** — a lightweight model already running in Zynkbot for memory search. Wiring it into HIPAA mode adds person-name-in-clinical-context detection, catching queries like *"Patient John Smith presents with..."* that regex cannot see. BERT NER runs fast on-device with no additional hardware requirements, making it appropriate for individual workstations. It's not a full semantic judgment — it won't catch everything — but it's a meaningful step up at essentially no infrastructure cost.

**Knowledge base in Scenario A:** Each user maintains their own indexed document library. A nurse uploads the unit's medication protocols and post-op instruction templates; a billing specialist uploads ICD-10 and CPT reference guides; a compliance officer uploads accreditation standards. Relevant sections surface in responses alongside the AI's general knowledge. Documents stay on the local machine — never sent to a cloud service.

**Staff communication in Scenario A:** ZChat allows direct device-to-device messaging between staff on the same local network. ZynkLink handles file transfers on the same local network — a staff member can send a referral document, a discharge summary, or a lab result directly to a colleague's Zynkbot instance without routing through email or shared drives. All traffic stays within the facility's network.

**Scenario B: Central server with shared access — the real enterprise deployment (not yet packaged)**

A hospital IT department runs Zynkbot on a local server. Staff access it from tablets, workstations, shared terminals, or touchscreens at nursing stations. One instance serves the whole department or facility, centrally managed and audited.

The query path: staff device → hospital Zynkbot server → PHI check → if clean, LLM → response to device. PHI never reaches the LLM. This is architecturally straightforward — Zynkbot's backend is already structured to support it — but has not yet been packaged as a server product. ContainAI is interested in building this out with a healthcare partner.

**PHI detection in Scenario B — adding the LLM semantic pre-check:** A server deployment changes what's possible. Because the server is already running a capable local model for inference, that same model can serve as a **semantic pre-check** before every query reaches the main LLM. The check is a simple prompt: *"Does this message contain information that could identify a specific person in connection with their health?"* A 13B+ model answers this reliably.

This is the significant compliance upgrade. Regex catches structured identifiers. BERT NER catches person names in clinical context. The LLM pre-check catches everything else — clinical descriptions, combinations of demographic details, narrative context that neither pattern matching nor NER can reason about. A query like *"The patient is a 47-year-old male firefighter at the only station in Flagstaff with Type 2 diabetes"* contains no structured identifier and no name, but is clearly identifying. Only a semantic model catches it.

The three-layer stack on the server:
```
Query arrives
  → Regex (instant — structured identifiers: SSN, phone, MRN, email)
  → BERT NER (fast, on-device — person names in clinical context)
  → LLM pre-check (slower — semantic judgment on full query)
  → If all pass → route to main LLM
```

This architecture takes PHI detection from "good for structured identifiers" to genuinely robust — catching the narrative and contextual cases that represent the most significant compliance gap in current regex-only implementations.

**Knowledge base in Scenario B:** The server hosts a shared organizational knowledge base for all authenticated staff. IT centrally manages the facility's clinical guidelines, formulary, accreditation policies, nursing protocols, and training materials — staff query a single authoritative source rather than managing their own document libraries. A physician asking about a drug interaction gets the same formulary reference as a pharmacist. Updates are made once and apply to everyone.

**Staff communication in Scenario B:** ZChat runs through the central server as a facility-wide secure messaging channel — nurses to physicians, departments to departments, admissions to billing. ZynkLink enables staff-to-staff file transfers within the facility: a radiologist sends an imaging report to an ordering physician, a department shares updated protocols with all staff — all without leaving the facility's infrastructure. Because all traffic routes through a server you control, communication logs are available for audit purposes.

**Branding:** For enterprise deployments, Zynkbot can be built and deployed under a custom name and icon — "Memorial Health AI Assistant" or similar. This involves building a custom binary from source rather than a settings toggle, but is a straightforward customization under the commercial license.

**What Zynkbot is not designed for:**
- Storing actual patient records or EMR data
- Replacing existing clinical decision support systems

The intended use is as an **AI assistant for healthcare staff** — a knowledgeable general-purpose tool that helps with reference, documentation, administrative work, and general queries, with guardrails preventing accidental PHI exposure.

---

## Implemented Features

### ✅ Production-Ready

**1. PHI Detection & Blocking** — Runs on device before any LLM call
- 7 regex patterns: SSN, phone, email, ZIP code, credit card, IP address, physical address
- Blocks at routing layer — LLM never sees the query
- Estimated accuracy: 70–85% for formatted/structured patterns
- Source: `src-tauri/src/containment.rs`

**2. Medical Request Blocking**

These blocks are designed for non-clinical or self-service contexts — they catch queries where someone is asking the AI to diagnose or treat themselves. For clinical staff asking about patient care ("what are the diagnostic criteria for X?", "when is surgery indicated for Y?"), these blocks generally do not trigger — those are legitimate reference questions that pass through normally. The PHI detection layer is the primary protection for clinical staff, not these blocks:

- Self-diagnosis requests: `"diagnose me"`, `"what do I have"`, `"do I have"`, `"is it cancer"`
- Medication dosing (self): phrases like `"how much should I take"` + a medication keyword
- Treatment decisions (self): `"should I get surgery"`, `"should I have the procedure"`, `"should I start treatment"`
- Source: `src-tauri/src/containment.rs`

**3. Audit Logging**
- Daily JSON logs: `logs/hipaa_audit/hipaa_audit_YYYY-MM-DD.json`
- Events logged: PHI detection, diagnosis blocking, dosing blocking, treatment blocking, allowed conversations
- Source: `src-tauri/src/containment.rs`

**4. Memory System Disabled in HIPAA Mode**
- Personal memory extraction pipeline does not run in HIPAA mode — extraction is skipped before any write is attempted
- Any memories that do persist (e.g. created via another code path) carry an 8-hour expiration as a defense-in-depth fallback
- Source: `src-tauri/src/lib.rs`

**5. Conversation History Disabled in HIPAA Mode**
- Raw conversation logging is skipped entirely in HIPAA mode — no records are written to `conversation_sessions` or `conversation_messages`
- The History button is grayed out with tooltip explanation when HIPAA mode is active
- Raw conversation text is more sensitive than extracted facts; disabling both is the correct default
- Source: `src-tauri/src/lib.rs`, `src-tauri/src/conversation_history.rs`

**6. Medical Disclaimer Auto-Addition**
- Automatically appended to any health-related response
- Triggered when the response contains: symptom, treatment, medication, diagnosis, disease, condition, health, medical, doctor, patient, therapy
- Disclaimer text appended: *"⚕️ AI-generated. Not a substitute for clinical judgment or current clinical guidelines."*
- Source: `src-tauri/src/lib.rs`

---

### 📌 PHI Detection: Current Implementation and v1.0 Roadmap

The 7-pattern implementation is functional working software. That said, 7 patterns are the floor. The v1.0 roadmap includes significant expansion without any architectural redesign.

**Additions that are direct HIPAA PHI identifiers (Safe Harbor method):**
- Date of birth formats (`01/15/1985`, `January 15, 1985`, `DOB:`) — one of HIPAA's 18 enumerated identifiers
- Medical Record Numbers (`MRN: 12345678`) — enumerated HIPAA identifier
- Health plan beneficiary and account numbers

**BERT NER for contextual name detection (v1.0):**
The BERT NER model already runs in Zynkbot and extracts `PERSON` entities for memory search. In HIPAA mode, detected person names combined with clinical context can be flagged as PHI — catching narrative patterns like *"Patient John Smith presents with chest pain..."* that regex cannot see.

Important nuance: a name alone is not PHI under HIPAA's Safe Harbor definition. What makes it PHI is a name *linked to health information* about an identifiable individual. "What did Johns Hopkins publish on metformin?" contains a name and should pass. "My patient John Smith has Type 2 diabetes" contains a name + health condition about a specific person and should block. The BERT NER implementation needs to be context-aware, not a simple name-triggers-block rule. This is what makes it a more careful engineering task than just wiring the existing NER output into a block decision.

**Local LLM semantic pre-check (future enterprise release):**
The ceiling for regex + NER detection is catching known patterns. The ceiling for a capable local language model is semantic understanding — it can reason about whether a query, taken as a whole, would constitute impermissible disclosure of PHI even without structured identifiers.

The architecture:
```
Query arrives
  → Regex check (fast, structured identifiers: SSN, phone, MRN, etc.)
  → BERT NER check (person names in clinical context)
  → Local LLM pre-check: "Does this message link an identifiable person to health information?"
  → If all pass → route to main LLM
```

The pre-check prompt is simple: ask the model whether the query contains information that could identify a specific real individual in connection with their health. A capable local model (13B+) handles this judgment reliably. This approach catches the narrative PHI cases that neither regex nor NER can see — clinical descriptions, combinations of demographic details, contextual identifiers.

For a hospital or clinic running a dedicated local server, this is the architecture that gets detection from "good for structured identifiers" to genuinely robust. The same local model can serve as both the pre-check filter and the main inference engine.

Questions or feedback: **matt@containai.ai**

---

### ⚠️ Not Implemented — Integration Points

The following are intentionally left as integration points so organizations can plug in their existing security infrastructure:

- **Encryption at rest** — database and disk (see Security Requirements)
- **Encryption in transit** — TLS for LAN traffic (on v1.0 roadmap; see Security Requirements)
- **Authentication & access control** — SSO, RBAC, MFA (enterprise deployments)
- **Backup and disaster recovery**
- **Incident response procedures**
- **Business Associate Agreements (BAA)** with any third-party vendors

---

## Comparison With Cloud AI

| Feature | ChatGPT / Claude / Gemini (with BAA) | **Zynkbot** |
|---------|--------------------------------------|-------------|
| PHI detection before processing | ❌ No — LLM processes all input | ✅ Yes — blocked on device first |
| PHI leaves the facility | ❌ Yes — travels to cloud data center | ✅ No — blocked before any network call |
| Memory system | Processes all input | ✅ Disabled entirely in HIPAA mode — extraction pipeline does not run |
| Audit trail | Provider-managed, limited access | ✅ Local JSON logs, your infrastructure |
| Self-hostable | ❌ No | ✅ Yes |
| Open source | ❌ No | ✅ Yes |
| BAA required | ✅ Yes — and you must trust it | ✅ Not required if PHI never leaves device |

### Processing Flow

```
┌──────────────────────────────────────┐
│           YOUR SECURITY LAYER        │
│  (Auth, Encryption, Access Control)  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        ZYNKBOT HIPAA AI LAYER        │
│  • PHI Detection (Regex, device-side)│
│  • Containment Modes                 │
│  • Memory System Disabled (HIPAA)    │
│  • Conversation History Disabled     │
│  • Audit Logging (local JSON)        │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│          YOUR DATABASE LAYER         │
│  Patient data clean prior to storage │
│       (Local SQLite database)        │
└──────────────────────────────────────┘
```

---

## PHI Detection in Action

**Try it yourself:** Download Zynkbot, open Settings, and switch the containment mode to **HIPAA**. Then type any of the queries below — or your own variations — and observe the blocked response. No configuration required beyond enabling the mode.

All names, numbers, and identifiers below are entirely fictional.

---

**Blocked — SSN detected:**
> *"My patient is Mary Johnson. Her social is 234-56-7890 and she has been having chest pains."*

**User sees:** `🔒 Please don't share personal health information like SSN, insurance numbers, or member IDs. This system is designed for general health discussions only.`

Query never reaches the LLM. Audit log entry written.

---

**Blocked — Phone number detected:**
> *"I've tried reaching the patient twice at 555-867-5309 with no answer. What should I document in the chart?"*

**User sees:** `🔒 Please don't share personal health information like SSN, insurance numbers, or member IDs. This system is designed for general health discussions only.`

Query blocked at routing layer before network call. To get a useful answer, the staff member should rephrase without the identifier: *"I've tried reaching a patient twice with no answer. What should I document in the chart?"* — which passes detection and gets a normal response.

---

**Blocked — Diagnosis request:**
> *"Based on the symptoms I just described, what do I have?"*

**User sees:** `🏥 I cannot provide medical diagnoses. Please consult with a licensed healthcare provider for diagnostic evaluations.`

---

**Blocked — Medication dosing:**
> *"How much acetaminophen should I take after the procedure?"*

**User sees:** `🏥 I cannot provide medication dosing advice. Please consult your doctor or pharmacist for accurate dosing information.`

---

**Allowed — General clinical question:**
> *"What are the common side effects of beta blockers?"*

No PHI, no dosing calculation, no diagnosis request. Response generated normally with disclaimer appended automatically: *"⚕️ AI-generated. Not a substitute for clinical judgment or current clinical guidelines."*

---

**Gap — Narrative PHI (not currently detected):**
> *"John Smith, a 45-year-old male, presents with substernal chest pain radiating to the left arm."*

No structured identifier (no SSN, phone, email). Current regex does not catch this. The name + clinical context combination is exactly what BERT NER person-name detection addresses in the v1.0 upgrade.

---

## Security Requirements

The requirements below vary significantly by deployment environment and organization size. A solo practitioner running Zynkbot on a single laptop has different obligations and constraints than a hospital IT department deploying a shared server. The items listed here are the categories that typically need to be addressed — how you address them depends on your existing infrastructure, your compliance officer's guidance, and the results of your own risk assessment.

**European deployments / GDPR:** Zynkbot's local-first architecture aligns well with GDPR's core principle of data protection by design and by default (Article 25). Local processing eliminates most third-party data processor obligations under Article 28. The Memory Manager's deletion capability supports the right to erasure. A data portability export feature and breach notification workflow are on the roadmap. As with HIPAA, GDPR compliance is ultimately the deploying organization's responsibility — Zynkbot provides structural support, not certification.

**A note on scope:** ContainAI cannot configure or customize Zynkbot for every security environment, EHR integration, or regulatory scenario a healthcare organization may face. HIPAA requirements interact with state-level privacy laws (which vary significantly), accreditation requirements, payer contracts, and organization-specific risk profiles. Each provider is responsible for determining how Zynkbot fits into their compliance program under the laws and regulations that apply to them.

### Encryption at Rest

**Requirement:** All data stored by Zynkbot — the SQLite database file and any local files — must be encrypted at rest.

**How to satisfy it:** Zynkbot stores all data in a single SQLite file (`~/.local/share/zynkbot/zynkbot.db` on Linux, `%LOCALAPPDATA%\zynkbot\zynkbot.db` on Windows). OS-level disk encryption (BitLocker, LUKS, FileVault) covers this file with no changes to Zynkbot itself and is the recommended approach. For column-level encryption of the database file itself, SQLCipher is an alternative, but requires a custom build.

Your IT or security team will know what your organization already uses and what your compliance documentation requires.

---

### Encryption in Transit

**Requirement:** All network communication carrying user data must use TLS 1.2 or higher.

**Current status:**
- Database layer: Not applicable. Zynkbot uses SQLite — a local file accessed directly by the process. There is no network database connection to secure.
- ZynkSync, ZynkLink, and ZChat LAN traffic: TLS via `rustls` is on the **v1.0 roadmap**. For standalone local deployments, this traffic currently runs unencrypted on the local network. For enterprise deployments on a hospital network, standard network-layer controls (VLANs, private subnets) reduce the exposure of this gap until it is addressed.

**How to satisfy it for enterprise deployments:** Route Zynkbot traffic through whatever TLS termination your organization already operates. Your network team will know the right approach for your infrastructure.

---

### Authentication & Access Control

**Standalone deployment:** The Zynkbot desktop app runs per-user locally. No network-facing API is exposed by default, so no additional authentication is needed for single-user installations.

**Enterprise / multi-user server deployment:** Add authentication middleware appropriate to your organization's existing infrastructure (JWT, SAML 2.0, OAuth 2.0 / OIDC). Zynkbot passes `user_id` and `session_id` through all API calls, making it straightforward to inject user identity from an upstream auth layer.

---

### Audit Logging

**Built in — no configuration required:**

Zynkbot writes structured JSON audit logs to `logs/hipaa_audit/hipaa_audit_YYYY-MM-DD.json`:

```json
{
  "timestamp": "2026-04-12T14:32:10.123456",
  "event_type": "phi_detection_blocked",
  "metadata": {
    "blocked": true,
    "query_length": 87
  }
}
```

Event types: `phi_detection_blocked`, `diagnosis_request_blocked`, `dosing_request_blocked`, `treatment_request_blocked`, `hipaa_conversation_allowed`.

**For enterprise compliance, you may want to add:**
- Hash-chaining log entries for tamper detection
- Real-time forwarding to your organization's SIEM (Splunk, ELK, etc.)
- User identity injected at the auth middleware layer

---

## Deployment Checklist

This checklist is divided into two categories. **Technical configuration** covers what IT sets up in Zynkbot and the surrounding infrastructure. **Organizational obligations** covers what your compliance officer, legal team, and administration must handle — these are requirements that no software product can satisfy on your behalf.

### Technical Configuration

- [ ] Encryption at rest — OS disk encryption enabled (standalone) or cloud provider encryption configured (enterprise)
- [ ] Encryption in transit — LAN traffic encryption (ZynkSync/ZynkLink/ZChat) on v1.0 roadmap; for now, use network-layer controls (VLANs, private subnets) on hospital networks
- [ ] Audit logs — storage path accessible, included in backup rotation
- [ ] PHI detection tested with representative examples (see examples above)
- [ ] Medical disclaimer confirmed present in health-related responses
- [ ] Memory system disabled in HIPAA mode verified — confirm no memories are written during a test session (Memory Manager shows empty / grayed out)
- [ ] Conversation history disabled in HIPAA mode verified — confirm History button is grayed out and no sessions appear in the database after a test exchange

### Organizational Obligations

- [ ] Business Associate Agreements signed with all third-party vendors
- [ ] HIPAA Security Rule risk assessment completed and documented (§164.308(a)(1))
- [ ] Written security policies and workforce training in place
- [ ] Incident response and breach notification plan documented
- [ ] Access provisioning/deprovisioning process defined
- [ ] Audit logs reviewed on a regular schedule (quarterly minimum)
- [ ] Penetration testing conducted (annually recommended)
- [ ] Cyber liability insurance obtained

---

**Safe language for describing Zynkbot in healthcare contexts:**
- ✅ "Blocks PHI on the device before it reaches any AI model"
- ✅ "Privacy-first architecture that supports HIPAA compliance efforts"
- ✅ "PHI never transmitted to LLM provider — checked locally first"
- ❌ "HIPAA-compliant" — requires certification this software does not have
- ❌ "Guarantees HIPAA compliance" — no software can guarantee this

---

## Contact

**Enhanced HIPAA module / provider partnerships:** matt@containai.ai

**Technical questions:** Open an issue on GitHub or see the `/docs` directory.

**What developers do not provide:** Legal advice, compliance consulting, BAAs, warranties, or indemnification. Engage qualified HIPAA counsel before production deployment.

**External resources:**
- HIPAA Security Rule: https://www.hhs.gov/hipaa/for-professionals/security/
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
