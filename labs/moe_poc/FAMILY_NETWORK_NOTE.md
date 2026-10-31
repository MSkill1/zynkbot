# One model for the house: a direction note

**Status:** Direction, not a plan. Nothing here is scheduled or measured.
**Date:** 29 October 2026
**Source:** Matt's framing after the ZynkCluster step 1 test, written down so it is not lost.
Scheduled work lives in `ROADMAP.md`; this is the idea behind it.

## The idea

Zynkbot started from Parenting Mode: an assistant that knows a family and behaves
differently for a child than for a parent. Followed through, that is not a mode on one
device. It is a **family network**: the household's phones, laptops and a desktop, linked
and synced through ZynkSync, with permissions that say who may ask what and who may see
which memories. Parents, children, and an older relative under Elder Mode are all accounts
on the same network, not separate installs.

ZynkCluster is what makes that network worth having. Once the devices are linked and wired,
the spare memory in the house can hold one model far larger than any single device could
run, and every account on the network can "ask the big one" when the question deserves it.
The local small model keeps answering the everyday things in the meantime.

## Why it matters

- **The sensitive questions are the valuable ones.** Medical results, the family's
  finances, a parent's care plan. People want a capable assistant for exactly these, and
  these are exactly what they will not send to a hosted service. A household model can be
  handed the records directly because they never leave the house.
- **Permissions become natural.** A child asks the house model about homework and gets
  age-appropriate answers; a parent asks about the mortgage; the grandparent's Elder Mode
  reminders draw on the same memory store with its own boundary. One model, one family
  memory, per-person rules. This is the standing-instructions tier
  (always-loaded, user-authored rules) applied per account.
- **The hardware can be cheap.** Step 1 ran on a desktop and a laptop that were already
  owned. The RAM-first plan prices a three-tower cluster for an 80%-tier model at roughly a
  used-car-repair budget. If that holds, the same setup serves a school classroom, a
  clinic, or an education programme somewhere without reliable connectivity or with data
  that may not leave the building. Out of the box: the small local model. When a stronger
  model is wanted and a few machines can be wired together: the household model.

## What would have to be true

Honest list, so the note does not oversell itself.

1. Step 2 of the RAM-first plan has to show that the speed model predicts real used
   towers, wired, within about a third. Step 1 only measured one WiFi pair.
2. Prefill has to be tamed (prefix caching, pipelined prefill), or "wait a minute" becomes
   "wait ten".
3. llama.cpp's RPC server is unauthenticated. It has to sit behind ZynkSync pairing and
   mTLS before any family uses it, which is already the plan's position.
4. Per-account permissions over a shared memory store are a design problem Zynkbot has
   not yet solved; Parenting Mode and Elder Mode are the two concrete cases to design from.
5. "Easy to set up" for a non-expert is a product question on top of all of the above,
   and the hardest to measure.

## Related

- [`ZYNKCLUSTER_RAM_FIRST_PLAN.md`](ZYNKCLUSTER_RAM_FIRST_PLAN.md) — the hardware and
  measurement plan, with step 1 results
- [`README.md`](README.md) — the expert-parallel architecture
- [`../standing_instructions/`](../standing_instructions/) — the always-loaded rules tier this
  note's per-person permissions would build on
- `docs/ROADMAP.md`, v1.1 section — Elder Mode and Parenting Mode as scheduled work
