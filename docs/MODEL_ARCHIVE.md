# Sovereign Model Archive

**What it would take to mirror the world's open-weight language models onto hardware you own**

*Measured 2026-10-09 against the live Hugging Face Hub. All figures in this document
are from direct measurement, not estimates, except where explicitly labelled.*

---

## The short answer

Mirroring **everything** is not a storage problem, it's a supply-chain problem — and it
buys you very little. Mirroring **everything that matters** costs about **16 TB, two days,
and $700 in drives**, and it is worth doing.

The distribution is brutally top-heavy. The 29 frontier models that represent the actual
state of the art across China, the US and Europe total **15.79 TB**. Expanding to all
3.14 million model repositories on the Hub costs roughly **15,880 TB** — a 1,000× increase
in cost to acquire fine-tunes, quantisations and abandoned experiments, the overwhelming
majority of which are derivatives of the models already in the first 16 TB.

The binding constraint is not disk. It is that **you cannot run most of what you would
download**, and that enterprise drive supply is booked into 2027–2028.

---

## What is actually out there

Two independent measurement methods, which converge:

| Method | Repos measured | Result |
|---|---|---|
| Sum of `main_branch_size` over a Hub repo-stats snapshot, scaled to current repo count | 1,160,000 measured → 3,135,619 current | **15.88 PB** |
| Sum of weight bytes computed from per-dtype safetensors parameter counts | 900,224 with weight metadata | **15.89 PB** |

Current Hub population: **3,135,619 public model repositories**. Median repository size is
**269 MB**; mean is **5.06 GB**. The largest single repository measured is **32.9 TB**.

> **A caution on Hub metadata.** The raw safetensors parameter totals sum to 146
> *quadrillion* parameters, which is nonsense: 95% of that comes from **ten repositories
> with fabricated metadata**, the worst claiming 39.2 quadrillion parameters on 228
> downloads. Any scan of this kind must trim outliers or it will produce a storage
> estimate an order of magnitude too high. The 15.89 PB figure above caps repositories at
> 3T parameters, just above the largest genuine model.

### Concentration

| Bucket | Share of all model bytes |
|---|---|
| Largest 100 repos | 5.9% |
| Largest 1,000 repos | 18.1% |
| Largest 10,000 repos | 45.3% |

---

## The budget curve

What you get for what you spend. `usable TB` is measured payload; `raw` assumes
dual-parity plus slack (×1.4); drive cost at **$30 per raw TB**, the Oct-2026 retail rate.

| Tier | Contents | Usable TB | Raw TB | Drives | Days @1 Gbps | Days @10 Gbps |
|---|---|---|---|---|---|---|
| **A** | 29 frontier models, as published | 16 | 22 | $663 | 2 | <1 |
| **A′** | same, duplicate formats filtered | 14 | 20 | $595 | 2 | <1 |
| **B** | top 1,000 repos by downloads | 5 | 8 | $226 | 1 | <1 |
| **C** | top 5,000 by downloads | 78 | 110 | $3,288 | 9 | 1 |
| **D** | top 10,000 by downloads | 392 | 549 | $16,455 | 45 | 6 |
| **E** | top 50,000 by downloads | 1,794 | 2,512 | $75,367 | 208 | 26 |
| **F** | every model repo (~3.14M) | 15,880 | 22,232 | $666,960 | 1,838 | 230 |

Note the shape: the **1,000 most-downloaded repositories cost only 5 TB**, because
popular models are mostly small — embedding models, 7B chat models, quantisations.
Tier B is cheaper than Tier A while containing almost none of the frontier. You want
both, and together they are still under 25 TB.

Tier F is a five-year download at gigabit, and the drive cost excludes chassis, HBAs,
JBOD shelves, ECC RAM and power. It is also, right now, unpurchasable at any speed:
nearline HDD supply is sold out through 2026 with agreements booked into 2027–2028.

---

## The frontier set

29 repositories, **15.79 TB** as published. Full measured table with checksummable file
counts in [`labs/model_archive/data/frontier_sizes.tsv`](../labs/model_archive/data/frontier_sizes.tsv).

| Repo | Size | Params | Licence | Gated |
|---|---|---|---|---|
| meta-llama/Llama-3.1-405B-Instruct | 2,443.8 GB | 406B | llama3.1 | **manual** |
| moonshotai/Kimi-K3 | 1,561.0 GB | 2,780B | other | no |
| zai-org/GLM-5.2 | 1,506.7 GB | 753B | mit | no |
| moonshotai/Kimi-K2-Instruct | 1,029.2 GB | 1,026B | other | no |
| deepseek-ai/DeepSeek-V4-Pro | 864.7 GB | 1,599B | mit | no |
| meta-llama/Llama-4-Maverick-17B-128E | 803.2 GB | 402B | other | **manual** |
| zai-org/GLM-5.3 | 755.7 GB | 753B | other | no |
| deepseek-ai/DeepSeek-V3.2 | 689.5 GB | 685B | mit | no |
| deepseek-ai/DeepSeek-R1 | 688.6 GB | 684B | mit | no |
| mistralai/Mistral-Large-3-675B-Instruct | 681.5 GB | 675B* | apache-2.0 | no |
| tencent/Hy3 | 597.6 GB | 299B | apache-2.0 | no |
| moonshotai/Kimi-K2.6 | 595.2 GB | 1,027B | other | no |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | 573.5 GB | 1,024B | mit | no |
| deepseek-ai/DeepSeek-V4.1-Flash | 510.3 GB | 763B | mit | no |
| MiniMaxAI/MiniMax-H3 | 498.5 GB | 33B | other | no |
| Qwen/Qwen3-235B-A22B | 470.2 GB | 235B | apache-2.0 | no |
| stepfun-ai/Step-3.5-Flash | 398.8 GB | 199B | apache-2.0 | no |
| zai-org/GLM-5.3-Flash | 328.4 GB | 321B | mit | no |
| MiniMaxAI/MiniMax-M2.7 | 230.2 GB | 229B | other | no |
| openai/gpt-oss-120b | 195.8 GB | 117B | apache-2.0 | no |
| *(9 more under 80 GB)* | 367.9 GB | | apache-2.0 | no |

\* Mistral publishes no safetensors parameter metadata; 675B is from the repo name. Every
other parameter count in this table is read from Hub metadata.

**Filtering pays.** Llama-3.1-405B is 2.44 TB only because the repo carries both
safetensors (812 GB) and a duplicate `original/` consolidated checkpoint (1,632 GB).
Excluding duplicate formats cuts the frontier tier from 15.79 TB to roughly 14.2 TB with
no capability loss.

---

## The part that actually bites: you cannot run most of it

Holding weights is not the same as having access to a model. A 4-bit quantisation needs
roughly `params × 0.56` bytes of fast memory, plus overhead:

| Model | Q4 weights | Runs on |
|---|---|---|
| Kimi-K3 (2.78T) | 1,557 GB | **nothing you can buy as one box** |
| DeepSeek-V4-Pro (1.6T) | 895 GB | 1.5 TB dual-socket server, CPU speed only |
| GLM-5.2 / DeepSeek-V3.2 (~700B) | ~420 GB | 512 GB server with GPU offload |
| gpt-oss-120b | 65 GB | 128 GB workstation, comfortably |
| gemma-4-31B / Qwen3.8-27B | ~18 GB | **one 24 GB GPU — genuinely interactive** |

This inverts the intuition behind the whole exercise. The models most at risk of being
withdrawn are the giant frontier ones, and those are exactly the ones a private archive
cannot *use* — they become museum pieces awaiting hardware. The models that deliver real
sovereign capability are the 7B–120B class, which total **under 1 TB** for the whole set
(see [`tiers/runnable.txt`](../labs/model_archive/tiers/runnable.txt), measured at 0.98 TB).

Archive the giants for insurance. Depend on the mid-size ones.

---

## Which denial vectors mirroring actually closes

| Vector | Does a mirror fix it? |
|---|---|
| Model withdrawn, repo deleted, lab shuts down | **Yes.** This is the core win, and it is real. |
| Licence revoked / terms changed retroactively | **Partly.** You keep using what you hold; redistribution is the exposed part. |
| Gated download approval revoked | **Only if you pull first.** Llama requires manual approval *at download time*. |
| Export controls, sanctions, geo-blocking | **Yes**, for what you already hold. |
| Hub rate-limits or blocks you mid-acquisition | **No** — this is an acquisition-phase risk (see below). |
| You lack hardware to run the weights | **No.** Mirroring does nothing for this. |
| Inference stack rots (CUDA, vLLM, kernels) | **No, unless you archive the stack too.** |
| Fire, flood, theft, drive failure | **No.** One server is a single point of failure. |

Two of these deserve emphasis:

**Gated models are the genuine urgency.** `meta-llama/*` repos are `gated: manual` — a
human approves your access request. If you want Llama weights, that approval is a door
that can close. Everything non-gated can be fetched any time; gated weights are the only
part of this with a real clock on it.

**Archive the runtime, not just the weights.** A safetensors file is inert. You need the
tokenizer, the chat template, a compatible `transformers`/`vLLM`/`llama.cpp` build, and a
CUDA toolchain that still compiles. Pin and mirror container images and source tarballs
alongside the weights, or in five years you will own 16 TB of files you cannot load. This
is the most commonly skipped step and the one most likely to render an archive useless.

---

## Licence reality

Of the 3,000 most-downloaded repos, parsed from card metadata:

| Licence | Repos | Redistribution |
|---|---|---|
| apache-2.0 | 1,649 | Free |
| mit | 404 | Free |
| *(unstated)* | 381 | **Assume none** |
| other (bespoke) | 263 | Read each one |
| cc-by-nc-4.0 / -sa | 84 | **Non-commercial only** |
| gemma | 40 | Google terms apply |
| llama3.1 / 3.2 | 36 | Notice + acceptable-use policy |

Roughly **68% are Apache-2.0 or MIT** — you may keep, run and redistribute them freely.
Nothing stops you *retaining and using* weights you lawfully downloaded under any of these.
The restrictions bite on **redistribution**: if the archive is for you, you are almost
entirely clear; if you intend to re-serve weights to others, the 263 bespoke "other"
licences (which include Kimi and GLM-5.3) and the non-commercial set each need reading.
This is not legal advice; the bespoke licences genuinely vary.

---

## Operational constraints during acquisition

**Rate limits.** The Hub enforces per-window limits on the resolver endpoint used for file
downloads: roughly 3,000 requests per 5-minute window anonymous (per IP), 5,000
authenticated free, 12,000 PRO, and far higher for Enterprise. Exceeding one returns
HTTP 429. For Tier A this is a non-issue — 15.79 TB across ~3,000 files is few requests
for very many bytes. For Tier D and above, request volume becomes the limiting factor
rather than bandwidth, and the fetcher must back off rather than retry hard. Always send
an `HF_TOKEN`. Rotating tokens to evade limits is both detectable and a terms violation;
don't.

**Mirror a second source.** For Chinese models, ModelScope carries releases that sometimes
appear there first or exclusively. A sovereign archive with one upstream has one
chokepoint.

**Integrity is available and should be used.** The Hub exposes a **SHA256 per LFS file**,
so every byte is verifiable at download and re-verifiable forever after. Silent corruption
is what actually kills decade-scale archives. Store the manifest, verify on write, and
scrub on a schedule.

---

## Recommended plan

Staged, so each stage is independently useful:

1. **Today — gated first (~3.2 TB).** Pull `meta-llama/*` while approval holds. It is the
   only component with a closing door.
2. **This week — frontier insurance (14 TB filtered).** The 29-model set, duplicate formats
   excluded, checksummed. Two days at gigabit, ~$600 of drives.
3. **This week — the runnable tier (1 TB).** The 7B–120B models you will actually use daily.
   Keep this on fast storage, separate from the archive bulk.
4. **Alongside — the runtime (~100 GB).** Pinned container images, inference engine source,
   CUDA installers, tokenizers. Without this the rest is inert.
5. **Then stop and reassess.** Tier C (78 TB) is defensible if you want breadth of
   fine-tunes. Tier D (392 TB, $16k, 45 days) needs a concrete reason. Tier F is not a
   project, it's an institution.
6. **Not one server.** "No one can ever deny access" includes fire and drive failure.
   Dual-parity plus one offline copy, or the single-server plan defeats itself.

Total for stages 1–4: **~18 TB, under $1,000 in drives, a long weekend of downloading.**
That captures essentially all of the capability and nearly all of the real risk reduction.

---

## Tooling

Working, tested scripts in [`labs/model_archive/`](../labs/model_archive/):

| Script | Purpose |
|---|---|
| `hf_manifest.sh` | Build a verifiable manifest (path, bytes, SHA256) with rate-limit backoff |
| `mirror.sh` | Download a manifest, verifying every file; resumable and idempotent |
| `verify.sh` | Re-verify an archive against its manifest (scheduled scrub) |
| `plan.sh` | Turn any manifest into storage, cost and time |

Tested end-to-end: download → verify → single-byte corruption → detect → repair → verify
clean → idempotent no-op.

---

## What this does not solve

A complete archive of every open-weight model still leaves you without the frontier
*closed* models, which are the actual state of the art and are not downloadable at any
price. Mirroring gives you durable, independent access to the open tier — which in
October 2026 is genuinely close behind, and in the 7B–120B class is entirely sufficient
for real work. That is a strong position. It is not the same as immunity.
