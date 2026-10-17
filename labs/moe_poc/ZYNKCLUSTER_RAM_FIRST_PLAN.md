# ZynkCluster: RAM-First Plan

**Running a frontier-class open model on used office PCs**

**Status:** Design notes, not yet tested
**Date:** 17 October 2026
**Source:** working session 16–17 October 2026. Full raw notes are kept locally at
`~/Documents/Zynk_related/claude_notes/ZynkCluster_moe_discussion_2026-10-17.md`
(not in this repository).
**Relationship to [`README.md`](README.md):** that document sizes ZynkCluster in GPU
VRAM and treats CPU nodes as impractical. This document supersedes that sizing for the
big-model path. The expert-parallel architecture in the README is unchanged and is still
the destination — what changes is the hardware it runs on and the order of the work.

---

## The pivot in one paragraph

The earlier plan asked how to split a model across graphics cards. That makes the
constraint VRAM, which is the most expensive memory you can buy. The constraint is
actually *capacity*: the whole model has to sit in memory somewhere. Because every large
open model is a mixture of experts, only a small slice of it is read for any one word, so
the per-word cost stays near what one PC already handles — what you need is somewhere to
*keep* the rest. Used office towers full of cheap DDR4 provide that capacity at roughly
$7–11 per GB of installed memory. Graphics cards buy speed, not capability, and can be added later.
Speed is deliberately secondary: "ask the big one and wait" is the product.

**Goal.** Run an open-weight model clearly bigger than one PC can hold — about 90% of the
paid flagships, enough that a household can leave corporate AI. Today's ceiling on Matt's
desktop (RTX 3090 24 GB, 62 GB DDR4, i9-10900F, MSI Z490, 4 DDR4 slots, 128 GB max) is
gpt-oss-120b at 4-bit, about 65 GB.

---

## Candidates

Totals and published repository sizes are measured — they come from
`data/frontier_sizes.tsv` in the `zynk-models` repository (measured 2026-10-09 against
the Hub). Active-parameter counts were the main open question in the session notes and are
now resolved; see [Active parameters, confirmed](#active-parameters-confirmed) for sources
and the one derivation.

| Model | Total | Active | 4-bit footprint | PCs @64 GB | Read per word | Decode ceiling |
|---|---|---|---|---|---|---|
| MiniMax-M2.7 | 229B | **9.8B** | ~130 GB | 3 | 5.4 GB | 4.6–6.5 w/s |
| Qwen3-235B-A22B | 235B | 22B | ~130 GB | 3 | 12.1 GB | 2.1–2.9 w/s |
| GLM-5.3-Flash | 321B | 18B | ~180 GB | 4 | 9.9 GB | 2.5–3.5 w/s |
| DeepSeek-V3.2 | 685B | 37B | ~380 GB | 7 | 20.4 GB | 1.2–1.7 w/s |
| GLM-5.3 | 753B | **~40B** | ~420 GB | 8 | 22.0 GB | 1.1–1.6 w/s |
| Kimi-K2.6 | 1,027B | 32B | ~570 GB | 10 | 17.6 GB | 1.4–2.0 w/s |
| DeepSeek-V4-Pro | 1,599B | 49B | ~900 GB | 15 | 27.0 GB | 0.9–1.3 w/s |
| Kimi-K3 | 2,780B | **~104B** | ~1.6 TB | 25+ | 57.2 GB | 0.4–0.6 w/s |

Capability tiers from the session, unchanged: Qwen3-235B and MiniMax-M2.7 around the 80%
mark; GLM-5.3-Flash and DeepSeek-V3.2 a step below the top; GLM-5.3 and DeepSeek-V4-Pro
top tier (~90%); Kimi-K3 out of reach.

**Footprint** is total parameters at 4-bit plus about 10% for quantisation metadata,
embeddings kept at higher precision, and a modest KV cache. **Read per word** is active
parameters at 4-bit plus the same 10%. **Decode ceiling** is read-per-word divided by
25–35 GB/s; it is an upper bound that ignores compute and network, and it is explained
next.

DeepSeek-V3.2's 685B is what the repository holds: 671B of main model plus a 14B
multi-token-prediction module. Kimi-K3 ships as a native MXFP4 checkpoint, so its 1.56 TB
published size already *is* the 4-bit size.

---

## Why memory bandwidth sets the speed

For one word, a mixture-of-experts model reads its active parameters once. At 4-bit that
is the "read per word" column above — 22 GB for GLM-5.3. Two consumer DDR4 channels at
2933–3200 MT/s offer 47–51 GB/s on paper and deliver 25–35 GB/s in practice, so 22 GB
takes about 0.7 seconds. That is the whole of the 1.1–1.6 words/s figure.

Two consequences, both important:

**Adding PCs does not make a layer split faster.** In a layer split each machine holds
some layers and they take turns. The cluster still reads 22 GB per word in total; it is
just spread over eight machines reading 2.75 GB each, one after another. Eight nodes at
30 GB/s finish in the same 0.7 s as one node that somehow had 420 GB. More nodes buy
capacity only. This is why step 1 below is worth doing immediately and why step 3 exists.

**An expert split is what parallelises the read.** For a given word the k active experts
live on different machines, so those reads happen at the same time and effective bandwidth
scales with the number of machines holding active experts. That is the 2–4 words/s target,
and it is the architecture already described in [`README.md`](README.md).

Decode is memory-bound, not compute-bound: GLM-5.3 needs about 80 GFLOP per word
(2 × active parameters), which is ~0.2 s on one office CPU against 0.7 s of reading. The
arithmetic has slack. **Prefill does not** — see below.

Each tower must have both memory channels populated. Two 32 GB sticks is simultaneously
the cheap way to reach 64 GB and the fast way; one 64 GB stick would halve bandwidth and
therefore halve every figure in the table.

---

## The real bottleneck is prefill, not decode

This did not come up in the session and it changes the product.

Decode reads each weight once per word. Prefill — reading the question and its memory
context before the first word comes out — reuses each weight across every prompt token, so
it is bound by arithmetic instead, at roughly 2 × active parameters per token. For GLM-5.3
that is 80 GFLOP per prompt token. A 4–8 core office CPU does perhaps 0.1–0.2 TFLOPS
effective on 4-bit GEMM, and in a layer split the machines again take turns, so the
cluster has one machine's worth of arithmetic available.

That works out to roughly **7–13 minutes per 1,000 prompt tokens** for a 40B-active model
on a layer-split cluster of office towers. Decode of a 300-word answer adds about 3.5
minutes. A "think hard" request carrying a page of memory context is therefore a
ten-minute request, not the minute or two the plan assumes.

Two checks agree on the arithmetic: the FLOP estimate above, and reported CPU prompt
throughput for gpt-oss-120b (5.1B active) of 30–50 tok/s on a strong modern desktop, which
is about 0.4 TFLOPS effective and scales down as expected for weaker CPUs and 8× the
active parameters.

Four mitigations, best first:

1. **Cache the prefix.** A household assistant sends a stable system prompt and a slowly
   changing memory context ahead of a short question. Keep that prefix's KV cache between
   requests and the recurring cost disappears; only the new question is prefilled. This is
   the single biggest win and it needs no new cluster code.
2. **Keep the delta small.** Send a tight memory context — a few hundred tokens of
   retrieved memories, not everything available.
3. **Pipeline prefill.** Chunk the prompt and pipeline the chunks across the layer split,
   so all eight machines work at once instead of taking turns. That is close to an 8×
   improvement and is arguably a better first piece of custom code than the expert split.
   llama.cpp's RPC mode does not do this today.
4. **Prefer low active-parameter models.** Prefill and decode both scale with active
   parameters, so this is the lever that moves everything.

**Recommendation on the first-model question.** Active parameters, not total, decide what
the thing feels like to use. MiniMax-M2.7 is the outlier: 9.8B active is a quarter of
GLM-5.3's, it needs 3 PCs instead of 8, and it should prefill and decode roughly four
times faster — about $1,000 of hardware for a usable 80%-tier assistant. Build that first
and use it to validate the measurements; treat the 90% tier as the second build, by which
point prefix caching and pipelined prefill will exist and memory may be cheaper. If
M2.7's quality disappoints in practice, GLM-5.3-Flash on 4 PCs is the fallback at 18B
active. This is a recommendation, not a decision.

---

## Network budget

A representative hidden-state vector is 12 KB (hidden size ~6,144 at fp16; Kimi-K3 is
7,168). All four cases below assume eight nodes, k=8 active experts and ~75 MoE layers.

| Case | Per request | On gigabit |
|---|---|---|
| Layer split, decode | ~84 KB per word (7 boundaries × 12 KB) | 0.1% of the link |
| Layer split, prefill (4k prompt) | ~336 MB | ~3 s |
| Expert split, decode | ~14 MB per word (75 layers × 8 nodes × 12 KB each way) | ~43 MB/s at 3 w/s, ~35% of the link |
| Expert split, prefill (4k prompt) | ~58 GB | ~8 min |

**Answers to the open question.** Gigabit is not merely enough for a layer split, it is
over-provisioned by three orders of magnitude — 100 Mbit would do, and the session's "a
few KB per word" was right. Gigabit is also adequate for expert-split *decode*, so 2.5 GbE
is headroom rather than a requirement there.

What actually matters for an expert split is **round-trip latency, not bandwidth**: each
word needs ~75 sequential round trips, one per MoE layer. Wired at 0.1–0.3 ms that is
8–23 ms per word against several hundred ms of compute — negligible. Over WiFi at 2–10 ms
plus jitter it becomes 150–750 ms per word and costs 1.5–3× the speed. So: **wire the
cluster.** WiFi is tolerable for a layer split (7 hops per word) and not for an expert
split.

Expert-split prefill is the exception that spoils the picture — 58 GB for a 4k prompt,
because every prompt token must reach each of its k experts. It is ~170× the layer split's
traffic for the same prompt. The two splits want opposite weight layouts, so a cluster has
to choose, and this is a second reason step 1 is the right place to start.

---

## Hardware and cost (October 2026)

Memory is about 5× last year's price and still rising; new fabs land 2027–28 with relief
expected in 2028. Check eBay **Sold items** for real prices rather than asking prices.

- **Used office tower** — tower/MT form factor, not the flat small-form-factor ones:
  ~$150. If a card might go in later, require a free full-length slot and a 450 W+ supply
  with an 8-pin PCIe cable, or budget $60–80 for a supply swap.
- **64 GB per PC** — two used 32 GB DDR4 UDIMM non-ECC sticks, DDR4-2933/3200: $250–500.
  Server RDIMM/ECC Registered memory is cheaper per GB and **does not work** in consumer
  boards; do not buy it by mistake.
- **Eight PCs plus a switch:** ~$3,500–5,500 with no graphics cards. That holds GLM-5.3 or
  DeepSeek-V3.2 class.
- **Three PCs plus a switch:** ~$1,200–2,000. Holds MiniMax-M2.7 or Qwen3-235B.
- **Matt's desktop to 128 GB** with used DDR4: $400–800, which fits gpt-oss-120b at CPU
  speed.
- **Cards, if speed is wanted later** — 16 GB is the common size; 24 GB appears only on
  the RTX 3090 / 3090 Ti / 4090 and RX 7900 XTX. Used RTX 3060 12 GB $200–250; Arc A770
  16 GB $250–300; RTX 4060 Ti 16 GB $350–450; RTX 5060 Ti 16 GB $429 list but $600–800
  street; used RTX 3090 24 GB $700–1,000, the best price per GB; Tesla P40 24 GB ~$300.
  One 16 GB card per tower roughly triples the decode figures above. Note that a card only
  accelerates the share of layers it holds — on a node holding 52 GB of a layer split, a
  16 GB card covers about 30%.

The laptop has no usable GPU and is fine as a CPU node, but a slow node in a layer split
adds its full turn to every word; give it fewer layers than the towers. Raspberry Pis were
rejected — too little memory and too little bandwidth to be worth a hop.

---

## Plan

**1. Free test, today.** llama.cpp's RPC mode already splits a model across machines by
layers with no custom code: an `rpc-server` on each node and `--rpc host:port,...` on the
main process. Run gpt-oss-20b across desktop + laptop and confirm the mechanism end to
end. The same mode would run a DeepSeek-V3.2-class model over 7–8 PCs today at ~1–2
words/s. Note that llama.cpp's RPC backend is explicitly experimental and unauthenticated
— trusted wired LAN only, never exposed, which is also the argument for moving it behind
ZynkSync's existing pairing and mTLS once it leaves the bench.
*Done when:* a model answers across two machines, and the per-token wall time is within a
third of read-per-word ÷ measured bandwidth.

**2. Build the cluster and measure the three costs separately.** Memory reading, network
hops and arithmetic, measured independently rather than as one aggregate words/s number.
The point is a model that predicts speed on hardware you have not bought yet to within
about a third. Measure prefill and decode separately — per §"The real bottleneck", they
are bound by different things and mixing them hides the problem.
*Done when:* predicted and measured words/s agree within a third on a configuration not
used to fit the model.

**3. Write ZynkCluster's expert-level split.** Experts for one word read in parallel
across machines — the 2–4 words/s target and the architecture in
[`README.md`](README.md). Prefix KV caching and pipelined prefill (§"The real bottleneck")
should be weighed against this for priority; they may deliver more felt improvement per
line of code. Cards only when speed matters after that.

**4. Zynkbot "think hard" path.** A button, or "Zynkbot, think hard about this": the
question plus its memory context go to the cluster, the local model keeps answering
normally in the meantime, and the result arrives as a notification. The cluster is an
optional back end, never a dependency — if it is off, Zynkbot behaves exactly as it does
today.

---

## Active parameters, confirmed

The session left three active-parameter counts unconfirmed. All three are now settled.

- **MiniMax-M2.7 — 9.8B active** of 229.9B, 8 of 256 experts over 62 layers, from the
  model's own technical report (arXiv:2605.26494). The session's "~10B?" was right.
- **GLM-5.3 — ~40B active** of 753B: 78 decoder layers (3 dense, 75 MoE), 256 routed
  experts with 8 chosen per token plus 1 shared expert. The session's "~40B?" was right.
  One aggregator lists active parameters as 753.9B, which is a labelling error — its own
  expert configuration implies ~40B.
- **Kimi-K3 — ~104B active**, settling the session's "50–104B" at the top of the range.
  The published `config.json` gives 93 layers (1 dense), 896 experts with 16 chosen per
  token plus 2 shared, hidden size 7,168 and MoE intermediate size 3,072. The routed-expert
  path alone is therefore 92 × 18 × 3 × 7,168 × 3,072 ≈ **109B**, which confirms the
  widely reported 104B and rules out the ~50B community figure. Practically this makes
  Kimi-K3 doubly out of reach: 1.6 TB to hold *and* 2–3× DeepSeek-V4-Pro's cost per word.

Also worth recording from the measured data: `zynk-models/data/frontier_sizes.tsv` reports
GLM-5.2 and GLM-5.3 with identical parameter counts (753,329,940,480). That is Hub
metadata, probably a carried-over config, and is a minor thing to re-check at some point.

---

## Open questions

- **First model.** MiniMax-M2.7 on 3 PCs is recommended above, on the strength of its 9.8B
  active count; GLM-5.3-Flash on 4 PCs is the fallback. Matt's call.
- **M2.7's real-world quality** at 80%-tier claims, which is the one thing the arithmetic
  cannot tell us. Worth testing through a hosted API before buying any hardware.
- **Whether pipelined prefill should come before the expert split** in step 3. Depends on
  what step 2 measures for a realistic memory-context prompt.
- **Effective DDR4 bandwidth on actual used towers** — the 25–35 GB/s assumption carries
  every speed figure here and is cheap to measure on the first machine bought.
- ~~Active parameters for MiniMax-M2.7, GLM-5.3, Kimi-K3~~ — resolved above.
- ~~Network: is gigabit enough~~ — resolved above. Wire it; latency, not bandwidth, is the
  constraint for an expert split.

---

## Sibling repository change

`MiniMaxAI/MiniMax-H3` should come out of `zynk-models/tiers/runnable.txt`. It is an
omni-modal video generator rather than a text model, and its licence grant excludes the
United States by default — US deployment requires a separate application to MiniMax, with
a content-compliance commitment, not a right under the community licence.

Two consequences of that removal, both measured from the repository:

- H3 is **498.5 GB of the runnable tier's 983 GB** and 280 of its 470 files. Removing it
  halves the tier to ~0.48 TB and 190 files, so `tiers/runnable.manifest`,
  `docs/MODEL_ARCHIVE.md` and the size figures in that repository's `README.md` need
  regenerating alongside.
- `tiers/frontier.txt` is a judgement call rather than a clear removal. That tier exists to
  preserve state-of-the-art capability across jurisdictions for the archive, and H3 is
  genuinely frontier for video. Keeping it there with a licence annotation is defensible;
  keeping it in *runnable* is not.

---

## What this does not claim

- Nothing here has been measured on a cluster. Every speed figure is derived from
  bandwidth, FLOP and packet-size arithmetic, and derived figures are wrong in the
  direction of optimism more often than not.
- The footprint column assumes a 4-bit quantisation of the stated quality exists for each
  model and that 10% of overhead covers embeddings and KV cache. Long contexts will exceed
  that allowance.
- Capability tiers ("80%", "90%") are the session's judgement from published benchmarks,
  not measurements, and benchmark standing moves quickly.
- Prices are October 2026 observations in a rising memory market and will not hold.

---

## References

**Models and measurements**

- `zynk-models/data/frontier_sizes.tsv` — measured repository bytes, file counts and
  parameter totals, 2026-10-09
- `zynk-models/docs/MODEL_ARCHIVE.md` — the archive feasibility analysis this builds on
- MiniMax-M2.7 technical report — https://arxiv.org/pdf/2605.26494
- Kimi-K3 configuration — https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json
- GLM-5.3 model card (NVIDIA) — https://build.nvidia.com/z-ai/glm-5-3/modelcard
- GLM-5.3 analysis — https://artificialanalysis.ai/models/glm-5-3
- DeepSeek-V3.2 — https://www.modular.com/models/deepseek-v3-2
- MiniMax H3 licence territory exclusion — https://we0.ai/articles/why-minimax-h3-s-open-weight
  and MiniMax's own reply — https://x.com/MiniMax_AI/status/2084490811652333633

**Runtime**

- llama.cpp RPC mode — https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc
- Petals, for the layer-split comparison — https://github.com/bigscience-workshop/petals

---

**Document Version:** 1.0
**Last Updated:** 17 October 2026
**Author:** Matt Skillman (notes from a working session with Claude)
**License:** See main repository LICENSE file
