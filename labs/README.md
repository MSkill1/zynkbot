# Labs — research and design notes

Experiments and design notes that are not scheduled work. Scheduled work lives in `docs/ROADMAP.md`; a labs note moves there only when it is decided and dated. Each folder says what it is and what came of it. Newest activity first.

| Folder | What it is | Where it stands (2026-10-23) |
|---|---|---|
| `local_model/` | The on-device model for Android: candidate list, build measurements, the plan with its 13 steps, kept-context results, the memory bench tables and the four-type relationship bench | **Active, v1.0.** The engine runs on the Pixel and the OnePlus; Qwen3-4B-Instruct-2507 Q4_K_M chosen as the shipped "ZB model" |
| `memory_bench/` | `cases.json`: 84 extraction cases (69 expected facts, 26 no-fact messages) and 44 relationship cases that `memory_bench.rs` runs against any backend, cloud or GGUF; since 2026-10-24 extraction is scored as call 2 of the pipeline, titles and dates included | **In use.** Chose the on-device model, the four relationship types and the call 2 wording (seven rounds, 2026-10-24; results note in `~/Documents/Zynk_related/memory_bench/`); the seed of a future fine-tuning set |
| `wake_verifier/` | The personal "Hey Zynk" verifier: clip collection, profile building, the voice-sample website, and whether the verifier could act as a lock | **Shipped in part** (verifier v3 ships, self-trains on the phone); the lock idea is an experiment only |
| `assistant_role/` | Zynkbot as the phone's digital assistant (the Android assistant role), what it allows and what it costs | Design note, agreed direction; gated on on-device checks |
| `zynksync_improvements/` | Semantic conflict detection during sync and other sync ideas | Design phase, aimed at v1.0 |
| `kits/` | "Kits": packaged uses of the memory system (the Kitchen kit ships free; others are one-time purchases) | Design; the tier decision is on the roadmap |
| `standing_instructions/` | An always-loaded memory tier of user-written rules, the foundation for Parenting and Elder Mode | Speculative; not before v1.0 |
| `check_in/` | "Did you ever…?": one question a day about a stale intention at its expiry; app logic, no training | Built 2026-10-29 as a beta 5 experiment; note kept as the spec |
| `snapin_platform/`, `zynkforge/` | The snap-in platform: plan and embedding-dimension notes | Labs plan (2026-09), not scheduled |
| `moe_poc/` | ZynkCluster: running a frontier-class open model across several ordinary machines; a RAM-first plan for used office PCs with its step 1 results; `FAMILY_NETWORK_NOTE.md`, the household direction behind it (one model for the house, Parenting and Elder Mode as accounts on a family network); illustrative coordinator and node code | **Step 1 measured 2026-10-29:** gpt-oss-20b answered split across the desktop and the laptop with llama.cpp's RPC mode, no custom code (a layer split over WiFi; the expert split is still step 3). Wired rerun pending an adapter; steps 2–4 untested. Illustrative code only |
| `collaboration_history/` | How the project is built: the two-session, four-device beta test day of 2026-09-12/13 | Record, not a plan |
| `AI_Collaboration_Case_Study_ZynkBot_mTLS.md` | Case study of the mutual-TLS transport being built with an AI pair | Record |
