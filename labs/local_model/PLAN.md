# On-device model: the plan

*Branch `local-model`, started 2026-10-22. Order is by payoff per day of work. Status
marks: done · building · next · later · decision (Matt's call, usually prompt wording).*

## Quantisation check (2026-10-22, step 9, Pixel 10 Pro XL, same question)

| | Q4_0 (ran first, cooler phone, 5 threads) | Q4_K_M (ran second, 6 threads) |
|---|---|---|
| Prompt read | 923 tokens, 46 tokens/s | 984 tokens, 37 tokens/s |
| Answer written | 44 tokens in ~15 s | 26 tokens in ~13 s |

Reading about a quarter faster with Q4_0; writing within noise. Not the big win hoped
for; Q4_K_M stays the default for its quality. Worth re-running when a 2B is compared.

## Where it stands (Pixel 10 Pro XL, Qwen3-4B Q4_K_M, 2026-10-22)

| | First run | After tuning |
|---|---|---|
| Prompt reading | 10 tokens/s | 39–49 tokens/s |
| Answer writing | ~2 tokens/s | 3–6 tokens/s (slower once the phone is warm) |
| Short question, ~900-token prompt | 96 s | 37 s |
| Long answer, 418 tokens | — | 169 s |
| Apps killed by Android at model load | 15 | 0 |

Reading is no longer the limit. Writing speed and answer length are, plus the memory
pipeline running on the same model after every message.

## Steps

1. **Engine builds for Android** — done. llama-cpp-2 0.1.160 (Gemma 4, Qwen3.5), the
   Android stub removed, model file installed from the Zynkbot folder.
2. **Engine tuned for the phone** — done. ARMv8.2 dot-product/fp16 kernels
   (`GGML_CPU_ARM_ARCH` in `.cargo/config.toml`; a stale configure hid it for two builds),
   fast cores only, 4K context, 512-token prompt pieces.
3. **Background yields to the user** — building. A user's turn registers itself; a memory
   call on the model stops between tokens when one is waiting and runs again after. Fixes
   the message that sat three minutes behind relationship classification.
4. **Keep the unchanged prompt between turns** — next. Today the model's memory is
   cleared before every call and the whole prompt is read again. If the start of the new
   prompt matches the previous one, the engine can keep that part and read only the rest.
   Needs the prompt ordered stable-first: voice, then the conversation so far, then the
   recalled memories and the new message. Engine change plus prompt structure (decision:
   the reorder touches prompt text).
5. **Streaming** — next. The local path returns the whole answer at once; the cloud paths
   stream. The token loop already exists; feed the same channel.
6. **The reply is one job** — done 2026-10-22, with a lesson. The reply prompt asks only
   for the answer, short. A closing "REMEMBER: yes/no" flag was tried first (Matt's
   design, approved) and the 4B model wrote the flag and nothing else, even for "I have a
   cat named Louie" (and a web-search answer came back blank the same way). Removed the
   same day: the small model does the last instruction it is given, so the reply gets one.
   The yes/no now lives in the background extraction call, which answers NONE or lists
   facts; it costs about 3 s (105-token prompt) per message. Original text of the item: The prompt now asks for a MEMORY_EXTRACT line and
   the answer in one reply; a 4B model did the first and stopped (empty bubble), or
   repeats the question. Proposal: the reply prompt asks only for the answer, with a
   length instruction for local models; memory work moves to step 7. Wording for Matt.
7. **Memory pipeline for a slow model** — built 2026-10-22 (local models only): after
   every reply a background extraction call lists up to three facts (or NONE); each fact
   runs the existing storage path (duplicates, contradictions, relationships) as its own
   memory; all of it yields to the user's next message. Still open: Android slows the app
   off screen, so this may also wait for the screen or the charger. Original text: (a) A yes/no call: does this message
   hold something worth remembering? (b) only on yes: extraction, then relationships, each
   its own small call. (c) A note where memories appear: with a local model they take a
   few minutes. Android slows an app's CPU off screen (a background call read at 20
   tokens/s vs 39 in front), so this work may also wait for the screen or the charger.
8. **A 2B model side by side** — later. Qwen3.5-2B and Gemma 4 E2B write roughly twice
   as fast as a 4B. Scored on the memory bench (ROADMAP) before choosing.
9. **Quantisation the phone CPU likes** — later, a measurement not a feature. The engine
   rearranges Q4_0 files at load for ARM's dot-product layout; Q4_K_M (what we run) is
   not rearranged. Helps reading, which is now the smaller cost; same file size, so
   writing speed (memory-bandwidth bound) should not change much. One download and one
   comparison on the same phone. Not phone-specific.
10. **Unload the model when idle** — later (ROADMAP, Android background power): drop it
    after a few idle minutes, reload on demand.
11. **Model file management** — later: the Models page says what is installed, which
    files will and will not run (architecture, size), a download list. Includes the
    two-model idea (Matt 2026-10-22): a small everyday model plus, optionally, a larger
    one kept for quality answers offline when a minute or two is acceptable; the page
    and the user guide explain the trade. ROADMAP has the item. Also (Matt 2026-10-22):
    the installers and first run on Windows, Linux and Android offer to download the
    chosen everyday model, with its size stated and the user's consent, so a fresh
    install can answer offline without hunting for a file.
12. **GPU/NPU** — later, optional: OpenCL/Vulkan on Qualcomm phones for the OnePlus test;
    the Pixel's TPU is Google-only.
13. **Fine-tune on Zynkbot's own tasks** — after the bench exists: an Apache-2.0 base, a
    training set from the bench and consenting logs, LoRA on a rented GPU, quantise, ship.

## Kept context on the phone (2026-10-22, first build with it)

| Turn | Prompt | Read |
|---|---|---|
| Plain question, cold after restart | 1,318 tokens, 0 kept | 39 s |
| Cat and trip message | 693 tokens, 260 kept | 11 s; answer 53 tokens in 16 s |
| Weather, first pass | 718 tokens, 260 kept | 23 s |
| Weather, answer pass with pages | 1,326 tokens, 252 kept | 48 s |

Only the 260-token voice part was kept, because the history window slid by one message
each turn and changed the text right after it. Fixed the same day: the window moves in
blocks of six. Also seen: the phone throttles after a few minutes of model work (CPU at
about 55% of its top speed, skin 38 °C), and reading fell from 38 to 20 tokens/s and
writing to under one word a second. Less work per turn is the only remedy on the CPU;
a 2B model is the next lever. The relationship call stepped aside for the user's
message as designed, but only after finishing a 512-token piece (22 s); background
reads now go in 128-token pieces.

## 2B versus 4B, first look (2026-10-22, Pixel, Qwen3.5-2B Q4_K_M)

| | Qwen3-4B | Qwen3.5-2B |
|---|---|---|
| Prompt read, cool phone | 39–49 tokens/s | 112–134 tokens/s |
| Answer written, cool | 3–6 tokens/s | ~6 tokens/s (104 tokens in 17 s) |
| Answer written, hot phone | ~1 | ~2.4 (74 tokens in 31 s) |
| Kept context between turns | works (455–490 tokens kept) | **does not apply**: 0 kept every turn |

Why the kept context fails on Qwen3.5: it is a hybrid model (part attention, part recurrent
state; `LLM_ARCH_QWEN35` is in llama.cpp's hybrid list), and a recurrent state cannot be
rolled back to an earlier position, so the engine's "drop everything after the shared
part" returns false and the whole prompt is read again. Gemma 4 is not hybrid and should
keep context. Possible fix for hybrids, later: save a checkpoint of the state right after
the stable part (voice + conversation) and restore it next turn, then read only the rest;
llama.cpp has state save/restore per sequence. Memory cost is the state size.

Quality on the second test message was poor: recall brought four unrelated memories from
test chatter (Spanish phrases, a speech) and the 2B refused and rambled. The bench
decides; recall quality on a test-polluted memory store is its own problem.

## Memory bench, first two rounds (2026-10-22; labs/memory_bench, memory_bench.rs)

Extraction: 59 expected facts in 50 messages, 20 messages that should store nothing.
Relationships: 30 pairs in round 1, 34 in round 2 (four mislabelled cases corrected — a new
fact that adds a detail is "elaborates" — and four restatement cases added as "supports").

| Model | Facts found | Invented | Junk on the 20 | Relationships r1 | r2, sharper definitions |
|---|---|---|---|---|---|
| Anthropic | 57/59 | 1 | 0 | 24/30 | 31/34 |
| OpenAI | 55/59 | 0 | 0 | 22/30 | — |
| Qwen3-4B Q4_K_M | 49 → 51/59 | 1 | 0 | 18/30 | 25/34 |
| Gemma 4 E4B q4_0 | 7/59 (answers NONE to plain facts) | 0 | 0 | 21/30 | — |
| Qwen3.5-2B Q4_K_M | 25/59 | 3 | 0 | 15/30 | — |

What moved the relationship numbers: the definitions. "Elaborates" = adds a detail the
memory did not have; "supports" = the same thing again with nothing new; "none" whenever
the two facts share no subject, person, place or topic. The 4B went from 6/12 to 11/12 on
elaborates and 1/4 to 4/4 on reminds-of. Still weak for the 4B: "none" (1/4; it links
unrelated facts as reminds-of) and restatements (2/6; it calls them elaborates). No model
stored junk on any of the 20 no-fact messages, so the two false memories of the afternoon
were the old one-prompt flow's doing. Numbers in a fact were being dropped by the parser
for every model (fixed in round 2). Decision from the numbers: Qwen3-4B is the memory
model on the phone; the 2B is for chat speed only; Gemma 4 needs an extraction-prompt
change before it can be judged.

### Confidence floor: measured and not adopted (2026-10-22 evening)

Sweep on the 4B's 34 relationship cases: floor 0.0→25 correct, 0.5→25, 0.6→25, 0.7→25,
0.8→25, 0.9→27. The model reports 0.85–0.98 on every link, wrong ones included; the three
links it invented between unrelated facts came at 0.85, 0.85, 0.92, and its correct
"reminds of" links at 0.92. A floor at 0.9 gains two and sits on top of the real links.
Conclusion: the 4B's confidence carries no information, so the floor is left at 0.0
(mechanism and tests kept). What would address the "links unrelated facts" habit: the
links it invents are all "reminds of", the loosest type, and graph traversal pulls linked
memories into later prompts, so a wrong "reminds of" pollutes recall (the Spanish test
phrases recalled for a dessert question). Option for Matt: do not store "reminds of" from
the on-device model (loses 4 correct weak links on the bench, removes 3 wrong ones; the
wrong ones cost more than the weak ones are worth).

## Seen on 2026-10-22 and not yet addressed

- A web-search turn is two model passes. The second one carried the fetched pages: 1,851
  tokens, 64 s to read. For local models the search results need a token cap, and that
  pass should not also carry recalled memories (18 were recalled for "what's the weather
  in Morocco").
- The thread heuristic picked 5 fast cores once and 6 another time on the same phone;
  harmless, but the top-frequency reading moves. Pin to a count once measured.

## Measurements to keep taking

Every change: the same two messages on the same phone, cold and warm — a short question
and the "favourite breakfast" memory message — with `Prompt read in`, `Generated`,
`Local model inference` and `Total request time` from the log, and whether anything was
killed for memory.
