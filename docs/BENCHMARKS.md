# Benchmarks: measuring the memory system

Zynkbot's memory is the part of the product that has to be trusted, so it is measured, not
asserted. This page records the three benchmarks the project runs, what each one measures,
the numbers they have produced, and how anyone with the repository can run them again.
Everything here is reproducible from the code in this repository; the benches are ordinary
Rust tests, marked `#[ignore]` so they run only when asked.

Dates are the day a result was produced. Numbers are copied from the result files the
benches write, never from memory.

## In short

- **Making memories** (does the model turn what the user said into the right facts, and
  nothing else?): on 84 messages with 69 expected facts and 26 messages that should store
  nothing, Claude Sonnet finds 69 of 69 and stores nothing wrong; the 4-billion-parameter
  model that runs on the phone finds 66 of 69 and stores nothing wrong on the 26, after a
  fine-tune 68 of 69. (2026-10-24, 2026-10-28)
- **Linking memories** (does the model say how a new fact relates to what is stored:
  updates it, contradicts it, is related, or unrelated?): Claude 44 of 44; the phone model
  28 of 44, after the fine-tune 38 of 44. (2026-10-24, 2026-10-28)
- **Finding memories** (when the user asks a question, is the right memory among the ones
  handed to the model?): on 40 questions against a real year of one user's memories, the
  right memory is in the top 10 for 36 of 40 and in the top 4 for 32 of 40. (2026-10-27)
- **The fine-tuned phone model** beats the base model it was trained from on every column
  of the memory bench, at the same speed, on cases it never saw. (2026-10-28)

What these numbers do not claim: the memory and linking cases are synthetic (written to
cover the kinds of thing people say, not drawn from anyone's life); the recall bench is one
user's data and 40 questions; every case has one scorer. The limits section says more.

## 1. The memory bench: making and linking memories

**Where:** `labs/memory_bench/cases.json` (the cases), `zynkbot_rust/src-tauri/src/memory_bench.rs` (the runner).

**What it measures.** Zynkbot extracts memories in a pipeline of three model calls per
message: the reply, then a separate call that lists the facts in the message (call 2),
then a call that says how each new fact relates to the memories already stored (call 3).
The bench runs calls 2 and 3 on a fixed set of cases and scores them.

- *Extraction:* 84 messages. 58 hold 69 expected facts in all; a produced fact counts as
  found when it contains every required term of an expected fact (for example "cat" and
  "louie"). 26 messages should produce nothing: questions, pasted third-party text,
  translation requests, television speech, greetings; a fact produced from one of those
  counts as junk. A produced fact that matches no expectation counts as invented. Two
  cases check that a date stated relatively ("yesterday") lands on the right calendar day.
- *Relationships:* 44 cases, each a new fact and one existing memory, with the expected
  type among `updates`, `contradicts`, `related`, `none`. A case passes when the model names
  that type for that memory.

**Results, 2026-10-24** (the prompt wording that shipped in beta 5; cloud models through
the user's own API key, the phone model on this desktop's engine):

| Model | Facts found (of 69) | Invented | Junk messages (of 26) | Relationships (of 44) |
|---|---|---|---|---|
| Claude Sonnet 5 | 69 | 0 | 0 | 44 |
| GPT-5.5 | 68 | 3 | 4 | 42 |
| Grok 4.5 | 67 | 1 | 1 | 44 |
| Mistral Large | 68 | 9 | 14 | 34 |
| Qwen3-4B Q4_0 (the phone model) | 66 | 13 | 0 | 29–30 |

**What the bench changed.** The extraction prompt went through seven wording rounds on
2026-10-24, each scored on the same cases before it was kept. The phone model went from
51 facts found under the old one-line format to 66 under the final wording; the two
examples that mattered most were a fact stated inside a question ("Raspberry pie is my
favourite; what's a good recipe?" holds one fact) and its negative twin ("Could I put salt
in my coffee?" holds none). One wording round (round 4, which showed an empty-answer
example up front) dropped the phone model to 34 and was reverted: a small model copies the
example it is shown. The full round-by-round table is in the project's result files.

**Relationship types.** The scheme was reduced on 2026-10-23 from six types to four. On
the 44 cases Claude scores 43–44 under every wording tried; the phone model's two habits,
refusing to say `none` and over-reaching for `updates`, did not move with wording (32, 30
and 31 of 44 across three variants), which is why the app treats the phone model's
`updates` as a question to the user, and why the fine-tune below targets this step.

## 2. The recall bench: finding memories

**Where:** `labs/recall_bench/cases.json`, `zynkbot_rust/src-tauri/src/recall_bench.rs`.

**What it measures.** When the user asks something, Zynkbot searches its memories and hands
the best matches to the model. The bench takes a copy of a real database (one user, about a
year of daily use, 617 memories on 2026-10-27), asks 40 questions whose answer is known to
be in a specific memory, and reports where that memory ranks under the production search
(meaning similarity plus a boost for shared names). A question counts as found for the phone
when the memory is in the top 4 (the phone's prompt holds 4) and for a cloud model in the
top 10 (today 30). The bench also tries four ways of writing the stored text to see which
one recall finds best.

**Results, 2026-10-27** (40 questions, 617 memories):

| Stored text variant | In top 4 | In top 10 | Mean reciprocal rank |
|---|---|---|---|
| A: the fact as stored | 32 / 40 | 36 / 40 | 0.71 |
| B: title + fact | 33 / 40 | 36 / 40 | 0.72 |
| C: fact + the user's words | 25 / 40 | 32 / 40 | 0.53 |
| D: the user's words only | 21 / 40 | 26 / 40 | 0.47 |

Reading: the fact the extractor wrote is what recall should search (A and B), not the raw
message it came from (C and D). The four questions that rank badly under every variant
are short questions with no name in them ("Where do I live?", "How old is my house?");
a word-match component for those is on the roadmap.

**The weighting study, 2026-10-26.** Seven formulas for combining meaning similarity with
the name boost were scored on the same 40 questions. Every formula that uses names tied
exactly (31 of 40 in the top 4); meaning alone lost five questions. The formula was left
as it was: the ratio makes no measurable difference on this set, and a change without a
measured gain is a change for its own sake.

**A note on 2026-10-28.** A run that day on a newer copy of the same database (648
memories, a day of use later) scored lower: 23 of 40 in the top 10, against 36 the day
before. The cause is not established; the obvious candidates are the new memories
themselves (a newer memory on the same subject outranking the expected older one, which
the case file would then need refreshing for) and the search. It is recorded here, unresolved,
so the 10-27 numbers are read with it in mind.

## 3. The fine-tuned phone model

**Where:** `labs/finetune/` on the `finetune` branch (scripts, data tools, results).

**Method.** The phone runs Qwen3-4B-Instruct-2507 compressed to 4 bits (Apache-2.0). A
LoRA adapter was trained on this desktop (one consumer GPU, about ten minutes a run) from
two sources: 933 of the project author's own messages, labelled by a stronger model
(Claude) using the exact prompt the phone sends, and 349 relationship cases written by the
same model, all filtered by the bench's own pass rules. The memory bench's cases were
never in the training data, so it is a held-out test. The adapter is merged into the base
model and compressed to the same 4-bit file the phone uses.

**Results, 2026-10-28**, phone prompt, all 128 bench cases:

| Model | Facts found (of 69) | Invented | Junk (of 26) | Relationships (of 44) | Extraction time | Relationship time |
|---|---|---|---|---|---|---|
| Base Qwen3-4B Q4_0 | 66 | 7 | 0 | 28 | 0.30 s | 0.99 s |
| mem3 (messages only) | 68 | 0 | 0 | 26 | 0.32 s | 0.21 s |
| mem4 (+ relationship cases) | **68** | **1** | **0** | **38** | 0.31 s | **0.21 s** |

The six relationship misses of mem4 are five `updates` called `contradicts` (the
signal-word distinction) and one `none` called `related`.

**What went wrong on the way, kept here because it is useful.** The first run (mem1)
trained on the whole text of each example, 90% of which is the same prompt; it learned the
prompt and found 9 facts of 69. The fix was to take the loss on the answer only. Runs two
to four then looked slow and broken inside the app (7 s a call against 0.3 s) while running
at full speed in llama.cpp's own tool; the cause was that the app chose the chat format by
the model's file name, and a file named `mem3-Q4_0.gguf` got a bare format with no chat
markers. The app now reads the architecture the file declares, and the export names files
after the base model. All of the "tuned model is worse" numbers before that fix were taken
under the wrong format.

**Status.** mem4 is a candidate for the phone, named **Sophro-4B-memory-v1** (from
*sophrosyne*, the Greek word for soundness of mind and self-restraint, which is the habit the
training is for: store what was said, nothing more). File `Sophro-4B-memory-v1-Q4_0.gguf`,
2.37 GB. It has not yet been used day to day; that test comes before any decision to ship it.

## 4. Running the benches

All commands run in `zynkbot_rust/src-tauri`. Each bench is an ignored Rust test; results
land as JSON and Markdown in the folder named by the `*_OUT` variable (default
`~/Documents/Zynk_related/<bench>/`).

```
# Memory bench: making and linking memories. Backends are provider names (keys from the
# app's settings file) or the path of a GGUF file. MEMORY_BENCH_SLIM=1 uses the phone's
# prompt; MEMORY_BENCH_SKIP_EXTRACTION=1 runs the relationship cases only.
MEMORY_BENCH_BACKENDS=anthropic,/path/to/model.gguf MEMORY_BENCH_SLIM=1 \
  LD_LIBRARY_PATH=lib/vosk cargo test --features cuda --lib memory_bench -- --ignored --nocapture

# Recall bench: finding memories, against a COPY of a database (never the live file).
RECALL_BENCH_DB=/path/copy.db LD_LIBRARY_PATH=lib/vosk \
  cargo test --features cuda --lib recall_bench -- --ignored --nocapture

# Weighting study on the same copy.
RECALL_BENCH_DB=/path/copy.db LD_LIBRARY_PATH=lib/vosk \
  cargo test --features cuda --lib weight_bench -- --ignored --nocapture
```

The fine-tune pipeline (export prompts, teacher answers, train, export, bench) is described
step by step in `labs/finetune/README.md` on the `finetune` branch.

## 5. Limits, stated plainly

- The memory and relationship cases are synthetic: written to cover the kinds of thing
  people say, including the junk kinds, not drawn from anyone's private life. 84 and 44
  cases are enough to rank models and wording against each other, not to put a confidence
  interval on a single number.
- The recall bench is one user's database and 40 questions, scored by where a known memory
  ranks. It says nothing yet about how often an answer was *right*; that needs a second
  bench with a human scorer.
- Every case has one expected answer written by one person. A disputed case (the
  relationship bench has two where the teacher model disagreed with the case) is kept as
  written and noted.
- The fine-tuned model's held-out test is the bench above; it has not been scored on a
  second, independent case set, and it has not been used in daily life yet.
- The project is open source. The benches, the cases and the scripts are in the repository
  under the project licence so that anyone can run them, extend the cases, or dispute a
  number. The one thing not in the repository is the recall bench's database copy, which
  is a real person's memories.
