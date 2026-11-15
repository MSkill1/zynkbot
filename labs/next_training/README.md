# The next training run of the phone model

Started 2026-11-15 (Matt). This folder collects what the next run of Zynkbot's own
model has to fix or add, with the evidence for each item, so that the run can be planned
from one page. The training pipeline itself lives in `labs/finetune/` on the `finetune`
branch (export the app's prompts, label them with a teacher model, filter by the bench's
rules, LoRA, merge, quantize, score). The model archive and the candidate list live in
`~/zynk-models` (private). Nothing large belongs here.

## Where the current model stands

Sophro-4B-memory-v1 (run mem4, 2026-10-28) is a LoRA fine-tune of Qwen3-4B-Instruct-2507,
shipped as the first-run download since 0.9.6. On the memory bench it finds 68 of 69
facts and gets 38 of 44 relationships right, against 66 and 28 for the base model
(`docs/BENCHMARKS.md`, sections 1 and 3). Standard benchmarks showed no loss from the
tuning. Its known faults are listed below.

## 1. Before anything is trained: the prompts changed

The extraction prompt gained three rules on 2026-11-12 and was reworded twice on
2026-11-15 (`memory_prompts.rs`; `docs/BENCHMARKS.md` section 1, results of 2026-11-15).
The training set was built from the previous prompt text. Step 1 of the pipeline
(`cargo test --lib finetune_export -- --ignored`) must be run again so the prompts in the
training data are the ones the phone now sends, and step 2 (`make_targets.py`) must
relabel every message under the new prompt. Training on the old prompt text would teach
the model to answer a prompt the app no longer uses.

If the nightly memory review (`docs/ROADMAP.md`, "Periodic memory review") ships before
the run, its prompt is a third job to export and label.

## 2. Extraction: what the model still gets wrong

Measured on the bench with the final 2026-11-15 wording, three runs each. The bench
cases are held out and must stay out of the training data; the training examples below
are to be written or drawn from real messages in the same shape, not copied from the cases.

- **An unnamed thing is not a fact.** "I do have that serum thing" stores "Matt has a
  serum thing in his shower" (case e86, junk in every run; the cloud model dropped it under
  the new wording, the phone model did not). On the phone on 2026-11-12 the real message
  produced "has serum thing" and "has new products". The fine-tune taught the model to keep
  every "has" line; the prompt cannot unteach it. Needed: examples where a message points
  at a thing with no name ("that stuff", "the new one", "the serum thing") and the target is
  NONE, and examples where the exchange before names it and the target uses the name.
- **A pronoun takes its name from the exchange before.** "He's fine, just needed his
  shots" after "How did Louie's vet visit go?" is about Louie (case e83). Examples: a
  short answer with "he", "she", "it", "they" or "there", a context line that names the
  thing, and a target that uses the name.
- **A plan is not a state.** "Indiana, to see my sister" after "Where are you going in
  April?" was stored as living in Indiana until the 2026-11-15 wording (case e84). Examples
  of answers to "where are you going", "when", "with whom" whose targets are plans marked
  passing, not facts about where the user lives.
- **One thing with its details is one fact.** A long rambling message (case e88, 60 words
  about getting up at 4:30 for the gym and an afternoon slump) comes out as two facts on the
  cloud model and as one or two on the phone. Decide which is wanted and label the
  training examples that way; today the bench expects one.
- **Something the user only mentions is not about the user.** "My neighbor was going on
  about some new air fryer thing" (case e87) is nothing to keep; the phone model now gets
  this one right but the pattern (a friend's, a neighbour's, a colleague's thing in passing)
  needs enough examples to hold.
- **Third person, stated as fact.** Targets start with the user's name and state the
  fact; never "Matt mentions that", "Matt says that", "Matt knows that" (seen from a
  3-billion-parameter model on the desktop on 2026-11-15: "Albert mentions Laurimar is
  Maria's daughter").

Real material: the user's own messages are the data source (933 were used last time).
The messages behind the deleted hair-care memories of 2026-11-12 (desktop ids 931 to 933)
are in the conversation history and are good negatives.

## 3. Relationships: 38 of 44, two habits

The phone model refuses to say `none` and reaches for `updates` when `related` is right
(`docs/BENCHMARKS.md`, section 1, "Relationship types"); the app works around it by
treating the phone's `updates` as a question to the user. The training set had 349
relationship cases from `make_rel_cases.py`. Next run: more `none` and `related` pairs
than `updates`, including pairs that share a name but not a subject (the same person,
two unrelated facts), and the bench's relationship section grows by the same kinds.

## 4. Candour, and the base model question

The product line (Matt, 2026-11-03, `docs/ROADMAP.md` "Phone model base"): Zynkbot
answers everything a free adult may ask, plainly, with the evidence. No moral preface, no
"it is important to note" caveats, no softened or substituted answer, no refusal of a
legal question; a false premise is corrected in a clause and then the question is
answered; uncertainty is stated as uncertainty. The two questions the models refuse today
(a suicide method, a drug recipe) stay refused. The opposite failure is also a failure:
the probe found one model stating "differences between racial groups" as fact, which is
not candour but a different substituted answer (`docs/BENCHMARKS.md` section 3).

What training can and cannot do here:

- Hedging and refusals on trained topics are trainable: track 2 of
  `labs/finetune/README.md` (behaviour) was planned for exactly this and waits on the
  values file and the evaluation suite (`labs/finetune/eval/suite.jsonl` holds the
  categories with a few examples; the content is Matt's to write). Each category needs
  questions, the answer wanted, and the answer not wanted.
- Positions baked into the base model are not. Qwen3 refuses Tiananmen and gives
  Beijing's wording on Taiwan and Xinjiang, and the memory fine-tune did not move them;
  "uncensored" Qwen variants keep them, since they are trained positions, not refusals.
  Leaving Qwen means a new base and the memory fine-tune run again on it. Candidates
  already fetched in `~/zynk-models/models`: gemma-4-E2B and E4B, Ministral-3-8B,
  Olmo-3-7B, Phi-4-mini, Qwen3-8B, Qwen3.5-9B. What decides it: speed on the OnePlus and
  the Pixel at Q4, the memory bench after the fine-tune, the candour probe, and the
  licence. The roadmap puts the decision before v1.0.
- The probe: `labs/sophro_eval/candour/ask.py`, 28 questions on race, politics, China,
  religion, sex and health plus the two refuse-worthy ones, each asked with and without
  Zynkbot's system prompt. Read the answers pair by pair; the automatic grade counts
  caveat sentences and misread the system prompt's effect last time.

## 5. Checks after the run

1. The memory bench, three runs, phone prompt (`MEMORY_BENCH_SLIM=1`), against the new
   GGUF; compare with section 1 of `docs/BENCHMARKS.md`.
2. The candour probe, read pair by pair.
3. The standard benchmarks (`labs/sophro_eval/run_all.sh`) for regressions.
4. On both phones: install, the first-run download checksum, speed on a 40-word message,
   and the hair-care message of 2026-11-12 typed again (expected: hair memories recalled,
   no "serum thing" stored).

## Open questions for Matt

- Keep Qwen3-4B as the base, or run the candidates? (Decides whether this is one run or a
  week of them.)
- The values file and the behaviour suite: who writes the first version, and when.
- One fact or two for a message like e88.
- Hindi and other languages (`docs/ROADMAP.md`, "A language switch"): in this run or later.
