# Fine-tuning the phone model (branch `finetune`, started 2026-10-28)

Goal, from `~/Documents/Zynk_related/ZYNKBOT_MODEL_TRAINING_PLAN.md` (2026-09-22) and
`labs/local_model/PLAN.md` step 13: a model Zynkbot ships that does Zynkbot's own jobs
well on a phone and answers honestly inside the user's own framework instead of a
vendor's defaults. Two tracks, in this order:

1. **Zynkbot's jobs** (measurable today): memory extraction, relationship
   classification and the reply with recalled memories, in the exact prompt formats the
   app sends. The memory bench (`labs/memory_bench`) and the recall bench score them.
2. **Behaviour** (needs Matt's values file and evaluation suite first, stage 3 of the
   training plan): no refusals or softening on honest questions, no substituted values,
   false premises corrected as qualifications and then answered.

## Base model

`Qwen/Qwen3-4B-Instruct-2507`, Apache-2.0, the same model the ZB download ships
(Q4_0 of it). Trainable weights in `~/zynkbot-finetune/base/Qwen3-4B-Instruct-2507`
(safetensors, bf16, about 8 GB).

## Machine

This desktop: RTX 3090, 24 GB. A LoRA on a 4B model in bf16 fits; QLoRA (4-bit base)
leaves more room for longer sequences. No rented GPU needed for the 4B.

## Layout (nothing large lives in the repo)

- `labs/finetune/` (this folder, committed): scripts, the evaluation suite, notes.
- `~/zynkbot-finetune/base/` base weights; `data/` prompts and targets (JSONL);
  `runs/` adapters, merged models, GGUF files; `tools/llama.cpp` for convert and
  quantize (`build/bin/llama-quantize`, `llama-completion`, `llama-perplexity`).
- Python: `~/.venvs/zynk-finetune` (torch, unsloth, trl, peft, datasets).

## Pipeline

1. **Export the app's own prompts** — `cargo test --lib finetune_export -- --ignored`
   (Rust, `src/finetune_export.rs`) writes `data/prompts_extraction.jsonl` and
   `data/prompts_relationships.jsonl` from the memory bench cases, using the same prompt
   builders the phone uses (slim extraction prompt, links-only prompt). The prompt text
   comes from the app, so training and production never drift.
2. **Targets** — `make_targets.py`: a teacher model (Claude through the Anthropic API,
   on Matt's key) answers each prompt; `check_targets.py` keeps only answers that pass the
   bench's own criteria (every `must` term present, NONE where nothing is expected, the
   expected relationship type). Rejected ones are listed for a human to write by hand.
3. **Train** — `train_lora.py`: Unsloth, LoRA r=16 on all linear layers, bf16 base, 2
   epochs, learning rate 2e-4, sequence length 2048, chat template from the tokenizer.
   Writes the adapter to `runs/<name>/adapter`.
4. **Export** — `export_gguf.sh <name>`: merge the adapter into the base, convert to
   GGUF (f16), quantize to Q4_0 (what the phone runs), print the sha256 and size.
5. **Score** — the memory bench against the new file: set `MEMORY_BENCH_BACKENDS` to the
   GGUF path and run `cargo test --lib memory_bench -- --ignored`. Compare with the
   base model's numbers in `~/Documents/Zynk_related/memory_bench/RESULTS_2026-10-24.md`.
   Then the phone: copy the GGUF to the Pixel's models folder and run the on-device
   questions.

## Evaluation suite for behaviour (`eval/`)

`eval/suite.jsonl` holds the categories from the training plan with a few examples each
to show the shape; the content is Matt's to write (values file, modes, the premise
cases). `run_eval.py` runs a suite through a GGUF with `llama-completion` and writes the answers
side by side for review; a judge step can be added once the suite is real.

## Rules

- Training data from Matt's own threads only with his say-so per export; never from
  anyone else's device. The bench cases are synthetic.
- A fine-tuned file ships only after the memory bench shows it at least matches the
  base model and the phone run is clean.
- The hidden Qwen `<think>` channel stays off: the 2507 Instruct model has none.

## Status

- **2026-10-28:** environment, base weights, tools and scripts in place. Pipeline proven
  end to end with a throwaway run (`runs/smoke`: 5 steps on 6 placeholder rows, 6 s of
  training; merge; f16 GGUF 8.05 GB; Q4_0 2.37 GB, the same size class as the ZB file;
  `llama-completion` answered from it). The smoke adapter is not a model: discard it.
  Prompts exported (84 extraction, 44 relationships). Teacher answers not yet made (the
  desktop had no cloud key that morning). Next: targets, then the first real run
  (`mem1`), then the memory bench against the base Q4_0 and `mem1-Q4_0.gguf` side by side.
- **2026-10-28, mem1 (first real run): worse than the base, and why.** 927 of Matt's own
  messages with teacher answers, LoRA r=16, 2 epochs. Bench on the phone's prompt
  (`MEMORY_BENCH_SLIM=1`): base Q4_0 found 66 of 69 expected facts, 7 invented, 0 junk,
  28/44 relationships, 0.44 s a case; mem1 found 9 of 69, 8/44, 5 s a case. Cause: the
  loss was taken over the whole text, and 90% of every example is the same 700-token
  prompt, so the model learned the prompt (given it raw, it recites it and stops) and
  not the answer. Fix in `train_lora.py`: prompt/completion rows, loss on the answer
  only (`completion_only_loss`). Also the data was half "no facts" rows; mem2 caps those
  at ~37%. The base model's bench numbers above are the bar every run has to clear.
