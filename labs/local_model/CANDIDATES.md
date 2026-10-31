# On-device model: how candidates are found, and the first list

*2026-10-22. Research note for the `local-model` branch. Nothing here is decided.*

## Target

A phone model for Zynkbot's own jobs: deciding what in a chat message is worth
remembering and pulling it out as facts (the memory-decision bench), plain chat when no
cloud key is set, and the Elder Mode assistant. Test phones: Pixel 10 Pro XL (16 GB RAM,
Tensor G5) and a OnePlus. A 4B-parameter model at 4-bit is about 2.5 GB on disk and in
RAM; a 2B about 1.5 GB. Android kills background apps holding that much, so the model is
loaded on demand and dropped after a few idle minutes (ROADMAP, Android background power).

## How the search is done

Memory of model names is not a method. Three sources, in this order, repeatable:

1. **Hugging Face API, by publisher.** `https://huggingface.co/api/models?author=<org>&pipeline_tag=text-generation&sort=createdAt&direction=-1`
   for the publishers that ship small instruct models (google, Qwen, LiquidAI,
   meta-llama, microsoft, HuggingFaceTB, ibm-granite, allenai, nvidia, mistralai), then
   `?search=<family>` for each family, keeping instruct models of 5B or less and dropping
   base, vision, audio, coder, math and quantised copies. Each hit's `config.json` gives
   the architecture; the model page's tags give the licence and whether it is gated.
2. **Hugging Face API, by downloads with the `gguf` filter** (`filter=gguf&sort=downloads`):
   what people actually run on CPUs. Noisy (fine-tunes, "uncensored" copies), but a
   family that is missing here has no community behind it.
3. **Can llama.cpp run it?** The architecture must appear in `src/llama-arch.cpp` of the
   llama.cpp bundled in the `llama-cpp-2` crate Zynkbot builds (0.1.133 today; 0.1.160 is
   the newest, 2026-10-15). A model the engine cannot load is not a candidate, whatever its
   scores.

Then the bench decides, not the leaderboards: the memory-extraction set (ROADMAP, decided
2026-10-07) scored on the desktop first, then speed and memory on the two phones.

## First list (2026-10-22)

| Model | Size | Licence | Architecture | In crate 0.1.133 | Note |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 4B | Apache-2.0 | qwen3_5 | **no** | 7.6M downloads; the strongest small general model by uptake |
| Qwen/Qwen3.5-2B | 2B | Apache-2.0 | qwen3_5 | **no** | same family, half the memory |
| google/gemma-4-E4B-it | ~4B effective | Apache-2.0 (per HF tags) | gemma4 | **no** | 4.0M downloads; QAT 4-bit variants published by Google |
| google/gemma-4-E2B-it | ~2B effective | Apache-2.0 (per HF tags) | gemma4 | **no** | 2.8M downloads |
| LiquidAI/LFM2.5-2.6B | 2.6B | LFM licence ("other") | lfm2 | yes | built for phones; 680k downloads of the GGUF |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | LFM licence ("other") | lfm2 | yes | smallest serious option |
| ibm-granite/granite-4.2-3b | 3B | Apache-2.0 | granite | yes | enterprise-trained, strong on structured output claims |
| HuggingFaceTB/SmolLM3-3B | 3B | Apache-2.0 | smollm3 | yes | fully open training recipe |
| Qwen/Qwen3-4B | 4B | Apache-2.0 | qwen3 | yes | the previous generation; the fallback if the crate is not bumped |

Dropped on sight: anything gated, the "Qwen3.8" and "Spark" repos (third-party distils
of unclear origin), models over 5B (Qwen3-8B and Qwen3.5-9B from the roadmap note stay as
desktop reference points only, not phone candidates).

## What this means for the engine

The two strongest families need a newer llama.cpp than the crate Zynkbot pins. Bumping
`llama-cpp-2` from 0.1.133 to 0.1.160 is part of the Android engine work, and it touches
the desktop build too, so it is checked there first.
