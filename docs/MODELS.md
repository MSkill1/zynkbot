# Zynkbot Local Model Guide

This guide lists recommended local LLM models for offline inference with Zynkbot.

## A Note on Model Claims

Performance descriptions in this guide are based on published benchmarks from model creators and the community. Benchmarks measure specific curated datasets — your results on real conversations may differ. Claims like "best coding model" reflect benchmark rankings at a point in time and are open to interpretation. When in doubt, download and test the model yourself.

---

## Installation Directory

Download models to: `zynkbot_rust/src-tauri/models/user/`

Models will appear in the app's model selector after placing them in this directory.

---

## Recommended Models (Included in Installer)

**Local models vs. API models:** API models (Claude, GPT-4o, Grok) produce the most consistent results for memory extraction, contradiction detection, and long-form reasoning. Local models work well for conversation but vary in how precisely they follow Zynkbot's internal structured instructions — this affects how reliably memories are created and contradictions flagged. Results depend on the model, hardware, and question type. As open-source models continue to improve, so will local model performance in Zynkbot.

**How a local model is prompted.** Any model loaded from a `.gguf` file gets a shorter prompt than a cloud model: the last 8 messages of the conversation (each cut to 400 characters), up to 4 recalled memories (each cut to 300 characters), and a shorter opening. Cloud models get 40 messages and 30 memories. This is why a local model sometimes forgets something said ten messages ago that a cloud model would still have.

**Sophro.** Sophro-4B-memory-v1 is a fine-tune of Qwen3-4B-Instruct-2507 trained on memory extraction: store what was said, nothing more. Since 0.9.6 it is the Zynkbot model that a fresh install offers. Its results are in [BENCHMARKS.md](BENCHMARKS.md). It is being used day to day on the developer's devices before any decision to ship it; the installer still offers Qwen3-4B.

### The Zynkbot model: Sophro-4B-memory-v1 ⭐ **Recommended**
- **Size:** 2.4GB
- **File:** `Sophro-4B-memory-v1-Q4_0.gguf`, a Q4_0 fine-tune of Qwen3-4B-Instruct-2507 (Apache-2.0), published at https://huggingface.co/GenerallyTso/Sophro-4B-memory-v1
- **Use Case:** The model Zynkbot is built around. Phone-sized, fast, no key needed
- **How you get it:** The setup wizard offers to download it at first run. On a phone it is the only model offered. A download that is interrupted continues where it stopped, and the file is checked against its checksum before it is used. If you choose Later, the download stays on the Current Model page
- **Download by hand:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/GenerallyTso/Sophro-4B-memory-v1/resolve/main/Sophro-4B-memory-v1-Q4_0.gguf
  ```

The models below are alternatives. The setup wizard offers them on a computer, not on a phone.

### 1. Gemma 4 12B (Google)
- **Size:** 7.1GB
- **File:** `gemma-4-12b-it-Q4_K_M.gguf` (Apache-2.0), checksum `0a270ec9fe6b…`
- **Use Case:** The clear pick for a computer with about 10 GB of memory free: the closest a local model comes to the cloud models
- **On the benches (2026-11-09, BENCHMARKS.md section 5):** 24 of 25 reply cases; 67 of 69 facts found, 2 invented, no junk, 45 of 49 relationships. Its one miss is an "I've noted that" on a statement.
- **Download by hand:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/unsloth/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf -O gemma-4-12b-it-Q4_K_M.gguf
  ```

### 2. Gemma 4 E4B (Google)
- **Size:** 5.0GB
- **File:** `gemma-4-E4B-it-Q4_K_M.gguf` (Apache-2.0), checksum `85a896a04755…`
- **Use Case:** Good replies at half the size; the pick for an 8 GB machine
- **On the benches (2026-11-09, BENCHMARKS.md section 5):** 21 of 25; every statement and recall right; when a personal detail is not in the memories it asks for a web search rather than guessing. 65 of 69 facts, 2 invented, no junk, 41 of 49 relationships.
- **Download by hand:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf -O gemma-4-E4B-it-Q4_K_M.gguf
  ```

### 3. Qwen3.5 9B (Alibaba)
- **Size:** 5.7GB
- **File:** `Qwen3.5-9B-Q4_K_M.gguf` (Apache-2.0), checksum `03b74727a860…`
- **Use Case:** The Chinese entry: strong recall, never invents a missing detail
- **On the benches (2026-11-09, BENCHMARKS.md section 5):** 21 of 25; on three statements it talks about saving ("I'll remember that"), the Qwen habit Sophro shares. 64 of 69 facts, 3 invented, no junk, 35 of 49 relationships. Like every Qwen3 model it refuses Tiananmen and answers Taiwan and Xinjiang in Beijing's terms (section 3).
- **Download by hand:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf -O Qwen3.5-9B-Q4_K_M.gguf
  ```

### 4. Ministral 3 8B Instruct (Mistral, France)
- **Size:** 5.2GB
- **File:** `Ministral-3-8B-Instruct-2512-Q4_K_M.gguf` (Apache-2.0), checksum `33e7a72cf5e6…`
- **Use Case:** The European entry: warm, natural conversation
- **On the benches (2026-11-09, BENCHMARKS.md section 5):** 20 of 25; every statement and recall right, but on two of the four missing-detail cases it made up an answer (a price, a colour), and it stored a memory from 10 of the 26 messages that held none. Kept in the list so the difference can be seen; not the one to trust with a fact.
- **Download by hand:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF/resolve/main/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf -O Ministral-3-8B-Instruct-2512-Q4_K_M.gguf
  ```

The list changed on 2026-11-09. Until then it offered Qwen3 8B, DeepSeek R1 Distill Llama 8B and Llama 3.1 8B Lexi Uncensored V2; they still work if present, and their formats are still detected.

---

## Additional Models

### Small Models (< 3GB) - For Older Hardware

#### Phi-3 Mini 3.8B
- **Size:** 2.3GB
- **Strengths:** Microsoft's efficient model, good reasoning for size
- **Download:**
  ```bash
  wget https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf
  ```

#### TinyLlama 1.1B
- **Size:** 0.6GB
- **Strengths:** Extremely fast, runs on anything
- **Use Case:** Testing, very low-end hardware
- **Download:**
  ```bash
  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
  ```

---

### Medium Models (3-8GB) - Best Balance

#### Mistral 7B Instruct v0.2
- **Size:** 4.1GB
- **Strengths:** General purpose, well-balanced, popular
- **Download:**
  ```bash
  wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
  ```

#### OpenHermes 2.5 Mistral 7B
- **Size:** 4.1GB
- **Strengths:** Fine-tuned for helpful assistant behavior
- **Download:**
  ```bash
  wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
  ```

#### Llama 3.1 8B Instruct
- **Size:** 4.9GB
- **Strengths:** Meta's latest, strong general capabilities
- **Download:**
  ```bash
  wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
  ```

#### Neural Chat 7B v3.3
- **Size:** 4.1GB
- **Strengths:** Conversational, empathetic responses
- **Download:**
  ```bash
  wget https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF/resolve/main/neural-chat-7b-v3-3.Q4_K_M.gguf
  ```

---

### Large Models (8-16GB) - For Powerful Hardware

#### Qwen 2.5 14B Instruct
- **Size:** 8.3GB
- **Strengths:** Excellent coding, reasoning, multilingual
- **Download:**
  ```bash
  wget https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf
  ```

#### Qwen 2.5 32B Instruct (Q3 quantization)
- **Size:** ~14GB
- **Strengths:** Excellent reasoning, coding, and multilingual capabilities
- **Requirements:** 16GB+ RAM
- **Download:**
  ```bash
  wget https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q3_K_M.gguf
  ```

---

### Specialized Models

#### Code Llama 13B Instruct
- **Size:** 7.4GB
- **Strengths:** Specialized for code generation and completion
- **Download:**
  ```bash
  wget https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf
  ```

#### Nous Hermes 2 Yi 34B (Q3 quantization)
- **Size:** 15GB
- **Strengths:** Creative writing, roleplay, long context
- **Download:**
  ```bash
  wget https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF/resolve/main/nous-hermes-2-yi-34b.Q3_K_M.gguf
  ```

---

## Thinking Models (Chain-of-Thought)

Some models — currently **DeepSeek R1** and **Qwen3** — are "thinking models." Before generating a response they produce an internal reasoning block (`<think>...</think>`) where the model works through the problem step by step before committing to an answer. This can significantly improve accuracy on math, logic, and multi-step problems.

**How Zynkbot handles this:**

Zynkbot automatically strips the `<think>...</think>` block from the displayed response. The model still reasons internally — you just see the final answer. For most everyday use, the quality difference is minimal. The reasoning process has the most impact on hard math, logic puzzles, and multi-step problems.

- **Qwen3:** Chain-of-thought is also short-circuited at the prompt level, so responses are fast.
- **DeepSeek R1:** Reasoning runs but is hidden from the display. Responses may be slightly slower than Qwen3 as a result.

**Potential future improvement:**

An opt-in "deep reasoning" mode — where the model is given a larger token budget and the reasoning process is optionally visible — is something worth exploring if there is community interest.

---

## Model Format: GGUF

Zynkbot uses **GGUF format** models (via llama.cpp). These are:
- Optimized for CPU inference
- Quantized for smaller size and faster speed
- Compatible with llama.cpp-based runners

### Quantization Levels

- **Q2_K:** Smallest, lowest quality
- **Q3_K_M:** Small, decent quality
- **Q4_K_M:** ⭐ **Recommended** - Best balance of size/quality
- **Q5_K_M:** Larger, better quality
- **Q6_K:** Largest, near-original quality
- **Q8_0:** Almost lossless, very large

For most users, **Q4_K_M** is a good place to start 

---

## Hardware Requirements

### Minimum (3B models):
- **CPU:** Dual-core 2GHz+
- **RAM:** 4GB
- **Storage:** 2GB free

### Recommended (7B models):
- **CPU:** Quad-core 3GHz+ (AVX2 support)
- **RAM:** 8GB
- **Storage:** 10GB free

---

## Performance Tips

### 1. Use Appropriate Model Size
- Laptop/Mobile: 3B models
- Desktop: 7B-13B models
- Workstation: 13B-32B models

### 2. GPU Acceleration
If you have an NVIDIA GPU, Zynkbot will automatically use it for faster inference if CUDA is installed

### 3. Lower Quantization for Speed
If a model is too slow, try a lower quantization (Q3 or Q2) of the same model.

---

## Finding More Models

### HuggingFace Collections

Browse thousands of GGUF models:
- **TheBloke's Models:** https://huggingface.co/TheBloke (most popular)
- **Bartowski's Models:** https://huggingface.co/bartowski (latest releases)
- **LM Studio Models:** https://huggingface.co/lmstudio-community

### Search Tips

On HuggingFace, search for:
- Model name + "GGUF"
- Filter by "TheBloke" or "bartowski"
- Look for `Q4_K_M.gguf` files

---

## Using Downloaded Models

1. **Place in models directory:**
   ```bash
   mv model-name.gguf ~/zynkbot/zynkbot_rust/src-tauri/models/user/
   ```

2. **Restart Zynkbot** if already running

3. **Select model in UI:**
   - Click model dropdown in chat interface
   - Choose your downloaded model
   - Start chatting!

---

## API Models (No Download Needed)

For best quality responses without local downloads, use API models:

- **Claude (Anthropic)** - Best overall, most intelligent
- **GPT-4 (OpenAI)** - Strong general capabilities
- **Grok (xAI)** - Fast, conversational

Add API keys via: **⚙️ System Controls → Current Model**

---

## Troubleshooting

### Model doesn't appear in dropdown
- Verify file is in `models/user/` directory
- Ensure file ends with `.gguf`
- Restart Zynkbot

### Model loads but crashes
- Model may be too large for available RAM
- Try a smaller model or lower quantization (Q3/Q2)

### Model is very slow
- Use smaller model (3B instead of 7B)
- Use lower quantization (Q3 instead of Q4)
- Close other applications to free RAM
- Consider using API models instead

### Out of memory errors
- Model requires more RAM than available
- Try Q2 quantization or smaller parameter count
- Close other applications

---

## Model Recommendations by Use Case

### General Chat & Assistance
- Gemma 4 12B ⭐ (closest to the cloud models; needs the RAM)
- Gemma 4 E4B (the same family at half the size)
- Qwen3.5 9B (strong recall)

### Code Generation
- Qwen3 8B ⭐ (best coding model in 2026)
- Qwen 2.5 14B (advanced)
- Llama 3.1 8B (general + code)

### Creative Writing
- Ministral 3 8B ⭐ (warm, natural voice)
- Nous Hermes 2 Yi 34B (long stories)
- Neural Chat 7B (conversational)

### Reasoning & Analysis
- Gemma 4 12B ⭐ (best local relationship detection on the memory bench)
- Qwen3.5 9B (thinking model; Zynkbot turns the thinking off)
- Qwen 2.5 32B (advanced, needs 16GB+ RAM)

### Low-Resource Devices
- TinyLlama 1.1B (ultra-light)
- Phi-3 Mini 3.8B (strong reasoning for size)
- Qwen 2.5 7B Instruct (smallest recommended 7B — see Additional Models)

---

**Last Updated:** 2026-11-09
**For Issues:** https://github.com/MSkill1/zynkbot/issues
