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

### 1. Qwen3 8B ⭐ **Best All-Around — Recommended Starting Point**
- **Size:** 5.0GB
- **Speed:** Fast on modern CPUs; significantly faster with NVIDIA GPU + CUDA
- **Use Case:** General conversation, memory-augmented chat, code, technical writing
- **Strengths:** Tops HumanEval in the 7-8B class (2026); excellent instruction-following; multilingual; designed with explicit think/no-think modes that work reliably in Zynkbot's structured prompting format
- **Download:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf -O Qwen3-8B-Q4_K_M.gguf
  ```

### 2. DeepSeek R1 Distill Llama 8B ⭐ **Reasoning Model — Analytical Tasks**
- **Size:** 4.7GB
- **Speed:** Moderate on CPU — generates more tokens due to chain-of-thought reasoning
- **Use Case:** Logic, analysis, multi-step problems, math, anything where accuracy matters more than speed
- **Strengths:** MIT license; distilled from DeepSeek R1, which matched GPT-4 on reasoning benchmarks; one of the most talked-about open-source models of 2025
- **Note:** DeepSeek R1 Distill was trained to reason out loud (chain-of-thought). Zynkbot suppresses the thinking output to keep responses concise. This works well for direct questions but may produce less thorough answers than models designed with explicit think/no-think modes like Qwen3. Best for analytical and math-heavy tasks; Qwen3 is the better default for general conversation and memory accuracy.
- **Download:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf -O DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
  ```

### 3. Llama 3.1 8B Lexi Uncensored V2 ⭐ **Creative / Unfiltered**
- **Size:** 4.9GB
- **Speed:** Fast on modern CPUs
- **Use Case:** Creative writing, roleplay, gray-area topics, unfiltered responses
- **Strengths:** No content filtering; widely regarded as the top uncensored 8B model in 2026
- **Memory quality note:** Uncensored fine-tunes like Lexi prioritize response freedom over instruction-following precision. In testing, Lexi's memory extraction tended to produce broad summaries of known context rather than isolating the specific new fact from a message. Conversations still work well, but memory entries may be less precise than with Qwen3 or DeepSeek. If accurate long-term memory is a priority, Qwen3 is the better choice. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — KI-007.
- **Download:**
  ```bash
  cd ~/zynkbot/zynkbot_rust/src-tauri/models/user
  wget https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf -O Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf
  ```

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
If you have an NVIDIA GPU, Zynkbot will automatically use it for faster inference.

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

Add API keys via: **⚙️ Settings → API Keys**

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
- DeepSeek R1 Distill 8B (reasoning, accuracy)
- Llama 3.1 8B Lexi Uncensored (quality, unfiltered)
- Qwen3 8B (balanced, instruction-following)

### Code Generation
- Qwen3 8B ⭐ (best coding model in 2026)
- Qwen 2.5 14B (advanced)
- Llama 3.1 8B (general + code)

### Creative Writing
- Llama 3.1 8B Lexi Uncensored ⭐ (unfiltered, vivid)
- Nous Hermes 2 Yi 34B (long stories)
- Neural Chat 7B (conversational)

### Reasoning & Analysis
- DeepSeek R1 Distill 8B ⭐ (chain-of-thought, shows its work)
- Qwen3 8B (strong logic)
- Qwen 2.5 32B (advanced, needs 16GB+ RAM)

### Low-Resource Devices
- TinyLlama 1.1B (ultra-light)
- Phi-3 Mini 3.8B (strong reasoning for size)
- Qwen 2.5 7B Instruct (smallest recommended 7B — see Additional Models)

---

**Last Updated:** 2026-06-01
**For Issues:** https://github.com/MSkill1/zynkbot/issues
