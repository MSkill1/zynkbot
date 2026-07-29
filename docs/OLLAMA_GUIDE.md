# Ollama Setup Guide

Ollama lets you run open-source LLMs locally — no API key, no subscription, no data leaving your machine. Zynkbot treats it as a first-class backend alongside Claude, GPT-4, and Grok.

---

## Desktop Setup

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com). Ollama runs as a background service on port `11434`.

### 2. Pull a model

From a terminal:

```bash
ollama pull llama3.1:8b
```

Or pull directly from within Zynkbot (see step 3). Recommended models by hardware:

| Hardware | Model | Notes |
|---|---|---|
| 8–16 GB RAM | `llama3.1:8b` | Good general-purpose baseline |
| 16–32 GB RAM | `qwen2.5:14b` | Better reasoning, still fast |
| 32 GB+ RAM | `qwen2.5:32b` | Near API-model quality |
| Any (fast) | `qwen2.5:7b` or `phi3:mini` | Low-latency for quick questions |

GPU acceleration (NVIDIA CUDA) is automatically used by Ollama when available — responses go from ~10 tokens/sec to 50–100+.

### 3. Configure in Zynkbot

1. Open **Settings → API Keys**
2. Scroll to **Custom / Ollama**
3. Enter the URL: `http://localhost:11434/v1`
4. Leave the API key field blank — Ollama doesn't require one
5. Click **Fetch Models** — Zynkbot will list available models
6. Select your model from the model picker

To pull a new model without leaving Zynkbot: type the model name in the pull field and click **Pull**. This runs `ollama pull <name>` in the background and streams progress.

---

## Android — Use Your Desktop's Ollama

When your Android phone is paired with your desktop via ZynkSync, the desktop acts as an Ollama proxy over your home network. Your phone sends chat requests to the desktop, the desktop queries its local Ollama, and the response streams back — no cloud involved. Android cannot run Ollama locally in Phase 1 — this proxy is the only way to use Ollama on Android. ZynkSync must be paired and running on the desktop for this to work; if the desktop shows as offline in ZynkSync, the connection will fail.

### Setup

1. Make sure Ollama is running and configured on your desktop (steps above)
2. Pair your phone and desktop via **Settings → ZynkSync** on both devices
3. On Android, open **Settings → API Keys → Ollama (Local AI)**
4. Tap **"Connect to Ollama on [your PC name]"** — Zynkbot auto-detects paired desktops
5. Select the model from the picker

Both devices must be on the same WiFi network for this to work.

---

## Other OpenAI-compatible Servers

Zynkbot's custom endpoint works with any server that implements the OpenAI chat completions API:

| Server | Default URL |
|---|---|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| llama-server (llama.cpp) | `http://localhost:8080/v1` |

Enter the URL in Settings → API Keys → Custom / Ollama. Leave the API key blank unless your server requires one.

---

## Troubleshooting

**"Connected but no models found"** — Pull a model first: `ollama pull llama3.1:8b`

**"Can't reach Ollama"** — Ollama may not be running. Start it with `ollama serve` or check that the background service is active.

**Android can't connect to desktop Ollama** — Verify both devices are on the same network and that ZynkSync is paired and running on the desktop.

**Slow responses** — Ollama defaults to CPU inference if no GPU is detected. Response speed depends heavily on hardware. API models (Claude, GPT-4) are faster for most consumer hardware.
