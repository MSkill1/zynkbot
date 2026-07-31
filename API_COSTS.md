# Honest API Cost Estimates

> **⚠️ Speculative estimates.** Prices change frequently. Always check each
> provider's current pricing page and watch your usage dashboard during your
> first month.

Zynkbot uses a **bring-your-own-key** model — instead of paying $20/month per
provider, you pay per use. For most people this is significantly cheaper, and
you get all four providers in one app with your memory staying local.

## Estimated monthly cost by usage pattern

| Usage pattern | Messages/month | Est. cost |
|---|---|---|
| Light (a few exchanges a day) | 100–150 | $1.50–4 |
| Moderate (real daily use) | 400–600 | $5–15 |
| Heavy (hours daily, long conversations) | 1,500+ | $25–60+ |

> **Honest caveat:** Heavy users on top-tier models may exceed subscription
> pricing. Subscriptions are flat-rate; APIs are metered. Light and moderate
> users usually win with APIs.

**Comparison:** Two subscription apps (Claude + ChatGPT) = **$40/month**, two
separate memories, no Grok, no Mistral. Zynkbot moderate use ≈ **$5–15/month**, all four
providers, one memory graph on your device.

## What one exchange actually costs

Each Zynkbot message is heavier than a bare chat message because the memory
system retrieves relevant context before sending:

- System prompt + retrieved memories + conversation history + your message → ~1,500–3,000 input tokens
- Model reply → ~300–800 output tokens
- Small memory processing calls (what to store, contradiction detection)

**Total: ~2,500–4,000 tokens per exchange.** Mid-tier models
(Claude Sonnet / GPT-4o-class) cost roughly **$0.01–0.03 per exchange**.
Budget tiers (Haiku / GPT-4o-mini) are often under a cent.

## Ensemble mode multiplier

Ensemble queries multiple models and synthesizes answers — roughly **4–5× the
cost** of a normal exchange (~$0.05–0.15 per question on mid-tier models). Use
it for questions where accuracy matters, not as a default.

## Local models cost $0

Local GGUF models running on your hardware have no per-message cost. The
tradeoff is speed and capability. Mixing modes (local for casual use, API for
harder questions) is a legitimate cost strategy.

## Practical tips

- **Set spending limits** in each provider's console on day one — caps
  worst-case to a fixed dollar amount.
- **Default to mid- or budget-tier models**; switch to top-tier only when a
  question deserves it.
- **Start new conversations when the topic changes** — message #40 in a long
  thread carries the full context of the previous 39, so it costs several times
  what message #2 costs.
- **Use Ensemble deliberately**, not habitually.
- **Use local models for casual chatter** if your hardware supports them.

---

*Estimates written mid-2026. Token prices have historically fallen over time.
If your real costs differ meaningfully, [open an issue](https://github.com/MSkill1/zynkbot/issues) —
real user data beats estimation.*
