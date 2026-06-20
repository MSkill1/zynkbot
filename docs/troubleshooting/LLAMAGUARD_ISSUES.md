# LlamaGuard - Why It Was Removed

> **For Contributors:** This document explains why LlamaGuard was completely removed from Zynkbot's safety system. It exists to prevent wasting time re-implementing a feature that was thoroughly tested and failed. If you're considering adding LlamaGuard support, read this entire document first.

## TL;DR
**LlamaGuard has been completely removed from the codebase.** It was tested multiple times and consistently failed. DO NOT re-implement it.

## Date: January 8, 2026
**Status:** Attempted re-enablement, failed again (see commit history)

## Problems Encountered

### 1. **Parsing Failures - Blocks Everything**
```
User: "tell me how to write a children's book"
LlamaGuard: [BLOCK] (Detected: )  ← EMPTY category = parsing failed
```

- LlamaGuard GGUF outputs malformed responses
- Parser expects "safe" or "unsafe\nS1\nS2"
- Instead gets gibberish or incomplete tokens
- Falls back to blocking everything with no reason

### 2. **10 Second Delay Per Message**
- LlamaGuard is a full LLM (1B parameters)
- Runs inference on EVERY message in Guardian/Sovereign mode
- CPU inference: 8-10 seconds per check
- **UNUSABLE for interactive chat**

### 3. **Double BOS Token Issue**
See commit `306e55f`: "CRITICAL - LlamaGuard double BOS token causing garbage output"
- GGUF quantization has token formatting issues
- Generates malformed output
- Has been debugged multiple times, never fully fixed

## Why We Keep Trying (And Why It Fails)

**The Appeal:**
- LlamaGuard has 11 safety categories (vs toxic-bert's 6)
- Purpose-built for AI safety
- Detects weapons, self-harm, child exploitation

**The Reality:**
- Quantized GGUF version is broken
- Would need full fp16 model (4GB+ memory)
- Still 10 seconds per check
- Not worth it

## Current Solution (What We Use Instead)

**LlamaGuard has been completely removed.** Here's what Zynkbot uses now:

### Guardian/Sovereign Modes:
- **toxic-bert** (6 categories, fast on-device inference)
- Fast, reliable, good enough
- Users selecting these modes aren't trying to jailbreak
- Location: `src-tauri/src/lib.rs` (Candle Safety module)

### Child Mode:
- **OpenAI Moderation API** (11 categories, network round-trip)
- Accurate, fast, professionally maintained
- Worth the API cost for children's safety
- Location: `src-tauri/src/containment.rs`

### HIPAA Mode:
- **toxic-bert + PHI regex** (fast on-device inference)
- Pre-LLM PHI detection and blocking
- Documentation explains limitations
- Users responsible for their own compliance
- Location: `src-tauri/src/containment.rs`

## If Someone Tries To Re-Add LlamaGuard

**Before spending days on this, ask yourself:**
1. Did you read this entire document?
2. Have you tested it on benign queries like "write a children's book"?
3. Did you measure the actual response time (not just inference time)?
4. Did you verify it outputs valid "safe"/"unsafe" responses consistently?
5. Why not just use toxic-bert (fast, works) or OpenAI API (accurate, maintained)?

**Seriously, don't waste days debugging this again. It was removed for good reasons.**

## Related Issues
- Commit `306e55f` - Double BOS token
- Commit `3a38eb1` - Attempted re-enablement (Jan 8 2026)
- Commit `efc4f72` - Reverted (Jan 8 2026)

## Alternative: Async Background Check (Future)
If we REALLY want LlamaGuard:
- Run check asynchronously (don't block UI)
- Only flag in logs, don't block message
- Use for audit trail, not enforcement
- Still probably not worth it
