# model_archive

Tools for building a **verifiable, offline mirror of open-weight language models** on
hardware you control. Measured feasibility analysis, tier sizing and the recommended
staging plan live in [`docs/MODEL_ARCHIVE.md`](../../docs/MODEL_ARCHIVE.md).

The premise matches Zynkbot's: a local model you hold is a model nobody can take away.
These scripts are the acquisition side of that — they get the weights onto your disk and
prove, cryptographically and repeatedly, that they are still intact.

## Why checksums are the whole point

The Hub exposes a SHA256 for every LFS file. An archive without verification is a rumour:
silent corruption, truncated transfers and bit rot are what actually destroy decade-scale
archives, and all three are invisible until the day you need the file. Every script here
is built around the manifest.

## Usage

```bash
# 1. Build a manifest: path, size and SHA256 for every file in every repo.
#    EXCLUDE drops duplicate formats - Llama's original/ dir alone is 1.6 TB.
EXCLUDE='^(onnx|original)/' ./hf_manifest.sh tiers/frontier.txt > frontier.manifest

# 2. See what it will cost before committing.
./plan.sh frontier.manifest                  # defaults: $30/raw TB, x1.4 parity, 100 MB/s
./plan.sh frontier.manifest 42 1.5 250       # your own drive price, parity and link

# 3. Pull it. Resumable, idempotent, verifies every byte.
./mirror.sh frontier.manifest /srv/models

# 4. Scrub on a schedule. This is not optional.
./verify.sh frontier.manifest /srv/models
```

Set `HF_TOKEN` always — it raises the rate limit from ~3,000 to ~5,000 resolver requests
per 5-minute window, and it is **required** for gated repos:

```bash
export HF_TOKEN=hf_...
```

`meta-llama/*` is `gated: manual`, meaning a human approves your request. Pull gated
weights first; that approval is the only part of this with a door that can close.

## Tiers

| File | Contents | Size |
|---|---|---|
| `tiers/frontier.txt` | 29 frontier models, all jurisdictions | 15.79 TB as published, ~14.2 TB filtered |
| `tiers/runnable.txt` | the subset that runs on hardware you can own | 0.98 TB |
| `tiers/runnable.manifest` | generated manifest for the above, 471 files | — |

`runnable.txt` is the one to start with. Everything in it runs at 4-bit on a single 24 GB
GPU or a 128 GB workstation. The frontier tier is insurance; this tier is capability.

## Measured data

`data/frontier_sizes.tsv` — per-repo measured bytes, file count, parameter count, licence
and gating status, as of 2026-10-09. Regenerate by re-running `hf_manifest.sh`; model
repos change and these figures will drift.

## Notes and limits

- **Rate limits, not bandwidth, bind at scale.** Above ~10,000 repos the request count
  becomes the constraint. The fetcher backs off on 429 with exponential delay. Do not
  rotate tokens to evade limits; it violates the Hub's terms.
- **One upstream is one chokepoint.** For Chinese models, consider mirroring ModelScope
  as a second source.
- **Weights alone are inert.** Archive the runtime too — pinned inference engine builds,
  CUDA installers, tokenizers. Covered in the plan doc, stage 4.
- **Licences restrict redistribution, not retention.** ~68% of the popular set is
  Apache-2.0 or MIT. Bespoke "other" licences need reading before you re-serve anything.
- `plan.sh` cost figures assume bare drives only, and the Oct-2026 market is supply
  constrained — nearline inventory is booked into 2027–2028.
