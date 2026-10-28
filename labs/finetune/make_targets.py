#!/usr/bin/env python3
"""Teacher answers for the exported prompts (2026-10-28).

Reads prompts_*.jsonl, asks Claude (Anthropic API, ANTHROPIC_API_KEY) each prompt exactly
as the phone would send it, and writes raw_*.jsonl with the answer added. Nothing is
filtered here; check_targets.py does that against the bench's own criteria.

  ANTHROPIC_API_KEY=... ~/.venvs/zynk-finetune/bin/python make_targets.py extraction
  ANTHROPIC_API_KEY=... ~/.venvs/zynk-finetune/bin/python make_targets.py relationships
"""
import json, os, sys, time, urllib.request
from pathlib import Path

DATA = Path.home() / "zynkbot-finetune" / "data"
MODEL = os.environ.get("TEACHER_MODEL", "claude-sonnet-5")


def ask(prompt: str) -> str:
    body = json.dumps({"model": MODEL, "max_tokens": 800, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        v = json.load(r)
    return "".join(b.get("text", "") for b in v.get("content", []) if b.get("type") == "text").strip()


def main():
    task = sys.argv[1]
    src = DATA / f"prompts_{task}.jsonl"
    dst = DATA / f"raw_{task}.jsonl"
    done = {}
    if dst.exists():
        for line in dst.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["id"]] = r
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} prompts, {len(done)} already answered, teacher {MODEL}")
    with dst.open("a") as f:
        for r in rows:
            if r["id"] in done:
                continue
            t0 = time.time()
            r["target"] = ask(r["prompt"]); r["teacher"] = MODEL
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"  {r['id']} {time.time() - t0:4.1f}s {r['target'][:90].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
