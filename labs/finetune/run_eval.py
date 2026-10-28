#!/usr/bin/env python3
"""Run an evaluation suite through a GGUF with llama-completion (llama.cpp) and write the answers beside
the ideal ones for review (2026-10-28). A judge step can be added once the suite is real.

  ~/.venvs/zynk-finetune/bin/python run_eval.py --gguf ~/zynkbot-finetune/runs/mem1/mem1-Q4_0.gguf \
      --suite eval/suite.jsonl --out ~/zynkbot-finetune/runs/mem1/eval_suite.md
"""
import argparse, json, subprocess, time
from pathlib import Path

LLAMA_CLI = Path.home() / "zynkbot-finetune" / "tools" / "llama.cpp" / "build" / "bin" / "llama-completion"


def answer(gguf: str, prompt: str, n_predict: int = 400) -> str:
    cmd = [str(LLAMA_CLI), "-m", gguf, "-p", prompt, "-n", str(n_predict), "--temp", "0", "-no-cnv",
           "--no-display-prompt", "-ngl", "99", "--log-disable"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--suite", default=str(Path(__file__).parent / "eval" / "suite.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.suite).read_text().splitlines() if l.strip()]
    lines = [f"# Eval: {Path(args.gguf).name} on {Path(args.suite).name} ({time.strftime('%Y-%m-%d %H:%M')})\n"]
    for r in rows:
        t0 = time.time()
        got = answer(args.gguf, r["prompt"])
        lines.append(f"## {r['id']} ({r['category']}, {time.time() - t0:.0f}s)\n\n**Prompt:** {r['prompt']}\n\n**Ideal:** {r['ideal']}\n\n**Got:**\n\n{got}\n")
        print(r["id"], f"{time.time() - t0:.0f}s")
    Path(args.out).write_text("\n".join(lines))
    print("->", args.out)


if __name__ == "__main__":
    main()
