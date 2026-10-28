#!/usr/bin/env python3
"""LoRA fine-tune of the phone model on Zynkbot's own prompts (2026-10-28).

Input: one or more JSONL files, each line {"prompt": "...", "target": "...", "id": "..."}.
The prompt is sent as the user turn and the target as the assistant turn through the
tokenizer's chat template, which is what the app's engine does with the same text.

  ~/.venvs/zynk-finetune/bin/python train_lora.py --name mem1 \
      --data ~/zynkbot-finetune/data/targets_extraction.jsonl \
             ~/zynkbot-finetune/data/targets_relationships.jsonl
"""
import argparse, json, os, random, time
from pathlib import Path

HOME = Path.home() / "zynkbot-finetune"


def load_rows(paths):
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("prompt") and r.get("target"):
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--base", default=str(HOME / "base" / "Qwen3-4B-Instruct-2507"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="for a smoke test")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA: 4-bit base")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from unsloth import FastLanguageModel  # noqa: E402 (must come before trl/transformers)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = load_rows(args.data)
    random.Random(args.seed).shuffle(rows)
    print(f"{len(rows)} training rows from {len(args.data)} file(s)")
    out = HOME / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq, dtype=None, load_in_4bit=args.load_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )

    # Prompt/completion rows: the loss is taken on the answer only. The first run (mem1,
    # 2026-10-28) trained on the whole text, so 90% of every example was the same 700-token
    # prompt; the model learned the prompt and forgot the job (9 of 69 bench facts found).
    # The wrapper is the one the app sends (llm/local_models.rs build_chatml_prompt with
    # no_think): user turn, then an assistant turn opened with an empty think block. mem2 was
    # trained on the tokenizer's plain template and did not stop inside the app (2026-10-28).
    def to_pair(r):
        prompt = f"<|im_start|>user\n{r['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        return {"prompt": prompt, "completion": r["target"] + "<|im_end|>"}

    ds = Dataset.from_list(rows).map(to_pair, remove_columns=[c for c in rows[0].keys()])
    print("example completion:", ds[0]["completion"][:160])

    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.lr, warmup_ratio=0.05,
        lr_scheduler_type="cosine", logging_steps=5, save_strategy="no",
        bf16=True, optim="adamw_8bit", seed=args.seed, report_to="none",
        max_length=args.max_seq, completion_only_loss=True,
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)
    t0 = time.time()
    stats = trainer.train()
    print(f"trained in {time.time() - t0:.0f}s: {stats.metrics}")
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    (out / "train_info.json").write_text(json.dumps({
        "name": args.name, "base": args.base, "rows": len(rows), "data": args.data,
        "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "max_steps": args.max_steps,
        "load_4bit": args.load_4bit, "metrics": stats.metrics, "when": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))
    print("adapter saved to", out / "adapter")


if __name__ == "__main__":
    main()
