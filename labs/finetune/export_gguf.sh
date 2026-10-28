#!/bin/bash
# Merge a LoRA adapter into the base, convert to GGUF and quantize to Q4_0 (the phone's
# format). Usage: export_gguf.sh <run name>   (2026-10-28)
set -euo pipefail
NAME="${1:?run name}"
HOME_FT=/home/matt/zynkbot-finetune
PY=/home/matt/.venvs/zynk-finetune/bin/python
LLAMA=$HOME_FT/tools/llama.cpp
RUN=$HOME_FT/runs/$NAME
BASE=${BASE:-$HOME_FT/base/Qwen3-4B-Instruct-2507}

echo "=== merging $RUN/adapter into $BASE"
$PY - "$BASE" "$RUN" <<'PYEOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, run = sys.argv[1], sys.argv[2]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, f"{run}/adapter").merge_and_unload()
m.save_pretrained(f"{run}/merged", safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(f"{run}/merged")
print("merged ->", f"{run}/merged")
PYEOF

echo "=== converting to GGUF f16"
# The file name carries the base model's name: the app chose the chat format by file
# name until 2026-10-28, and a file called mem3-Q4_0.gguf got the generic format.
OUT="$RUN/Qwen3-4B-$NAME-Q4_0.gguf"
$PY "$LLAMA/convert_hf_to_gguf.py" "$RUN/merged" --outtype f16 --outfile "$RUN/$NAME-f16.gguf"
echo "=== quantizing to Q4_0"
"$LLAMA/build/bin/llama-quantize" "$RUN/$NAME-f16.gguf" "$OUT" Q4_0 > "$RUN/quantize.log" 2>&1
ls -la "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
# The merged weights and the f16 file are 16 GB per run and can be remade from the
# adapter; keeping them filled the disk on the fourth run (2026-10-28).
rm -rf "$RUN/merged" "$RUN/$NAME-f16.gguf" "$RUN/checkpoints"
echo "=== kept: adapter + Q4_0 ($(du -sh "$RUN" | cut -f1))"
