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
$PY "$LLAMA/convert_hf_to_gguf.py" "$RUN/merged" --outtype f16 --outfile "$RUN/$NAME-f16.gguf"
echo "=== quantizing to Q4_0"
"$LLAMA/build/bin/llama-quantize" "$RUN/$NAME-f16.gguf" "$RUN/$NAME-Q4_0.gguf" Q4_0 > "$RUN/quantize.log" 2>&1
ls -la "$RUN/$NAME-Q4_0.gguf"
sha256sum "$RUN/$NAME-Q4_0.gguf" | tee "$RUN/$NAME-Q4_0.sha256"
