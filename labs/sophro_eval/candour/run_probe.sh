#!/usr/bin/env bash
# Candour probe (2026-11-03, Matt: "test it now to see how honest it is about controversial
# questions"). Each question in questions.txt goes to each model twice: with Zynkbot's own
# system prompt (the "As Zynkbot, be warm, calm, direct" block the app sends) and with no
# system prompt. Temperature 0, 350 tokens. Answers are saved whole; a rough classifier
# marks refusals and hedges, and a person reads the rest. Usage: ./run_probe.sh [out.md]
set -u
cd "$(dirname "$0")"
SERVER=/home/matt/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2/llama-server
PORT=8081
OUT=${1:-$HOME/Documents/Zynk_related/candour_probe_$(date +%F).md}
declare -A MODELS=(
  [base]=/home/matt/store/Zynk_related_models/Qwen3-4B-Instruct-2507-Q4_0.gguf
  [sophro]=/home/matt/store/loose_models/Sophro-4B-memory-v1-Q4_0.gguf
  [lexi]=/home/matt/store/zynkbot-models/user/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf
)
mkdir -p results
for name in base sophro lexi; do
  pkill -x llama-server 2>/dev/null; sleep 3
  nohup "$SERVER" -m "${MODELS[$name]}" -ngl 99 -c 8192 -np 4 --host 127.0.0.1 --port $PORT --no-jinja > "results/server_$name.log" 2>&1 &
  for _ in $(seq 1 90); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 2; done
  echo "=== $name $(date +%H:%M)"
  python3 -I ask.py "$name" "http://127.0.0.1:$PORT/v1/chat/completions" questions.txt "results/$name.json"
done
pkill -x llama-server 2>/dev/null
python3 -I report.py "$OUT" results/base.json results/sophro.json results/lexi.json
echo "=== done $(date +%H:%M): $OUT"
