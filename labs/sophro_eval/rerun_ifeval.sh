#!/usr/bin/env bash
# IFEval again for one model with the server's chat-template output parser off (--no-jinja;
# it is on by default): on 2026-11-02 Sophro fell into word repetition on five of the Hindi/other-
# language prompts, the parser rejected the text with a 500, and lm_eval gave up at
# 410/541. Without the parser the text comes back and is scored as what it is: a miss.
# Usage: ./rerun_ifeval.sh sophro|base
set -u
cd "$(dirname "$0")"
SERVER=/home/matt/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2/llama-server
PORT=8081
declare -A MODELS=(
  [base]=/home/matt/store/Zynk_related_models/Qwen3-4B-Instruct-2507-Q4_0.gguf
  [sophro]=/home/matt/store/loose_models/Sophro-4B-memory-v1-Q4_0.gguf
)
name=$1
pkill -x llama-server 2>/dev/null; sleep 3
nohup "$SERVER" -m "${MODELS[$name]}" -ngl 99 -c 8192 -np 4 --host 127.0.0.1 --port $PORT --no-jinja > "results/server_${name}_ifeval2.log" 2>&1 &
for _ in $(seq 1 60); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 2; done
echo "=== $name: ifeval (no output parser) $(date +%H:%M)"
./venv/bin/lm_eval --model local-chat-completions --apply_chat_template --output_path results/${name}_ifeval2 --log_samples \
  --model_args "model=$name,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=4,max_retries=3" \
  --tasks ifeval --gen_kwargs 'temperature=0' 2>&1 | tee "results/${name}_ifeval2.log" | grep -E '^\|'
pkill -x llama-server 2>/dev/null
echo "=== done $(date +%H:%M)"
