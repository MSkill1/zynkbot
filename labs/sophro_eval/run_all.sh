#!/usr/bin/env bash
# Regression check for Sophro against its base model on the standard benchmarks:
# IFEval (instruction following), GSM8K (arithmetic, 8-shot chain of thought, Llama
# prompt variant) and MMLU (knowledge, generative form, N questions per subject).
# Both models are the exact Q4_0 GGUF files the app ships, served by llama.cpp on the
# GPU, so the numbers describe what runs on the device, not a float checkpoint.
# Usage: ./run_all.sh [mmlu_per_subject]   (default 10; the full MMLU is ~6 h a model)
set -u
cd "$(dirname "$0")"
SERVER=/home/matt/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.25.2/llama-server
PORT=8081
MMLU_N=${1:-10}
declare -A MODELS=(
  [base]=/home/matt/store/Zynk_related_models/Qwen3-4B-Instruct-2507-Q4_0.gguf
  [sophro]=/home/matt/store/loose_models/Sophro-4B-memory-v1-Q4_0.gguf
)
COMMON=(--model local-chat-completions --apply_chat_template --output_path results --log_samples)
ARGS="model=NAME,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=4,max_retries=3"

stop_server() { pkill -x llama-server 2>/dev/null; sleep 3; }
start_server() {
  nohup "$SERVER" -m "$1" -ngl 99 -c 8192 -np 4 --host 127.0.0.1 --port $PORT --jinja > "results/server_$2.log" 2>&1 &
  for _ in $(seq 1 60); do curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0; sleep 2; done
  echo "server for $2 did not come up"; return 1
}

mkdir -p results
for name in base sophro; do
  stop_server
  start_server "${MODELS[$name]}" "$name" || exit 1
  A=${ARGS/NAME/$name}
  echo "=== $name: ifeval $(date +%H:%M)"
  ./venv/bin/lm_eval "${COMMON[@]}" --model_args "$A" --tasks ifeval --gen_kwargs 'temperature=0' \
    2>&1 | tee "results/${name}_ifeval.log" | grep -E '^\|'
  echo "=== $name: gsm8k $(date +%H:%M)"
  ./venv/bin/lm_eval "${COMMON[@]}" --model_args "$A" --tasks gsm8k_cot_llama --fewshot_as_multiturn --gen_kwargs 'temperature=0' \
    2>&1 | tee "results/${name}_gsm8k.log" | grep -E '^\|'
  echo "=== $name: mmlu ($MMLU_N per subject) $(date +%H:%M)"
  ./venv/bin/lm_eval "${COMMON[@]}" --model_args "$A" --tasks mmlu_generative --limit "$MMLU_N" \
    --gen_kwargs 'temperature=0,max_gen_toks=8' \
    --system_instruction 'Answer with only the letter of the correct choice: A, B, C or D. No explanation.' \
    2>&1 | tee "results/${name}_mmlu.log" | grep -E '^\|mmlu \(generative\)|^\| - mmlu'
done
stop_server
echo "=== done $(date +%H:%M)"
