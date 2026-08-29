#!/bin/bash
# Generate single-axis paraphrases (13 cells × N per task).
#
# Required env vars:
#   OPENROUTER_API_KEY  OpenRouter API key (https://openrouter.ai)
# Optional env vars:
#   PARA_GEN_ROOT       paraphrase_generation/ root (auto-detected)
#   MODEL               default: google/gemini-2.5-pro
#   INPUT_CSV           default: inputs/calvin_instructions_15.csv
#   NUM_PARAPHRASES     default: 3
set -euo pipefail
PARA_GEN_ROOT="${PARA_GEN_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
MODEL="${MODEL:-google/gemini-2.5-pro}"
INPUT_CSV="${INPUT_CSV:-${PARA_GEN_ROOT}/inputs/calvin_instructions_15.csv}"
NUM_PARAPHRASES="${NUM_PARAPHRASES:-3}"

python "${PARA_GEN_ROOT}/main_single.py" \
    --api_key "${OPENROUTER_API_KEY}" \
    --model "${MODEL}" \
    --input_csv "${INPUT_CSV}" \
    --num_paraphrases "${NUM_PARAPHRASES}"
