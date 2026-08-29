#!/bin/bash
# Generate compositional (obj × act) paraphrases on top of the single-axis output.
#
# Required env vars:
#   OPENROUTER_API_KEY  OpenRouter API key
#   VERIFIED_CSV        path to llm_verified_*.csv produced by run_02_single.sh
# Optional:
#   PARA_GEN_ROOT, MODEL, NUM_PARAPHRASES (defaults same as single)
set -euo pipefail
PARA_GEN_ROOT="${PARA_GEN_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
: "${VERIFIED_CSV:?set VERIFIED_CSV to the llm_verified_*.csv from run_02}"
MODEL="${MODEL:-google/gemini-2.5-pro}"
NUM_PARAPHRASES="${NUM_PARAPHRASES:-3}"

python "${PARA_GEN_ROOT}/main_compositional.py" \
    --api_key "${OPENROUTER_API_KEY}" \
    --model "${MODEL}" \
    --verified_csv "${VERIFIED_CSV}" \
    --num_paraphrases "${NUM_PARAPHRASES}"
