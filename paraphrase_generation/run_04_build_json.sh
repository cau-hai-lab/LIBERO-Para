#!/bin/bash
# Combine single-axis + compositional verified outputs into paraphrases.json.
#
# Required env vars:
#   SINGLE_CSV  llm_verified_*.csv from run_02
#   MERGED_CSV  llm_merged_verified_*.csv from run_03
# Optional:
#   PARA_GEN_ROOT, OUTPUT_DIR (default: llm_output), CAP (default: 3)
set -euo pipefail
PARA_GEN_ROOT="${PARA_GEN_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
: "${SINGLE_CSV:?set SINGLE_CSV}"
: "${MERGED_CSV:?set MERGED_CSV}"
OUTPUT_DIR="${OUTPUT_DIR:-${PARA_GEN_ROOT}/llm_output}"
CAP="${CAP:-3}"

python "${PARA_GEN_ROOT}/build_paraphrases_json.py" \
    --single_csv "${SINGLE_CSV}" \
    --merged_csv "${MERGED_CSV}" \
    --output_dir "${OUTPUT_DIR}" \
    --cap_per_cell_per_task "${CAP}"
