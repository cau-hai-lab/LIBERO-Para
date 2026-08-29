#!/bin/bash
# Build the input CSV for the paraphrase pipeline from a CALVIN annotations YAML.
#
# Required env vars (with sensible defaults below):
#   PARA_GEN_ROOT       paraphrase_generation/ root (auto-detected from this script)
#   CALVIN_PARA_ROOT    calvin-para/ root (default: ../calvin-para from PARA_GEN_ROOT)
#   ANNOTATIONS_YAML    canonical CALVIN annotations YAML
#   TASK_LIST           one task id per line (default: inputs/calvin_base15_tasks.txt)
#   OUTPUT_CSV          output (default: inputs/calvin_instructions_15.csv)
set -euo pipefail
PARA_GEN_ROOT="${PARA_GEN_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
CALVIN_PARA_ROOT="${CALVIN_PARA_ROOT:-$(cd "${PARA_GEN_ROOT}/.." && pwd)/calvin-para}"
ANNOTATIONS_YAML="${ANNOTATIONS_YAML:-${CALVIN_PARA_ROOT}/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml}"
TASK_LIST="${TASK_LIST:-${PARA_GEN_ROOT}/inputs/calvin_base15_tasks.txt}"
OUTPUT_CSV="${OUTPUT_CSV:-${PARA_GEN_ROOT}/inputs/calvin_instructions_15.csv}"

python "${PARA_GEN_ROOT}/build_input_csv.py" \
    --annotations_yaml "${ANNOTATIONS_YAML}" \
    --task_list "${TASK_LIST}" \
    --output_csv "${OUTPUT_CSV}"
