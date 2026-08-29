#!/bin/bash
# Phase B: Paraphrase inference. 1935 episodes × 1 trial each.
#  - RF: split into 2 chunks of 968+967, both on GPU 4 in parallel.
#  - FLOWER: single process on GPU 5 (1935).
# Output: paraphrase_eval/results/{rf,flower}/paraphrase_seed7/
set -euo pipefail
PARA_EVAL="${PARA_EVAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EP=$PARA_EVAL/episodes/paraphrase_seed7.json
RF_OUT=$PARA_EVAL/results/rf/paraphrase_seed7
FL_OUT=$PARA_EVAL/results/flower/paraphrase_seed7
mkdir -p $RF_OUT $FL_OUT $PARA_EVAL/logs

TOTAL=1935
HALF=968

GPU_ID=4 EPISODES_JSON=$EP EPISODES_OUTPUT_DIR=$RF_OUT \
    EPISODES_START_IDX=0 EPISODES_END_IDX=$HALF \
    bash $PARA_EVAL/run_rf_episodes.sh > $PARA_EVAL/logs/rf_paraphrase_chunk0.log 2>&1 &
PID_RF1=$!

GPU_ID=4 EPISODES_JSON=$EP EPISODES_OUTPUT_DIR=$RF_OUT \
    EPISODES_START_IDX=$HALF EPISODES_END_IDX=$TOTAL \
    bash $PARA_EVAL/run_rf_episodes.sh > $PARA_EVAL/logs/rf_paraphrase_chunk1.log 2>&1 &
PID_RF2=$!

GPU_ID=5 EPISODES_JSON=$EP EPISODES_OUTPUT_DIR=$FL_OUT \
    EPISODES_START_IDX=0 EPISODES_END_IDX=$TOTAL \
    bash $PARA_EVAL/run_flower_episodes.sh > $PARA_EVAL/logs/flower_paraphrase.log 2>&1 &
PID_FL=$!

wait $PID_RF1 $PID_RF2 $PID_FL
echo "ALL PARAPHRASE DONE"
