#!/bin/bash
# Phase A: Canonical baseline. 15 tasks × 20 trials × 5 seeds.
#  - RF: split each seed's 300 episodes into 2 chunks, run BOTH on GPU 4 in parallel.
#  - FLOWER: single process per seed on GPU 5.
# Output: paraphrase_eval/results/{rf,flower}/baseline_seed{S}/
set -euo pipefail
PARA_EVAL="${PARA_EVAL:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
EPISODES=$PARA_EVAL/episodes
SEEDS=(7 8 9 10 11)
TOTAL=300
HALF=150  # 0..150 and 150..300

mkdir -p $PARA_EVAL/results/rf $PARA_EVAL/results/flower $PARA_EVAL/logs

for S in "${SEEDS[@]}"; do
    echo "=== seed $S ==="
    EP=$EPISODES/baseline_seed${S}.json
    RF_OUT=$PARA_EVAL/results/rf/baseline_seed${S}
    FL_OUT=$PARA_EVAL/results/flower/baseline_seed${S}
    mkdir -p $RF_OUT $FL_OUT

    # RF on GPU 4: 2 parallel processes split 0:150 and 150:300
    GPU_ID=4 EPISODES_JSON=$EP EPISODES_OUTPUT_DIR=$RF_OUT \
        EPISODES_START_IDX=0 EPISODES_END_IDX=$HALF \
        bash $PARA_EVAL/run_rf_episodes.sh > $PARA_EVAL/logs/rf_baseline_seed${S}_chunk0.log 2>&1 &
    PID_RF1=$!

    GPU_ID=4 EPISODES_JSON=$EP EPISODES_OUTPUT_DIR=$RF_OUT \
        EPISODES_START_IDX=$HALF EPISODES_END_IDX=$TOTAL \
        bash $PARA_EVAL/run_rf_episodes.sh > $PARA_EVAL/logs/rf_baseline_seed${S}_chunk1.log 2>&1 &
    PID_RF2=$!

    # FLOWER on GPU 5: single process, full 300
    GPU_ID=5 EPISODES_JSON=$EP EPISODES_OUTPUT_DIR=$FL_OUT \
        EPISODES_START_IDX=0 EPISODES_END_IDX=$TOTAL \
        bash $PARA_EVAL/run_flower_episodes.sh > $PARA_EVAL/logs/flower_baseline_seed${S}.log 2>&1 &
    PID_FL=$!

    wait $PID_RF1 $PID_RF2 $PID_FL
    echo "  seed $S done"
done

echo "ALL BASELINE DONE"
