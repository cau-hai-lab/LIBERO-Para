#!/bin/bash
# Run RoboFlamingo on a paraphrase_eval episodes JSON.
# Usage:
#   GPU_ID=4 EPISODES_JSON=... EPISODES_OUTPUT_DIR=... \
#     EPISODES_START_IDX=0 EPISODES_END_IDX=968 \
#     ./run_rf_episodes.sh
set -euo pipefail
: "${EPISODES_JSON:?set EPISODES_JSON}"
: "${EPISODES_OUTPUT_DIR:?set EPISODES_OUTPUT_DIR}"
: "${EPISODES_START_IDX:=0}"
GPU_ID=${GPU_ID:-4}
PORT=${PORT:-$((20000 + RANDOM % 10000))}

CALVIN_PARA="${CALVIN_PARA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RF=$CALVIN_PARA/RoboFlamingo
CKPT=$CALVIN_PARA/checkpoints/roboflamingo/RoboFlamingo/checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth
DATASET=$CALVIN_PARA/calvin/dataset/task_D_D
CONF=$CALVIN_PARA/calvin/calvin_models/conf

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYOPENGL_PLATFORM=egl
# Pin EGL/PyBullet rendering to the same GPU as compute (otherwise it
# defaults to GPU 0 -> cross-GPU traffic + slowdown).
export EGL_VISIBLE_DEVICES=${EGL_DEV:-$GPU_ID}
export EGL_VISIBLE_DEVICE=${EGL_DEV:-$GPU_ID}
export EGL_DEVICE_ID=${EGL_DEV:-$GPU_ID}
export NCCL_BLOCKING_WAIT=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export EPISODES_JSON
export EPISODES_OUTPUT_DIR
export EPISODES_START_IDX
[ -n "${EPISODES_END_IDX:-}" ] && export EPISODES_END_IDX
[ -n "${INFERENCE_SEED:-}" ] && export INFERENCE_SEED
export RF_EP_LEN=${RF_EP_LEN:-200}

source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
conda activate "${RF_ENV:-calvin-para-rf}"
cd $RF
export PYTHONPATH=$RF:$RF/open_flamingo:${PYTHONPATH:-}

torchrun --nnodes=1 --nproc_per_node=1 --master_port=$PORT \
    robot_flamingo/eval/eval_calvin.py \
    --precision fp32 \
    --use_gripper \
    --window_size 12 \
    --eval_hist_size 12 \
    --fusion_mode post \
    --hist_window 1 \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --llm_name mpt_dolly_3b \
    --run_name "RFParaphraseEval" \
    --calvin_dataset $DATASET \
    --evaluate_from_checkpoint $CKPT \
    --calvin_conf_path $CONF \
    --workers 1
