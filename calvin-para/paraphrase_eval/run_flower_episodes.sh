#!/bin/bash
# Run FLOWER on a paraphrase_eval episodes JSON. Single process.
# Usage:
#   GPU_ID=5 EPISODES_JSON=... EPISODES_OUTPUT_DIR=... ./run_flower_episodes.sh
set -euo pipefail
: "${EPISODES_JSON:?set EPISODES_JSON}"
: "${EPISODES_OUTPUT_DIR:?set EPISODES_OUTPUT_DIR}"
: "${EPISODES_START_IDX:=0}"
GPU_ID=${GPU_ID:-5}

CALVIN_PARA="${CALVIN_PARA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FLOWER=$CALVIN_PARA/flower_vla_calvin
CKPT_DIR=$CALVIN_PARA/checkpoints/flower_calvin_abcd
DATASET=$CALVIN_PARA/calvin/dataset/task_D_D

export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES=${EGL_DEV:-$GPU_ID}
export EGL_VISIBLE_DEVICE=${EGL_DEV:-$GPU_ID}
export EGL_DEVICE_ID=${EGL_DEV:-$GPU_ID}
export EPISODES_JSON
export EPISODES_OUTPUT_DIR
export EPISODES_START_IDX
[ -n "${EPISODES_END_IDX:-}" ] && export EPISODES_END_IDX
[ -n "${INFERENCE_SEED:-}" ] && export INFERENCE_SEED

source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
conda activate "${FLOWER_ENV:-calvin-para-flower}"

cd $FLOWER
export PYTHONPATH=$FLOWER:$CALVIN_PARA/calvin/calvin_models:${PYTHONPATH:-}

EP_LEN=${EP_LEN:-200}
python flower/evaluation/flower_evaluate.py \
    train_folder=$CKPT_DIR \
    checkpoint=$CKPT_DIR/model.safetensors \
    dataset_path=$DATASET \
    num_sequences=1 \
    log_wandb=False \
    device=0 \
    log_dir=$CALVIN_PARA/paraphrase_eval/logs/flower \
    num_videos=0 \
    ep_len=$EP_LEN \
    +eval_cfg_overwrite.datamodule.datasets.lang_dataset.use_extracted_rel_actions=False
