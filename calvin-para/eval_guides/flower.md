# FLOWER

CALVIN ABCD-trained FlowerVLA, evaluated on scene D.

## 1. Clone

```bash
cd calvin-para
git clone https://github.com/intuitive-robots/flower_vla_calvin.git
```

## 2. Environment Setup

```bash
conda create -n calvin-para-flower python=3.9 -y
conda activate calvin-para-flower

cd flower_vla_calvin
pip install -e .
pip install -r requirements.txt
```

Install CALVIN itself into the same env (needed for `calvin_env` + the task oracle):

```bash
cd ../calvin
sh install.sh
```

## 3. Download Weights

```bash
hf download mbreuss/flower_calvin_abcd \
    --revision a46b456277a8ee5ce8d584c93f346b561b07e3b1 \
    --local-dir checkpoints/flower_calvin_abcd
```

`microsoft/Florence-2-large` (the VLM backbone) is pulled automatically at model-init time.

> The shipped `config.yaml` sets `model.load_pretrained: true` with the original author's
> cluster path. Our eval overrides it to `false` and loads `model.safetensors` directly,
> so the stale path is never used.

## 4. Dataset

Only the validation split is needed, since we never train.

```bash
# CALVIN task_D_D validation
cd calvin/dataset
sh download_data.sh D
```

`calvin/dataset/task_D_D/{training,validation}` must both exist; the loader touches
`training/` only for statistics, so symlinking it to `validation/` is enough.

## 5. Apply the CALVIN-Para patch

`flower/evaluation/flower_evaluate.py` needs the per-episode eval path
(`evaluate_policy_episodes`), which replaces CALVIN's default 1000-sequence
long-horizon rollout with one initial state + one instruction per episode,
logging per-step actions and EEF positions.

```bash
git apply ../patches/flower_episode_eval.patch
```

## 6. Build Episodes

```bash
python paraphrase_eval/build_episodes.py \
    --task_list        ../paraphrase_generation/inputs/calvin_base15_tasks.txt \
    --annotations_yaml calvin/calvin_models/conf/annotations/new_playtable_validation.yaml \
    --paraphrases_json ../paraphrase_generation/paraphrases.json \
    --output_dir       paraphrase_eval/episodes
```

Produces `baseline_seed7.json` (300 episodes) and `paraphrase_seed7.json` (1,935).
Pass `--baseline_seeds` for additional initial-state pools.

## 7. Run

```bash
conda activate calvin-para-flower

# canonical baseline
GPU_ID=0 \
EPISODES_JSON=paraphrase_eval/episodes/baseline_seed7.json \
EPISODES_OUTPUT_DIR=paraphrase_eval/results/flower/baseline_seed7 \
    bash paraphrase_eval/run_flower_episodes.sh

# paraphrase eval
GPU_ID=0 \
EPISODES_JSON=paraphrase_eval/episodes/paraphrase_seed7.json \
EPISODES_OUTPUT_DIR=paraphrase_eval/results/flower/paraphrase_seed7 \
    bash paraphrase_eval/run_flower_episodes.sh
```

Split across GPUs with `EPISODES_START_IDX` / `EPISODES_END_IDX`; vary inference noise
with `INFERENCE_SEED`. See `paraphrase_eval/run_paraphrase_all.sh` for the full sweep.

## Notes

- Checkpoint: [mbreuss/flower_calvin_abcd](https://huggingface.co/mbreuss/flower_calvin_abcd), trained on CALVIN **ABCD** and evaluated on **D**
- Model loads directly in-process; no inference server
- `ep_len=200` for single-task episodes (CALVIN's default 360 targets 5-task chains)
- Set `PYOPENGL_PLATFORM=egl` and pin `EGL_VISIBLE_DEVICES` to the same GPU as compute. Otherwise PyBullet renders on GPU 0 and throttles everything
- Reported: 99.67% canonical SR, 58.29% CALVIN-Para SR
