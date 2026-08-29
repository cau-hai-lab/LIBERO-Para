# Evaluation Guides

Guides for evaluating each model on CALVIN-Para.

## Supported Models

| Model | Guide | Weights |
|-------|-------|---------|
| FLOWER | [Guide](flower.md) | [mbreuss/flower_calvin_abcd](https://huggingface.co/mbreuss/flower_calvin_abcd) |
| RoboFlamingo | [Guide](roboflamingo.md) | [robovlms/RoboFlamingo](https://huggingface.co/robovlms/RoboFlamingo) |

## Quick Start

```bash
# 1. Install CALVIN (per-model conda env; see each guide)
cd calvin && sh install.sh

# 2. Build the shared episode files
python paraphrase_eval/build_episodes.py \
    --task_list        ../paraphrase_generation/inputs/calvin_base15_tasks.txt \
    --annotations_yaml calvin/calvin_models/conf/annotations/new_playtable_validation.yaml \
    --paraphrases_json ../paraphrase_generation/paraphrases.json \
    --output_dir       paraphrase_eval/episodes

# 3. Run a model
GPU_ID=0 \
EPISODES_JSON=paraphrase_eval/episodes/paraphrase_seed7.json \
EPISODES_OUTPUT_DIR=paraphrase_eval/results/<model>/paraphrase_seed7 \
    bash paraphrase_eval/run_<model>_episodes.sh
```

## Evaluation Protocol

- **Tasks**: 15 CALVIN tasks that are feasible as single-task episodes (no prerequisite manipulation)
- **Scene**: CALVIN scene **D** (`task_D_D` validation split); models are ABCD-trained
- **Baseline**: 15 tasks × 20 trials = 300 episodes per seed, canonical instructions
- **Paraphrase**: 1,935 episodes (43 taxonomy cells × 3 × 15 tasks), each paired to a baseline initial state
- **Episode length**: 200 steps (CALVIN's default 360 targets 5-task chains)
- **Metric**: success rate (%), plus cluster-aware NearGT / FarGT / Cross-NearGT trajectory classification
- **Action space**: 7-DoF (6 EEF pose + 1 gripper)

## Adding a New Model

Each integration needs three things:

1. **A per-episode eval path** in the model's own eval script, triggered by `EPISODES_JSON`.
   It reads episodes from JSON, resets the env to `init_state`, injects `lang_override`,
   and logs per-step `actions` + `eef_xyz` alongside the success flag. See the
   `patches/` directory for what this looks like for FLOWER and RoboFlamingo.
2. **A runner script** `paraphrase_eval/run_<model>_episodes.sh` that pins the GPU
   (compute *and* EGL) and forwards the `EPISODES_*` environment variables.
3. **A guide** in this directory.

Results land in `paraphrase_eval/results/<model>/<run>/eval_<task>__chunk<a>_<b>.json`
and are picked up by the analysis scripts automatically.
