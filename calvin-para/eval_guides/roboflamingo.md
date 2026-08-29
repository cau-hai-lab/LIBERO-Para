# RoboFlamingo

MPT-IFT-3B variant, CALVIN ABCD→D.

## 1. Clone

```bash
cd calvin-para
git clone https://github.com/RoboFlamingo/RoboFlamingo.git
```

## 2. Environment Setup

CALVIN needs Python 3.8.

```bash
conda create -n calvin-para-rf python=3.8 -y
conda activate calvin-para-rf

cd RoboFlamingo
pip install -r requirements.txt
pip install -e .
cd open_flamingo && pip install -e . && cd ..

cd ../calvin
sh install.sh
```

## 3. Download Weights

The policy checkpoint carries the **full** model (854 tensors / 3.22 B params / 12.87 GB fp32),
so the two scaffolding repos below contribute no weights at inference. See Notes.

```bash
# policy (17.60 GB), required
hf download robovlms/RoboFlamingo \
    checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth \
    --revision b08c0a2ac4d2a92927ddc066e3eb077319bc5a94 \
    --local-dir checkpoints/roboflamingo/RoboFlamingo

# OpenFlamingo base (4.19 GB), loaded then fully overwritten
hf download openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct checkpoint.pt \
    --revision 656bbbcd4508db84ccc83c02361011c6fe92ae93 \
    --local-dir checkpoints/roboflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct

# MPT LLM + tokenizer (5.25 GB); only the 2.2 MB of config/tokenizer files matter
hf download mosaicml/mpt-1b-redpajama-200b-dolly \
    --revision f0a13e41fcee2217cd701219ffa1eaef7fe955ea \
    --local-dir checkpoints/roboflamingo/mpt-1b-redpajama-200b-dolly
```

> **`mosaicml/mpt-1b-redpajama-200b-dolly` returns HTTP 401 to anonymous requests**
> (page, API, and file resolve alike; same for the non-dolly variant). Verified 2026-08-24
> without an HF token, so it may be gated rather than removed. Check while logged in.
> Only `config.json`, the six `trust_remote_code` modeling `.py` files, and the tokenizer
> are actually required; the 5.25 GB `pytorch_model.bin` is touched only because
> `AutoModelForCausalLM.from_pretrained` insists on a weights file.

The vision encoder is in none of the above. OpenCLIP downloads `ViT-L-14 / openai`
at model-construction time.

## 4. Point the model factory at your checkpoints

`robot_flamingo/models/factory.py` ships with `path_to/...` placeholders. Set the
`mpt_dolly_3b` entry's `lang_encoder_path`, `tokenizer_path`, and `openflamingo_checkpoint`
to the directories from step 3.

## 5. Apply the CALVIN-Para patch

`robot_flamingo/eval/eval_calvin.py` and `eval_utils.py` need the per-episode eval path
(`eval_one_epoch_calvin_episodes` / `_rollout_episode_logged`). The patch also replaces the
upstream author's hardcoded `/mnt/bn/robotics/...` paths and switches
`PYOPENGL_PLATFORM` from `osmesa` to `egl`.

```bash
git apply ../patches/roboflamingo_episode_eval.patch
```

## 6. Run

Build episodes first (see the [FLOWER guide](flower.md), step 6); the episode files are shared.

```bash
conda activate calvin-para-rf

GPU_ID=0 \
EPISODES_JSON=paraphrase_eval/episodes/paraphrase_seed7.json \
EPISODES_OUTPUT_DIR=paraphrase_eval/results/rf/paraphrase_seed7 \
EPISODES_START_IDX=0 EPISODES_END_IDX=968 \
    bash paraphrase_eval/run_rf_episodes.sh
```

RoboFlamingo is slow enough that we split 1,935 episodes into two chunks per GPU.

## Notes

- Checkpoint: [robovlms/RoboFlamingo](https://huggingface.co/robovlms/RoboFlamingo). The old id `roboflamingo/RoboFlamingo` still resolves (307, same sha)
- **Which variant**: `checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_dolly_3b_2.pth` at the repo root = **MPT-IFT-3B, ABCD→D**. The repo also holds MPT-3B / 4B / IFT-4B / 9B, a co-finetuned variant, and an `ABC_D/` directory for the harder split.
- Filename tokens map onto the run flags: `gripper`→`--use_gripper`, `post`→`--fusion_mode post`, `hist_1`→`--hist_window 1`, `aug_10_4`→`--rgb_pad 10 --gripper_pad 4`, `ws_12`→`--window_size 12 --eval_hist_size 12`
- `RF_EP_LEN=200` for single-task episodes (upstream default 360)
- Launched via `torchrun` even on a single GPU, because the eval script assumes DDP
- Reported: 92.33% canonical SR, 53.13% CALVIN-Para SR
