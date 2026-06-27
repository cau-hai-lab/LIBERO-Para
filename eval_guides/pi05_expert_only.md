# Pi 0.5 (Expert-Only)

Pi 0.5 fine-tuned by us on the **expert-only** subset of LIBERO-Goal
(`pi05_libero_expert_only` config in openpi, batch size 256, 30k steps).

Same codebase as Pi 0.5 (Base). See [shared setup in pi05.md](pi05.md#shared-setup).

## Download the checkpoint

Hosted on HuggingFace:
https://huggingface.co/HAI-Lab/pi05-libero_goal-expert_only

```bash
# inference-only: just params/ (~5.8 GB)
huggingface-cli download HAI-Lab/pi05-libero_goal-expert_only \
    --local-dir ./checkpoints/pi05_libero_expert_only \
    --include "_CHECKPOINT_METADATA" "params/**" "README.md"

# (optional) include train_state/ for resume training (~8.8 GB extra)
huggingface-cli download HAI-Lab/pi05-libero_goal-expert_only \
    --local-dir ./checkpoints/pi05_libero_expert_only
```

## Run (Expert-Only variant)

Same two-terminal flow as the base variant, but tell the server to load our
custom checkpoint via `policy:checkpoint`.

### Terminal 1: policy server (custom checkpoint)

```bash
conda activate libero-para-pi05-server
cd eval_scripts/pi05

CKPT=$(realpath ./checkpoints/pi05_libero_expert_only)
uv run scripts/serve_policy.py \
    --env LIBERO --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero_expert_only \
    --policy.dir "$CKPT"
```

### Terminal 2: LIBERO-Para client

Only the `--model_name` and `--output_dir` change from the base variant —
the client just talks to whatever model the server has loaded.

```bash
conda activate libero-para-pi05-client    # or: conda activate openpi-libero
cd /path/to/LIBERO-Para
export MUJOCO_GL=egl
python eval_scripts/examples/eval_pi05.py \
    --host localhost --port 8000 \
    --model_name pi05-expert-only \
    --gpu 0 --seed 7 \
    --output_dir ./logs_para/pi05-expert-only/seed7/
```

## Docker variant

```bash
cd eval_scripts/pi05
CKPT=$(realpath ./checkpoints/pi05_libero_expert_only)
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_expert_only --policy.dir $CKPT"
docker compose -f examples/libero/compose.yml up --build
```

## Evaluating on Original LIBERO Suites

```bash
python eval_scripts/examples/eval_pi05.py \
    --host localhost --port 8000 \
    --model_name pi05-expert-only \
    --gpu 0 --seed 7 \
    --bddl_dir libero/libero/bddl_files/libero_goal \
    --init_dir libero/libero/init_files/libero_goal \
    --output_dir ./logs_para/pi05-expert-only-goal/seed7/
```

## Notes

- Same architecture as base (PaliGemma 2 + flow-matching expert)
- Base for training: Pi 0.5 (`gs://openpi-assets/checkpoints/pi05_base/params`)
- Training data: LIBERO-Goal **expert demonstrations only**
- Config name: `pi05_libero_expert_only`
- Batch size: 256, steps: 30 000
- LR schedule: cosine, warmup 10 k, peak 5e-5
- EMA decay: 0.999
- Frozen: image encoder + most LLM layers (kept `*_1*` LLM layer trainable)
- HuggingFace mirror: [`HAI-Lab/pi05-libero_goal-expert_only`](https://huggingface.co/HAI-Lab/pi05-libero_goal-expert_only)
- Reported as `Pi05_expert` in LIBERO-Para paper tables
