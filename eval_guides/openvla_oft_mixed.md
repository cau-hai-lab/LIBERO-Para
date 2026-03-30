# OpenVLA-OFT (Mixed)

Same codebase as OpenVLA-OFT (Goal). Follow the [shared setup in the Goal guide](openvla_oft_goal.md#shared-setup) first.

## Run (Mixed variant)

```bash
conda activate libero-para-openvla-oft
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_openvla_oft.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
    --gpu 0 --seed 7 \
    --output_dir ./logs_para/openvla-oft-mixed/seed7/
```

## Notes

- Same setup as Goal variant — only checkpoint string differs
- See [Goal guide](openvla_oft_goal.md) for full setup instructions
