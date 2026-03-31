# OpenVLA-OFT (Goal)

Same codebase as OpenVLA-OFT (Mixed). See [shared setup below](#shared-setup).

## Shared Setup

### 1. Clone

```bash
cd eval_scripts
git clone https://github.com/moojink/openvla-oft.git openvla-oft
cd openvla-oft
```

### 2. Environment Setup

```bash
conda create -n libero-para-openvla-oft python=3.10 -y
conda activate libero-para-openvla-oft

pip install -e .
pip install "numpy<2" robosuite==1.4.0 bddl==1.0.1 future easydict matplotlib gym==0.25.2 msgpack msgpack-numpy
pip install flash-attn --no-build-isolation
```

### 3. Install LIBERO-Para

```bash
cd ../../
pip install --config-settings editable_mode=compat -e .
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
```

## Run (Goal variant)

```bash
conda activate libero-para-openvla-oft
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_openvla_oft.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
    --gpu 0 --seed 7 \
    --output_dir ./logs_para/openvla-oft-goal/seed7/
```

## Evaluating on Original LIBERO Suites

The same script supports standard LIBERO suites via `--bddl_dir` and `--init_dir`. Mode is auto-detected.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_openvla_oft.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
    --gpu 0 --seed 7 \
    --bddl_dir libero/libero/bddl_files/libero_spatial \
    --init_dir libero/libero/init_files/libero_spatial \
    --goal_bddl_dir libero/libero/bddl_files/libero_spatial \
    --output_dir ./logs_para/openvla-oft-spatial/seed7/
```

## Notes

- Checkpoint is auto-downloaded from HuggingFace
- Same codebase for goal and mixed — only checkpoint string differs
- Action chunking: 8 open-loop steps
- Camera resolution: 256x256
- Paper: [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)
