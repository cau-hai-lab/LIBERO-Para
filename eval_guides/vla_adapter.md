# VLA-Adapter

## 1. Clone

```bash
cd eval_scripts
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git vla-adapter
cd vla-adapter
```

## 2. Environment Setup

```bash
conda create -n libero-para-vla-adapter python=3.10 -y
conda activate libero-para-vla-adapter

pip install -e .
pip install "numpy<2" robosuite==1.4.0 bddl==1.0.1 future easydict matplotlib gym==0.25.2 msgpack msgpack-numpy
```

Flash Attention (check your CUDA version with `nvidia-smi`):
```bash
pip install flash-attn --no-build-isolation
```

## 3. Install LIBERO-Para

```bash
cd ../../
pip install --config-settings editable_mode=compat -e .
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
```

## 4. Download Checkpoint

Download the LIBERO-Goal-Pro checkpoint from [HuggingFace](https://huggingface.co/VLA-Adapter) and place it in `eval_scripts/vla-adapter/outputs/`:

```bash
cd eval_scripts/vla-adapter
mkdir -p outputs
# Download from: https://huggingface.co/VLA-Adapter/LIBERO-Goal-Pro
# Place into: outputs/LIBERO-Goal-Pro/
```

## 5. Run

```bash
conda activate libero-para-vla-adapter
export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_vla_adapter.py \
    --pretrained_checkpoint eval_scripts/vla-adapter/outputs/LIBERO-Goal-Pro \
    --gpu 0 --seed 7 \
    --output_dir ./logs_para/vla-adapter/seed7/
```

For quick testing:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_vla_adapter.py \
    --pretrained_checkpoint eval_scripts/vla-adapter/outputs/LIBERO-Goal-Pro \
    --gpu 0 --seed 7 --max_tasks 10 \
    --output_dir ./logs_para/vla-adapter/seed7/
```

## Evaluating on Original LIBERO Suites

The same script supports standard LIBERO suites via `--bddl_dir` and `--init_dir`. Mode is auto-detected.

```bash
CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_vla_adapter.py \
    --pretrained_checkpoint eval_scripts/vla-adapter/outputs/LIBERO-Goal-Pro \
    --gpu 0 --seed 7 \
    --bddl_dir libero/libero/bddl_files/libero_spatial \
    --init_dir libero/libero/init_files/libero_spatial \
    --goal_bddl_dir libero/libero/bddl_files/libero_spatial \
    --output_dir ./logs_para/vla-adapter-spatial/seed7/
```

## Notes

- No server needed — model loads directly in the eval script
- Checkpoint: [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) (~3GB per model)
- Uses Pro version by default (`use_pro_version=True`)
- Action chunking: 8 open-loop steps
- Camera resolution: 256x256
- Paper: [arXiv:2509.09372](https://arxiv.org/abs/2509.09372)
