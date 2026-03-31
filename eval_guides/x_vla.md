# X-VLA

## 1. Clone

```bash
cd eval_scripts
git clone https://github.com/huggingface/lerobot.git x-vla
cd x-vla
```

## 2. Environment Setup

```bash
conda create -n libero-para-xvla python=3.12 -y
conda activate libero-para-xvla
conda install ffmpeg -c conda-forge

pip install -e ".[xvla]"
pip install robosuite==1.4.0 bddl==1.0.1 mujoco future easydict matplotlib gym==0.25.2
```

## 3. Install LIBERO-Para

```bash
cd ../../
pip install -e .
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
```

## 4. Run

```bash
conda activate libero-para-xvla
export MUJOCO_GL=egl
python eval_scripts/examples/eval_x_vla.py \
    --policy_path lerobot/xvla-libero \
    --gpu 0 \
    --seed 7 \
    --output_dir ./logs_para/x-vla/seed7/
```

For quick testing with limited tasks:

```bash
python eval_scripts/examples/eval_x_vla.py \
    --policy_path lerobot/xvla-libero \
    --gpu 0 --seed 7 --max_tasks 10 \
    --output_dir ./logs_para/x-vla/seed7/
```

## Evaluating on Original LIBERO Suites

The same script supports standard LIBERO suites (spatial, object, goal, etc.) via `--bddl_dir` and `--init_dir`. Mode is auto-detected from filenames.

```bash
python eval_scripts/examples/eval_x_vla.py \
    --policy_path lerobot/xvla-libero \
    --gpu 0 --seed 7 \
    --bddl_dir libero/libero/bddl_files/libero_spatial \
    --init_dir libero/libero/init_files/libero_spatial \
    --output_dir ./logs_para/x-vla-spatial/seed7/
```

## Notes

- No server needed — model loads directly in the eval script
- Checkpoint: [lerobot/xvla-libero](https://huggingface.co/lerobot/xvla-libero) (auto-downloaded from HuggingFace)
- Uses `domain_id=3` for LIBERO tasks
- Camera resolution: 360x360
- Control mode: absolute (default)
- Paper: [arXiv:2510.10274](https://arxiv.org/pdf/2510.10274)
