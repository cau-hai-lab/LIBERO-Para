# Evaluation Guides

Guides for evaluating each model on LIBERO-Para.

## Supported Models

| Model | Guide | Example Script |
|-------|-------|----------------|
| OpenVLA-OFT (Goal) | [Guide](openvla_oft_goal.md) | [Script](../eval_scripts/examples/eval_openvla_oft_goal.py) |
| OpenVLA-OFT (Mixed) | [Guide](openvla_oft_mixed.md) | [Script](../eval_scripts/examples/eval_openvla_oft_mixed.py) |
| Pi0.5 | [Guide](pi05.md) | [Script](../eval_scripts/examples/eval_pi05.py) |
| X-VLA | [Guide](x_vla.md) | [Script](../eval_scripts/examples/eval_x_vla.py) |
| VLA-Adapter | [Guide](vla_adapter.md) | [Script](../eval_scripts/examples/eval_vla_adapter.py) |
| Xiaomi Robotics 0 | [Guide](xiaomi_robotics_0.md) | [Script](../eval_scripts/examples/eval_xiaomi_robotics_0.py) |

## Quick Start

```bash
# 1. Install LIBERO-Para
pip install -e .

# 2. Follow the model-specific guide to install dependencies

# 3. Run eval script
python eval_scripts/examples/eval_<model_name>.py \
    --suite libero_para \
    --num_episodes 20 \
    --device cuda:0
```

## Evaluation Protocol

- **Task Suite**: `libero_para` (paraphrase variants of LIBERO tasks)
- **Episodes per task**: 20 (fixed init states)
- **Metric**: Task success rate (%)
- **Action space**: 7-DoF (6 EEF pose + 1 gripper)
