# Xiaomi Robotics 0

## 1. Clone

```bash
cd eval_scripts
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-0.git xiaomi-robotics-0
cd xiaomi-robotics-0
```

## 2. Model Server Environment

```bash
conda create -n libero-para-xiaomi-mibot python=3.12 -y
conda activate libero-para-xiaomi-mibot

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1

pip uninstall -y ninja && pip install ninja
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

sudo apt-get install -y libegl1 libgl1 libgles2
```

## 3. Eval Client Environment

```bash
conda create -n libero-para-xiaomi-libero python=3.10 -y
conda activate libero-para-xiaomi-libero

cd eval_libero
ln -s ../../../ LIBERO
pip install -r requirements.txt
pip install --config-settings editable_mode=compat -e ./LIBERO
pip install easydict
sudo apt update && sudo apt install -y xvfb
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
cd ../
```

## 4. Run

### Terminal 1: Start model server

```bash
conda activate libero-para-xiaomi-mibot
cd eval_scripts/xiaomi-robotics-0
CUDA_VISIBLE_DEVICES=0 python deploy/server.py --model XiaomiRobotics/Xiaomi-Robotics-0-LIBERO --port 10086
```

### Terminal 2: Run evaluation

```bash
conda activate libero-para-xiaomi-libero
cd ../../
export MUJOCO_GL=egl
python eval_scripts/examples/eval_xiaomi_robotics_0.py --port 10086 --seed 7 --output_dir ./logs_para/xiaomi/seed7/
```

## Evaluating on Original LIBERO Suites

The same script supports standard LIBERO suites via `--bddl_dir` and `--init_dir`. Mode is auto-detected.

```bash
python eval_scripts/examples/eval_xiaomi_robotics_0.py \
    --port 10086 --seed 7 \
    --bddl_dir libero/libero/bddl_files/libero_spatial \
    --init_dir libero/libero/init_files/libero_spatial \
    --goal_bddl_dir libero/libero/bddl_files/libero_spatial \
    --output_dir ./logs_para/xiaomi-spatial/seed7/
```

## Notes

- Requires two terminals: model server (mibot env) + eval client (libero env)
- Checkpoint: [XiaomiRobotics/Xiaomi-Robotics-0-LIBERO](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-LIBERO) (auto-downloaded)
- Paper: [arXiv:2602.12684](https://arxiv.org/abs/2602.12684)
