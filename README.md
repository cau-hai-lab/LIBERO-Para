<div align="center">

<h1>LIBERO-Para</h1>
<p><b>A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models</b></p>

Chanyoung Kim<sup>1&#42;</sup>,
Minwoo Kim<sup>1&#42;</sup>,
Minseok Kang<sup>1</sup>,
Hyunwoo Kim<sup>2</sup>,
Dahuin Jung<sup>2&dagger;</sup>

<sup>1</sup>Soongsil University &nbsp;&nbsp; <sup>2</sup>Chung-Ang University

<sub>&#42; Equal contribution &nbsp;&nbsp; &dagger; Corresponding author</sub>

<p>
[Paper (coming soon)]
</p>

<img src="images/LIBERO-Para.png" width="900">

</div>

---

## Evaluation Guides

Each model is evaluated using a custom standalone script under `eval_scripts/examples/`, which directly interfaces with the model's inference server or loads the model directly.

| Model | Guide | Script | Status |
|-------|-------|--------|--------|
| OpenVLA-OFT (Goal) | [Guide](eval_guides/openvla_oft_goal.md) | [Script](eval_scripts/examples/eval_openvla_oft.py) | Verified |
| OpenVLA-OFT (Mixed) | [Guide](eval_guides/openvla_oft_mixed.md) | [Script](eval_scripts/examples/eval_openvla_oft.py) | Verified |
| Pi0.5 | [Guide](eval_guides/pi05.md) | — | TODO |
| X-VLA | [Guide](eval_guides/x_vla.md) | [Script](eval_scripts/examples/eval_x_vla.py) | Verified |
| VLA-Adapter | [Guide](eval_guides/vla_adapter.md) | [Script](eval_scripts/examples/eval_vla_adapter.py) | Verified |
| Xiaomi-Robotics-0 | [Guide](eval_guides/xiaomi_robotics_0.md) | [Script](eval_scripts/examples/eval_xiaomi_robotics_0.py) | Verified |

## PRIDE Metric

**PRIDE** (Paraphrase Robustness Index in Robotic Instructional DEviation) evaluates how robustly a VLA model handles paraphrased instructions. It computes a Paraphrase Distance (PD) from keyword similarity (S_K) and structural similarity (S_T), then measures the ratio of PD-weighted successes to total possible PD, normalized to 0-100. Unlike plain success rate, PRIDE gives more credit for succeeding on harder, more deviated paraphrases.

> See [metrics/README.md](metrics/README.md) for the full formulation and [PRIDE_metric_playground.ipynb](metrics/PRIDE_metric_playground.ipynb) for interactive exploration.

## Metrics & Analysis

### Setup

```bash
conda create -n libero-para python=3.10 -y
conda activate libero-para
pip install -r metrics/requirements.txt
python -m spacy download en_core_web_sm
```

### Quick Start

Xiaomi-Robotics-0 example results are included in `logs_para/example_xiaomi-robotics-0/`.

> **Note**: Action trajectories are stripped from the example logs to reduce file size. Only metadata and success/failure results are included.

```bash
python metrics/analyze_results.py \
    --model_path logs_para/example_xiaomi-robotics-0
```

See [metrics/README.md](metrics/README.md) for full usage details.

## Project Structure

```
LIBERO-Para/
├── libero/                    # LIBERO-Para benchmark core
├── metrics/                   # PRIDE metric & analysis tools
│   ├── analyze_results.py     # Heatmap, bar charts, PRIDE computation
│   ├── PRIDE_metric_playground.ipynb
│   └── libero_para_metadata.csv
├── eval_guides/               # Per-model setup & evaluation guides
├── eval_scripts/
│   ├── examples/              # Per-model eval scripts
│   ├── openvla-oft/           # git clone (see guide)
│   ├── x-vla/                 # git clone (see guide)
│   ├── vla-adapter/           # git clone (see guide)
│   └── xiaomi-robotics-0/     # git clone (see guide)
├── logs_para/                 # Evaluation results
│   └── example_xiaomi-robotics-0/  # Example data (seed7)
├── images/
├── benchmark_scripts/
└── scripts/
```

## TODO

- [ ] Add obj preserved vs paraphrased visualization
- [ ] Eval Guide & Script
  - [x] OpenVLA-OFT (Goal)
  - [x] OpenVLA-OFT (Mixed)
  - [ ] Pi0.5
  - [x] X-VLA
  - [x] VLA-Adapter
  - [x] Xiaomi-Robotics-0

## Acknowledgement

This project is built upon [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) by Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone.
