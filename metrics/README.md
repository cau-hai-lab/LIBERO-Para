# Metrics

Tools for analyzing LIBERO-Para evaluation results.

## PRIDE: Paraphrase Robustness Index in Robotic Instructional DEviation

PRIDE quantifies how robustly a VLA model handles paraphrased instructions by incorporating the linguistic deviation between the original and paraphrased instruction into the evaluation.

It uses two similarity scores:

- **Keyword Similarity (S_K)**: Measures preservation of task-critical content words (actions, objects) between original and paraphrased instructions, computed via Sentence-BERT embedding cosine similarity.
- **Structural Similarity (S_T)**: Measures syntactic divergence using normalized tree edit distance (TED) on dependency parse trees with POS/dependency-relation labels.

**Paraphrase Distance (PD)** combines these into a single deviation score:

```
PD_i = 1 - (alpha * S_K_i + (1 - alpha) * S_T_i)
```

The model-level **PRIDE score** is the ratio of achieved PD-weighted successes to the total possible PD, normalized to 0-100:

```
PRIDE(alpha) = sum(success_i * PD_i) / sum(PD_i) * 100
```

`alpha` controls the relative weight of keyword vs. structural similarity (default `alpha=0.5`). PRIDE = 100 means all episodes succeeded, PRIDE = 0 means all failed. Unlike plain SR, PRIDE gives more credit for succeeding on harder (more deviated) paraphrases.

## Files

- `libero_para_metadata.csv` — Paraphrase metadata with keyword/structural similarity scores
- `analyze_results.py` — Heatmap, bar charts, and PRIDE metric computation
- `PRIDE_metric_playground.ipynb` — Interactive notebook for exploring PRIDE with sentence pairs

## Usage

### Single model analysis

```bash
python metrics/analyze_results.py \
    --model_path logs_para/example_xiaomi-robotics-0
```

Output will be saved to `metrics/output/example_xiaomi-robotics-0/`.

> **Note**: The example results (`logs_para/example_xiaomi-robotics-0/`) contain only seed7. The numbers reported in the paper are averaged across 5 seeds, so the computed metrics may differ.

### Multi-model PRIDE comparison

```bash
python metrics/analyze_results.py \
    --base_dir logs_para \
    --output_dir ./metrics/output
```

### Re-plot from existing results

```bash
python metrics/analyze_results.py \
    --plot_only \
    --output_dir ./metrics/output
```

### Model-average heatmap

```bash
python metrics/analyze_results.py \
    --average_models \
        "Xiaomi:metrics/output/detail/Xiaomi-Robotics-0/Xiaomi-Robotics-0_aggregated_cells.json" \
        "ModelB:metrics/output/detail/ModelB/ModelB_aggregated_cells.json" \
    --output_dir ./metrics/output/average
```

### Interactive playground

```bash
jupyter notebook metrics/PRIDE_metric_playground.ipynb
```

## Outputs

| Output | Description |
|--------|-------------|
| `visualization/paraphrase_heatmap.png` | 4x11 obj x act SR heatmap |
| `visualization/act_type_bars.png` | SR per act paraphrase type |
| `visualization/act_category_bars.png` | SR per act category (lexical/structural/pragmatical) |
| `visualization/pride_sweep.png` | PRIDE score across alpha (SK/ST weight) |
| `results/pride_comparison.csv` | PRIDE scores for all models |
| `results/*_aggregated_cells.json` | Per-cell SR data |

