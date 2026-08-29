# CALVIN-Para evaluation results

15 single-task CALVIN tasks × 1,935 LLM-generated paraphrases (43 cells × 3 per task),
evaluated on FLOWER and RoboFlamingo.

**Methodology**: cluster-aware Near-GT / Far-GT envelopes (v2). For dynamic tasks (block
manipulation), baseline successes are clustered by gripper grasp xy (≈ block xy) into K=2
sub-groups (left vs right table side). Each sub-group gets its own GT mean trajectory and
τ_max envelope, removing the bias from multi-modal initial block positions.

## Headline numbers

| metric | FLOWER | RoboFlamingo |
|---|---:|---:|
| CALVIN canonical SR | 99.67% | 92.33% |
| CALVIN-Para SR | **58.29%** | **53.13%** |
| PRIDE (α=0.5) | 48.35 | 43.50 |
| Obj-preserving (None + AD) SR | 77.57% | 69.84% |
| Obj-paraphrased (SP-Ctx + SP-Hab) SR | 39.90% | 37.17% |
| NearGT (% of failures) | 2.87% | 7.64% |
| FarGT-Unmatched (% of failures) | 95.38% | 79.11% |
| FarGT-Cross-NearGT (% of failures) | 1.75% | 13.25% |

When these models fail on a paraphrase, they overwhelmingly **wander off** the canonical
trajectory rather than almost-succeeding or executing a neighboring task.

Object renaming dominates: action-only paraphrases (Obj=None) sit mostly at 0 to 15% FarGT
(exceptions: Act-Ques 35%, Act-Hint 44%), while renaming the object pushes FarGT to
**48 to 76%**. Object addition/deletion falls in between at 13 to 47%. This matches
LIBERO-Para Finding 2.

## Files

Numbers only. Every figure is regenerated from the analysis scripts (see Regenerating).

| file | contents |
|---|---|
| `<model>__summary.csv` | headline metrics |
| `<model>__sr_per_task.csv` | per-task SR (15 tasks) |
| `<model>__category_breakdown.csv` | per-cell n / SR / NearGT / FarGT / Cross-NearGT (43 rows) |
| `overview/<Model>_paraphrase_heatmap.csv` | 4x11 obj x act SR table |
| `overview/<Model>_paraphrase_counts.csv` | episodes per cell (uniformly 45; the canonical cell is 0) |

> **Note on `category_breakdown` counts**: the `n` column counts only episodes that could be
> matched to a trajectory envelope, not all 1,935. FLOWER retains 1,928 (7 dropped);
> RoboFlamingo retains 1,780 (155 dropped, 8%), with per-cell n ranging 39-45. Success rates
> in `overview/` use the full 1,935 for both models and are the fair comparison; the
> NearGT/FarGT splits carry this sampling caveat.

## Per-task baseline τ_max (cluster-aware)

Static tasks have a single tight envelope; dynamic tasks split into 2 clusters, each with
its own narrow envelope:

| task | K | τ_max (s) |
|---|---|---|
| open_drawer / close_drawer | 1 | 0.008 / 0.011 |
| move_slider_left / right | 1 | 0.016 / 0.011 |
| turn_on / off_lightbulb | 1 | 0.025 / 0.019 |
| turn_on / off_led | 1 | 0.014 / 0.014 |
| lift_red_block_table | 2 | 0.016 (left) / 0.009 (right) |
| lift_blue_block_table | 2 | 0.012 / 0.013 |
| lift_red_block_slider | 2 | 0.020 / 0.036 |
| push_red_block_left | 2 | 0.026 / 0.014 |
| push_red_block_right | 2 | 0.041 / 0.026 |
| rotate_red_block_left | 2 | 0.029 / 0.016 |
| rotate_red_block_right | 2 | 0.016 / 0.020 |

## Regenerating

```bash
conda activate calvin-para-analysis   # needs spacy + sentence-transformers for PRIDE

# 1. LIBERO-Para-style SR heatmaps (both models)
python paraphrase_eval/analysis/libero_style_heatmap.py

# 2. Cluster-aware NearGT / FarGT / Cross-NearGT
python paraphrase_eval/analysis/compute_neargt_fargt_v2.py --model flower

# 3. Failure-mode heatmaps and per-cell tables
python paraphrase_eval/analysis/paraphrase_table.py --model flower \
    --neargt_cache paraphrase_eval/analysis/cache/flower__neargt_fargt_v2.json

# 4. Headline summary incl. PRIDE
python paraphrase_eval/analysis/model_summary.py flower

# 5. Scene figures (needs the CALVIN env, not the analysis env)
python paraphrase_eval/analysis/render_scene.py
```

Replace `--model flower` with `--model rf` for RoboFlamingo.

## Why clustering matters

An earlier single-envelope version built one GT envelope per task. For block tasks that
envelope is artificially wide, because baseline trajectories scatter across the two block
positions, so many genuine wander-offs landed inside it and were mislabeled NearGT
(37% of failures). Per-cluster envelopes bring that to 2.9%, which is what makes the
"models wander off rather than almost-succeed" conclusion meaningful.
