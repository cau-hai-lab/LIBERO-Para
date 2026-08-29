"""LIBERO-Para-style 4x11 (obj x act) paraphrase SR heatmap for CALVIN-Para.

Reproduces the exact look of
  260301_analysis/paraphrase_table/heatmap_results/overview/<Model>_paraphrase_heatmap.png
(plot_heatmap() in generate_paraphrase_table.py) but reads CALVIN-Para
per-episode result JSONs instead of LIBERO structured logs.

Usage:
  python libero_style_heatmap.py                       # both models, seed7
  python libero_style_heatmap.py --model flower
  python libero_style_heatmap.py --seeds seed7 inference_seed8 --output_dir ...
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PARA_EVAL = Path(__file__).resolve().parents[1]

# CALVIN metadata codes -> LIBERO-Para canonical type names
OBJ_CODE_TO_TYPE = {
    "none": "None",
    "ad": "addition_deletion",
    "spc": "same_polarity_contextual",
    "sph": "same_polarity_habitual",
}
ACT_CODE_TO_TYPE = {
    "none": "None",
    "ad": "addition_deletion",
    "spc": "same_polarity_contextual",
    "sph": "same_polarity_habitual",
    "coord": "coordination",
    "subord": "subordination",
    "need": "need_statement",
    "embed": "embedded_imperative",
    "perm": "permission_directive",
    "ques": "question_directive",
    "hint": "hint",
}

# --- identical to generate_paraphrase_table.py ---
OBJ_TYPES_ORDER = [
    "None",
    "addition_deletion",
    "same_polarity_contextual",
    "same_polarity_habitual",
]
ACT_TYPES_ORDER = [
    "None",
    "addition_deletion",
    "same_polarity_contextual",
    "same_polarity_habitual",
    "coordination",
    "subordination",
    "need_statement",
    "embedded_imperative",
    "permission_directive",
    "question_directive",
    "hint",
]
OBJ_DISPLAY = {
    "None": "None",
    "addition_deletion": "Addition",
    "same_polarity_contextual": "SP-contextual",
    "same_polarity_habitual": "SP-habitual",
}
ACT_DISPLAY = {
    "None": "None",
    "addition_deletion": "Addition",
    "same_polarity_contextual": "SP-contextual",
    "same_polarity_habitual": "SP-habitual",
    "coordination": "Coordination",
    "subordination": "Subordination",
    "need_statement": "Need",
    "embedded_imperative": "Embedded",
    "permission_directive": "Permission",
    "question_directive": "Question",
    "hint": "Hint",
}

MODEL_DISPLAY = {"flower": "FLOWER", "rf": "RoboFlamingo"}


def get_colormap():
    """Custom red-to-green colormap: aggressive red for low success, green only for high."""
    colors = [
        (0.0,  "#67001f"),
        (0.10, "#d73027"),
        (0.30, "#fc8d59"),
        (0.50, "#fee090"),
        (0.65, "#d9ef8b"),
        (0.80, "#66bd63"),
        (0.92, "#1a9850"),
        (1.0,  "#00441b"),
    ]
    return LinearSegmentedColormap.from_list(
        "custom_red_green", [(pos, c) for pos, c in colors], N=256
    )


def load_seed_cells(model: str, seed_dir: str):
    """Return {(obj_type, act_type): sr_pct} for one paraphrase run directory."""
    pattern = os.path.join(PARA_EVAL, "results", model, seed_dir, "eval_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no eval_*.json under results/{model}/{seed_dir}")

    bucket = defaultdict(list)
    n_ep = 0
    for f in files:
        d = json.load(open(f))
        for ep in d.get("episodes", []):
            md = ep.get("metadata", {})
            if md.get("kind") != "paraphrase":
                continue
            obj = OBJ_CODE_TO_TYPE.get(md.get("object_type"))
            act = ACT_CODE_TO_TYPE.get(md.get("action_type"))
            if obj is None or act is None:
                continue
            bucket[(obj, act)].append(bool(ep.get("success", False)))
            n_ep += 1

    cells = {k: 100.0 * float(np.mean(v)) for k, v in bucket.items()}
    counts = {k: len(v) for k, v in bucket.items()}
    return cells, counts, n_ep


def build_frames(model: str, seeds):
    """Mean/std SR across the given seed dirs -> (heatmap_df, std_df, count_df, info)."""
    per_seed, counts_ref, info = [], None, []
    for s in seeds:
        cells, counts, n_ep = load_seed_cells(model, f"paraphrase_{s}")
        per_seed.append(cells)
        counts_ref = counts
        info.append((s, n_ep))

    mean = pd.DataFrame(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER, dtype=float)
    std = pd.DataFrame(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER, dtype=float)
    cnt = pd.DataFrame(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER, dtype=float)
    for obj in OBJ_TYPES_ORDER:
        for act in ACT_TYPES_ORDER:
            vals = [c[(obj, act)] for c in per_seed if (obj, act) in c]
            if vals:
                mean.loc[obj, act] = float(np.mean(vals))
                std.loc[obj, act] = float(np.std(vals)) if len(vals) > 1 else 0.0
                cnt.loc[obj, act] = counts_ref.get((obj, act), np.nan)
    return mean, std, cnt, info


def plot_heatmap(heatmap_df, std_df, png_path, pdf_path, model_name="", show_std=False):
    """4x11 paraphrase combination heatmap; (None, None) is masked (= canonical)."""
    mask = np.zeros_like(heatmap_df.values, dtype=bool)
    none_row = list(heatmap_df.index).index("None")
    none_col = list(heatmap_df.columns).index("None")
    mask[none_row, none_col] = True
    mask |= heatmap_df.isna().values

    annot_array = np.empty_like(heatmap_df.values, dtype=object)
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            val = heatmap_df.iloc[i, j]
            if mask[i, j] or np.isnan(val):
                annot_array[i, j] = ""
            elif show_std:
                annot_array[i, j] = f"{val:.1f}\n±{std_df.iloc[i, j]:.1f}"
            else:
                annot_array[i, j] = f"{val:.1f}"

    fig, ax = plt.subplots(figsize=(18, 5.5))
    obj_display = [OBJ_DISPLAY.get(o, o) for o in heatmap_df.index]
    act_display = [ACT_DISPLAY.get(a, a) for a in heatmap_df.columns]

    sns.heatmap(
        heatmap_df.values.astype(float),
        annot=annot_array,
        fmt="",
        cmap=get_colormap(),
        mask=mask,
        cbar_kws={"label": "Success Rate (%)", "shrink": 0.8},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 16 if show_std else 20, "fontweight": "bold"},
        xticklabels=act_display,
        yticklabels=obj_display,
    )

    title = "Success Rate by Paraphrase Type Combination"
    if model_name:
        title = f"{model_name}: {title}"
    ax.set_xlabel("Act Paraphrase Type", fontsize=16, fontweight="normal", labelpad=10)
    ax.set_ylabel("Obj Paraphrase Type", fontsize=16, fontweight="normal", labelpad=10)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=16, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16, fontweight="bold")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("Success Rate (%)", fontsize=14, fontweight="normal")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["flower", "rf"],
                    help="model result dirs under paraphrase_eval/results/")
    ap.add_argument("--seeds", nargs="+", default=["seed7"],
                    help="run suffixes: seed7, inference_seed8, ...")
    ap.add_argument("--output_dir", default=str(PARA_EVAL.parent / "RESULTS" / "overview"))
    ap.add_argument("--show_std", action="store_true",
                    help="annotate mean ± std across seeds (only useful with >1 seed)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.model:
        name = MODEL_DISPLAY.get(model, model)
        print(f"[{name}] seeds={args.seeds}")
        mean, std, cnt, info = build_frames(model, args.seeds)
        for s, n in info:
            print(f"    {s}: {n} paraphrase episodes")

        n_cells = int(mean.notna().sum().sum())
        overall = float(np.nansum((mean * cnt).values) / np.nansum(cnt.values))
        print(f"    {n_cells} cells populated, episode-weighted overall SR = {overall:.2f}%")

        plot_heatmap(mean, std,
                     str(out_dir / f"{name}_paraphrase_heatmap.png"),
                     str(out_dir / f"{name}_paraphrase_heatmap.pdf"),
                     model_name=name, show_std=args.show_std)

        csv = mean.copy()
        csv.index = [OBJ_DISPLAY[o] for o in csv.index]
        csv.columns = [ACT_DISPLAY[a] for a in csv.columns]
        csv.round(2).to_csv(out_dir / f"{name}_paraphrase_heatmap.csv")

        # per-cell sample count (episodes per seed; (None,None) is the canonical cell)
        cnt_csv = cnt.fillna(0).astype(int)
        cnt_csv.index, cnt_csv.columns = csv.index, csv.columns
        cnt_csv.to_csv(out_dir / f"{name}_paraphrase_counts.csv")
        if len(args.seeds) > 1:
            sd = std.copy()
            sd.index, sd.columns = csv.index, csv.columns
            sd.round(2).to_csv(out_dir / f"{name}_paraphrase_heatmap_std.csv")
        print(f"  Saved: {out_dir / f'{name}_paraphrase_heatmap.csv'}")


if __name__ == "__main__":
    main()
