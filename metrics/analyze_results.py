#!/usr/bin/env python3
"""
generate_paraphrase_table.py

Generates a 4x11 (obj x act) paraphrase success rate heatmap from
structured JSON eval logs.

Directory structure expected:
    model_path/
        seed7/
            eval0.json  ... eval9.json
            meta.json
            summary.json
        seed8/
        ...
        seed11/

Each eval{N}.json has:
    {
      "eval_id": N,
      "original_instruction": ...,
      "episodes": [
        {
          "paraphrase_type": "act" | "obj" | "comp",
          "categories": [...],
          "subcategories": [...],  # comp: [obj_subcat, act_subcat]
          "success": true/false,
          ...
        }, ...
      ]
    }

Usage:
    python generate_paraphrase_table.py \
        --model_path ../logs_para/xiaomi \
        --model_name "OpenVLA-OFT" \
        --output_dir ./metrics/output
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ============================================================================
# Constants
# ============================================================================

# Y-axis (obj paraphrase types)
OBJ_TYPES_ORDER = [
    "None",
    "addition_deletion",
    "same_polarity_contextual",
    "same_polarity_habitual",
]

# X-axis (act paraphrase types)
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

# Fixed model display order
MODEL_ORDER = [
    "OpenVLA-OFT-goal",
    "OpenVLA-OFT_mixed",
    "Pi05",
    "Pi05-ExpertOnly",
    "X-VLA",
    "VLA-Adapter",
    "Xiaomi-Robotics-0",
]

# Fixed model colors (consistent across all plots)
MODEL_COLORS_FIXED = {
    "OpenVLA-OFT-goal":   "#1f77b4",  # blue
    "OpenVLA-OFT_mixed":  "#ff7f0e",  # orange
    "Pi05":               "#d62728",  # red
    "Pi05-ExpertOnly":    "#8c564b",  # brown
    "X-VLA":              "#9467bd",  # purple
    "VLA-Adapter":        "#2ca02c",  # green
    "Xiaomi-Robotics-0":  "#17becf",  # cyan
}

# Display labels (shorter names for heatmap axes)
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

# Act paraphrase categories (indices into ACT_TYPES_ORDER, excluding "None" at 0)
ACT_CATEGORIES = {
    "Lexical":     ["addition_deletion", "same_polarity_contextual", "same_polarity_habitual"],
    "Structural":  ["coordination", "subordination"],
    "Pragmatic":   ["need_statement", "embedded_imperative", "permission_directive",
                    "question_directive", "hint"],
}

CATEGORY_COLORS = {
    "Lexical":    "#2166ac",
    "Structural": "#d6604d",
    "Pragmatic":  "#7b3294",
}

# Act types excluding None (for bar charts)
ACT_TYPES_PARA = ACT_TYPES_ORDER[1:]

ACT_DISPLAY_SHORT = {
    "addition_deletion": "Addition",
    "same_polarity_contextual": "SP-ctx",
    "same_polarity_habitual": "SP-hab",
    "coordination": "Coord",
    "subordination": "Subord",
    "need_statement": "Need",
    "embedded_imperative": "Embedded",
    "permission_directive": "Permission",
    "question_directive": "Question",
    "hint": "Hint",
}

# Map each act type to its category
ACT_TYPE_TO_CATEGORY = {}
for cat, types in ACT_CATEGORIES.items():
    for t in types:
        ACT_TYPE_TO_CATEGORY[t] = cat


def discover_seeds(model_path: str) -> List[str]:
    """Find all seed directories under model_path."""
    seed_dirs = sorted(glob.glob(os.path.join(model_path, "seed*")))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
    if not seed_dirs:
        print(f"[ERROR] No seed directories found in: {model_path}")
        sys.exit(1)
    print(f"Found {len(seed_dirs)} seeds: {[os.path.basename(d) for d in seed_dirs]}")
    return seed_dirs


def load_eval_jsons(seed_dir: str) -> List[dict]:
    """Load all eval{N}.json files from a seed directory."""
    eval_files = sorted(glob.glob(os.path.join(seed_dir, "eval*.json")))
    # Exclude eval_unknown.json if exists
    eval_files = [f for f in eval_files if "unknown" not in os.path.basename(f)]
    
    all_episodes = []
    for ef in eval_files:
        with open(ef, "r") as f:
            data = json.load(f)
        eval_id = data.get("eval_id", -1)
        for ep in data.get("episodes", []):
            ep["_eval_id"] = eval_id
            all_episodes.append(ep)
    
    return all_episodes


def episode_to_cell(ep: dict) -> Tuple[str, str]:
    """
    Map an episode to (obj_low, act_low) cell coordinates.

    Rules:
        act  → obj_low = "None",             act_low = subcategories[0]
        obj  → obj_low = subcategories[0],    act_low = "None"
        comp → obj_low = subcategories[0],    act_low = subcategories[1]
    """
    ptype = ep["paraphrase_type"]
    subcats = ep.get("subcategories", [])

    if ptype == "act":
        return ("None", subcats[0] if subcats else "unknown")
    elif ptype == "obj":
        return (subcats[0] if subcats else "unknown", "None")
    elif ptype == "comp":
        obj_sub = subcats[0] if len(subcats) > 0 else "unknown"
        act_sub = subcats[1] if len(subcats) > 1 else "unknown"
        return (obj_sub, act_sub)
    else:
        return ("unknown", "unknown")


# ============================================================================
# Aggregation
# ============================================================================

def aggregate_seed(episodes: List[dict]) -> Dict[Tuple[str, str], dict]:
    """
    Aggregate episodes within one seed into per-cell stats.
    Returns: {(obj_low, act_low): {"successes": int, "total": int}}
    """
    cells = defaultdict(lambda: {"successes": 0, "total": 0})
    for ep in episodes:
        obj_low, act_low = episode_to_cell(ep)
        cells[(obj_low, act_low)]["total"] += 1
        if ep.get("success", False):
            cells[(obj_low, act_low)]["successes"] += 1
    return dict(cells)


def aggregate_across_seeds(
    model_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """
    Load all seeds, compute per-seed cell stats, then average across seeds.

    Returns:
        heatmap_df: DataFrame with mean success rates (index=obj, columns=act)
        std_df: DataFrame with std of success rates
        per_seed_stats: {seed_name: {(obj, act): {successes, total, sr}}}
        full_df: DataFrame with all rows (obj, act, mean, std, n_seeds, per-seed SRs)
    """
    seed_dirs = discover_seeds(model_path)
    per_seed_stats = {}
    all_cell_srs = defaultdict(list)  # {(obj, act): [sr_seed1, sr_seed2, ...]}
    seed_names = []

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        seed_names.append(seed_name)
        episodes = load_eval_jsons(seed_dir)
        print(f"  {seed_name}: {len(episodes)} episodes loaded")

        cell_stats = aggregate_seed(episodes)

        # Compute per-seed SR for each cell
        for cell_key, stats in cell_stats.items():
            sr = stats["successes"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["success_rate"] = sr
            all_cell_srs[cell_key].append(sr)

        per_seed_stats[seed_name] = cell_stats

    # Build full dataframe rows
    rows = []
    for obj_low in OBJ_TYPES_ORDER:
        for act_low in ACT_TYPES_ORDER:
            srs = all_cell_srs.get((obj_low, act_low), [])
            mean_sr = np.mean(srs) * 100 if srs else np.nan
            std_sr = np.std(srs) * 100 if len(srs) > 1 else 0.0
            var_sr = np.var(srs) * 10000 if len(srs) > 1 else 0.0  # variance in %^2

            row = {
                "obj_type": obj_low,
                "act_type": act_low,
                "mean_sr": mean_sr,
                "std_sr": std_sr,
                "var_sr": var_sr,
                "n_seeds": len(srs),
            }
            # Add per-seed SR columns
            for i, sname in enumerate(sorted(seed_names)):
                cell = per_seed_stats.get(sname, {}).get((obj_low, act_low), {})
                seed_sr = cell.get("success_rate", np.nan)
                row[sname] = round(seed_sr * 100, 2) if not np.isnan(seed_sr) else np.nan
            rows.append(row)

    full_df = pd.DataFrame(rows)

    heatmap_df = full_df.pivot(index="obj_type", columns="act_type", values="mean_sr")
    heatmap_df = heatmap_df.reindex(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER)

    std_df = full_df.pivot(index="obj_type", columns="act_type", values="std_sr")
    std_df = std_df.reindex(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER)

    return heatmap_df, std_df, per_seed_stats, full_df


# ============================================================================
# CSV Export
# ============================================================================

def save_aggregated_csv(full_df: pd.DataFrame, output_dir: str, model_name: str):
    """
    Save aggregated CSV with columns:
        obj_type, act_type, mean_sr, std_sr, var_sr, n_seeds, seed7, seed8, ...
    Also computes overall summary rows (total paraphrase avg, per-type avg, etc.)
    """
    # --- Per-cell CSV ---
    csv_path = os.path.join(output_dir, f"{model_name}_aggregated.csv")
    
    # Round numeric columns
    export_df = full_df.copy()
    for col in export_df.columns:
        if col not in ("obj_type", "act_type", "n_seeds"):
            export_df[col] = export_df[col].round(2)
    
    # Sort by obj_type order, then act_type order
    obj_order = {v: i for i, v in enumerate(OBJ_TYPES_ORDER)}
    act_order = {v: i for i, v in enumerate(ACT_TYPES_ORDER)}
    export_df["_obj_order"] = export_df["obj_type"].map(obj_order)
    export_df["_act_order"] = export_df["act_type"].map(act_order)
    export_df = export_df.sort_values(["_obj_order", "_act_order"]).drop(columns=["_obj_order", "_act_order"])
    
    export_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # --- Summary CSV (marginals + overall) ---
    summary_rows = []
    
    # Filter out (None, None) = original instruction
    para_df = full_df[~((full_df["obj_type"] == "None") & (full_df["act_type"] == "None"))].copy()
    
    # Overall paraphrase average
    overall_mean = para_df["mean_sr"].mean()
    overall_std = para_df["mean_sr"].std()
    summary_rows.append({
        "category": "ALL_PARAPHRASE",
        "type": "overall",
        "mean_sr": round(overall_mean, 2),
        "std_across_cells": round(overall_std, 2),
        "n_cells": len(para_df),
    })

    # Per obj_type marginal (average across all act_types)
    for obj in OBJ_TYPES_ORDER:
        subset = para_df[para_df["obj_type"] == obj]
        if len(subset) > 0:
            summary_rows.append({
                "category": "obj_marginal",
                "type": obj,
                "mean_sr": round(subset["mean_sr"].mean(), 2),
                "std_across_cells": round(subset["mean_sr"].std(), 2),
                "n_cells": len(subset),
            })

    # Per act_type marginal (average across all obj_types)
    for act in ACT_TYPES_ORDER:
        subset = para_df[para_df["act_type"] == act]
        if len(subset) > 0:
            summary_rows.append({
                "category": "act_marginal",
                "type": act,
                "mean_sr": round(subset["mean_sr"].mean(), 2),
                "std_across_cells": round(subset["mean_sr"].std(), 2),
                "n_cells": len(subset),
            })

    # act-only average (obj=None, act!=None)
    act_only = para_df[(para_df["obj_type"] == "None") & (para_df["act_type"] != "None")]
    if len(act_only) > 0:
        summary_rows.append({
            "category": "paraphrase_scope",
            "type": "act_only",
            "mean_sr": round(act_only["mean_sr"].mean(), 2),
            "std_across_cells": round(act_only["mean_sr"].std(), 2),
            "n_cells": len(act_only),
        })

    # obj-only average (obj!=None, act=None)
    obj_only = para_df[(para_df["obj_type"] != "None") & (para_df["act_type"] == "None")]
    if len(obj_only) > 0:
        summary_rows.append({
            "category": "paraphrase_scope",
            "type": "obj_only",
            "mean_sr": round(obj_only["mean_sr"].mean(), 2),
            "std_across_cells": round(obj_only["mean_sr"].std(), 2),
            "n_cells": len(obj_only),
        })

    # comp average (obj!=None, act!=None)
    comp = para_df[(para_df["obj_type"] != "None") & (para_df["act_type"] != "None")]
    if len(comp) > 0:
        summary_rows.append({
            "category": "paraphrase_scope",
            "type": "comp",
            "mean_sr": round(comp["mean_sr"].mean(), 2),
            "std_across_cells": round(comp["mean_sr"].std(), 2),
            "n_cells": len(comp),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f"{model_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")


# ============================================================================
# Saving per-seed JSON
# ============================================================================

def save_per_seed_json(per_seed_stats: dict, output_dir: str, model_name: str):
    """Save per-seed cell success rates as JSON."""
    for seed_name, cell_stats in per_seed_stats.items():
        out = {}
        for (obj_low, act_low), stats in cell_stats.items():
            key = f"{obj_low}+{act_low}"
            out[key] = {
                "obj_low": obj_low,
                "act_low": act_low,
                "successes": stats["successes"],
                "total": stats["total"],
                "success_rate": round(stats.get("success_rate", 0.0) * 100, 2),
            }

        path = os.path.join(output_dir, f"{model_name}_{seed_name}_cells.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved: {path}")


def save_aggregated_json(heatmap_df: pd.DataFrame, std_df: pd.DataFrame,
                         output_dir: str, model_name: str):
    """Save the aggregated (mean across seeds) heatmap data as JSON."""
    result = {}
    for obj_low in OBJ_TYPES_ORDER:
        for act_low in ACT_TYPES_ORDER:
            key = f"{obj_low}+{act_low}"
            mean_val = heatmap_df.loc[obj_low, act_low]
            std_val = std_df.loc[obj_low, act_low]
            result[key] = {
                "obj_low": obj_low,
                "act_low": act_low,
                "mean_sr": round(float(mean_val), 2) if not np.isnan(mean_val) else None,
                "std_sr": round(float(std_val), 2) if not np.isnan(std_val) else None,
            }

    path = os.path.join(output_dir, f"{model_name}_aggregated_cells.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path}")


# ============================================================================
# Heatmap Visualization
# ============================================================================

def get_colormap():
    """Custom red-to-green colormap: aggressive red for low success, green only for high."""
    colors = [
        (0.0,  "#67001f"),   # 0%  - dark crimson
        (0.10, "#d73027"),   # 10%
        (0.30, "#fc8d59"),   # 30%
        (0.50, "#fee090"),   # 50%
        (0.65, "#d9ef8b"),   # 65%
        (0.80, "#66bd63"),   # 80%
        (0.92, "#1a9850"),   # 92%
        (1.0,  "#00441b"),   # 100%
    ]
    return LinearSegmentedColormap.from_list(
        "custom_red_green", [(pos, c) for pos, c in colors], N=256
    )


def plot_heatmap(
    heatmap_df: pd.DataFrame,
    std_df: pd.DataFrame,
    png_path: str,
    model_name: str = "",
):
    """
    Plot 4x11 paraphrase combination heatmap.
    (None, None) cell is masked.
    """
    # Build mask: (None, None) = original, mask it
    mask = np.zeros_like(heatmap_df.values, dtype=bool)
    if "None" in heatmap_df.index and "None" in heatmap_df.columns:
        none_row = list(heatmap_df.index).index("None")
        none_col = list(heatmap_df.columns).index("None")
        mask[none_row, none_col] = True

    # Build annotation strings
    annot_array = np.empty_like(heatmap_df.values, dtype=object)
    for i, obj in enumerate(heatmap_df.index):
        for j, act in enumerate(heatmap_df.columns):
            val = heatmap_df.iloc[i, j]
            std = std_df.iloc[i, j]
            if mask[i, j]:
                annot_array[i, j] = ""
            elif np.isnan(val):
                annot_array[i, j] = "N/A"
            else:
                annot_array[i, j] = f"{val:.1f}"

    fig, ax = plt.subplots(figsize=(18, 5.5))
    cmap = get_colormap()

    obj_display = [OBJ_DISPLAY.get(o, o) for o in heatmap_df.index]
    act_display = [ACT_DISPLAY.get(a, a) for a in heatmap_df.columns]

    sns.heatmap(
        heatmap_df.values.astype(float),
        annot=annot_array,
        fmt="",
        cmap=cmap,
        mask=mask,
        cbar_kws={"label": "Success Rate (%)", "shrink": 0.8},
        linewidths=0.8,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 20, "fontweight": "bold"},
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
    plt.close()
    print(f"  Saved: {png_path}")


# ============================================================================
# Action-axis Bar Charts
# ============================================================================

def _extract_act_marginals(heatmap_df: pd.DataFrame,
                           baseline_sr: float = None) -> Dict[str, float]:
    """
    From a 4x11 heatmap_df, compute act-axis marginal SR.
    For each act type, average SR across all obj types.

    If baseline_sr is provided, include act="None" marginal
    using baseline_sr for the (None, None) cell.
    """
    marginals = {}

    # Include "None" if baseline_sr is given
    if baseline_sr is not None:
        vals = [baseline_sr]  # (None, None) cell
        for obj in OBJ_TYPES_ORDER:
            if obj == "None":
                continue
            v = heatmap_df.loc[obj, "None"]
            if not np.isnan(v):
                vals.append(v)
        marginals["None"] = np.mean(vals) if vals else np.nan

    for act in ACT_TYPES_PARA:
        vals = []
        for obj in OBJ_TYPES_ORDER:
            v = heatmap_df.loc[obj, act]
            if not np.isnan(v):
                vals.append(v)
        marginals[act] = np.mean(vals) if vals else np.nan
    return marginals


def _extract_act_marginals_from_json(json_path: str) -> Dict[str, float]:
    """Extract act marginals from aggregated_cells.json."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect per act type
    act_vals = defaultdict(list)
    for key, val in data.items():
        obj_low = val["obj_low"]
        act_low = val["act_low"]
        sr = val["mean_sr"]
        if act_low == "None":
            continue  # skip obj-only rows for act analysis
        if obj_low == "None" and act_low == "None":
            continue
        if sr is not None:
            act_vals[act_low].append(sr)

    marginals = {}
    for act in ACT_TYPES_PARA:
        vals = act_vals.get(act, [])
        marginals[act] = np.mean(vals) if vals else np.nan
    return marginals


def plot_act_type_bars(
    model_marginals: Dict[str, Dict[str, float]],
    output_dir: str,
    filename_prefix: str = "act_type_bars",
):
    """
    Chart 1 (1-column): Mean SR per act type, bar color = category.
    If "None" is present in marginals, show it as first bar (gray).
    """
    # Determine act types to plot
    has_none = any("None" in m for m in model_marginals.values())
    act_types = (["None"] + ACT_TYPES_PARA) if has_none else ACT_TYPES_PARA
    n_types = len(act_types)
    x = np.arange(n_types)

    means, bar_colors = [], []
    for act in act_types:
        vals = [m.get(act, np.nan) for m in model_marginals.values()]
        vals = [v for v in vals if not np.isnan(v)]
        means.append(np.mean(vals) if vals else 0)
        if act == "None":
            bar_colors.append("#555555")
        else:
            cat = ACT_TYPE_TO_CATEGORY.get(act, "Lexical")
            bar_colors.append(CATEGORY_COLORS.get(cat, "#333333"))

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.bar(x, means, 0.7, color=bar_colors,
           edgecolor="white", linewidth=0.5, alpha=0.85)

    for xi, m in zip(x, means):
        ax.text(xi, m + 0.8, f"{m:.1f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    act_labels = []
    for a in act_types:
        if a == "None":
            act_labels.append("None")
        else:
            act_labels.append(ACT_DISPLAY_SHORT.get(a, a))
    ax.set_xticks(x)
    ax.set_xticklabels(act_labels, rotation=45, ha="right", fontsize=8, fontweight="bold")
    for tick_label, act in zip(ax.get_xticklabels(), act_types):
        if act == "None":
            tick_label.set_color("#555555")
        else:
            cat = ACT_TYPE_TO_CATEGORY.get(act, "")
            tick_label.set_color(CATEGORY_COLORS.get(cat, "black"))

    _draw_category_spans(ax, act_types)

    ax.set_xlabel("Act Paraphrase Type", fontsize=9, fontweight="bold", labelpad=6)
    ax.set_ylabel("Success Rate (%)", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_act_category_bars(
    model_marginals: Dict[str, Dict[str, float]],
    output_dir: str,
    filename_prefix: str = "act_category_bars",
):
    """
    Chart 2 (1-column): Mean SR per category (Lexical/Structural/Pragmatic).
    Error bar = std across models' category-level means.
    """
    categories = list(ACT_CATEGORIES.keys())
    x = np.arange(len(categories))

    cat_means, cat_colors = [], []
    for cat in categories:
        cat_types = ACT_CATEGORIES[cat]
        per_model = []
        for marginals in model_marginals.values():
            vals = [marginals.get(t, np.nan) for t in cat_types]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                per_model.append(np.mean(vals))
        cat_means.append(np.mean(per_model) if per_model else 0)
        cat_colors.append(CATEGORY_COLORS[cat])

    fig, ax = plt.subplots(figsize=(4, 3.5))

    ax.bar(x, cat_means, 0.55, color=cat_colors,
           edgecolor="white", linewidth=0.5, alpha=0.85)

    for xi, m in zip(x, cat_means):
        ax.text(xi, m + 1.0, f"{m:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    for tick_label, cat in zip(ax.get_xticklabels(), categories):
        tick_label.set_color(CATEGORY_COLORS.get(cat, "black"))

    ax.set_ylabel("Success Rate (%)", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()



def _draw_category_spans(ax, act_types):
    """Draw category bracket lines above the x-axis area."""
    # Find index ranges for each category
    for cat, cat_types in ACT_CATEGORIES.items():
        indices = [i for i, a in enumerate(act_types) if a in cat_types]
        if not indices:
            continue
        color = CATEGORY_COLORS[cat]
        x_left = min(indices) - 0.4
        x_right = max(indices) + 0.4
        x_center = (x_left + x_right) / 2

        y_pos = 1.02
        h = 0.02

        ax.plot([x_left, x_right], [y_pos, y_pos],
                color=color, linewidth=2.5, clip_on=False,
                transform=ax.get_xaxis_transform())
        ax.plot([x_left, x_left], [y_pos - h, y_pos],
                color=color, linewidth=2.5, clip_on=False,
                transform=ax.get_xaxis_transform())
        ax.plot([x_right, x_right], [y_pos - h, y_pos],
                color=color, linewidth=2.5, clip_on=False,
                transform=ax.get_xaxis_transform())
        ax.text(x_center, y_pos + 0.03, cat,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color, transform=ax.get_xaxis_transform(), clip_on=False)


def generate_action_bar_charts(
    model_heatmaps: Dict[str, pd.DataFrame],
    output_dir: str,
    baseline_srs: Dict[str, float] = None,
):
    """
    Entry point: given {model_name: heatmap_df}, generate all 3 action bar charts.
    baseline_srs: {model_name: baseline SR for (None,None) cell}.
                  If None, "None" bar is omitted.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract marginals
    model_marginals = {}
    model_baselines = {}
    for model, hdf in model_heatmaps.items():
        bl = baseline_srs.get(model) if baseline_srs else None
        model_marginals[model] = _extract_act_marginals(hdf, baseline_sr=bl)
        # Baseline = overall paraphrase mean (all cells excl None,None)
        vals = []
        for obj in OBJ_TYPES_ORDER:
            for act in ACT_TYPES_ORDER:
                if obj == "None" and act == "None":
                    continue
                v = hdf.loc[obj, act]
                if not np.isnan(v):
                    vals.append(v)
        model_baselines[model] = np.mean(vals) if vals else 0

    # 2. Plot
    print("\n  [Action Bars 1/3] Per act type...")
    plot_act_type_bars(model_marginals, output_dir)

    print("  [Action Bars 2/3] Per category...")
    plot_act_category_bars(model_marginals, output_dir)

    print("  [Action Bars] Done.")


# ============================================================================
# PRIDE computation (SK + ST + PD from pre-computed CSV)
# ============================================================================


def load_pride_csv(csv_path: str) -> dict:
    """
    Load pre-computed SK/ST CSV.
    Key by instruction text (unique per paraphrase).
    Returns lookup: instruction_text -> {"sk": keyword_sim, "st": structural_sim}
    """
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        key = str(row["new_instruction"]).strip().lower()
        lookup[key] = {
            "sk": float(row["keyword_similarity"]),
            "st": float(row["structural_similarity"]),
        }

    print(f"  Lookup: {len(lookup)} unique instructions (from {len(df)} CSV rows)")
    if lookup:
        sample = list(lookup.items())[:3]
        for k, v in sample:
            print(f"    \"{k[:50]}...\" → sk={v['sk']:.3f}, st={v['st']:.3f}")

    return lookup


def compute_model_pride(model_path: str, pride_lookup: dict,
                        alphas: List[float] = None) -> dict:
    """
    Compute PRIDE score for a single model.

    PD_i = 1 - (alpha * SK_i + (1 - alpha) * ST_i)
    PRIDE(alpha) = sum(success_i * PD_i) / sum(PD_i) * 100
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0]

    seed_dirs = discover_seeds(model_path)

    # Collect per-episode (sk, st, success) for alpha sweep
    episode_data = []
    n_total = 0
    n_matched = 0
    unmatched_keys = []

    for seed_dir in seed_dirs:
        episodes = load_eval_jsons(seed_dir)
        for ep in episodes:
            key = str(ep.get("paraphrased_instruction", "")).strip().lower()
            n_total += 1

            info = pride_lookup.get(key)
            if info is None:
                if len(unmatched_keys) < 5:
                    unmatched_keys.append(key[:60])
                continue

            n_matched += 1
            sk = info["sk"]
            st = info["st"]
            success = 1.0 if ep.get("success", False) else 0.0
            episode_data.append((sk, st, success))

    print(f"  Matching: {n_matched}/{n_total} episodes ({n_total - n_matched} unmatched)")
    if unmatched_keys:
        print(f"  Sample unmatched: {unmatched_keys[:3]}")
    if episode_data:
        sks = [d[0] for d in episode_data]
        sts = [d[1] for d in episode_data]
        print(f"  Similarity ranges: SK=[{min(sks):.3f}, {max(sks):.3f}], "
              f"ST=[{min(sts):.3f}, {max(sts):.3f}]")

    n_episodes = len(episode_data)
    n_success = int(sum(s for _, _, s in episode_data))
    sr = round(n_success / n_episodes * 100, 1) if n_episodes > 0 else 0

    results = {"sr": sr, "n_episodes": n_episodes, "n_success": n_success}

    for alpha in alphas:
        actual = 0.0
        total = 0.0
        for sk, st, success in episode_data:
            pd = 1.0 - (alpha * sk + (1.0 - alpha) * st)
            total += pd
            if success:
                actual += pd
        score = round(actual / total * 100, 1) if total > 0 else 0
        results[f"pride_a{alpha:.2f}"] = score

    results["sk_score"] = results.get("pride_a1.00", 0)
    results["st_score"] = results.get("pride_a0.00", 0)
    results["pride"] = results.get("pride_a0.50", 0)

    return results


ALPHA_SWEEP = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]


def discover_models(base_dir: str) -> List[Tuple[str, str]]:
    """Discover model directories under base_dir."""
    models = []
    for entry in sorted(os.listdir(base_dir)):
        model_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(model_dir):
            continue
        seed_dirs = glob.glob(os.path.join(model_dir, "seed*"))
        if seed_dirs:
            models.append((entry, model_dir))
    return models


PRIDE_CSV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libero_para_metadata.csv")


def batch_pride(base_dir: str, output_dir: str, pride_csv: str,
                baseline_sr: float = None):
    """
    Run PRIDE computation for all models under base_dir.
    Also generates action-axis bar charts and PRIDE sweep.
    """
    models = discover_models(base_dir)
    if not models:
        print(f"[ERROR] No models found under: {base_dir}")
        return

    print(f"Loading PRIDE CSV: {pride_csv}")
    pride_lookup = load_pride_csv(pride_csv)
    print(f"  {len(pride_lookup)} paraphrase entries loaded")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Collect heatmaps for action bar charts ----
    model_heatmaps = {}
    for model_name, model_path in models:
        print(f"\n--- Aggregating {model_name} for action bars ---")
        heatmap_df, _, _, _ = aggregate_across_seeds(model_path)
        model_heatmaps[model_name] = heatmap_df

    # ---- Generate action bar charts ----
    print(f"\n{'#'*75}")
    print(f"  Generating Action-axis Bar Charts")
    print(f"{'#'*75}")
    bl_dict = {m: baseline_sr for m in model_heatmaps} if baseline_sr else None
    generate_action_bar_charts(model_heatmaps, output_dir, baseline_srs=bl_dict)

    # ---- PRIDE sweep ----
    print(f"\n{'#'*75}")
    print(f"  PRIDE Sweep")
    print(f"{'#'*75}")

    rows = []
    for model_name, model_path in models:
        print(f"\n--- {model_name} ---")
        result = compute_model_pride(model_path, pride_lookup, alphas=ALPHA_SWEEP)
        row = {"model": model_name, "sr": result["sr"]}
        for alpha in ALPHA_SWEEP:
            row[f"a{alpha:.2f}"] = result[f"pride_a{alpha:.2f}"]
        rows.append(row)

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda r: order_map.get(r["model"], len(MODEL_ORDER)))

    print(f"\n[PRIDE]")
    print("=" * 75)
    print(f"{'Model':<30} {'SR':>8} {'PRIDE':>10} {'S_K':>10} {'S_T':>10}")
    print(f"{'':30} {'':>8} {'(SK.5/ST.5)':>10} {'(SK1/ST0)':>10} {'(SK0/ST1)':>10}")
    print("-" * 75)
    for r in rows:
        print(f"{r['model']:<30} {r['sr']:>8.1f} {r['a0.50']:>10.1f} "
              f"{r['a1.00']:>10.1f} {r['a0.00']:>10.1f}")
    print("=" * 75)

    df = pd.DataFrame(rows)
    rename_map = {}
    for a in ALPHA_SWEEP:
        rename_map[f"a{a:.2f}"] = f"SK={a:.1f}/ST={1-a:.1f}"
    df = df.rename(columns=rename_map)
    csv_path = os.path.join(output_dir, "pride_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_path = os.path.join(output_dir, "pride_sweep.png")
    plot_pride_sweep(csv_path, plot_path, mode_label="PRIDE")



def plot_pride_sweep(csv_path: str, output_path: str, mode_label: str = "PD"):
    """
    Line plot: x = SK weight (0.0 → 1.0), y = PRIDE score.
    One line per model. SR shown as dashed horizontal.
    Shows per-model slope and average slope.
    """
    df = pd.read_csv(csv_path)

    alpha_cols = [c for c in df.columns if c.startswith("SK=")]
    alphas = []
    for c in alpha_cols:
        sk_val = float(c.split("/")[0].split("=")[1])
        alphas.append(sk_val)

    sorted_pairs = sorted(zip(alphas, alpha_cols))
    alphas_sorted = [p[0] for p in sorted_pairs]
    cols_sorted = [p[1] for p in sorted_pairs]

    # ★ Sort by score at α=0 (top line in chart = top row in table)
    df["_score"] = df[cols_sorted[0]]  # α=0.0 score
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["_score"])

    # ★ Wider figure + more table space
    fig, (ax, ax_tbl) = plt.subplots(1, 2, figsize=(11, 5),
                                      gridspec_kw={"width_ratios": [2.5, 1.5]})

    slopes = []
    model_names_list = []
    model_colors_used = []
    for idx, (_, row) in enumerate(df.iterrows()):
        model = row["model"]
        sr = row["sr"]
        scores = [row[c] for c in cols_sorted]
        color = MODEL_COLORS_FIXED.get(model, plt.cm.tab10(idx / max(len(df), 1)))

        slope = np.polyfit(alphas_sorted, scores, 1)[0]
        slopes.append(slope)
        model_names_list.append(model)
        model_colors_used.append(color)

        ax.plot(alphas_sorted, scores, "-o", color=color, linewidth=2.5,
                markersize=6, alpha=0.85)
        ax.axhline(y=sr, color=color, linestyle=":", linewidth=1, alpha=0.25)

    ax.set_xticks(alphas_sorted)
    x_labels = []
    for a in alphas_sorted:
        if a == 0.0:
            x_labels.append(f"{a:.1f}\n$S_K$=0\n$S_T$=1")
        elif a == 0.5:
            x_labels.append(f"{a:.1f}\n$S_K$=.5\n$S_T$=.5")
        elif a == 1.0:
            x_labels.append(f"{a:.1f}\n$S_K$=1\n$S_T$=0")
        else:
            x_labels.append(f"{a:.1f}")
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_xlabel("α", fontsize=11, fontweight="bold")
    ax.set_ylabel("PRIDE Score", fontsize=11, fontweight="bold")
    ax.set_ylim(20, 85)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.tick_params(axis="y", labelsize=9)

    ax_tbl.axis("off")

    avg_slope = np.mean(slopes)
    table_data = []
    cell_colors = []
    for i, (m, s) in enumerate(zip(model_names_list, slopes)):
        color = model_colors_used[i]
        table_data.append([m, f"{s:+.1f}"])
        rgba = plt.cm.colors.to_rgba(color)
        cell_colors.append([(rgba[0], rgba[1], rgba[2], 0.15), "white"])
    table_data.append(["Average", f"{avg_slope:+.1f}"])
    cell_colors.append(["#F5F5DC", "#F5F5DC"])

    # ★ colWidths: Model gets more space, Slope gets less
    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=["Model", "Slope"],
        colWidths=[0.42, 0.18],
        loc="center",
        cellLoc="left",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 2.3)

    for j in range(2):
        tbl[0, j].set_facecolor("#D8D8D8")
        tbl[0, j].set_text_props(fontweight="bold", fontsize=10)
        tbl[len(table_data), j].set_text_props(fontweight="bold", fontsize=10)
    # Color model names + bold slope values
    for i, (m, color) in enumerate(zip(model_names_list, model_colors_used)):
        tbl[i + 1, 0].set_text_props(color=color, fontweight="bold")
        tbl[i + 1, 1].set_text_props(fontweight="bold")

    fig.suptitle(f"PRIDE α Sweep — {mode_label}",
                 fontsize=12, fontweight="bold", y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Avg slope: {avg_slope:+.1f} pp per 1.0 SK shift")



# ============================================================================
# Model-wise average heatmap
# ============================================================================

def average_heatmap(json_paths: List[Tuple[str, str]], output_dir: str,
                    baseline_sr: float = None, pride_sweep_dir: str = None):
    """
    Load multiple models' aggregated_cells.json, average SR cell-by-cell,
    and plot a single "average across models" heatmap + action bar charts.
    """
    os.makedirs(output_dir, exist_ok=True)

    cell_srs = defaultdict(list)

    model_heatmaps = {}

    for model_name, json_path in json_paths:
        print(f"  Loading: {model_name} from {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        # Build per-model heatmap for action bars
        rows = []
        for key, val in data.items():
            obj_low = val["obj_low"]
            act_low = val["act_low"]
            mean_sr = val["mean_sr"]
            if mean_sr is not None:
                cell_srs[(obj_low, act_low)].append(mean_sr)
            rows.append({"obj": obj_low, "act": act_low, "sr": mean_sr if mean_sr is not None else np.nan})

        mdf = pd.DataFrame(rows)
        hdf = mdf.pivot(index="obj", columns="act", values="sr")
        hdf = hdf.reindex(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER)
        model_heatmaps[model_name] = hdf

    # Build averaged heatmap
    rows = []
    for obj in OBJ_TYPES_ORDER:
        for act in ACT_TYPES_ORDER:
            srs = cell_srs.get((obj, act), [])
            mean = np.mean(srs) if srs else np.nan
            std = np.std(srs) if len(srs) > 1 else 0.0
            rows.append({"obj": obj, "act": act, "mean_sr": mean, "std_sr": std,
                          "n_models": len(srs)})

    df = pd.DataFrame(rows)

    heatmap_df = df.pivot(index="obj", columns="act", values="mean_sr")
    heatmap_df = heatmap_df.reindex(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER)

    std_df = df.pivot(index="obj", columns="act", values="std_sr")
    std_df = std_df.reindex(index=OBJ_TYPES_ORDER, columns=ACT_TYPES_ORDER)

    print(f"\n{'='*60}")
    print("Model-wise Average Heatmap (43 cells)")
    print(f"{'='*60}")
    for obj in OBJ_TYPES_ORDER:
        for act in ACT_TYPES_ORDER:
            if obj == "None" and act == "None":
                continue
            val = heatmap_df.loc[obj, act]
            std = std_df.loc[obj, act]
            if not np.isnan(val):
                print(f"  {OBJ_DISPLAY.get(obj, obj):>15} × {ACT_DISPLAY.get(act, act):<15}  "
                      f"{val:5.1f} ± {std:4.1f}")

    para_vals = []
    for obj in OBJ_TYPES_ORDER:
        for act in ACT_TYPES_ORDER:
            if obj == "None" and act == "None":
                continue
            v = heatmap_df.loc[obj, act]
            if not np.isnan(v):
                para_vals.append(v)
    print(f"\n  Overall avg: {np.mean(para_vals):.1f} ± {np.std(para_vals):.1f}")

    csv_path = os.path.join(output_dir, "model_average_heatmap.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    png_path = os.path.join(output_dir, "model_average_heatmap.png")
    plot_heatmap(heatmap_df, std_df, png_path,
                 model_name="Model Average")

    # ---- Action bar charts from individual models ----
    print(f"\n  Generating Action-axis Bar Charts...")
    bl_dict = {m: baseline_sr for m in model_heatmaps} if baseline_sr else None
    generate_action_bar_charts(model_heatmaps, output_dir, baseline_srs=bl_dict)

    # ---- PRIDE sweep plots (if CSV exists) ----
    if pride_sweep_dir:
        csv_path = os.path.join(pride_sweep_dir, "pride_comparison.csv")
        if os.path.exists(csv_path):
            plot_path = os.path.join(output_dir, "pride_sweep.png")
            print(f"\n  Generating PRIDE sweep plot from {csv_path}...")
            plot_pride_sweep(csv_path, plot_path, mode_label="PRIDE")
        else:
            print(f"  [SKIP] {csv_path} not found")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrase success rate heatmap from structured JSON logs."
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to model log directory containing seed* folders.",
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Display name for the model (used in title & filenames). "
             "Defaults to last component of model_path.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        help="Output directory for heatmap images and JSONs.",
    )
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Base directory with multiple model subdirs. "
             "Runs PRIDE comparison across all models.",
    )
    parser.add_argument(
        "--pride_csv", type=str, default=PRIDE_CSV_DEFAULT,
        help="Path to pre-computed SK/ST CSV.",
    )
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Skip computation, re-plot from existing pride_comparison_*.csv in output_dir.",
    )
    parser.add_argument(
        "--average_models", nargs="+", default=None,
        help="Compute model-wise average heatmap. Specs as 'Name:path/to/aggregated_cells.json'",
    )
    parser.add_argument(
        "--baseline_sr", type=float, default=None,
        help="Baseline SR for (None,None) cell in action bar charts. "
             "Adds 'Original' bar. E.g., --baseline_sr 95.0",
    )
    parser.add_argument(
        "--pride_sweep_dir", type=str, default=None,
        help="Directory containing pride_comparison.csv. "
             "If provided, PRIDE sweep plots are generated in --average_models mode.",
    )
    args = parser.parse_args()

    # Average models mode
    if args.average_models:
        print("=== Model-wise Average Heatmap ===")
        json_paths = []
        for spec in args.average_models:
            name, path = spec.split(":", 1)
            json_paths.append((name, path))
        average_heatmap(json_paths, args.output_dir, baseline_sr=args.baseline_sr,
                        pride_sweep_dir=args.pride_sweep_dir)
        return

    # Plot-only mode
    if args.plot_only:
        print("=== Plot-only mode ===")
        csv_path = os.path.join(args.output_dir, "pride_comparison.csv")
        if os.path.exists(csv_path):
            plot_path = os.path.join(args.output_dir, "pride_sweep.png")
            plot_pride_sweep(csv_path, plot_path, mode_label="PRIDE")
        else:
            print(f"  [SKIP] {csv_path} not found")
        return

    # Batch mode
    if args.base_dir:
        batch_pride(args.base_dir, args.output_dir, args.pride_csv,
                    baseline_sr=args.baseline_sr)
        return

    # Single model mode
    if args.model_path is None:
        print("[ERROR] --model_path or --base_dir required")
        sys.exit(1)

    dir_name = os.path.basename(args.model_path.rstrip("/"))
    if args.model_name is None:
        args.model_name = dir_name

    # Output goes into metrics/output/<dir_name>/
    output_base = os.path.join(args.output_dir, dir_name)
    vis_dir = os.path.join(output_base, "visualization")
    res_dir = os.path.join(output_base, "results")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Path:  {args.model_path}")
    print(f"Output: {output_base}")
    print("=" * 70)

    # 1. Aggregate
    print("\n[1/6] Aggregating across seeds...")
    heatmap_df, std_df, per_seed_stats, full_df = aggregate_across_seeds(args.model_path)

    # 2. Save JSONs (detail)
    print("\n[2/6] Saving JSON results...")
    save_per_seed_json(per_seed_stats, res_dir, args.model_name)
    save_aggregated_json(heatmap_df, std_df, res_dir, args.model_name)

    # 3. Save CSVs (detail + overview)
    print("\n[3/6] Saving CSV results...")
    save_aggregated_csv(full_df, res_dir, args.model_name)

    # 4. Plot heatmap
    print("\n[4/6] Plotting heatmap...")
    heatmap_png = os.path.join(vis_dir, "paraphrase_heatmap.png")
    plot_heatmap(
        heatmap_df, std_df, heatmap_png,
        model_name=args.model_name,
    )

    # 5. Action bar charts (single model)
    print("\n[5/6] Generating action-axis bar charts...")
    act_bar_dir = vis_dir
    bl_dict = {args.model_name: args.baseline_sr} if args.baseline_sr else None
    generate_action_bar_charts(
        {args.model_name: heatmap_df},
        act_bar_dir,
        baseline_srs=bl_dict,
    )

    # 6. PRIDE
    print(f"\n[6/6] Computing PRIDE (α sweep)...")
    pride_lookup = load_pride_csv(args.pride_csv)

    result = compute_model_pride(args.model_path, pride_lookup, alphas=ALPHA_SWEEP)

    print(f"\n[PRIDE]")
    print(f"{'='*65}")
    print(f"  Model:   {args.model_name}")
    print(f"  SR:      {result['sr']:.1f}")
    print(f"  PRIDE:   {result['pride']:.1f}  (SK=0.5/ST=0.5)")
    print(f"  S_K:     {result['sk_score']:.1f}  (SK=1.0/ST=0.0)")
    print(f"  S_T:     {result['st_score']:.1f}  (SK=0.0/ST=1.0)")
    print(f"  ---")
    for a in ALPHA_SWEEP:
        print(f"  SK={a:.1f}/ST={1-a:.1f}:  {result[f'pride_a{a:.2f}']:.1f}")
    print(f"{'='*65}")

    row = {"model": args.model_name, "sr": result["sr"]}
    for a in ALPHA_SWEEP:
        row[f"SK={a:.1f}/ST={1-a:.1f}"] = result[f"pride_a{a:.2f}"]
    pride_row = pd.DataFrame([row])
    csv_path = os.path.join(res_dir, "pride_comparison.csv")
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[existing["model"] != args.model_name]
        pride_row = pd.concat([existing, pride_row], ignore_index=True)
    pride_row.to_csv(csv_path, index=False)
    print(f"  Appended to: {csv_path}")

    plot_path = os.path.join(vis_dir, "pride_sweep.png")
    plot_pride_sweep(csv_path, plot_path, mode_label="PRIDE")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()