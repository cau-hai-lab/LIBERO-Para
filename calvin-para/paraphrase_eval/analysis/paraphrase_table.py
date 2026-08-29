"""obj×act paraphrase SR heatmap (4×11) from CALVIN paraphrase eval results.

Adapted from LIBERO-Para metrics/analyze_results.py.

Aggregates 1935 paraphrase episodes by (object_type, action_type) cell and
produces:
  1) SR table (obj × act)
  2) NearGT/FarGT/Cross-NearGT breakdown table (uses cache from
     compute_neargt_fargt.py if available)
  3) Per-task SR table

Outputs CSV + simple text tables (heatmap PNG optional via --plot).
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

CALVIN_PARA = Path(__file__).resolve().parents[2]

OBJ_ORDER = ["none", "ad", "spc", "sph"]
ACT_ORDER = [
    "none", "ad", "spc", "sph",
    "coord", "subord",
    "need", "embed", "perm", "ques", "hint",
]
OBJ_LABELS = {
    "none": "Obj=None",
    "ad": "Obj-AD",
    "spc": "Obj-SP-Ctx",
    "sph": "Obj-SP-Hab",
}
ACT_LABELS = {
    "none": "Act=None",
    "ad": "Act-AD",
    "spc": "Act-SP-Ctx",
    "sph": "Act-SP-Hab",
    "coord": "Act-Coord",
    "subord": "Act-Subord",
    "need": "Act-Need",
    "embed": "Act-Embed",
    "perm": "Act-Perm",
    "ques": "Act-Ques",
    "hint": "Act-Hint",
}


def load_paraphrase_results(model_results_dir: str):
    rows = []
    for ef in sorted(glob.glob(os.path.join(model_results_dir, "eval_*.json"))):
        d = json.load(open(ef))
        for ep in d["episodes"]:
            md = ep["metadata"]
            rows.append({
                "task_id": d["task_id"],
                "paraphrase_id": md.get("paraphrase_id"),
                "cell_id": md.get("cell_id"),
                "object_type": md.get("object_type"),
                "action_type": md.get("action_type"),
                "paraphrase_index": md.get("paraphrase_index"),
                "lang_override": ep.get("lang_override"),
                "success": bool(ep.get("success", False)),
                "num_steps": ep.get("num_steps"),
            })
    return pd.DataFrame(rows)


def sr_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["object_type", "action_type"])["success"]
    pivot = (g.mean() * 100).unstack().reindex(index=OBJ_ORDER, columns=ACT_ORDER)
    return pivot.round(1)


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.groupby(["object_type", "action_type"]).size().unstack(fill_value=0)
    return pivot.reindex(index=OBJ_ORDER, columns=ACT_ORDER, fill_value=0)


def per_task_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("task_id")["success"]
    return pd.DataFrame({"sr_pct": (g.mean() * 100).round(1), "n": g.size()})


def plot_heatmap(values: pd.DataFrame, png_path: str, title: str,
                 cbar_label: str, vmin: float = 0, vmax: float = 100,
                 cmap_name: str = "RdYlGn", invert: bool = False,
                 mask_none_none: bool = True) -> None:
    """4x11 heatmap with annotations.

    invert=True flips the colormap (useful for FarGT% where high = bad).
    mask_none_none hides the (Obj=None, Act=None) cell since it's the canonical baseline.
    """
    df = values.reindex(index=OBJ_ORDER, columns=ACT_ORDER)
    arr = df.values.astype(float)
    mask = np.zeros_like(arr, dtype=bool)
    if mask_none_none and "none" in df.index and "none" in df.columns:
        mask[OBJ_ORDER.index("none"), ACT_ORDER.index("none")] = True

    annot = np.empty_like(arr, dtype=object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i, j]:
                annot[i, j] = ""
            elif np.isnan(arr[i, j]):
                annot[i, j] = "—"
            else:
                annot[i, j] = f"{arr[i, j]:.0f}"

    cmap = plt.get_cmap(cmap_name + ("_r" if invert else ""))
    fig, ax = plt.subplots(figsize=(15, 4.5))
    sns.heatmap(arr, annot=annot, fmt="", cmap=cmap, mask=mask,
                vmin=vmin, vmax=vmax,
                cbar_kws={"label": cbar_label, "shrink": 0.85},
                linewidths=0.6, linecolor="white",
                xticklabels=[ACT_LABELS[a] for a in ACT_ORDER],
                yticklabels=[OBJ_LABELS[o] for o in OBJ_ORDER],
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                ax=ax)
    ax.set_xlabel("Action paraphrase type", fontsize=12, labelpad=8)
    ax.set_ylabel("Object paraphrase type", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved heatmap -> {png_path}")


def fargt_pivot_from_cache(cache_path: str) -> pd.DataFrame:
    d = json.load(open(cache_path))
    df = pd.DataFrame(d["episodes"])
    g = df.groupby(["object_type", "action_type"])
    fargt_pct = (g["is_fargt"].mean() * 100).unstack().reindex(index=OBJ_ORDER, columns=ACT_ORDER)
    return fargt_pct.round(1)


def cross_neargt_pivot_from_cache(cache_path: str) -> pd.DataFrame:
    d = json.load(open(cache_path))
    df = pd.DataFrame(d["episodes"])
    g = df.groupby(["object_type", "action_type"])
    cross_pct = (g["is_cross_neargt"].mean() * 100).unstack().reindex(index=OBJ_ORDER, columns=ACT_ORDER)
    return cross_pct.round(1)


def category_breakdown_from_cache(cache_path: str) -> pd.DataFrame:
    """Use compute_neargt_fargt.py cache to break down by category."""
    d = json.load(open(cache_path))
    df = pd.DataFrame(d["episodes"])
    g = df.groupby(["object_type", "action_type"])
    out = pd.DataFrame({
        "n": g.size(),
        "success_pct": (g["success"].mean() * 100).round(1),
        "neargt_pct": (g["is_neargt"].mean() * 100).round(1),
        "fargt_pct": (g["is_fargt"].mean() * 100).round(1),
        "cross_neargt_pct": (g["is_cross_neargt"].mean() * 100).round(1),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results_root", default=str(CALVIN_PARA / "paraphrase_eval/results"))
    ap.add_argument("--paraphrase_seed", default="seed7")
    ap.add_argument("--neargt_cache", default=None,
                    help="optional: cache JSON from compute_neargt_fargt.py")
    ap.add_argument("--output_dir", default=str(CALVIN_PARA / "paraphrase_eval/analysis/outputs"))
    args = ap.parse_args()

    paraphrase_dir = os.path.join(args.results_root, args.model, f"paraphrase_{args.paraphrase_seed}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_paraphrase_results(paraphrase_dir)
    print(f"loaded {len(df)} paraphrase episodes for {args.model}")
    print(f"overall SR: {df['success'].mean()*100:.1f}%")

    # SR pivot
    sr = sr_table(df)
    counts = count_table(df)
    print("\n=== SR (%) by obj × act cell ===")
    print(sr.fillna("-").to_string())
    print("\n=== n per cell ===")
    print(counts.to_string())

    # per-task SR
    pt = per_task_table(df)
    print("\n=== Per-task SR ===")
    print(pt.to_string())

    sr.to_csv(out_dir / f"{args.model}__sr_obj_x_act.csv")
    counts.to_csv(out_dir / f"{args.model}__counts_obj_x_act.csv")
    pt.to_csv(out_dir / f"{args.model}__sr_per_task.csv")
    print(f"\nwrote SR tables -> {out_dir}/{args.model}__*.csv")

    # SR heatmap (always)
    plot_heatmap(sr, str(out_dir / f"{args.model}__heatmap_sr.png"),
                 title=f"{args.model.upper()} — Success Rate (%) by Paraphrase Cell",
                 cbar_label="Success Rate (%)", vmin=0, vmax=100,
                 cmap_name="RdYlGn", invert=False)

    # category breakdown if neargt cache available
    if args.neargt_cache and os.path.exists(args.neargt_cache):
        cb = category_breakdown_from_cache(args.neargt_cache)
        print("\n=== Category breakdown (NearGT/FarGT/Cross-NearGT) ===")
        print(cb.to_string())
        cb.to_csv(out_dir / f"{args.model}__category_breakdown.csv")
        print(f"\nwrote category breakdown -> {out_dir}/{args.model}__category_breakdown.csv")

        fargt_pivot = fargt_pivot_from_cache(args.neargt_cache)
        cross_pivot = cross_neargt_pivot_from_cache(args.neargt_cache)
        print("\n=== FarGT % by obj × act cell ===")
        print(fargt_pivot.fillna("-").to_string())
        print("\n=== Cross-NearGT % by obj × act cell ===")
        print(cross_pivot.fillna("-").to_string())
        fargt_pivot.to_csv(out_dir / f"{args.model}__fargt_obj_x_act.csv")
        cross_pivot.to_csv(out_dir / f"{args.model}__cross_neargt_obj_x_act.csv")

        plot_heatmap(fargt_pivot, str(out_dir / f"{args.model}__heatmap_fargt.png"),
                     title=f"{args.model.upper()} — FarGT % by Paraphrase Cell (high = wandering off)",
                     cbar_label="FarGT %", vmin=0, vmax=100,
                     cmap_name="RdYlGn", invert=True)
        plot_heatmap(cross_pivot, str(out_dir / f"{args.model}__heatmap_cross_neargt.png"),
                     title=f"{args.model.upper()} — Cross-NearGT % (executes another task)",
                     cbar_label="Cross-NearGT %", vmin=0, vmax=30,
                     cmap_name="Reds", invert=False)


if __name__ == "__main__":
    main()
