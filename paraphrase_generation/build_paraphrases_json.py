"""Combine single-axis + compositional verified CSVs into a single
paraphrases.json with stable cell ids and one paraphrase_index per row.

Outputs:
  - <output_dir>/paraphrases.json          (full list)
  - <output_dir>/paraphrase_summary.csv    (counts per cell, per task)
  - prints sampled rows to stdout for spot-check.

Each row in the final JSON looks like:
{
  "paraphrase_id": "p_00001",
  "base_task_id": "open_drawer",
  "base_original": "pull the handle to open the drawer",
  "object_type": "sp_habitual" | "none",
  "action_type": "embedded_imperative" | "none",
  "cell_id": "obj_sph__act_embed",
  "paraphrase_index": 1,
  "paraphrase": "could you pull the handle to open the compartment?"
}
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# Short codes for low-types so cell_id stays compact.
OBJ_LOW_CODES = {
    "same_polarity_habitual": "sph",
    "same_polarity_contextual": "spc",
    "addition_deletion": "ad",
}
ACT_LOW_CODES = {
    "same_polarity_habitual": "sph",
    "same_polarity_contextual": "spc",
    "addition_deletion": "ad",
    "coordination": "coord",
    "subordination": "subord",
    "ellipsis": "ellip",
    "need_statement": "need",
    "embedded_imperative": "embed",
    "permission_directive": "perm",
    "question_directive": "ques",
    "hint": "hint",
}


def make_cell_id(high, low_field):
    """high in {obj, act, merged}; low_field is e.g. 'embedded_imperative'
    or for merged: 'same_polarity_habitual+embedded_imperative'."""
    if high == "obj":
        return f"obj_{OBJ_LOW_CODES.get(low_field, low_field)}__act_none"
    if high == "act":
        return f"obj_none__act_{ACT_LOW_CODES.get(low_field, low_field)}"
    if high == "merged":
        obj_low, act_low = low_field.split("+")
        return f"obj_{OBJ_LOW_CODES.get(obj_low, obj_low)}__act_{ACT_LOW_CODES.get(act_low, act_low)}"
    raise ValueError(f"unknown high: {high}")


def normalise_low(high, low_field):
    if high == "obj":
        return OBJ_LOW_CODES.get(low_field, low_field), "none"
    if high == "act":
        return "none", ACT_LOW_CODES.get(low_field, low_field)
    if high == "merged":
        obj_low, act_low = low_field.split("+")
        return OBJ_LOW_CODES.get(obj_low, obj_low), ACT_LOW_CODES.get(act_low, act_low)
    raise ValueError(f"unknown high: {high}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_csv", required=True,
                    help="llm_verified_*.csv from main_single.py")
    ap.add_argument("--merged_csv", required=True,
                    help="llm_merged_verified_*.csv from main_compositional.py")
    ap.add_argument("--output_dir", default="llm_output")
    ap.add_argument("--cap_per_cell_per_task", type=int, default=3,
                    help="if more than N paraphrases survived per (task, cell), keep first N")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sdf = pd.read_csv(args.single_csv)
    mdf = pd.read_csv(args.merged_csv)
    df = pd.concat([sdf, mdf], ignore_index=True)

    # bucket per (task, cell_id) and cap
    buckets = defaultdict(list)
    for _, row in df.iterrows():
        cell_id = make_cell_id(row["high"], row["low"])
        buckets[(row["eval"], cell_id, row["original_instruction"])].append({
            "high": row["high"],
            "low": row["low"],
            "paraphrase": row["new_instruction"],
        })

    rows = []
    pid = 0
    for (task_id, cell_id, orig), entries in sorted(buckets.items()):
        # keep first N (already pseudo-randomised by LLM; deterministic for repro)
        kept = entries[: args.cap_per_cell_per_task]
        # capture obj/act type strings from any entry (they all share cell)
        any_e = kept[0]
        obj_type, act_type = normalise_low(any_e["high"], any_e["low"])
        for i, e in enumerate(kept, start=1):
            pid += 1
            rows.append({
                "paraphrase_id": f"p_{pid:05d}",
                "base_task_id": task_id,
                "base_original": orig,
                "object_type": obj_type,
                "action_type": act_type,
                "cell_id": cell_id,
                "paraphrase_index": i,
                "paraphrase": e["paraphrase"],
            })

    out_json = out_dir / "paraphrases.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {len(rows)} paraphrases -> {out_json}")

    # summary csv
    cell_counts = Counter(r["cell_id"] for r in rows)
    task_counts = Counter(r["base_task_id"] for r in rows)
    summary_rows = []
    for cell, n in sorted(cell_counts.items()):
        summary_rows.append({"axis": "cell", "id": cell, "count": n})
    for tid, n in sorted(task_counts.items()):
        summary_rows.append({"axis": "task", "id": tid, "count": n})
    summary_path = out_dir / "paraphrase_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"summary -> {summary_path}")
    print(f"  unique cells: {len(cell_counts)}  (target 43)")
    print(f"  unique tasks: {len(task_counts)}  (target 20)")
    print(f"  per-cell mean count: {sum(cell_counts.values())/max(len(cell_counts),1):.1f}")
    print(f"  per-task mean count: {sum(task_counts.values())/max(len(task_counts),1):.1f}")

    # spot check
    print("\n--- 8 random samples ---")
    for r in random.sample(rows, min(8, len(rows))):
        print(f"[{r['cell_id']}] {r['base_task_id']}\n"
              f"  orig: {r['base_original']}\n"
              f"  para: {r['paraphrase']}\n")


if __name__ == "__main__":
    main()
