"""Build input CSV for paraphrase pipeline from a CALVIN annotations YAML.

Output schema matches what main.py expects: columns `key, instruction`.

  key         = CALVIN task id (e.g. open_drawer)
  instruction = canonical English instruction (first item in YAML list)

Optional task-subset file: one task id per line. If omitted, all keys in the
YAML are exported.
"""
import argparse
import csv
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_yaml", required=True,
                    help="path to CALVIN annotations yaml (e.g. new_playtable_validation.yaml)")
    ap.add_argument("--task_list", default=None,
                    help="optional file with one task id per line; defaults to all keys")
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    with open(args.annotations_yaml) as f:
        anns = yaml.safe_load(f)

    if args.task_list:
        with open(args.task_list) as f:
            task_ids = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        task_ids = list(anns.keys())

    rows = []
    missing = []
    for tid in task_ids:
        if tid not in anns:
            missing.append(tid)
            continue
        val = anns[tid]
        instr = val[0] if isinstance(val, list) else str(val)
        rows.append({"key": tid, "instruction": instr})

    if missing:
        print(f"WARN: {len(missing)} tasks missing from yaml: {missing}")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "instruction"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
