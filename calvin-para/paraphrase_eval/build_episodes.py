"""Build per-episode JSON files for canonical baseline + paraphrase eval.

Baseline:
  - For each task in 15-task subset, sample 20 trials = 20 different
    initial states, each tagged with the canonical instruction.
  - Output: <out>/baseline_seed{S}.json  (300 episodes)

Paraphrase:
  - Read paraphrases.json (1935 entries).
  - For each paraphrase, attach an init_state for that base_task
    (deterministic from --paraphrase_seed by indexing into the same
    list used for baseline).
  - Output: <out>/paraphrase_seed{S}.json  (1935 episodes)

Each episode entry:
  {
    "episode_idx": int,
    "init_state": {"robot_obs": [...], "scene_obs": [...]},
    "task_id":    "open_drawer",
    "lang_override": "<instruction>",
    "metadata": {
      "kind": "baseline" | "paraphrase",
      "trial_idx": int (baseline only),
      "paraphrase_id": str (paraphrase only),
      "cell_id": str (paraphrase only),
      "object_type": str, "action_type": str, "paraphrase_index": int
    }
  }
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CALVIN_PARA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALVIN_PARA / "calvin/calvin_models"))
from calvin_agent.evaluation.multistep_sequences import get_sequences


def sample_initial_states(tasks, trials_per_task, base_n=10000):
    """For each task, return up to `trials_per_task` initial_state dicts where
    that task is the FIRST task in a sampled CALVIN sequence.
    """
    print(f"sampling {base_n} base sequences for initial-state pool ...")
    seqs = get_sequences(base_n)
    per_task = defaultdict(list)
    for init_state, seq in seqs:
        first = str(seq[0])
        if first in tasks and len(per_task[first]) < trials_per_task:
            per_task[first].append(init_state)
    for t in tasks:
        if len(per_task[t]) < trials_per_task:
            print(f"  WARN: {t} only has {len(per_task[t])}/{trials_per_task}")
    return per_task


def normalise_init_state(s):
    """get_sequences returns dicts of np arrays / lists; force JSON-serialisable."""
    out = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = [vv.tolist() if isinstance(vv, np.ndarray) else vv for vv in v]
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_list", required=True)
    ap.add_argument("--annotations_yaml", required=True,
                    help="canonical annotations yaml (for baseline lang_override)")
    ap.add_argument("--paraphrases_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--baseline_seeds", nargs="+", type=int, default=[7, 8, 9, 10, 11])
    ap.add_argument("--paraphrase_seed", type=int, default=7,
                    help="which seed's init-state pool to use for paraphrase episodes")
    ap.add_argument("--trials_per_task", type=int, default=20)
    ap.add_argument("--base_n", type=int, default=20000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [ln.strip() for ln in open(args.task_list) if ln.strip() and not ln.startswith("#")]
    print(f"tasks: {tasks}")

    import yaml
    canonical = yaml.safe_load(open(args.annotations_yaml))
    canonical_text = {t: (canonical[t][0] if isinstance(canonical[t], list) else canonical[t])
                      for t in tasks}

    # ---- Baseline: per seed, 20 trials per task ----
    # We seed get_sequences via numpy/random RNG by setting np.random.seed before sampling.
    for seed in args.baseline_seeds:
        np.random.seed(seed)
        per_task = sample_initial_states(tasks, args.trials_per_task, base_n=args.base_n)

        episodes = []
        idx = 0
        for t in tasks:
            for trial_i, init in enumerate(per_task[t]):
                episodes.append({
                    "episode_idx": idx,
                    "init_state": normalise_init_state(init),
                    "task_id": t,
                    "lang_override": canonical_text[t],
                    "metadata": {
                        "kind": "baseline",
                        "trial_idx": trial_i,
                        "seed": seed,
                    },
                })
                idx += 1

        out_path = out_dir / f"baseline_seed{seed}.json"
        out_path.write_text(json.dumps(episodes))
        print(f"  wrote {len(episodes)} baseline episodes -> {out_path}")

    # ---- Paraphrase: 1935 episodes, each tied to its base_task's init_state ----
    np.random.seed(args.paraphrase_seed)
    per_task_para = sample_initial_states(tasks, args.trials_per_task, base_n=args.base_n)

    paraphrases = json.loads(Path(args.paraphrases_json).read_text())

    # group paraphrases by task, then INTERLEAVE so contiguous chunks have
    # similar task distribution (avoids workload imbalance when running RF in
    # 2-process split mode).
    by_task = defaultdict(list)
    for p in paraphrases:
        by_task[p["base_task_id"]].append(p)
    task_order = [t for t in tasks if t in by_task]
    max_per_task = max(len(by_task[t]) for t in task_order)

    interleaved = []
    for k in range(max_per_task):
        for t in task_order:
            if k < len(by_task[t]):
                interleaved.append(by_task[t][k])

    cursor = defaultdict(int)
    episodes = []
    skipped = []
    for p in interleaved:
        t = p["base_task_id"]
        if t not in per_task_para or not per_task_para[t]:
            skipped.append(p["paraphrase_id"])
            continue
        init = per_task_para[t][cursor[t] % len(per_task_para[t])]
        cursor[t] += 1
        episodes.append({
            "episode_idx": len(episodes),
            "init_state": normalise_init_state(init),
            "task_id": t,
            "lang_override": p["paraphrase"],
            "metadata": {
                "kind": "paraphrase",
                "paraphrase_id": p["paraphrase_id"],
                "cell_id": p["cell_id"],
                "object_type": p["object_type"],
                "action_type": p["action_type"],
                "paraphrase_index": p["paraphrase_index"],
                "base_original": p["base_original"],
                "seed": args.paraphrase_seed,
            },
        })

    out_path = out_dir / f"paraphrase_seed{args.paraphrase_seed}.json"
    out_path.write_text(json.dumps(episodes))
    print(f"\nwrote {len(episodes)} paraphrase episodes -> {out_path}")
    if skipped:
        print(f"  skipped {len(skipped)} paraphrases for missing tasks: {skipped[:5]}...")


if __name__ == "__main__":
    main()
