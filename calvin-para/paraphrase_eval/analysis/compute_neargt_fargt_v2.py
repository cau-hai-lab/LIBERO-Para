"""v2 — Cluster-aware Near-GT / Far-GT / Cross-Near-GT.

Block-manipulation tasks have multi-modal initial block positions (~2 clusters
in X, "left" vs "right"). Building one envelope from ALL baseline successes
yields a wide τ_max that masks real wandering. v2 splits each task's baseline
successes into K position clusters and uses a per-cluster envelope.

Pipeline:
1) For each task, gather baseline successful trajectories.
2) Estimate "block xy" per baseline trial = gripper xyz at z-minimum point.
3) Cluster trials by xy (KMeans, K chosen per task; default 2 for block tasks,
   1 for static tasks).
4) Build per-cluster GT mean trajectory + τ_max (max DTW within cluster).
5) Paraphrase episodes share the same init_state pool as baseline (cursor
   cycle), so each paraphrase episode gets the cluster of its paired baseline
   trial.
6) For each paraphrase ep:
   - Compute DTW to its cluster's GT.
   - Classify success / NearGT / FarGT.
   - For FarGT: cross-check against every (other_task, cluster) pair.

Outputs: cache/<model>__neargt_fargt_v2.json, summary printed.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

CALVIN_PARA = Path(__file__).resolve().parents[2]

N_RESAMPLE = 50

STATIC_TASKS = {
    "open_drawer", "close_drawer",
    "move_slider_left", "move_slider_right",
    "turn_on_lightbulb", "turn_off_lightbulb",
    "turn_on_led", "turn_off_led",
}
DEFAULT_K_DYNAMIC = 2  # left/right cluster


def resample_trajectory(traj, n):
    T = len(traj)
    if T == 0: return np.zeros((n, 3))
    if T == 1: return np.tile(traj[0], (n, 1))
    old_t = np.linspace(0, 1, T); new_t = np.linspace(0, 1, n)
    out = np.zeros((n, 3))
    for d in range(3): out[:, d] = np.interp(new_t, old_t, traj[:, d])
    return out


def compute_dtw(a, b):
    dist, _ = fastdtw(a, b, dist=euclidean)
    return float(dist / len(a))


def extract_xyz(ep):
    xyz = ep.get("eef_xyz")
    if xyz is None: return np.zeros((0, 3))
    arr = np.array([p for p in xyz if p is not None], dtype=float)
    return arr if arr.size else np.zeros((0, 3))


def grasp_xy(eef_arr):
    """Estimate block xy as gripper position at z-minimum."""
    if len(eef_arr) < 5: return None
    return eef_arr[np.argmin(eef_arr[:, 2])][:2]


def load_episodes_by_task(model_results_dir):
    by_task = defaultdict(list)
    for ef in sorted(glob.glob(os.path.join(model_results_dir, "eval_*.json"))):
        d = json.load(open(ef))
        for ep in d.get("episodes", []):
            by_task[d["task_id"]].append(ep)
    return dict(by_task)


def build_clustered_gt(baseline_dir, k_per_task=None) -> Dict[str, dict]:
    """Per task: K clusters, each with mean GT + τ_max + member trial_idxs."""
    if k_per_task is None: k_per_task = {}
    by_task = load_episodes_by_task(baseline_dir)
    gt = {}
    for task_id, eps in by_task.items():
        succ = [ep for ep in eps if ep.get("success", False)]
        if not succ:
            print(f"  WARN: no baseline successes for {task_id}; skipping")
            continue

        # decide K
        K = k_per_task.get(task_id)
        if K is None:
            K = 1 if task_id in STATIC_TASKS else DEFAULT_K_DYNAMIC

        trajs = [resample_trajectory(extract_xyz(ep), N_RESAMPLE) for ep in succ]
        trial_idxs = [ep["metadata"].get("trial_idx", -1) for ep in succ]

        # cluster by grasp xy if K>1
        if K > 1:
            xys = np.array([grasp_xy(extract_xyz(ep)) for ep in succ])
            if any(x is None for x in xys):
                print(f"  WARN: {task_id} has missing grasp xy, falling back to K=1")
                K = 1

        if K == 1:
            cluster_labels = np.zeros(len(succ), dtype=int)
        else:
            km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(xys)
            cluster_labels = km.labels_

        clusters = {}
        for c in range(K):
            members = [i for i, l in enumerate(cluster_labels) if l == c]
            if len(members) < 1: continue
            ctrajs = [trajs[i] for i in members]
            mean_traj = np.mean(ctrajs, axis=0)
            within_dtws = [compute_dtw(t, mean_traj) for t in ctrajs]
            tau = max(within_dtws) if within_dtws else 0.0
            cluster_centroid_xy = (np.array([grasp_xy(extract_xyz(succ[i])) for i in members]).mean(axis=0).tolist()
                                    if K > 1 else None)
            clusters[c] = {
                "gt_traj": mean_traj.tolist(),
                "tau_max": tau,
                "n_baseline": len(members),
                "trial_idxs": [trial_idxs[i] for i in members],
                "centroid_xy": cluster_centroid_xy,
            }
        gt[task_id] = {"K": K, "clusters": clusters}
    return gt


def build_trial_to_cluster(gt: Dict[str, dict]) -> Dict[Tuple[str, int], int]:
    """Map (task_id, baseline_trial_idx) -> cluster id."""
    out = {}
    for task_id, info in gt.items():
        for c, cd in info["clusters"].items():
            for ti in cd["trial_idxs"]:
                out[(task_id, ti)] = c
    return out


def _classify_one(args):
    ep, task_id, cluster_id, gt_dict, tau_dict = args
    xyz = extract_xyz(ep)
    if len(xyz) == 0: return None
    ep_traj = resample_trajectory(xyz, N_RESAMPLE)

    own_gt = np.array(gt_dict[(task_id, cluster_id)])
    own_tau = tau_dict[(task_id, cluster_id)]
    dtw_own = compute_dtw(ep_traj, own_gt)
    success = bool(ep.get("success", False))

    record = {
        "task_id": task_id,
        "cluster_id": int(cluster_id),
        "paraphrase_id": ep["metadata"].get("paraphrase_id"),
        "cell_id": ep["metadata"].get("cell_id"),
        "object_type": ep["metadata"].get("object_type"),
        "action_type": ep["metadata"].get("action_type"),
        "paraphrase_index": ep["metadata"].get("paraphrase_index"),
        "lang_override": ep.get("lang_override"),
        "num_steps": ep.get("num_steps"),
        "success": success,
        "dtw_own_gt": round(dtw_own, 6),
        "own_tau_max": round(own_tau, 6),
    }

    if success:
        record["category"] = "Success"
        record["is_neargt"] = False
        record["is_fargt"] = False
        record["is_cross_neargt"] = False
        return record

    if dtw_own <= own_tau:
        record["category"] = "NearGT"
        record["is_neargt"] = True
        record["is_fargt"] = False
        record["is_cross_neargt"] = False
        return record

    # FarGT — cross-check vs every (other_task, cluster) GT
    cross_dtws = {}
    for (other_task, other_cluster), other_gt in gt_dict.items():
        if (other_task, other_cluster) == (task_id, cluster_id):
            cross_dtws[(other_task, other_cluster)] = dtw_own
        else:
            cross_dtws[(other_task, other_cluster)] = compute_dtw(ep_traj, np.array(other_gt))
    others = {k: v for k, v in cross_dtws.items() if k != (task_id, cluster_id)}
    nearest = min(others, key=others.get)
    nearest_dtw = others[nearest]
    nearest_tau = tau_dict[nearest]

    record["category"] = "Cross-NearGT" if nearest_dtw <= nearest_tau else "Unmatched"
    record["is_neargt"] = False
    record["is_fargt"] = True
    record["is_cross_neargt"] = nearest_dtw <= nearest_tau
    record["nearest_other_task"] = nearest[0]
    record["nearest_other_cluster"] = int(nearest[1])
    record["nearest_other_dtw"] = round(nearest_dtw, 6)
    record["nearest_other_tau_max"] = round(nearest_tau, 6)
    return record


def classify_paraphrase(paraphrase_dir, baseline_episodes_json, gt, n_workers=8):
    """Use paraphrase episode metadata's `paraphrase_index` to map to baseline trial.

    Better: directly compare init_state dicts between paraphrase ep and baseline
    ep, since cursor cycle in build_episodes.py guarantees same dicts repeat.
    """
    by_task = load_episodes_by_task(paraphrase_dir)

    # build baseline init_state lookup: (task, init_dict_hash) -> trial_idx
    baseline_eps = json.load(open(baseline_episodes_json))
    base_init_to_trial = {}
    for be in baseline_eps:
        key = (be["task_id"], json.dumps(be["init_state"], sort_keys=True))
        base_init_to_trial[key] = be["metadata"]["trial_idx"]

    # map (task, trial_idx) -> cluster
    trial_to_cluster = build_trial_to_cluster(gt)

    # paraphrase episodes JSON (for init_state lookup)
    para_eps_json = json.load(open(
        os.path.join(os.path.dirname(baseline_episodes_json), "paraphrase_seed7.json")
    ))
    para_idx_to_init = {p["episode_idx"]: p["init_state"] for p in para_eps_json}

    # gt_dict / tau_dict keyed by (task, cluster_id)
    gt_dict = {}
    tau_dict = {}
    for task_id, info in gt.items():
        for c, cd in info["clusters"].items():
            gt_dict[(task_id, c)] = cd["gt_traj"]
            tau_dict[(task_id, c)] = cd["tau_max"]

    args_list = []
    skipped = 0
    for task_id, eps in by_task.items():
        if task_id not in gt:
            print(f"  WARN: paraphrase task {task_id} missing in baseline GT; skipping")
            continue
        for ep in eps:
            init = para_idx_to_init.get(ep["episode_idx"])
            if init is None:
                skipped += 1; continue
            key = (task_id, json.dumps(init, sort_keys=True))
            trial = base_init_to_trial.get(key)
            if trial is None:
                skipped += 1; continue
            cluster = trial_to_cluster.get((task_id, trial))
            if cluster is None:
                skipped += 1; continue
            args_list.append((ep, task_id, cluster, gt_dict, tau_dict))

    if skipped:
        print(f"  skipped {skipped} paraphrase eps with no matching baseline trial / cluster")

    print(f"  classifying {len(args_list)} paraphrase episodes ({n_workers} workers)...")
    with Pool(processes=n_workers) as pool:
        out = pool.map(_classify_one, args_list)
    return [r for r in out if r is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--results_root", default=str(CALVIN_PARA / "paraphrase_eval/results"))
    ap.add_argument("--episodes_root", default=str(CALVIN_PARA / "paraphrase_eval/episodes"))
    ap.add_argument("--baseline_seed", default="seed7")
    ap.add_argument("--paraphrase_seed", default="seed7")
    ap.add_argument("--cache_dir", default=str(CALVIN_PARA / "paraphrase_eval/analysis/cache"))
    ap.add_argument("--n_workers", type=int, default=8)
    args = ap.parse_args()

    baseline_dir = os.path.join(args.results_root, args.model, f"baseline_{args.baseline_seed}")
    paraphrase_dir = os.path.join(args.results_root, args.model, f"paraphrase_{args.paraphrase_seed}")
    baseline_episodes_json = os.path.join(args.episodes_root, f"baseline_{args.baseline_seed}.json")

    print(f"[{args.model}] building cluster-aware GT from {baseline_dir}")
    gt = build_clustered_gt(baseline_dir)
    for t, info in sorted(gt.items()):
        print(f"  {t:30s} K={info['K']}")
        for c, cd in info["clusters"].items():
            cstr = f"centroid_xy={[round(x,3) for x in cd['centroid_xy']]}" if cd['centroid_xy'] else ""
            print(f"    cluster {c}: n={cd['n_baseline']}  τ_max={cd['tau_max']:.4f}  {cstr}")

    print(f"\n[{args.model}] classifying paraphrase episodes from {paraphrase_dir}")
    results = classify_paraphrase(paraphrase_dir, baseline_episodes_json, gt, n_workers=args.n_workers)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.cache_dir, f"{args.model}__neargt_fargt_v2.json")
    json.dump({
        "model": args.model,
        "n_episodes": len(results),
        "gt_summary": {
            t: {
                "K": info["K"],
                "clusters": {str(c): {"tau_max": cd["tau_max"], "n_baseline": cd["n_baseline"],
                                       "centroid_xy": cd["centroid_xy"]}
                              for c, cd in info["clusters"].items()},
            } for t, info in gt.items()
        },
        "episodes": results,
    }, open(out_path, "w"))
    print(f"\nwrote {len(results)} episodes -> {out_path}")

    from collections import Counter
    cats = Counter(r["category"] for r in results)
    total = sum(cats.values())
    print(f"\noverall ({total} eps):")
    for c, n in cats.most_common():
        print(f"  {c:15s} {n:>5}  ({n/total*100:.1f}%)")
    failures = total - cats.get("Success", 0)
    print(f"\n--- failures-only breakdown ({failures}) ---")
    for c in ["NearGT", "Cross-NearGT", "Unmatched"]:
        n = cats.get(c, 0)
        print(f"  {c:15s} {n:>5}  ({n/failures*100:.1f}% of failures)")


if __name__ == "__main__":
    main()
