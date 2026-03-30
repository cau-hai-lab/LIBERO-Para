"""
Xiaomi-Robotics-0 LIBERO-Para eval with structured JSON logging.

Standalone script — no LIBERO benchmark registration needed.
Uses libero_goal envs (10) + libero_para bddl instructions.

Run from: eval_scripts/examples/

Usage:
    # 1. Start Xiaomi server first (in mibot env):
    #    conda activate libero-para-xiaomi-mibot
    #    CUDA_VISIBLE_DEVICES=0 bash scripts/deploy.sh XiaomiRobotics/Xiaomi-Robotics-0-LIBERO 1 1

    # 2. Run this script (in libero env):
    #    conda activate libero-para-xiaomi-libero
    #    export MUJOCO_GL=egl
    #    python eval_scripts/examples/eval_xiaomi_robotics_0.py \
    #        --host 0.0.0.0 --port 10086 \
    #        --seed 7 \
    #        --output_dir ./logs_para/xiaomi/seed7/
"""

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np
import imageio
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Patch torch.load for LIBERO compatibility
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": False}
)

os.environ["MUJOCO_GL"] = "egl"


# =====================================================================================
# BDDL Parsing Utilities
# =====================================================================================

KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}


def parse_bddl_instruction(bddl_path: str) -> str:
    """Parse the (:language ...) instruction from a BDDL file."""
    with open(bddl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("(:language"):
                instruction = line.replace("(:language", "").rstrip(")").strip()
                return instruction
    return ""


def parse_bddl_filename(filename: str) -> dict:
    """
    Parse BDDL filename into structured metadata.

    Patterns:
        act_{category}_{subcategory}_eval{id}_ver{id}.bddl
        obj_{category}_{subcategory}_eval{id}_ver{id}.bddl
        comp_{cat1}+{cat2}_{subcat1}+{subcat2}_eval{id}_ver{id}.bddl
    """
    basename = os.path.basename(filename).replace(".bddl", "")

    if "_eval" not in basename:
        logging.warning(f"Cannot parse BDDL filename (no _eval found): {filename}")
        return {
            "paraphrase_type": "unknown",
            "categories": [],
            "subcategories": [],
            "eval_id": -1,
            "variant_id": -1,
        }

    prefix_part, eval_ver_part = basename.rsplit("_eval", 1)

    eval_str, ver_str = eval_ver_part.split("_ver")
    eval_id = int(eval_str)
    variant_id = int(ver_str)

    if prefix_part.startswith("comp_"):
        paraphrase_type = "comp"
        body = prefix_part[5:]

        first_plus = body.index("+")
        cat1 = body[:first_plus]
        remainder = body[first_plus + 1 :]

        cat2 = None
        subcat1 = None
        subcat2 = None
        for known_cat in KNOWN_CATEGORIES:
            if remainder.startswith(known_cat + "_"):
                cat2 = known_cat
                after_cat2 = remainder[len(known_cat) + 1 :]
                subcat1, subcat2 = after_cat2.rsplit("+", 1)
                break

        if cat2 is None:
            categories = [body]
            subcategories = [body]
        else:
            categories = [cat1, cat2]
            subcategories = [subcat1, subcat2]

    elif prefix_part.startswith("act_"):
        paraphrase_type = "act"
        body = prefix_part[4:]
        cat, subcat = _split_category_subcategory(body)
        categories = [cat]
        subcategories = [subcat]

    elif prefix_part.startswith("obj_"):
        paraphrase_type = "obj"
        body = prefix_part[4:]
        cat, subcat = _split_category_subcategory(body)
        categories = [cat]
        subcategories = [subcat]
    else:
        paraphrase_type = "unknown"
        categories = []
        subcategories = []

    return {
        "paraphrase_type": paraphrase_type,
        "categories": categories,
        "subcategories": subcategories,
        "eval_id": eval_id,
        "variant_id": variant_id,
    }


def _split_category_subcategory(body: str) -> tuple:
    for known_cat in KNOWN_CATEGORIES:
        if body.startswith(known_cat + "_"):
            return known_cat, body[len(known_cat) + 1 :]
    return body, ""


def extract_eval_id(bddl_file: str) -> int:
    """Extract eval_id from bddl filename. e.g. 'act_..._eval3_ver1.bddl' -> 3"""
    m = re.search(r"eval(\d+)", bddl_file)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract eval_id from: {bddl_file}")


# =====================================================================================
# Xiaomi Model Utilities
# =====================================================================================


def _quat2axisangle(quat):
    """Copied from robosuite."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def hash_data_to_seed(data, max_bytes=4):
    """Compute stable hash for model inputs (from Xiaomi's code)."""

    def custom_encoder(obj):
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "numpy",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tobytes().hex(),
            }
        if isinstance(obj, Image.Image):
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()
            return {
                "__type__": "PIL.Image",
                "mode": obj.mode,
                "size": obj.size,
                "content_hash": img_hash,
            }
        if isinstance(obj, set):
            return sorted(list(obj))
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    json_str = json.dumps(
        data,
        default=custom_encoder,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    hex_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    seed_int = int(hex_hash, 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))
    return seed_int


# =====================================================================================
# JSON Logger (same as X-VLA version for consistency)
# =====================================================================================


class StructuredLogger:
    """Manages structured JSON logging per eval_id."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._cache = {}

    def _get_eval_path(self, eval_id: int) -> str:
        if eval_id == -1:
            return os.path.join(self.log_dir, "eval_unknown.json")
        return os.path.join(self.log_dir, f"eval{eval_id}.json")

    def _load_eval(self, eval_id: int) -> dict:
        if eval_id in self._cache:
            return self._cache[eval_id]
        path = self._get_eval_path(eval_id)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = {"eval_id": eval_id, "original_instruction": None, "episodes": []}
        self._cache[eval_id] = data
        return data

    def _flush_eval(self, eval_id: int):
        data = self._cache.get(eval_id)
        if data is None:
            return
        path = self._get_eval_path(eval_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def log_episode(
        self,
        task_id: int,
        bddl_file: str,
        paraphrased_instruction: str,
        success: bool,
        num_steps: int,
        actions: list,
        proprio_states: list,
        action_chunk_boundaries: list,
        rewards: list,
        replay_video_path: str,
        initial_state_idx: int,
        episode_idx: int,
    ):
        bddl_basename = os.path.basename(bddl_file)
        parsed = parse_bddl_filename(bddl_basename)
        eval_id = parsed["eval_id"]

        eval_data = self._load_eval(eval_id)

        episode_record = {
            "task_id": task_id,
            "bddl_file": bddl_basename,
            "paraphrase_type": parsed["paraphrase_type"],
            "categories": parsed["categories"],
            "subcategories": parsed["subcategories"],
            "variant_id": parsed["variant_id"],
            "paraphrased_instruction": paraphrased_instruction,
            "success": bool(success),
            "num_steps": int(num_steps),
            "actions": actions,
            "proprio_states": proprio_states,
            "action_chunk_boundaries": action_chunk_boundaries,
            "rewards": rewards,
            "replay_video_path": replay_video_path,
            "initial_state_idx": int(initial_state_idx),
            "episode_idx": int(episode_idx),
        }

        eval_data["episodes"].append(episode_record)
        self._flush_eval(eval_id)

    def save_meta(self, meta_dict: dict):
        path = os.path.join(self.log_dir, "meta.json")
        with open(path, "w") as f:
            json.dump(meta_dict, f, indent=2)

    def save_summary(self, total_episodes: int, total_successes: int):
        per_eval = {}
        per_category = {}

        for eval_id, data in self._cache.items():
            episodes = data["episodes"]
            if len(episodes) == 0:
                continue

            successes = sum(1 for ep in episodes if ep["success"])
            per_eval[f"eval{eval_id}"] = {
                "total": len(episodes),
                "successes": successes,
                "success_rate": successes / len(episodes),
            }

            for ep in episodes:
                if ep["paraphrase_type"] == "comp":
                    cat_key = f"comp_{'+'.join(ep['categories'])}_{'+'.join(ep['subcategories'])}"
                else:
                    cat_key = (
                        f"{ep['paraphrase_type']}_{ep['categories'][0]}_{ep['subcategories'][0]}"
                        if ep["categories"]
                        else "unknown"
                    )

                if cat_key not in per_category:
                    per_category[cat_key] = {"total": 0, "successes": 0}
                per_category[cat_key]["total"] += 1
                if ep["success"]:
                    per_category[cat_key]["successes"] += 1

        for key in per_category:
            stats = per_category[key]
            stats["success_rate"] = (
                stats["successes"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        summary = {
            "overall_success_rate": (
                total_successes / total_episodes if total_episodes > 0 else 0.0
            ),
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "per_eval": per_eval,
            "per_category": per_category,
        }

        path = os.path.join(self.log_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# =====================================================================================
# Main
# =====================================================================================

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _get_repo_root():
    """Get LIBERO-Para repo root from this script's location."""
    return str(Path(__file__).resolve().parent.parent.parent)


def main():
    repo_root = _get_repo_root()
    default_bddl = os.path.join(repo_root, "libero/libero/bddl_files/libero_para")
    default_init = os.path.join(repo_root, "libero/libero/init_files/libero_para")
    default_goal = os.path.join(repo_root, "libero/libero/bddl_files/libero_goal")

    parser = argparse.ArgumentParser(description="Xiaomi-Robotics-0 LIBERO-Para Eval")
    parser.add_argument(
        "--bddl_dir",
        type=str,
        default=default_bddl,
        help="Path to libero_para bddl files (paraphrased instructions)",
    )
    parser.add_argument(
        "--init_dir",
        type=str,
        default=default_init,
        help="Path to libero_para init_files (eval0~9.pruned_init)",
    )
    parser.add_argument(
        "--goal_bddl_dir",
        type=str,
        default=default_goal,
        help="Path to libero_goal bddl files (for env creation)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="./logs_para/xiaomi/")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--replan_steps", type=int, default=10)
    parser.add_argument("--max_tasks", type=int, default=-1, help="Limit number of tasks for debugging")
    parser.add_argument("--save_video", action="store_true", help="Save replay videos")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 1. Scan paraphrase bddl files ----
    bddl_files = sorted(
        [f for f in os.listdir(args.bddl_dir) if f.endswith(".bddl")]
    )
    logging.info(f"Found {len(bddl_files)} paraphrase bddl files")

    if args.max_tasks > 0:
        bddl_files = bddl_files[: args.max_tasks]
        logging.info(f"Limited to {len(bddl_files)} tasks (debug mode)")

    # Parse all bddl files: extract eval_id and instruction
    tasks = []
    for bddl_file in bddl_files:
        bddl_path = os.path.join(args.bddl_dir, bddl_file)
        instruction = parse_bddl_instruction(bddl_path)
        eval_id = extract_eval_id(bddl_file)
        tasks.append(
            {"bddl_file": bddl_file, "instruction": instruction, "eval_id": eval_id}
        )

    eval_ids_needed = sorted(set(t["eval_id"] for t in tasks))
    logging.info(f"Eval IDs needed: {eval_ids_needed}")

    # ---- 2. Load init states (per eval_id) ----
    init_states = {}
    for eval_id in eval_ids_needed:
        init_path = os.path.join(args.init_dir, f"eval{eval_id}.pruned_init")
        init_states[eval_id] = torch.load(init_path)
        logging.info(
            f"  Loaded init_states for eval{eval_id}: {len(init_states[eval_id])} states"
        )

    # ---- 3. Create envs (10 envs from libero_goal bddl) ----
    from libero.libero.envs import OffScreenRenderEnv

    goal_bddl_files = sorted(
        [f for f in os.listdir(args.goal_bddl_dir) if f.endswith(".bddl")]
    )
    logging.info(f"Found {len(goal_bddl_files)} goal bddl files")

    envs = {}
    for idx, goal_bddl in enumerate(goal_bddl_files):
        goal_bddl_path = os.path.join(args.goal_bddl_dir, goal_bddl)
        env = OffScreenRenderEnv(
            bddl_file_name=goal_bddl_path,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        env.reset()
        envs[idx] = env
        logging.info(f"  Created env {idx}: {goal_bddl}")

    logging.info(f"Total envs created: {len(envs)}")

    # ---- 4. Connect to Xiaomi server ----
    xiaomi_root = os.path.join(repo_root, "eval_scripts/xiaomi-robotics-0")
    sys.path.insert(0, xiaomi_root)
    from deploy.client import Client

    client = Client(args.host, args.port)
    logging.info(f"Connected to Xiaomi server at {args.host}:{args.port}")

    # ---- 5. Setup structured logger ----
    structured_logger = StructuredLogger(log_dir=args.output_dir)

    meta = {
        "model_name": "Xiaomi-Robotics-0",
        "model_family": "xiaomi_mot",
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "bddl_dir": args.bddl_dir,
        "total_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    structured_logger.save_meta(meta)

    # ---- 6. Eval loop ----
    total_episodes = 0
    total_successes = 0
    start_time = time.time()

    for task_idx, task_info in enumerate(tqdm(tasks, desc="Eval")):
        bddl_file = task_info["bddl_file"]
        instruction = task_info["instruction"]
        eval_id = task_info["eval_id"]

        # Select env and init state
        env = envs[eval_id]
        init_state = init_states[eval_id][0]  # Use first init state

        # Reset env
        raw_obs = env.reset()
        raw_obs = env.set_init_state(init_state)

        # Wait for stabilization
        for _ in range(args.num_steps_wait):
            raw_obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        # Trajectory recording
        all_actions = []
        all_proprio_states = []
        all_rewards = []
        action_chunk_boundaries = []
        replay_images = []

        action_plan = collections.deque()
        success = False

        for step in range(args.max_steps):
            # Get observation images
            img = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(
                raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]
            )
            img = convert_to_uint8(img)
            wrist_img = convert_to_uint8(wrist_img)

            if args.save_video:
                replay_images.append(img.copy())

            # Record proprio state
            eef_pos = raw_obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = raw_obs.get("robot0_eef_quat", np.zeros(4))
            gripper_qpos = raw_obs.get("robot0_gripper_qpos", np.zeros(2))
            proprio = np.concatenate(
                (eef_pos, _quat2axisangle(eef_quat.copy()), gripper_qpos)
            )
            all_proprio_states.append(proprio.tolist())

            # Query model when action plan is empty
            if len(action_plan) <= 0:
                action_chunk_boundaries.append(step)

                base_obs = Image.fromarray(img)
                left_wrist_obs = Image.fromarray(wrist_img)
                state = np.concatenate(
                    [
                        raw_obs["robot0_eef_pos"],
                        _quat2axisangle(raw_obs["robot0_eef_quat"]),
                        raw_obs["robot0_gripper_qpos"],
                        np.array([0.0] * 24),
                    ]
                )

                # Capitalize instruction (Xiaomi convention)
                lang = str(instruction).capitalize() + "."

                model_inputs = {
                    "task_id": "libero_all",
                    "state": state,
                    "base": base_obs,
                    "wrist_left": left_wrist_obs,
                    "language": lang,
                }
                temp_seed = hash_data_to_seed(
                    {**model_inputs, "language": lang}
                )
                model_inputs["seed"] = temp_seed

                action_chunk = client(**model_inputs)[0, :, :-1]
                action_plan.extend(action_chunk[0 : args.replan_steps, 0:7])

            action = action_plan.popleft().tolist()
            all_actions.append(action)

            raw_obs, reward, done, info = env.step(action)
            all_rewards.append(float(reward))

            if done or env.check_success():
                success = True
                break

        num_steps = len(all_actions)
        total_episodes += 1
        if success:
            total_successes += 1

        # Save video
        video_path = ""
        if args.save_video and replay_images:
            suffix = "success" if success else "failure"
            video_dir = os.path.join(args.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(
                video_dir,
                f"eval{eval_id}_{os.path.splitext(bddl_file)[0]}_{suffix}.mp4",
            )
            imageio.mimwrite(video_path, replay_images, fps=10)

        # Log to structured JSON
        structured_logger.log_episode(
            task_id=task_idx,
            bddl_file=bddl_file,
            paraphrased_instruction=instruction,
            success=success,
            num_steps=num_steps,
            actions=all_actions,
            proprio_states=all_proprio_states,
            action_chunk_boundaries=action_chunk_boundaries,
            rewards=all_rewards,
            replay_video_path=video_path,
            initial_state_idx=eval_id,
            episode_idx=0,
        )

        # Print progress
        _sr = total_successes / total_episodes * 100
        print(
            f"  Task {task_idx} [eval{eval_id}]: "
            f"{'✓' if success else '✗'} | "
            f"{instruction[:55]} | SR: {_sr:.1f}%"
        )

        if (task_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (task_idx + 1) * (len(tasks) - task_idx - 1)
            print(
                f"[{task_idx+1}/{len(tasks)}] SR: {_sr:.1f}% | "
                f"Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h"
            )

    # ---- 7. Save summary & cleanup ----
    structured_logger.save_summary(total_episodes, total_successes)

    for env in envs.values():
        env.close()

    total_sr = total_successes / total_episodes * 100 if total_episodes > 0 else 0
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Model: Xiaomi-Robotics-0")
    print(f"Seed: {args.seed}")
    print(f"Total tasks: {total_episodes}")
    print(f"Success rate: {total_sr:.1f}%")
    print(f"Total time: {elapsed/3600:.1f}h")
    print(f"Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()