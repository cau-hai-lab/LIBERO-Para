"""
OpenVLA-OFT LIBERO-Para eval with structured JSON logging.

Standalone script — no LIBERO benchmark registration needed.
Uses libero_goal envs (10) + libero_para bddl instructions.
Supports both goal and mixed variants via --pretrained_checkpoint.

Run from: LIBERO-Para repo root

Usage:
    conda activate libero-para-openvla-oft
    export MUJOCO_GL=egl

    # Goal variant
    CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_openvla_oft.py \
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
        --gpu 0 --seed 7 \
        --output_dir ./logs_para/openvla-oft-goal/seed7/

    # Mixed variant
    CUDA_VISIBLE_DEVICES=0 python eval_scripts/examples/eval_openvla_oft.py \
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal-mixed \
        --gpu 0 --seed 7 \
        --output_dir ./logs_para/openvla-oft-mixed/seed7/
"""

import argparse
import collections
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Patch torch.load for LIBERO compatibility
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": False}
)

os.environ["MUJOCO_GL"] = "egl"


def _get_repo_root():
    return str(Path(__file__).resolve().parent.parent.parent)


REPO_ROOT = _get_repo_root()
OPENVLA_OFT_ROOT = os.path.join(REPO_ROOT, "eval_scripts/openvla-oft")
sys.path.insert(0, OPENVLA_OFT_ROOT)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


# =====================================================================================
# BDDL Parsing Utilities
# =====================================================================================

KNOWN_CATEGORIES = {"lexical", "pragmatical", "structural"}


def parse_bddl_instruction(bddl_path: str) -> str:
    with open(bddl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("(:language"):
                return line.replace("(:language", "").rstrip(")").strip()
    return ""


def parse_bddl_filename(filename: str) -> dict:
    basename = os.path.basename(filename).replace(".bddl", "")
    if "_eval" not in basename:
        logging.warning(f"Cannot parse BDDL filename (no _eval found): {filename}")
        return {"paraphrase_type": "unknown", "categories": [], "subcategories": [], "eval_id": -1, "variant_id": -1}

    prefix_part, eval_ver_part = basename.rsplit("_eval", 1)
    eval_str, ver_str = eval_ver_part.split("_ver")
    eval_id = int(eval_str)
    variant_id = int(ver_str)

    if prefix_part.startswith("comp_"):
        paraphrase_type = "comp"
        body = prefix_part[5:]
        first_plus = body.index("+")
        cat1 = body[:first_plus]
        remainder = body[first_plus + 1:]
        cat2, subcat1, subcat2 = None, None, None
        for known_cat in KNOWN_CATEGORIES:
            if remainder.startswith(known_cat + "_"):
                cat2 = known_cat
                after_cat2 = remainder[len(known_cat) + 1:]
                subcat1, subcat2 = after_cat2.rsplit("+", 1)
                break
        if cat2 is None:
            categories, subcategories = [body], [body]
        else:
            categories, subcategories = [cat1, cat2], [subcat1, subcat2]
    elif prefix_part.startswith("act_"):
        paraphrase_type = "act"
        body = prefix_part[4:]
        cat, subcat = _split_category_subcategory(body)
        categories, subcategories = [cat], [subcat]
    elif prefix_part.startswith("obj_"):
        paraphrase_type = "obj"
        body = prefix_part[4:]
        cat, subcat = _split_category_subcategory(body)
        categories, subcategories = [cat], [subcat]
    else:
        paraphrase_type = "unknown"
        categories, subcategories = [], []

    return {"paraphrase_type": paraphrase_type, "categories": categories,
            "subcategories": subcategories, "eval_id": eval_id, "variant_id": variant_id}


def _split_category_subcategory(body: str) -> tuple:
    for known_cat in KNOWN_CATEGORIES:
        if body.startswith(known_cat + "_"):
            return known_cat, body[len(known_cat) + 1:]
    return body, ""


def extract_eval_id(bddl_file: str) -> int:
    m = re.search(r"eval(\d+)", bddl_file)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract eval_id from: {bddl_file}")


# =====================================================================================
# JSON Logger
# =====================================================================================

class StructuredLogger:
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

    def log_episode(self, task_id, bddl_file, paraphrased_instruction, success,
                    num_steps, actions, proprio_states, action_chunk_boundaries,
                    rewards, replay_video_path, initial_state_idx, episode_idx):
        bddl_basename = os.path.basename(bddl_file)
        parsed = parse_bddl_filename(bddl_basename)
        eval_id = parsed["eval_id"]
        eval_data = self._load_eval(eval_id)
        episode_record = {
            "task_id": task_id, "bddl_file": bddl_basename,
            "paraphrase_type": parsed["paraphrase_type"],
            "categories": parsed["categories"], "subcategories": parsed["subcategories"],
            "variant_id": parsed["variant_id"],
            "paraphrased_instruction": paraphrased_instruction,
            "success": bool(success), "num_steps": int(num_steps),
            "actions": actions, "proprio_states": proprio_states,
            "action_chunk_boundaries": action_chunk_boundaries,
            "rewards": rewards, "replay_video_path": replay_video_path,
            "initial_state_idx": int(initial_state_idx), "episode_idx": int(episode_idx),
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
            if not episodes:
                continue
            successes = sum(1 for ep in episodes if ep["success"])
            per_eval[f"eval{eval_id}"] = {
                "total": len(episodes), "successes": successes,
                "success_rate": successes / len(episodes),
            }
            for ep in episodes:
                if ep["paraphrase_type"] == "comp":
                    cat_key = f"comp_{'+'.join(ep['categories'])}_{'+'.join(ep['subcategories'])}"
                else:
                    cat_key = (f"{ep['paraphrase_type']}_{ep['categories'][0]}_{ep['subcategories'][0]}"
                               if ep["categories"] else "unknown")
                if cat_key not in per_category:
                    per_category[cat_key] = {"total": 0, "successes": 0}
                per_category[cat_key]["total"] += 1
                if ep["success"]:
                    per_category[cat_key]["successes"] += 1
        for key in per_category:
            stats = per_category[key]
            stats["success_rate"] = stats["successes"] / stats["total"] if stats["total"] > 0 else 0.0
        summary = {
            "overall_success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes, "total_successes": total_successes,
            "per_eval": per_eval, "per_category": per_category,
        }
        path = os.path.join(self.log_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# =====================================================================================
# Fake config for OpenVLA-OFT utils
# =====================================================================================

class FakeConfig:
    def __init__(self, args):
        self.model_family = "openvla"
        self.pretrained_checkpoint = args.pretrained_checkpoint
        self.use_l1_regression = True
        self.use_diffusion = False
        self.num_diffusion_steps = 50
        self.use_film = False
        self.num_images_in_input = 2
        self.use_proprio = True
        self.center_crop = True
        self.num_open_loop_steps = 8
        self.unnorm_key = ""
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.task_suite_name = "libero_goal"
        self.num_steps_wait = 10
        self.num_trials_per_task = 1
        self.initial_states_path = "DEFAULT"
        self.env_img_res = 256
        self.lora_rank = 32
        self.run_id_note = None
        self.local_log_dir = "./experiments/logs"
        self.use_wandb = False
        self.wandb_entity = ""
        self.wandb_project = ""
        self.seed = args.seed
        self.my_output_attentions = False


# =====================================================================================
# Main
# =====================================================================================

LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
LIBERO_ENV_RESOLUTION = 256


def resize_image_for_policy(img, resize_size):
    from experiments.robot.openvla_utils import resize_image_for_policy as _resize
    return _resize(img, resize_size)


def main():
    default_bddl = os.path.join(REPO_ROOT, "libero/libero/bddl_files/libero_para")
    default_init = os.path.join(REPO_ROOT, "libero/libero/init_files/libero_para")
    default_goal = os.path.join(REPO_ROOT, "libero/libero/bddl_files/libero_goal")

    parser = argparse.ArgumentParser(description="OpenVLA-OFT LIBERO-Para Eval")
    parser.add_argument("--bddl_dir", type=str, default=default_bddl)
    parser.add_argument("--init_dir", type=str, default=default_init)
    parser.add_argument("--goal_bddl_dir", type=str, default=default_goal)
    parser.add_argument("--pretrained_checkpoint", type=str,
                        default="moojink/openvla-7b-oft-finetuned-libero-goal")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=str, default="./logs_para/openvla-oft/")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "para", "original"],
                        help="para: libero_para paraphrases, original: standard LIBERO suites, auto: detect from bddl filenames")
    parser.add_argument("--max_tasks", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ---- 1. Scan bddl files ----
    bddl_files = sorted([f for f in os.listdir(args.bddl_dir) if f.endswith(".bddl")])
    logging.info(f"Found {len(bddl_files)} bddl files")

    if args.max_tasks > 0:
        bddl_files = bddl_files[:args.max_tasks]
        logging.info(f"Limited to {len(bddl_files)} tasks (debug mode)")

    mode = args.mode
    if mode == "auto":
        mode = "para" if any("_eval" in f for f in bddl_files) else "original"
    logging.info(f"Mode: {mode}")

    from libero.libero.envs import OffScreenRenderEnv

    if mode == "para":
        tasks = []
        for bddl_file in bddl_files:
            bddl_path = os.path.join(args.bddl_dir, bddl_file)
            instruction = parse_bddl_instruction(bddl_path)
            eval_id = extract_eval_id(bddl_file)
            tasks.append({"bddl_file": bddl_file, "instruction": instruction, "eval_id": eval_id})

        eval_ids_needed = sorted(set(t["eval_id"] for t in tasks))
        logging.info(f"Eval IDs needed: {eval_ids_needed}")

        init_states = {}
        for eval_id in eval_ids_needed:
            init_path = os.path.join(args.init_dir, f"eval{eval_id}.pruned_init")
            init_states[eval_id] = torch.load(init_path)
            logging.info(f"  Loaded init_states for eval{eval_id}: {len(init_states[eval_id])} states")

        goal_bddl_files = sorted([f for f in os.listdir(args.goal_bddl_dir) if f.endswith(".bddl")])
        envs = {}
        for idx, goal_bddl in enumerate(goal_bddl_files):
            goal_bddl_path = os.path.join(args.goal_bddl_dir, goal_bddl)
            env = OffScreenRenderEnv(
                bddl_file_name=goal_bddl_path,
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(0)
            env.reset()
            envs[idx] = env
            logging.info(f"  Created env {idx}: {goal_bddl}")
    else:
        tasks = []
        envs = {}
        init_states = {}
        for idx, bddl_file in enumerate(bddl_files):
            bddl_path = os.path.join(args.bddl_dir, bddl_file)
            instruction = parse_bddl_instruction(bddl_path)
            tasks.append({"bddl_file": bddl_file, "instruction": instruction, "eval_id": idx})
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(0)
            env.reset()
            envs[idx] = env
            logging.info(f"  Created env {idx}: {bddl_file}")
            init_file = os.path.join(args.init_dir, bddl_file.replace(".bddl", ".pruned_init"))
            if os.path.exists(init_file):
                init_states[idx] = torch.load(init_file)
            else:
                init_states[idx] = None

    logging.info(f"Total envs created: {len(envs)}")

    # ---- 4. Load OpenVLA-OFT model ----
    cfg = FakeConfig(args)
    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None

    processor = get_processor(cfg)

    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"unnorm_key {unnorm_key} not found!"
    cfg.unnorm_key = unnorm_key

    resize_size = get_image_resize_size(cfg)
    logging.info(f"Model loaded: {args.pretrained_checkpoint}, resize_size: {resize_size}")

    # ---- 5. Setup logger ----
    model_name = args.pretrained_checkpoint.rstrip("/").split("/")[-1]
    structured_logger = StructuredLogger(log_dir=args.output_dir)
    meta = {
        "model_name": model_name, "model_family": "openvla_oft",
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "seed": args.seed, "gpu": args.gpu,
        "max_steps": args.max_steps, "unnorm_key": cfg.unnorm_key,
        "bddl_dir": args.bddl_dir, "total_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    structured_logger.save_meta(meta)

    # ---- 6. Eval loop ----
    total_episodes = 0
    total_successes = 0
    start_time = time.time()

    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    for task_idx, task_info in enumerate(tqdm(tasks, desc="Eval")):
        bddl_file = task_info["bddl_file"]
        instruction = task_info["instruction"]
        eval_id = task_info["eval_id"]

        env = envs[eval_id]
        obs = env.reset()

        init_data = init_states.get(eval_id)
        if init_data is not None:
            obs = env.set_init_state(init_data[0])

        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        all_actions = []
        all_proprio_states = []
        all_rewards = []
        action_chunk_boundaries = []
        action_queue = collections.deque(maxlen=cfg.num_open_loop_steps)
        success = False

        for step in range(args.max_steps):
            img = get_libero_image(obs)
            wrist_img = get_libero_wrist_image(obs)
            img_resized = resize_image_for_policy(img, resize_size)
            wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
            gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
            proprio = np.concatenate((eef_pos, quat2axisangle(eef_quat.copy()), gripper_qpos))
            all_proprio_states.append(proprio.tolist())

            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_img_resized,
                "state": proprio.copy(),
            }

            if len(action_queue) == 0:
                action_chunk_boundaries.append(step)
                actions = get_action(
                    cfg, model, observation, instruction,
                    processor=processor, action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            all_actions.append(action.tolist())
            obs, reward, done, info = env.step(action.tolist())
            all_rewards.append(float(reward))

            if done:
                success = True
                break

        num_steps = len(all_actions)
        total_episodes += 1
        if success:
            total_successes += 1

        structured_logger.log_episode(
            task_id=task_idx, bddl_file=bddl_file,
            paraphrased_instruction=instruction, success=success,
            num_steps=num_steps, actions=all_actions,
            proprio_states=all_proprio_states,
            action_chunk_boundaries=action_chunk_boundaries,
            rewards=all_rewards, replay_video_path="",
            initial_state_idx=eval_id, episode_idx=0,
        )

        _sr = total_successes / total_episodes * 100
        print(f"  Task {task_idx} [eval{eval_id}]: {'OK' if success else 'FAIL'} | {instruction[:55]} | SR: {_sr:.1f}%")

        if (task_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (task_idx + 1) * (len(tasks) - task_idx - 1)
            print(f"[{task_idx+1}/{len(tasks)}] SR: {_sr:.1f}% | Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")

    # ---- 7. Save summary & cleanup ----
    structured_logger.save_summary(total_episodes, total_successes)

    for env in envs.values():
        env.close()

    total_sr = total_successes / total_episodes * 100 if total_episodes > 0 else 0
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.pretrained_checkpoint}")
    print(f"Seed: {args.seed}")
    print(f"Total tasks: {total_episodes}")
    print(f"Success rate: {total_sr:.1f}%")
    print(f"Total time: {elapsed/3600:.1f}h")
    print(f"Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
