"""
Pi 0.5 LIBERO-Para eval with structured JSON logging.

Pi 0.5 inference uses openpi's server/client split — this script is the
CLIENT (LIBERO sim + paraphrase loop), it connects over WebSocket to a
policy server that holds the JAX model.

Start the server first (in a separate terminal, libero-para-pi05-server env):
    # base variant
    uv run scripts/serve_policy.py --env LIBERO
    # expert-only variant
    uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \\
        --policy.config pi05_libero_expert_only \\
        --policy.dir /path/to/checkpoints/pi05_libero_expert_only

Then run this client (libero-para-pi05-client env, from LIBERO-Para repo root):
    conda activate libero-para-pi05-client
    export MUJOCO_GL=egl
    python eval_scripts/examples/eval_pi05.py \\
        --host localhost --port 8000 \\
        --gpu 0 --seed 7 \\
        --output_dir ./logs_para/pi05/seed7/
"""

import argparse
import collections
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")


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
        return {"paraphrase_type": "unknown", "categories": [], "subcategories": [],
                "eval_id": -1, "variant_id": -1}

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
    m = re.search(r'eval(\d+)', bddl_file)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract eval_id from: {bddl_file}")


def _torch_load_compat(path):
    """torch.load with version-agnostic `weights_only` handling.
    PyTorch < 1.13 has no weights_only kwarg; >=2.4 defaults to True which
    rejects pickled objects like LIBERO init states."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# =====================================================================================
# JSON Logger (identical schema to eval_x_vla.py for cross-model compatibility)
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
        per_eval, per_category = {}, {}
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
            s = per_category[key]
            s["success_rate"] = s["successes"] / s["total"] if s["total"] > 0 else 0.0

        summary = {
            "overall_success_rate": total_successes / total_episodes if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes, "total_successes": total_successes,
            "per_eval": per_eval, "per_category": per_category,
        }
        path = os.path.join(self.log_dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# =====================================================================================
# Main
# =====================================================================================

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _get_repo_root():
    return str(Path(__file__).resolve().parent.parent.parent)


def main():
    repo_root = _get_repo_root()
    default_bddl = os.path.join(repo_root, "libero/libero/bddl_files/libero_para")
    default_init = os.path.join(repo_root, "libero/libero/init_files/libero_para")
    default_goal = os.path.join(repo_root, "libero/libero/bddl_files/libero_goal")

    parser = argparse.ArgumentParser(description="Pi 0.5 LIBERO-Para Eval (server/client)")
    # Server connection
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize_size", type=int, default=224,
                        help="Pi 0.5 input image size")
    parser.add_argument("--replan_steps", type=int, default=5,
                        help="How many steps from each action chunk to execute before replanning")
    # Task selection
    parser.add_argument("--bddl_dir", type=str, default=default_bddl)
    parser.add_argument("--init_dir", type=str, default=default_init)
    parser.add_argument("--goal_bddl_dir", type=str, default=default_goal)
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "para", "original"],
                        help="auto: detect from filenames; para: libero_para; original: standard LIBERO")
    parser.add_argument("--max_tasks", type=int, default=-1)
    # Run identity
    parser.add_argument("--model_name", type=str, default="pi05",
                        help="logged in meta.json; e.g. pi05 or pi05-expert-only")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./logs_para/pi05/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 1. Scan bddl files ----
    bddl_files = sorted([f for f in os.listdir(args.bddl_dir) if f.endswith(".bddl")])
    logging.info(f"Found {len(bddl_files)} bddl files in {args.bddl_dir}")

    if args.max_tasks > 0:
        bddl_files = bddl_files[:args.max_tasks]
        logging.info(f"Limited to {len(bddl_files)} tasks (debug mode)")

    mode = args.mode
    if mode == "auto":
        mode = "para" if any("_eval" in f for f in bddl_files) else "original"
    logging.info(f"Mode: {mode}")

    from libero.libero.envs import OffScreenRenderEnv

    # ---- 2. Build envs ----
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
            init_states[eval_id] = _torch_load_compat(init_path)
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
            env.seed(args.seed)
            env.reset()
            envs[idx] = env
            logging.info(f"  Created env {idx}: {goal_bddl}")
    else:
        tasks, envs, init_states = [], {}, {}
        for idx, bddl_file in enumerate(bddl_files):
            bddl_path = os.path.join(args.bddl_dir, bddl_file)
            instruction = parse_bddl_instruction(bddl_path)
            tasks.append({"bddl_file": bddl_file, "instruction": instruction, "eval_id": idx})

            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
            )
            env.seed(args.seed)
            env.reset()
            envs[idx] = env
            logging.info(f"  Created env {idx}: {bddl_file}")

            init_file = os.path.join(args.init_dir, bddl_file.replace(".bddl", ".pruned_init"))
            if os.path.exists(init_file):
                init_states[idx] = _torch_load_compat(init_file)
            else:
                init_states[idx] = None

    logging.info(f"Total envs created: {len(envs)}")

    # ---- 3. Connect to policy server ----
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as _websocket_client_policy

    max_retries = 5
    client = None
    for retry in range(max_retries):
        try:
            client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
            logging.info(f"Connected to policy server at {args.host}:{args.port}")
            break
        except Exception as e:
            if retry < max_retries - 1:
                logging.warning(f"Connection failed (attempt {retry+1}/{max_retries}), retry in 10s: {e}")
                time.sleep(10)
            else:
                raise

    # ---- 4. Logger + meta ----
    structured_logger = StructuredLogger(log_dir=args.output_dir)
    meta = {
        "model_name": args.model_name, "model_family": "pi0.5",
        "host": args.host, "port": args.port,
        "resize_size": args.resize_size, "replan_steps": args.replan_steps,
        "seed": args.seed, "gpu": args.gpu,
        "max_steps": args.max_steps, "num_steps_wait": args.num_steps_wait,
        "bddl_dir": args.bddl_dir, "mode": mode,
        "total_tasks": len(tasks),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    structured_logger.save_meta(meta)

    # ---- 5. Eval loop ----
    total_episodes, total_successes = 0, 0
    start_time = time.time()

    for task_idx, task_info in enumerate(tqdm(tasks, desc="Eval")):
        bddl_file = task_info["bddl_file"]
        instruction = task_info["instruction"]
        eval_id = task_info["eval_id"]

        env = envs[eval_id]
        env.reset()

        init_data = init_states.get(eval_id)
        if init_data is not None:
            obs = env.set_init_state(init_data[0])
        else:
            obs = env.reset()

        # warm-up dummy steps (settle physics)
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        action_plan = collections.deque()
        all_actions, all_proprio_states, all_rewards = [], [], []
        action_chunk_boundaries = []
        success = False

        for step in range(args.max_steps):
            # proprio: openpi expects pos[3] + axis-angle[3] + gripper_qpos[2] = 8-D
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
            gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
            proprio = np.concatenate((eef_pos, quat2axisangle(eef_quat.copy()), gripper_qpos))
            all_proprio_states.append(proprio.tolist())

            if not action_plan:
                action_chunk_boundaries.append(step)

                # Images: openpi convention flip[::-1, ::-1] + resize-with-pad to resize_size
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": proprio,
                    "prompt": str(instruction),
                }
                action_chunk = client.infer(element)["actions"]
                if len(action_chunk) < args.replan_steps:
                    raise RuntimeError(
                        f"Policy returned only {len(action_chunk)} actions; need >= {args.replan_steps}"
                    )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()
            action_np = np.asarray(action)
            all_actions.append(action_np.tolist())

            obs, reward, done, info = env.step(action_np.tolist())
            all_rewards.append(float(reward))

            if env.check_success() or done:
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
        print(f"  Task {task_idx} [eval{eval_id}]: {'OK' if success else 'FAIL'} | "
              f"{instruction[:55]} | SR: {_sr:.1f}%")

        if (task_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (task_idx + 1) * (len(tasks) - task_idx - 1)
            print(f"[{task_idx+1}/{len(tasks)}] SR: {_sr:.1f}% | "
                  f"Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")

    # ---- 6. Summary & cleanup ----
    structured_logger.save_summary(total_episodes, total_successes)
    for env in envs.values():
        env.close()

    total_sr = total_successes / total_episodes * 100 if total_episodes > 0 else 0
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}  (Pi 0.5)")
    print(f"Server: {args.host}:{args.port}")
    print(f"Seed: {args.seed}")
    print(f"Total tasks: {total_episodes}")
    print(f"Success rate: {total_sr:.1f}%")
    print(f"Total time: {elapsed/3600:.1f}h")
    print(f"Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
