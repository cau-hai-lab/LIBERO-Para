"""Render the CALVIN playtable scene for the 15 CALVIN-Para tasks.

Resets the real calvin_env to the init_states actually used in our eval
(episodes/baseline_seed7.json) and grabs high-res static + gripper frames.

Outputs (RESULTS/scene/):
  scene_overview.png        single big static view of the playtable
  task_grid.png             15-panel grid, one initial state per task
  static/<task>.png         per-task full-res static frames
  gripper/<task>.png        per-task gripper-cam frames

Run with the calvin env:
  conda activate calvin_py38
  PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=5 python render_scene.py
"""
import argparse
import json
import os
from pathlib import Path

CALVIN_PARA = Path(__file__).resolve().parents[2]
CALVIN_ROOT = CALVIN_PARA / "calvin"
os.environ.setdefault("CALVIN_ROOT", str(CALVIN_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

TASKS = [
    ("open_drawer", "pull the handle to open the drawer"),
    ("close_drawer", "push the handle to close the drawer"),
    ("move_slider_left", "push the sliding door to the left side"),
    ("move_slider_right", "push the sliding door to the right side"),
    ("turn_on_lightbulb", "use the switch to turn on the light bulb"),
    ("turn_off_lightbulb", "use the switch to turn off the light bulb"),
    ("turn_on_led", "press the button to turn on the led light"),
    ("turn_off_led", "press the button to turn off the led light"),
    ("lift_red_block_table", "grasp and lift the red block"),
    ("lift_blue_block_table", "grasp and lift the blue block"),
    ("lift_red_block_slider", "lift the red block from the sliding cabinet"),
    ("push_red_block_right", "go push the red block right"),
    ("push_red_block_left", "go push the red block left"),
    ("rotate_red_block_left", "take the red block and rotate it to the left"),
    ("rotate_red_block_right", "take the red block and rotate it to the right"),
]


def build_env(res: int):
    """Same as calvin_env.get_env() but with the static camera bumped to `res`."""
    dataset_path = CALVIN_ROOT / "dataset" / "task_D_D" / "validation"
    cfg = OmegaConf.load(dataset_path / ".hydra" / "merged_config.yaml")
    cfg.env.cameras.static.width = res
    cfg.env.cameras.static.height = res
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    return hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)


def get_state(ep):
    """episodes JSON init_state -> (robot_obs, scene_obs) arrays."""
    import sys
    sys.path.insert(0, str(CALVIN_ROOT / "calvin_models"))
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    return get_env_state_for_initial_condition(ep["init_state"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(CALVIN_PARA / "paraphrase_eval/episodes/baseline_seed7.json"))
    ap.add_argument("--out_dir", default=str(CALVIN_PARA / "RESULTS" / "scene"))
    ap.add_argument("--res", type=int, default=800, help="static camera render size")
    ap.add_argument("--trial", type=int, default=0, help="which trial's init_state per task")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "static").mkdir(parents=True, exist_ok=True)
    (out / "gripper").mkdir(parents=True, exist_ok=True)

    episodes = json.loads(Path(args.episodes).read_text())
    # first matching trial per task
    picked = {}
    for ep in episodes:
        t = ep["task_id"]
        if t not in picked and ep["metadata"].get("trial_idx") == args.trial:
            picked[t] = ep

    print(f"building env (static camera {args.res}x{args.res}) ...")
    env = build_env(args.res)

    frames = {}
    for task_id, instruction in TASKS:
        ep = picked.get(task_id)
        if ep is None:
            print(f"  WARN: no init_state for {task_id}")
            continue
        robot_obs, scene_obs = get_state(ep)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        obs = env.get_obs()

        static = obs["rgb_obs"]["rgb_static"]
        gripper = obs["rgb_obs"]["rgb_gripper"]
        Image.fromarray(static).save(out / "static" / f"{task_id}.png")
        Image.fromarray(gripper).save(out / "gripper" / f"{task_id}.png")
        frames[task_id] = static
        print(f"  {task_id:26s} static={static.shape} gripper={gripper.shape}")

    # ---- scene overview: the first task's static frame, full res ----
    first = TASKS[0][0]
    if first in frames:
        Image.fromarray(frames[first]).save(out / "scene_overview.png")

    # ---- 15-panel grid ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def wrap(s, n=34):
        words, lines, cur = s.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > n:
                lines.append(cur); cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        return "\n".join(lines)

    fig, axes = plt.subplots(3, 5, figsize=(20, 14))
    for ax, (task_id, instruction) in zip(axes.flat, TASKS):
        img = frames.get(task_id)
        if img is None:
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.set_title(task_id, fontsize=13, fontweight="bold", pad=6)
        ax.text(0.5, -0.04, f'"{wrap(instruction)}"', transform=ax.transAxes,
                ha="center", va="top", fontsize=10.5, style="italic")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("CALVIN-Para — 15 tasks, initial state (scene D, static camera)",
                 fontsize=20, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.subplots_adjust(hspace=0.28)
    plt.savefig(out / "task_grid.png", dpi=150, bbox_inches="tight")
    plt.savefig(out / "task_grid.pdf", bbox_inches="tight")
    plt.close()
    print(f"\nwrote {out}/task_grid.png  (+ pdf, per-task static/ gripper/)")

    env.close()


if __name__ == "__main__":
    main()
