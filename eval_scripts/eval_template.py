"""
LIBERO-Para Evaluation Template

Each model's eval script should be based on this template.
Only modify MODEL_NAME, load_model(), and predict_action().
"""

import argparse
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="libero_para")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(args):
    """Load model. Implement per model."""
    raise NotImplementedError


def predict_action(model, obs, task_description):
    """Given obs and language instruction, return action. Implement per model."""
    raise NotImplementedError


def evaluate(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    model = load_model(args)

    num_tasks = task_suite.get_num_tasks()
    success_rates = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        bddl_file = task_suite.get_task_bddl_file_path(task_id)

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)

        init_states = task_suite.get_task_init_states(task_id)
        num_success = 0

        for ep in range(args.num_episodes):
            env.reset()
            env.set_init_state(init_states[ep])
            obs = env.get_observation()

            for step in range(args.max_steps):
                action = predict_action(model, obs, task_description)
                obs, reward, done, info = env.step(action)
                if done:
                    num_success += 1
                    break

        env.close()
        rate = num_success / args.num_episodes
        success_rates.append(rate)
        print(f"[Task {task_id}] {task.name}: {rate:.1%} ({num_success}/{args.num_episodes})")

    avg = np.mean(success_rates)
    print(f"\n[Result] {args.suite} average success rate: {avg:.1%}")
    return success_rates


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
