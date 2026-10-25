# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the task success rate on the training terrain, with many robots in parallel.

The robots are spread uniformly (round-robin) over all the cells of the terrain grid (row = difficulty,
column = terrain type), instead of letting the curriculum decide, and walk for one episode (20 s) with the
constant command vx = 1 m/s. An episode is a success if the robot never falls, i.e. if neither
``bad_orientation`` nor ``base_contact`` is triggered. The result is the overall success rate and the success
rate of every cell of the grid.

.. code-block:: bash

    uv run --extra isaacsim python scripts/evaluate_task_success_rate.py --checkpoint <model.pt> --out out.json
"""

import argparse
import json
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Evaluate the task success rate on the training terrain.")
parser.add_argument("--task", type=str, default="GaitScoreBo-Velocity-B1", help="Registered task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint to evaluate.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of robots in parallel.")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed, so that all the evaluations are comparable.")
parser.add_argument("--out", type=str, default=None, help="If given, the result is also written as JSON to this path.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

import GAIT_SCORE_BO.tasks  # noqa: F401


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point", overrides=hydra_args)
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    # constant command: forward at 1 m/s
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    # no randomization, and no curriculum: the terrain assigned to every robot stays fixed
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None

    terrain_names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # round-robin assignment of the robots to the cells (row = difficulty, column = terrain type)
        terrain = env.unwrapped.scene.terrain
        num_rows, num_cols = terrain.terrain_origins.shape[:2]
        num_cells = num_rows * num_cols
        num_envs = env.unwrapped.num_envs
        cell = torch.arange(num_envs, device=terrain.device) % num_cells
        rows = cell // num_cols
        cols = cell % num_cols
        terrain.terrain_types[:] = cols
        terrain.terrain_levels[:] = rows
        terrain.env_origins[:] = terrain.terrain_origins[rows, cols]
        env.unwrapped.reset()
        print(
            f"[INFO]: {num_envs} robots on {num_rows}x{num_cols}={num_cells} cells (~{num_envs / num_cells:.1f} each)"
        )

        checkpoint = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {checkpoint}")
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        termination_manager = env.unwrapped.termination_manager
        num_steps = math.ceil(env_cfg.episode_length_s / env.unwrapped.step_dt)
        fell = torch.zeros(num_envs, dtype=torch.bool, device=terrain.device)
        obs = env.get_observations()
        for _ in range(num_steps):
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            fell |= termination_manager.get_term("bad_orientation") | termination_manager.get_term("base_contact")

        success = (~fell).float()
        counts = torch.bincount(cell, minlength=num_cells).float()
        matrix = (torch.bincount(cell, weights=success, minlength=num_cells) / counts).reshape(num_rows, num_cols)

        result = {
            "checkpoint": checkpoint,
            "num_envs": num_envs,
            "episode_length_s": env_cfg.episode_length_s,
            "overall_success_rate": success.mean().item(),
            "num_rows": num_rows,
            "num_cols": num_cols,
            "terrain_names": terrain_names,
            "success_rate_matrix": matrix.tolist(),  # [row][column], row = difficulty (0 = easy)
            "envs_per_cell": counts.reshape(num_rows, num_cols).tolist(),
        }
        print(f"[INFO]: Overall success rate: {result['overall_success_rate'] * 100:.1f}%")

        if args_cli.out:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump(result, f)

        env.close()


if __name__ == "__main__":
    main()
