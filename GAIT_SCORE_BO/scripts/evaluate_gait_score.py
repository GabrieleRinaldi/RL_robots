# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute the Gait Score of a trained policy on one deterministic rollout.

A single robot walks for ``--steps`` steps with the constant command vx = 1 m/s, vy = 0, wz = 0, without
terminations, randomization or pushes. The Gait Score is::

    GS = forward displacement - mean tracking error - mean gravity error - mean torque

.. code-block:: bash

    uv run --extra isaacsim python scripts/evaluate_gait_score.py --checkpoint <model.pt> --out gait_score.json
"""

import argparse
import json
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Compute the Gait Score of a trained policy.")
parser.add_argument("--task", type=str, default="GaitScoreBo-Velocity-B1", help="Registered task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint to evaluate.")
parser.add_argument("--steps", type=int, default=1000, help="Number of evaluation steps.")
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

    # one single robot: one deterministic rollout, not a statistic over many copies
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    # constant command: forward at 1 m/s
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0

    # no resets during the evaluation: a fall must not teleport the robot, it keeps accumulating error
    env_cfg.terminations.time_out = None
    env_cfg.terminations.base_contact = None
    env_cfg.terminations.bad_orientation = None

    # no randomization: the initial state is identical in every evaluation
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        checkpoint = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {checkpoint}")
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        robot = env.unwrapped.scene["robot"]
        obs = env.get_observations()
        x_initial = robot.data.root_pos_w.torch[0, 0].item()

        tracking_sum = torch.zeros((), device=env.unwrapped.device)
        gravity_sum = torch.zeros((), device=env.unwrapped.device)
        torque_sum = torch.zeros((), device=env.unwrapped.device)
        resets = 0

        for _ in range(args_cli.steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            resets += int(dones[0])

            lin_vel = robot.data.root_lin_vel_b.torch[0]
            ang_vel = robot.data.root_ang_vel_b.torch[0]
            tracking_sum += (1.0 - lin_vel[0]) ** 2 + lin_vel[1] ** 2 + 0.5 * ang_vel[2] ** 2
            gravity_sum += torch.linalg.norm(robot.data.projected_gravity_b.torch[0, :2])
            torque_sum += robot.data.applied_torque.torch[0].abs().mean()

        forward_displacement = robot.data.root_pos_w.torch[0, 0].item() - x_initial
        tracking_mean = tracking_sum.item() / args_cli.steps
        gravity_mean = gravity_sum.item() / args_cli.steps
        torque_mean = torque_sum.item() / args_cli.steps
        gait_score = forward_displacement - tracking_mean - gravity_mean - torque_mean

        if resets:
            print(f"[WARN] the environment was reset {resets} times: a termination was not disabled.")

        result = {
            "checkpoint": checkpoint,
            "steps": args_cli.steps,
            "gait_score": gait_score,
            "forward_displacement": forward_displacement,
            "tracking_mean": tracking_mean,
            "gravity_mean": gravity_mean,
            "torque_mean": torque_mean,
        }
        print(json.dumps(result, indent=2))
        if args_cli.out:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump(result, f, indent=2)

        env.close()


if __name__ == "__main__":
    main()
