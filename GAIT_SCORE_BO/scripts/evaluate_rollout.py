# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record one rollout of a trained policy on a flat terrain and save it as a JSON file.

Two tests, both with a single robot and the same ground for every checkpoint:

* ``--test tracking``: random but deterministic velocity commands (same seed, same sequence), resampled every
  10 s in the range +-1.5 m/s and rad/s, for 60 s, without terminations. The result also contains the Cost of
  Transport of the session.
* ``--test push``: the robot stands still and receives random pushes (the base velocity is set to a random value
  in +-``--push_magnitude`` m/s on x and y) every 8-10 s, for 90 s. Falling resets the robot.

.. code-block:: bash

    uv run --extra isaacsim python scripts/evaluate_rollout.py --test tracking --checkpoint <model.pt> --out out.json
"""

import argparse
import json
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Record a tracking or push rollout of a trained policy.")
parser.add_argument("--test", type=str, required=True, choices=["tracking", "push"], help="Protocol to run.")
parser.add_argument("--task", type=str, default="GaitScoreBo-Velocity-B1", help="Registered task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint to evaluate.")
parser.add_argument("--out", type=str, required=True, help="Path of the JSON file with the recorded rollout.")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed, so that all the evaluations are comparable.")
parser.add_argument("--duration_s", type=float, default=None, help="Duration [s]. Default: 60 (tracking), 90 (push).")
parser.add_argument("--vel_range", type=float, default=1.5, help="Tracking: commands in [-vel_range, vel_range].")
parser.add_argument("--resample_interval_s", type=float, default=10.0, help="Tracking: seconds between two commands.")
parser.add_argument("--push_magnitude", type=float, default=4.0, help="Push: velocity range [m/s] on x and y.")
parser.add_argument("--push_interval_min", type=float, default=8.0, help="Push: minimum seconds between two pushes.")
parser.add_argument("--push_interval_max", type=float, default=10.0, help="Push: maximum seconds between two pushes.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

from GAIT_SCORE_BO.tasks.velocity.config.b1.env_cfg import FLAT_TERRAIN_CFG

G = 9.81  # m/s^2

push_times = []  # time [s] of every push


def push_and_log(env, env_ids, velocity_range):
    """Push the robot and remember when."""
    push_times.append(env.common_step_counter * env.step_dt)
    mdp.push_by_setting_velocity(env, env_ids, velocity_range)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point", overrides=hydra_args)
    agent_cfg.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    # one flat tile, no curriculum
    env_cfg.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None

    # no randomization: the initial state is identical in every evaluation
    env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None

    command = env_cfg.commands.base_velocity
    command.rel_standing_envs = 0.0
    env_cfg.terminations.time_out = None
    env_cfg.terminations.bad_orientation = None  # a recoverable tilt must not count as a fall
    if args_cli.test == "tracking":
        duration_s = args_cli.duration_s or 60.0
        r = args_cli.vel_range
        command.resampling_time_range = (args_cli.resample_interval_s, args_cli.resample_interval_s)
        command.ranges.lin_vel_x = (-r, r)
        command.ranges.lin_vel_y = (-r, r)
        command.ranges.ang_vel_z = (-r, r)
        env_cfg.terminations.base_contact = None  # no reset: the sequence of commands is never interrupted
    else:
        duration_s = args_cli.duration_s or 90.0
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        m = args_cli.push_magnitude
        env_cfg.events.push_robot = EventTerm(
            func=push_and_log,
            mode="interval",
            interval_range_s=(args_cli.push_interval_min, args_cli.push_interval_max),
            params={"velocity_range": {"x": (-m, m), "y": (-m, m)}},
        )

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        checkpoint = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {checkpoint}")
        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        foot_ids, foot_names = contact_sensor.find_sensors(".*_foot")
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        dt = env.unwrapped.step_dt
        mass = robot.data.body_mass.torch[0].sum().item()
        num_steps = int(duration_s / dt)

        # one row per step: command (3), velocity vx vy wz (3), feet in contact (4), done, power, position (3),
        # speed, torque of the 12 joints
        rows = []
        obs = env.get_observations()
        for _ in range(num_steps):
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            torque = robot.data.applied_torque.torch[0]
            lin_vel = robot.data.root_lin_vel_b.torch[0]
            ang_vel = robot.data.root_ang_vel_b.torch[0]
            contact = (contact_sensor.data.current_contact_time.torch[0, foot_ids] > 0.0).float()
            power = (robot.data.joint_vel.torch[0] * torque).abs().sum()
            speed = torch.linalg.norm(robot.data.root_lin_vel_w.torch[0])
            rows.append(
                torch.cat(
                    [
                        command_term.command[0],
                        lin_vel[:2],
                        ang_vel[2:3],
                        contact,
                        dones[:1].float(),
                        power.reshape(1),
                        robot.data.root_pos_w.torch[0],
                        speed.reshape(1),
                        torque,
                    ]
                )
            )
        data = torch.stack(rows).cpu().double()
        t = torch.arange(num_steps, dtype=torch.float64) * dt

        log = {
            "test": args_cli.test,
            "checkpoint": checkpoint,
            "seed": args_cli.seed,
            "dt": dt,
            "mass": mass,
            "foot_names": foot_names,
            "joint_names": list(robot.joint_names),
            "t": t.tolist(),
            "vx_cmd": data[:, 0].tolist(),
            "vy_cmd": data[:, 1].tolist(),
            "wz_cmd": data[:, 2].tolist(),
            "vx": data[:, 3].tolist(),
            "vy": data[:, 4].tolist(),
            "wz": data[:, 5].tolist(),
            "foot_contact": data[:, 6:10].bool().tolist(),
            "done": data[:, 10].bool().tolist(),
            "power": data[:, 11].tolist(),
            "pos_x": data[:, 12].tolist(),
            "pos_y": data[:, 13].tolist(),
            "base_height": data[:, 14].tolist(),
            "vel_norm": data[:, 15].tolist(),
            "torque_mean": data[:, 16:].abs().mean(dim=1).tolist(),
            "torque_joint": data[:, 16:].round(decimals=3).tolist(),
        }

        if args_cli.test == "tracking":
            # Cost of Transport: energy / (m g d), with d the length of the path walked
            distance = torch.linalg.norm(data[1:, 12:14] - data[:-1, 12:14], dim=1).sum().item()
            energy = data[1:, 11].sum().item() * dt
            log["cot"] = {
                "cot": energy / (mass * G * distance),
                "total_energy_j": energy,
                "total_distance_m": distance,
                "mass_kg": mass,
            }
            print(json.dumps(log["cot"], indent=2))
        else:
            falls = int(data[:, 10].sum().item())
            log["push_times_s"] = push_times
            log["falls"] = falls
            print(f"[INFO]: {falls} falls out of {len(push_times)} pushes")

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(log, f)
        print(f"[INFO]: Rollout of {num_steps} steps saved in {args_cli.out}")

        env.close()


if __name__ == "__main__":
    main()
