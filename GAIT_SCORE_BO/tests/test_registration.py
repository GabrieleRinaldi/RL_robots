# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import GAIT_SCORE_BO.tasks  # noqa: F401


def test_task_registrations():
    """The project task must expose valid environment and agent entry points."""
    spec = gym.spec("GaitScoreBo-Velocity-B1")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "GAIT_SCORE_BO.tasks.velocity.config.b1.env_cfg:B1VelocityEnvCfg"
    assert spec.kwargs["default_agent"] == "rsl_rl"
