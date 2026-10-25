# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the Unitree B1 robot (DC motor model, position control)."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

# the robot model is shipped with the project, in <project>/assets/b1/
B1_USD_PATH = Path(__file__).resolve().parents[6] / "assets" / "b1" / "b1_unitree_colors.usd"

B1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(B1_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationCfg(
            # Self-collisions are disabled: with this USD the collision meshes of thigh and hip overlap in the default
            # pose, which would terminate every episode (base_contact) as soon as self-collisions were enabled.
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # the B1 is much larger than a Go2: nominal spawn height is about 65 cm
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Effort and velocity limits come from the official Unitree URDF (<limit effort=... velocity=...> of each joint).
    # Stiffness and damping are control parameters (not in the URDF): 150 / 5, close to the official Unitree values for
    # the B2 (160 / 5), shared by the three joint groups. The armature is the rotor inertia of each joint (URDF).
    actuators={
        "hip_motors": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=91.0035,
            saturation_effort=91.0035,
            velocity_limit=19.69,
            stiffness=150.0,
            damping=5.0,
            friction=0.0,
            armature=0.00039249,
        ),
        "thigh_motors": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=93.33,
            saturation_effort=93.33,
            velocity_limit=23.32,
            stiffness=150.0,
            damping=5.0,
            friction=0.0,
            armature=0.00091885,
        ),
        "calf_motors": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=140.0,
            saturation_effort=140.0,
            velocity_limit=15.55,
            stiffness=150.0,
            damping=5.0,
            friction=0.0,
            armature=0.00091885,
        ),
    },
)
