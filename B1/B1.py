import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

B1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= "/home/rosario/Scrivania/Isaac_Sim_Lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/B1/b1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, # added this line
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*L_hip_joint": 0.0698132, # 4 gradi
            ".*R_hip_joint": -0.0698132, # -4 gradi
            "F[L,R]_thigh_joint": 0.523599,# 30 gradi
            "R[L,R]_thigh_joint": 0.523599, # 30 gradi
            ".*_calf_joint": -1.22173, # -70 gradi
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=45,
            saturation_effort=45,
            velocity_limit=21.0,
            stiffness=250, 
            damping=5,
            friction=0.0,
        ),
    },
)