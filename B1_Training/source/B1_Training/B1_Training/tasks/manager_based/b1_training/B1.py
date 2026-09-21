import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

B1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path= "/home/inf-04/B1/b1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Il B1 è molto più grande del Go2, l'altezza di spawn nominale è circa 65 cm
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            #".*L_hip_joint": 0.0698132, # 4 gradi
            #".*R_hip_joint": -0.0698132, # -4 gradi
            #"F[L,R]_thigh_joint": 0.523599,# 30 gradi
            #"R[L,R]_thigh_joint": 0.523599, # 30 gradi
            #".*_calf_joint": -1.22173, # -70 gradi

            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    # ==========================================================================================
    # VERSIONE PRECEDENTE (commentata, NON cancellata, per riferimento/confronto): un unico
    # gruppo con limiti di sforzo/velocità UNIFORMI (45 / 21.0) per tutti i giunti - non
    # corrispondono ai valori reali dell'URDF (/home/inf-04/B1/b1.urdf), che sono diversi per
    # tipo di giunto e per il calf in particolare molto più alti (140 Nm reali contro i 45
    # configurati qui, meno di un terzo).
    # --------------------------------------------------------------------------------------------
    # actuators={
    #         "base_legs": DCMotorCfg(
    #             joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    #             effort_limit=45,
    #             saturation_effort=45,
    #             velocity_limit=21.0,
    #             stiffness=250,
    #             damping=5,
    #             friction=0.0,
    #         ),
    #     },
    # ==========================================================================================

    # VERSIONE NUOVA (attiva): tre gruppi separati (hip, thigh, calf), con `effort_limit` e
    # `velocity_limit` presi direttamente dall'URDF ufficiale Unitree (/home/inf-04/B1/b1.urdf,
    # tag <limit effort=... velocity=...> di ciascun giunto - valori fisici misurati, non
    # tunabili). `stiffness`/`damping` invece NON sono nell'URDF (sono parametri di controllo,
    # non specifiche hardware): qui usiamo 150/5, vicini al valore ufficiale Unitree per il B2
    # (stiffness=160, damping=5 - vedi unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/
    # assets/robots/unitree.py, UNITREE_B2_CFG, lo stesso repository usato per il deployment
    # reale). Il B2 è più potente del B1 (200-320 Nm contro i 91-140 Nm del B1) eppure usa
    # stiffness=160 uniforme su tutti i giunti (non 250 come nella versione precedente) - da qui
    # la scelta di abbassare a 150, mantenendo comunque un'unica coppia stiffness/damping
    # condivisa fra i tre gruppi, esattamente come fa Unitree per il B2.
    actuators={
        "hip_motors": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=91.0035,
            saturation_effort=91.0035,
            velocity_limit=19.69,
            stiffness=150.0,
            damping=5.0,
            friction=0.0,
            # inerzia del rotore attorno all'asse di spin, da <link name="*_hip_rotor"><inertial>
            # in b1.urdf (ixx, asse X). mechanicalReduction=1 -> nessun fattore di riduzione da applicare.
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
            # da <link name="*_thigh_rotor"><inertial> in b1.urdf (iyy, asse Y).
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
            # da <link name="*_calf_rotor"><inertial> in b1.urdf (iyy, asse Y) - stesso rotore del thigh.
            armature=0.00091885,
        ),
    },
)
"""Configuration of Unitree B1 using DC-Motor actuator model in Position Control.

Limiti di sforzo/velocità per giunto presi dall'URDF ufficiale (/home/inf-04/B1/b1.urdf).
Stiffness/damping (150/5) allineati al riferimento ufficiale Unitree B2 (stiffness=160,
damping=5 in unitree_rl_lab), scalati leggermente al ribasso dato che il B1 ha motori meno
potenti del B2.
"""