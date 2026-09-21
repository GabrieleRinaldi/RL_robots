# B1 – Locomotion with Reinforcement Learning (Isaac Lab)

Training of a locomotion policy for the **Unitree B1** quadruped robot with **PPO** (RSL-RL library) inside **NVIDIA Isaac Lab**. The repository contains the robot model, the training environment code and the checkpoints of the last training run (from 0 to 18400 iterations) with the TensorBoard history.

## Contents

```
.
├── B1/                     Robot model: b1.usd, b1.urdf, meshes (.dae) and USD files generated from the URDF
└── B1_Training/            Isaac Lab project (installable Python extension)
    ├── scripts/
    │   ├── rsl_rl/         train.py, play.py, play_keyboard.py, play_gamepad.py
    │   ├── list_envs.py, zero_agent.py, random_agent.py
    ├── source/B1_Training/ Python package: environment definition and PPO configuration
    └── logs/rsl_rl/2026-08-19_22-49-01/
                            Checkpoints model_0.pt … model_18400.pt, params/, exported/, TensorBoard
```

## Requirements

The code was run with these versions:

| Component | Version |
|---|---|
| Operating system | Ubuntu 24.04 |
| GPU | NVIDIA (tested on NVIDIA L4, 24 GB) |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.3 |
| RSL-RL (`rsl-rl-lib`) | 5.4.2 |
| PyTorch | 2.13 (CUDA) |

Isaac Sim and Isaac Lab must be installed following the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). In the examples below, `ISAACLAB` is the Isaac Lab installation folder:

```bash
ISAACLAB=~/IsaacLab
```

If Isaac Lab is installed in a conda or venv environment that is already active, you can use `python` directly instead of `$ISAACLAB/isaaclab.sh -p`.

## Installation

1. Clone the repository, **outside** the Isaac Lab folder:

   ```bash
   git clone https://github.com/GabrieleRinaldi/RL_robots.git
   cd RL_robots
   ```

2. Install the package in editable mode, using the Python interpreter of Isaac Lab:

   ```bash
   $ISAACLAB/isaaclab.sh -p -m pip install -e B1_Training/source/B1_Training
   ```

   The robot model is found automatically: `B1.py` loads `B1/b1.usd` relative to the root of the repository, so the `B1/` and `B1_Training/` folders must stay next to each other.

3. Check the installation with a very short training run (2 iterations, 16 environments):

   ```bash
   cd B1_Training
   $ISAACLAB/isaaclab.sh -p scripts/rsl_rl/train.py --task Template-B1-Training-v0 --headless --num_envs 16 --max_iterations 2
   ```

   If "Learning iteration 0/2" and "Learning iteration 1/2" appear and the program ends without errors, the installation is correct. The test creates a small folder in `logs/rsl_rl/`, which you can delete.

All the following commands are run from the `B1_Training/` folder.

## Available tasks

| ID | Description |
|---|---|
| `Template-B1-Training-v0` | *Manager-based* environment used for training and for the included checkpoints |
| `Template-B1-Training-Direct-v0` | *Direct* variant inherited from the Isaac Lab template |

## Running a training

```bash
$ISAACLAB/isaaclab.sh -p scripts/rsl_rl/train.py --task Template-B1-Training-v0 --headless
```

Useful options:

| Option | Effect |
|---|---|
| `--headless` | No graphical window, much faster |
| `--num_envs N` | Number of parallel environments (default: 4096) |
| `--max_iterations N` | Number of iterations (default: 50000) |
| `--seed N` | Random seed |
| `--run_name NAME` | Suffix added to the run folder name |

The results are saved in `logs/rsl_rl/<date_time>/`: a `model_<iteration>.pt` checkpoint every 100 iterations, the parameters in `params/` and the TensorBoard files.

### Resuming a training

```bash
$ISAACLAB/isaaclab.sh -p scripts/rsl_rl/train.py --task Template-B1-Training-v0 --headless \
    --resume --load_run 2026-08-19_22-49-01 --checkpoint model_18400.pt
```

`--load_run` is the name of the run folder and `--checkpoint` is the file to resume from. Training resumes from the iteration number of the checkpoint and creates a new log folder.

## Trying a trained policy

With the graphical window and a single robot:

```bash
$ISAACLAB/isaaclab.sh -p scripts/rsl_rl/play.py --task Template-B1-Training-v0 --num_envs 1 \
    --load_run 2026-08-19_22-49-01 --checkpoint model_18400.pt
```

Without `--load_run` and `--checkpoint`, the latest checkpoint of the latest run found in `logs/rsl_rl/` is loaded. Other options: `--video` to record a video, `--real-time` for real-time speed.

The exported policy of the included run is in `logs/rsl_rl/2026-08-19_22-49-01/exported/` (`policy.pt` and `policy.onnx`).

### Driving the robot by hand

Two scripts allow you to command the robot's velocity with the trained policy.

**Keyboard** (local only, it reads the physical keyboard of the machine):

```bash
$ISAACLAB/isaaclab.sh -p scripts/rsl_rl/play_keyboard.py --task Template-B1-Training-v0 --num_envs 1
```

| Key | Command |
|---|---|
| `W` / `S` | Forward / backward |
| `A` / `D` | Sidestep left / right |
| `Q` / `E` | Counterclockwise / clockwise rotation |
| `Space` | Stop |

**Gamepad**:

```bash
$ISAACLAB/isaaclab.sh -p scripts/rsl_rl/play_gamepad.py --task Template-B1-Training-v0 --num_envs 1
```

With `--gamepad_source network --gamepad_port <port>` the gamepad is read from a UDP port. Other options common to both scripts: `--max_forward_speed`, `--max_lateral_speed`, `--max_yaw_rate`, `--camera_distance`, `--camera_height`, `--no_follow_camera`, `--deadzone` (gamepad only).

## TensorBoard

To see the training curves of the included run, from 0 to 18400 iterations:

```bash
$ISAACLAB/isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/2026-08-19_22-49-01
```

Then open the address it prints (usually `http://localhost:6006`). If the command runs on a remote server, add `--bind_all`.

## Main configuration

PPO configuration (`.../b1_training/agents/rsl_rl_ppo_cfg.py`):

| Parameter | Value |
|---|---|
| Steps per environment per iteration | 24 |
| Maximum iterations | 50000 |
| Save interval | 100 iterations |
| Learning epochs | 5 |
| Learning rate | 1e-3 |
| γ (gamma) / λ (lambda) | 0.99 / 0.95 |

Environment (`.../b1_training/b1_training_env_cfg.py`): 4096 parallel environments, simulation step 0.005 s with decimation 4, 20 s episodes. In play mode the environment uses 32 robots, or as many as you specify with `--num_envs`.

## Notes

- The robot model in `B1/` derives from the official Unitree URDF of the B1.
- The checkpoints in `logs/` are files of about 12 MB each (2 GB in total): the `git clone` is therefore heavy.
- The task IDs start with `Template-`, a prefix inherited from the Isaac Lab template. The same prefix is used by `scripts/list_envs.py` to filter the tasks to be listed.
