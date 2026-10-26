# How to install IsaacLab

For more information visit: https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html

This project uses IsaacLab 3.0.0.

### Installation of IsaacLab with IsaacSim (for others follow the doc). Ask for doubts.

```bash
#Installation of uv (python environment manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
#clone IsaacLab repository
git clone https://github.com/isaac-sim/IsaacLab.git --branch main
```

```bash
cd IsaacLab
#use the same version of IsaacLab of this project
git checkout 46d6cc3d7
```

```bash
#train of an example model with automatic installation of dependencies
uv run --extra all --extra isaacsim --extra video isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx
```

```bash
#play of the trained model
uv run isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct --viz viser
```

I've installed IsaacSim Streaming Client because I'm working in remote from a server. This include the necessity to add LIVESTREAM into the command to able the streaming

```bash
LIVESTREAM=2 uv run --extra all --extra isaacsim --extra video isaaclab play --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx --viz kit --num_envs=1
```

If there is the necessity to save a video the command is:

```bash
uv run --extra all --extra isaacsim --extra video isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx --video --video_length 200 --video_interval 800 --num_envs 64 --max_iterations 101
```
