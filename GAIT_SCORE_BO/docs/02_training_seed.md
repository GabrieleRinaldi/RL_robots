# Seed training

Training the seed model for 10k iterations with weights decided by the programmer.

Default weights:
- w_tracking = 1.5
- w_torque = -2e-5
- w_energy = -2e-5

Command to train the seed, to run from the project folder:

```bash
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task GaitScoreBo-Velocity-B1 --seed 42 --max_iterations 10001 --run_name seed
```

The training runs for 10001 iterations (from 0 to 10000), so that the last checkpoint is `model_10000.pt`.

The results will be available in a new folder, named after the date and time of the launch, with this content:

```
logs/rsl_rl/b1_velocity/2026-10-25_14-30-00_seed/
├── model_0.pt, model_100.pt, … model_10000.pt   # one checkpoint every 100 iterations
├── events.out.tfevents…           # training curves (for TensorBoard)
├── params/env.yaml, params/agent.yaml            # configuration used
├── git/                                # state of the code at launch
└── run.json                          # task and RL library of the run
```

TensorBoard curves can be visualized with this command:

```bash
uv run --extra isaacsim tensorboard --logdir logs/rsl_rl --port 6006
```

Gait Score of the seed model, computed on 1000 steps with the constant command vx = 1 m/s:

```bash
uv run --extra isaacsim python scripts/evaluate_gait_score.py --checkpoint logs/rsl_rl/b1_velocity/2026-10-25_14-30-00_seed/model_10000.pt --out logs/rsl_rl/b1_velocity/2026-10-25_14-30-00_seed/gait_score.json
```

The Gait Score is printed on the terminal and saved in the `.json` file.
