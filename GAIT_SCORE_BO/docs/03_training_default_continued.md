# Training Default-Continued

Training the Default-Continued model for 1.5k iterations with the same weights of the seed model.

Command to continue the training, to run from the project folder:

```bash
uv run --extra isaacsim isaaclab train --rl_library rsl_rl --task GaitScoreBo-Velocity-B1 --checkpoint logs/rsl_rl/b1_velocity/2026-10-25_14-30-00_seed/model_10000.pt --max_iterations 1500 --run_name default_continued
```

Replace the folder `2026-10-25_14-30-00_seed` with the folder of your seed training (see `02_training_seed.md`).

The training restarts from the weights, the optimizer and the iteration counter saved in the checkpoint, so the iterations go from 10000 to 11499 and the last checkpoint is `model_11499.pt`. The results will be available in a new folder:

```
logs/rsl_rl/b1_velocity/<date-and-time>_default_continued/
```

Gait Score of the Default-Continued model, computed on 1000 steps with the constant command vx = 1 m/s:

```bash
uv run --extra isaacsim python scripts/evaluate_gait_score.py --checkpoint logs/rsl_rl/b1_velocity/<date-and-time>_default_continued/model_11499.pt --out logs/rsl_rl/b1_velocity/<date-and-time>_default_continued/gait_score.json
```

Replace the folder `<date-and-time>_default_continued` with the folder of your Default-Continued training.
