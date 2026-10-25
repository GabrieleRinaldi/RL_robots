# Bayesian Optimization Training

Start from the seed model, and train for 1.5k iterations each trial with the new weights.

The weights (w_tracking, w_torque, w_energy) are optimized with Optuna (TPE sampler), maximizing the Gait Score of the policy obtained by each trial.

Search ranges:
- w_tracking in [0.5, 3.0]
- w_torque in [-1e-4, -1e-6]
- w_energy in [-1e-4, -1e-6]

Command to run the optimization (40 candidates, 4 GPUs with 2 candidates each), to run from the project folder:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
uv run --extra isaacsim python scripts/BO_weight.py --seed_checkpoint logs/rsl_rl/b1_velocity/2026-10-25_14-30-00_seed/model_10000.pt
```

Replace the folder `2026-10-25_14-30-00_seed` with the folder of your seed training (see `02_training_seed.md`).

The first command accepts the NVIDIA Omniverse EULA in advance, because Isaac Sim runs in the background and cannot ask for it.

If the optimization is interrupted, launching the same command again resumes the same study.

The results will be available in:

```
- logs/bo/bo_results.jsonl   # one line per candidate: weights and Gait Score
- logs/bo/bo_study.db                                # Optuna study
- logs/bo/bo_orchestrator/        # output of every training and evaluation
- logs/rsl_rl/b1_velocity/<date-and-time>_bo_r0_p0/  # trained candidates (r = round, p = point), with gait_score.json
```
