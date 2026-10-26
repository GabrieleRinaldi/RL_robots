# GAIT_SCORE_BO

Motion control policy for the Unitree B1 quadruped, trained with reinforcement learning (PPO) in Isaac Lab 3.0. The weights of the reward function are optimized with Bayesian Optimization, maximizing a Gait Score.

The registered task is `GaitScoreBo-Velocity-B1`.

First install Isaac Lab following the [installation guide](https://github.com/GabrieleRinaldi/RL_robots/blob/main/installation/IsaacLab.md) in the `main` branch. Then follow the guides in `docs/` in this order:

1. `01_clone_project.md`: clone and register the project
2. `02_training_seed.md`: train the seed model
3. `03_training_default_continued.md`: continue the seed with the default weights
4. `04_training_BO.md`: optimize the reward weights
5. `05_evaluation.md`: evaluate the models and plot the results
