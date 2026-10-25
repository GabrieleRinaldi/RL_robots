# GAIT_SCORE_BO

Motion control policy for the Unitree B1 quadruped, trained with reinforcement learning (PPO) in Isaac Lab 3.0. The weights of the reward function are optimized with Bayesian Optimization, maximizing a Gait Score.

The registered task is `GaitScoreBo-Velocity-B1`.

Follow the guides in `GAIT_SCORE_BO/docs/` in this order:

1. `00_IsaacLab_Installation.md`: install Isaac Lab
2. `01_clone_project.md`: clone and register the project
3. `02_training_seed.md`: train the seed model
4. `03_training_default_continued.md`: continue the seed with the default weights
5. `04_training_BO.md`: optimize the reward weights
6. `05_evaluation.md`: evaluate the models and plot the results
