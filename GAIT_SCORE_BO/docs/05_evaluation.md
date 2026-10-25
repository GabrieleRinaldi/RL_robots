# Evaluation

Tracking, push-robustness and task success rate of a model. The Gait Score is in the file of each training.

Commands to evaluate the seed model, to run from the project folder:

```bash
RUN=logs/rsl_rl/b1_velocity/2026-10-25_14-30-00_seed
CKPT=$RUN/model_10000.pt
```

```bash
uv run --extra isaacsim python scripts/evaluate_rollout.py --test tracking --checkpoint $CKPT --out $RUN/tracking.json
```

```bash
uv run --extra isaacsim python scripts/evaluate_rollout.py --test push --checkpoint $CKPT --out $RUN/push.json
```

```bash
uv run --extra isaacsim python scripts/evaluate_task_success_rate.py --checkpoint $CKPT --out $RUN/task_success_rate.json
```

Replace the first two lines with the folder and the checkpoint of the model to evaluate:
- Default-Continued: `RUN=logs/rsl_rl/b1_velocity/<date-and-time>_default_continued` and `CKPT=$RUN/model_11499.pt`
- BO-best: `RUN=logs/rsl_rl/b1_velocity/<date-and-time>_bo_rX_pY` of the best trial and `CKPT=$RUN/model_11499.pt`

## Figures

Command to plot the results of the evaluations of a model (it needs the `.json` files above):

```bash
uv run --extra isaacsim python scripts/plot_results.py --run $RUN
```

The figures are saved in `$RUN/figures/`:

```
tracking.png              # commanded and actual velocity
gait_diagram.png          # stance phases of the four feet (from 10 s to 12 s)
push_robustness.png       # speed and height of the base, the falls are marked
task_success_heatmap.png  # success rate for every terrain and difficulty
```
