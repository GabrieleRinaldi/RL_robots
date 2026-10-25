# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the results of the evaluations of a model, read from the JSON files saved in its run folder.

The figures are saved in ``<run>/figures/``:

* ``tracking.json`` (``evaluate_rollout.py --test tracking``): ``tracking.png`` and ``gait_diagram.png``
* ``push.json`` (``evaluate_rollout.py --test push``): ``push_robustness.png``
* ``task_success_rate.json`` (``evaluate_task_success_rate.py``): ``task_success_heatmap.png``

Only JSON and matplotlib are needed, no simulation. The figures of the files that are missing are skipped.

.. code-block:: bash

    uv run --extra isaacsim python scripts/plot_results.py --run <run folder>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# share of the 20 terrain columns given to every terrain type, in the order of ``terrain_names`` in the JSON
# (the proportions of COBBLESTONE_ROAD_CFG)
TERRAIN_PROPORTIONS = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]


def plot_tracking(d: dict, out_path: Path):
    """Commanded versus actual velocity (x, y, yaw) over time."""
    panels = [
        ("vx_cmd", "vx", "Linear velocity (x)", "m/s"),
        ("vy_cmd", "vy", "Linear velocity (y)", "m/s"),
        ("wz_cmd", "wz", "Angular velocity (z)", "rad/s"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (cmd_key, actual_key, title, unit) in zip(axes, panels, strict=True):
        ax.plot(d["t"], d[cmd_key], color="tab:red", linestyle="--", linewidth=1.0, label="Commanded")
        ax.plot(d["t"], d[actual_key], color="tab:blue", linewidth=0.8, label="Actual")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gait_diagram(d: dict, out_path: Path, t_start: float, t_end: float):
    """Stance phases of the four feet in a short time window."""
    t = np.array(d["t"])
    contacts = np.array(d["foot_contact"], dtype=bool)  # (steps, feet)
    window = np.where((t >= t_start) & (t <= t_end))[0]
    dt = d["dt"]

    fig, ax = plt.subplots(figsize=(12, 3))
    for foot in range(len(d["foot_names"])):
        # contiguous steps in contact, drawn as one bar
        stance = contacts[window, foot].astype(int)
        edges = np.diff(np.concatenate([[0], stance, [0]]))
        for start, stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
            ax.barh(foot, (stop - start) * dt, left=t[window[start]], height=0.6, color=plt.cm.tab10.colors[foot])
    ax.set_yticks(range(len(d["foot_names"])))
    ax.set_yticklabels(d["foot_names"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Gait diagram (stance phases)", fontsize=10)
    ax.set_xlim(t_start, t_end)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_push_robustness(d: dict, out_path: Path):
    """Norm of the base velocity and base height over time, with the falls (resets) marked."""
    t = d["t"]
    # reference standing height: median height while the robot is quiet (not in the middle of a push)
    quiet_heights = [h for h, v in zip(d["base_height"], d["vel_norm"], strict=True) if v < 0.5]
    standing_height = float(np.median(quiet_heights)) if quiet_heights else None

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, d["vel_norm"], color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel("||v|| [m/s]")
    axes[0].set_title("Base velocity norm", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, d["base_height"], color="tab:orange", linewidth=0.8, label="Base height")
    if standing_height is not None:
        axes[1].axhline(standing_height, color="tab:red", linestyle="--", linewidth=1.0, label="Standing height (ref.)")
    for step, fell in enumerate(d["done"]):
        if fell:
            axes[1].axvline(t[step], color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel("Height [m]")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Base height (dotted gray lines = fall/reset)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  falls: {d['falls']} out of {len(d['push_times_s'])} pushes")


def plot_task_success_heatmap(d: dict, out_path: Path):
    """Success rate for every difficulty (rows) and terrain type (columns, grouped by type)."""
    matrix = np.array(d["success_rate_matrix"])
    num_rows, num_cols = matrix.shape
    col_counts = [round(p * num_cols) for p in TERRAIN_PROPORTIONS]
    assert sum(col_counts) == num_cols, "the terrain proportions do not match the number of columns"

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto", origin="lower")
    col_start = 0
    for name, count in zip(d["terrain_names"], col_counts, strict=True):
        col_end = col_start + count
        if col_start > 0:
            ax.axvline(col_start - 0.5, color="black", linewidth=1.0)
        ax.text((col_start + col_end - 1) / 2, -0.8, name, ha="center", va="top", fontsize=8, rotation=20)
        col_start = col_end
    ax.set_xticks([])
    ax.set_yticks(range(num_rows))
    ax.set_ylabel("Terrain difficulty level")
    ax.set_title("Task success rate", fontsize=10)
    fig.colorbar(im, ax=ax).set_label("Success rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  overall success rate: {d['overall_success_rate']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot the results of the evaluations of a model.")
    parser.add_argument("--run", type=str, required=True, help="Run folder with the JSON files of the evaluations.")
    parser.add_argument("--t_start", type=float, default=10.0, help="Gait diagram: start of the time window [s].")
    parser.add_argument("--t_end", type=float, default=12.0, help="Gait diagram: end of the time window [s].")
    args = parser.parse_args()

    run = Path(args.run)
    out_dir = run / "figures"
    out_dir.mkdir(exist_ok=True)

    def load(name: str):
        path = run / name
        if not path.exists():
            print(f"{name} not found, skipped")
            return None
        with open(path) as f:
            return json.load(f)

    if (d := load("tracking.json")) is not None:
        plot_tracking(d, out_dir / "tracking.png")
        plot_gait_diagram(d, out_dir / "gait_diagram.png", args.t_start, args.t_end)
        print(f"  Cost of Transport: {d['cot']['cot']:.3f}")
    if (d := load("push.json")) is not None:
        plot_push_robustness(d, out_dir / "push_robustness.png")
    if (d := load("task_success_rate.json")) is not None:
        plot_task_success_heatmap(d, out_dir / "task_success_heatmap.png")
    print(f"Figures saved in {out_dir}")


if __name__ == "__main__":
    main()
