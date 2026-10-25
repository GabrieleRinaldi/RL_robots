# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bayesian Optimization of the reward weights (w_tracking, w_torque, w_energy).

This script does not run the simulation itself: for every candidate it launches, as subprocesses, the training
(resumed from the seed checkpoint with the candidate weights) and the Gait Score evaluation of the resulting policy.
Each subprocess runs on one GPU; ``--num_gpus`` x ``--slots_per_gpu`` candidates are evaluated in parallel.
The Gait Score is the value maximized by Optuna (TPE sampler).

.. code-block:: bash

    uv run --extra isaacsim python scripts/BO_weight.py --seed_checkpoint <path to model_10000.pt of the seed>

If the script is interrupted, launching it again with the same command resumes the same study.
"""

import argparse
import atexit
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISAACLAB = Path(sys.executable).parent / "isaaclab"  # Isaac Lab command line of the active environment

TASK_NAME = "GaitScoreBo-Velocity-B1"
EXPERIMENT_NAME = "b1_velocity"  # experiment_name of the RSL-RL agent: folder of the runs in logs/rsl_rl/
EVAL_STEPS = 1000

parser = argparse.ArgumentParser(description="Bayesian Optimization of the reward weights.")
parser.add_argument(
    "--seed_checkpoint", type=str, required=True, help="Checkpoint of the seed from which every candidate restarts."
)
parser.add_argument("--train_iterations", type=int, default=1500, help="Training iterations per candidate.")
parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs available.")
parser.add_argument("--slots_per_gpu", type=int, default=2, help="Parallel candidates per GPU.")
parser.add_argument(
    "--total_budget", type=int, default=40, help="Total number of candidates of the study (random ones included)."
)
parser.add_argument("--seed", type=int, default=42, help="Seed of the Optuna sampler.")
parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="Prefix of the study, of its files and of the run names, to keep two studies apart.",
)
args_cli = parser.parse_args()

if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("Y", "YES"):
    parser.error(
        "the subprocesses run Isaac Sim without a terminal, so they cannot ask to accept the NVIDIA Omniverse EULA. "
        "Accept it with `export OMNI_KIT_ACCEPT_EULA=YES` and launch the script again."
    )

SEED_CHECKPOINT = Path(args_cli.seed_checkpoint).resolve()
TRAIN_ITERATIONS = args_cli.train_iterations
NUM_GPUS = args_cli.num_gpus
SLOTS_PER_GPU = args_cli.slots_per_gpu
TOTAL_BUDGET = args_cli.total_budget
ROUND_SIZE = NUM_GPUS * SLOTS_PER_GPU  # candidates running together; the first round is sampled at random
SUFFIX = f"{args_cli.suffix}_" if args_cli.suffix else ""

# bounds centered on the default weights (w_tracking = 1.5, w_torque = w_energy = -2e-5)
PARAM_BOUNDS = {
    "w_tracking": (0.5, 3.0),
    "w_torque": (-1e-4, -1e-6),
    "w_energy": (-1e-4, -1e-6),
}

STUDY_NAME = f"{SUFFIX}b1_bayesian_optimization"
BO_DIR = PROJECT_ROOT / "logs" / "bo"
STUDY_DB_PATH = BO_DIR / f"{SUFFIX}bo_study.db"
RESULTS_JSONL = BO_DIR / f"{SUFFIX}bo_results.jsonl"  # one JSON line per candidate
SUBPROCESS_LOG_DIR = BO_DIR / f"{SUFFIX}bo_orchestrator"  # raw output of the training and evaluation subprocesses

TRAIN_TIMEOUT_S = 6 * 3600
EVAL_TIMEOUT_S = 30 * 60

_dispatch_lock = threading.Lock()
_dispatched_count = 0  # candidates already reserved, shared by all the workers
_results_lock = threading.Lock()
_children_lock = threading.Lock()
_children: set[subprocess.Popen] = set()  # running subprocesses, killed if the orchestrator is stopped


def _reserve_slot() -> bool:
    """Reserve one candidate if the budget is not exhausted."""
    global _dispatched_count
    with _dispatch_lock:
        if _dispatched_count >= TOTAL_BUDGET:
            return False
        _dispatched_count += 1
        return True


def _append_result(record: dict) -> None:
    with _results_lock, open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")


def _kill_children() -> None:
    with _children_lock:
        for proc in _children:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _run_subprocess(cmd: list[str], log_path: Path, timeout_s: int) -> int:
    """Run a command, with its output in ``log_path``. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        # a new session lets us kill the whole process group (Isaac Sim spawns children) on timeout or exit
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
        )
        with _children_lock:
            _children.add(proc)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            raise
        finally:
            with _children_lock:
                _children.discard(proc)
    return proc.returncode


def _find_run_dir(run_name: str) -> Path | None:
    """Most recent run folder of ``run_name`` (folders are named ``<date>_<time>_<run_name>``)."""
    matches = sorted(glob.glob(str(PROJECT_ROOT / "logs" / "rsl_rl" / EXPERIMENT_NAME / f"*_{run_name}")))
    return Path(matches[-1]) if matches else None


def _find_final_checkpoint(run_dir: Path) -> Path | None:
    """Checkpoint with the highest iteration."""
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        m = re.match(r"model_(\d+)\.pt", path.name)
        if m:
            checkpoints.append((int(m.group(1)), path))
    return max(checkpoints)[1] if checkpoints else None


def run_candidate(trial: optuna.Trial, gpu_id: int) -> dict:
    """Train and evaluate one candidate. The result has ``gait_score``, or ``error`` if something failed."""
    w_tracking = trial.suggest_float("w_tracking", *PARAM_BOUNDS["w_tracking"])
    w_torque = trial.suggest_float("w_torque", *PARAM_BOUNDS["w_torque"])
    w_energy = trial.suggest_float("w_energy", *PARAM_BOUNDS["w_energy"])

    round_idx = trial.number // ROUND_SIZE
    point_idx = trial.number % ROUND_SIZE
    run_name = f"{SUFFIX}bo_r{round_idx}_p{point_idx}"
    device = f"cuda:{gpu_id}"

    train_cmd = [
        str(ISAACLAB),
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        TASK_NAME,
        "--device",
        device,
        "--checkpoint",
        str(SEED_CHECKPOINT),
        "--max_iterations",
        str(TRAIN_ITERATIONS),
        "--run_name",
        run_name,
        # the angular tracking weight is always half of the linear one
        f"env.rewards.track_lin_vel_xy.weight={w_tracking}",
        f"env.rewards.track_ang_vel_z.weight={w_tracking / 2}",
        f"env.rewards.joint_torques.weight={w_torque}",
        f"env.rewards.energy.weight={w_energy}",
    ]
    train_log = SUBPROCESS_LOG_DIR / f"{run_name}_train.log"

    record = {
        "trial": trial.number,
        "run_name": run_name,
        "gpu": gpu_id,
        "w_tracking": w_tracking,
        "w_torque": w_torque,
        "w_energy": w_energy,
    }

    try:
        rc = _run_subprocess(train_cmd, train_log, TRAIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        record["error"] = f"training timeout after {TRAIN_TIMEOUT_S}s"
        return record
    if rc != 0:
        record["error"] = f"training exited with returncode {rc} (see {train_log})"
        return record

    run_dir = _find_run_dir(run_name)
    if run_dir is None:
        record["error"] = f"log folder not found for run_name={run_name}"
        return record

    final_ckpt = _find_final_checkpoint(run_dir)
    if final_ckpt is None:
        record["error"] = f"no checkpoint found in {run_dir}"
        return record
    record["checkpoint"] = str(final_ckpt)

    eval_out = run_dir / "gait_score.json"
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_gait_score.py",
        "--task",
        TASK_NAME,
        "--checkpoint",
        str(final_ckpt),
        "--steps",
        str(EVAL_STEPS),
        "--device",
        device,
        "--out",
        str(eval_out),
    ]
    eval_log = SUBPROCESS_LOG_DIR / f"{run_name}_eval.log"

    try:
        rc = _run_subprocess(eval_cmd, eval_log, EVAL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        record["error"] = f"evaluation timeout after {EVAL_TIMEOUT_S}s"
        return record
    if rc != 0:
        record["error"] = f"evaluation exited with returncode {rc} (see {eval_log})"
        return record
    if not eval_out.exists():
        record["error"] = f"evaluation output not found: {eval_out}"
        return record

    with open(eval_out) as f:
        record.update(json.load(f))
    return record


def worker(gpu_id: int, study: optuna.Study) -> None:
    while _reserve_slot():
        trial = study.ask()
        print(f"[worker gpu{gpu_id}] trial {trial.number} started", flush=True)
        t0 = time.time()
        result = run_candidate(trial, gpu_id)
        elapsed = time.time() - t0

        if "error" in result:
            print(f"[worker gpu{gpu_id}] trial {trial.number} FAILED: {result['error']}", flush=True)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            print(
                f"[worker gpu{gpu_id}] trial {trial.number} completed in {elapsed / 60:.1f} min, "
                f"gait_score={result['gait_score']:.4f}",
                flush=True,
            )
            study.tell(trial, result["gait_score"])

        result["elapsed_s"] = elapsed
        _append_result(result)


def main() -> None:
    global _dispatched_count
    SUBPROCESS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # the first round (ROUND_SIZE candidates) is sampled at random, then TPE takes over
    sampler = optuna.samplers.TPESampler(n_startup_trials=ROUND_SIZE, seed=args_cli.seed)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB_PATH}",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    # resuming: candidates left running by an interrupted execution are failed, finished ones count in the budget
    for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    _dispatched_count = len(study.trials)

    threads = []
    for gpu_id in range(NUM_GPUS):
        for _ in range(SLOTS_PER_GPU):
            # daemon threads: if the orchestrator is stopped, the exit handler kills the subprocesses and nothing waits
            t = threading.Thread(target=worker, args=(gpu_id, study), daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    print("\n=== Bayesian Optimization completed ===")
    print(f"Trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"No trial completed successfully: check the logs in {SUBPROCESS_LOG_DIR}")
    else:
        best = study.best_trial
        print(f"Best gait_score: {best.value:.4f}")
        print(f"Best weights: {best.params}")


if __name__ == "__main__":
    atexit.register(_kill_children)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # SIGINT already raises KeyboardInterrupt
    main()
