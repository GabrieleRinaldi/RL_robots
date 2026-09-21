# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guida interattivamente un checkpoint addestrato usando la tastiera. Solo locale (il controller
via carb.input legge la tastiera FISICA di questa macchina - non ha senso via rete/NoMachine come
per il gamepad, dato che la tastiera la stai gia' usando localmente sul server).

Controllo diretto sulle tre componenti del comando di velocita' (nessuna "modalita'" da
selezionare, ogni tasto pilota sempre la sua componente mentre resta premuto):

    W / S  -> velocita' lineare avanti / indietro (lin_vel_x), fino a --max_forward_speed.
    A / D  -> velocita' laterale/strafe sinistra / destra (lin_vel_y), fino a --max_lateral_speed.
    Q / E  -> velocita' angolare di imbardata, antiorario / orario (ang_vel_z), fino a
        --max_yaw_rate.
    SPAZIO -> stop immediato (azzera tutte e tre le componenti, utile se perdi il conto dei tasti
        premuti).

Nessun tasto premuto -> tutte e tre le componenti a zero, robot fermo.

La telecamera del viewport segue automaticamente il robot da dietro (distanza --camera_distance,
altezza --camera_height), ruotando insieme a lui cosi' resta sempre alle sue spalle. Disattivabile
con --no_follow_camera.

Con --spawn_terrain_col/--spawn_terrain_row il robot spawna su una cella precisa del terrain
generator invece che su quella assegnata di default (default: pyramid_stairs, riga 0). Disattivabile
con --no_forced_spawn.

Questo script NON lascia che il comando "base_velocity" si ricampioni da solo (lo facciamo
scrivendo direttamente nel suo stato interno ad ogni step) e disattiva il time_out dell'episodio:
per questo forziamo resampling_time_range e episode_length_s a valori enormi prima di creare
l'ambiente, cosi' l'episodio dura finche' non chiudiamo noi lo script (le terminazioni per caduta
vera - bad_orientation, base_contact - restano attive).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Guida un checkpoint RSL-RL con la tastiera.")
parser.add_argument("--num_envs", type=int, default=1, help="Numero di ambienti (di norma 1 per il teleop).")
parser.add_argument("--task", type=str, default=None, help="Nome del task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--max_forward_speed",
    type=float,
    default=1.5,
    help="Velocita' avanti/indietro (m/s) con W/S a fondo (binario: o zero o questo valore). Limite di training: 1.0 m/s.",
)
parser.add_argument(
    "--max_lateral_speed",
    type=float,
    default=1.0,
    help="Velocita' laterale/strafe (m/s) con A/D a fondo. Limite di training: 0.4 m/s.",
)
parser.add_argument(
    "--max_yaw_rate",
    type=float,
    default=1.5,
    help="Velocita' angolare (rad/s) con Q/E a fondo. Limite di training: 1.0 rad/s.",
)
parser.add_argument(
    "--camera_distance", type=float, default=3.0, help="Distanza (m) della telecamera dietro al robot."
)
parser.add_argument("--camera_height", type=float, default=1.3, help="Altezza (m) della telecamera sopra il robot.")
parser.add_argument(
    "--no_follow_camera",
    action="store_true",
    default=False,
    help="Disattiva la telecamera che segue il robot da dietro (resta quella di default, fissa).",
)
parser.add_argument(
    "--spawn_terrain_col",
    type=int,
    default=12,
    help=(
        "Colonna del terrain generator su cui far spawnare il robot. Con le proporzioni attuali"
        " di COBBLESTONE_ROAD_CFG in b1_training_env_cfg.py (num_cols=20): 0-1=flat, 2-3=random_rough,"
        " 4-5=hf_pyramid_slope, 6-7=hf_pyramid_slope_inv, 8-11=boxes, 12-15=pyramid_stairs,"
        " 16-19=pyramid_stairs_inv. Default 12 = pyramid_stairs. Se cambi le proporzioni dei"
        " sub_terrains nel env_cfg, questi numeri di colonna vanno ricalcolati di conseguenza."
    ),
)
parser.add_argument(
    "--spawn_terrain_row",
    type=int,
    default=0,
    help="Riga del terrain generator, cioe' livello di difficolta' (0=minima, num_rows-1=massima).",
)
parser.add_argument(
    "--no_forced_spawn",
    action="store_true",
    default=False,
    help="Disattiva lo spawn forzato su --spawn_terrain_col/--spawn_terrain_row (torna alla selezione automatica standard).",
)
# stesso meccanismo (e stesso nome argomento) di play.py, ma di default ON qui: per il teleop
# interattivo vogliamo il ritmo 1:1 fin da subito, non serve ricordarsi di passare il flag ogni volta.
parser.add_argument(
    "--real-time", dest="real_time", action="store_true", default=True, help="Run in real-time (default: on qui)."
)
parser.add_argument(
    "--no-real-time", dest="real_time", action="store_false", help="Disattiva il pacing real-time, gira piu' veloce possibile."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# NOTA: niente gestione di --headless qui: vogliamo sempre il viewport aperto per poter
# guidare il robot e vederlo muoversi.

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Da qui in poi tutto il resto."""

import math
import os
import time
import weakref

import torch

import carb
import carb.input
import omni.appwindow

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import B1_Training.tasks  # noqa: F401

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")


class KeyboardReader:
    """Legge la tastiera FISICA di questa macchina via carb.input e la mappa sulle tre componenti
    del comando di velocita' (lin_vel_x, lin_vel_y, ang_vel_z). Stato booleano per tasto (premuto/
    rilasciato), non un accumulatore - cosi' non c'e' rischio di sfasamenti se un evento di
    rilascio si perde.
    """

    _KEY_TO_AXIS = {
        # (nome_tasto): (indice_componente, segno)
        "W": (0, 1.0),
        "S": (0, -1.0),
        "A": (1, 1.0),
        "D": (1, -1.0),
        "Q": (2, 1.0),
        "E": (2, -1.0),
    }

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # weakref: cosi' l'oggetto puo' essere distrutto normalmente (stesso pattern usato per il
        # gamepad in play_gamepad.py)
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )
        self._pressed = {name: False for name in self._KEY_TO_AXIS}

    def __del__(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)

    def _on_event(self, event: carb.input.KeyboardEvent, *args, **kwargs):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "SPACE":
                for key in self._pressed:
                    self._pressed[key] = False
            elif name in self._pressed:
                self._pressed[name] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if name in self._pressed:
                self._pressed[name] = False
        return True

    def read(self) -> tuple[float, float, float]:
        """Ritorna (frazione_lin_vel_x, frazione_lin_vel_y, frazione_ang_vel_z), ciascuna -1/0/1."""
        axes = [0.0, 0.0, 0.0]
        for name, (idx, sign) in self._KEY_TO_AXIS.items():
            if self._pressed[name]:
                axes[idx] += sign
        return tuple(axes)


def _update_chase_camera(env, distance: float, height: float, env_index: int = 0):
    """Posiziona la telecamera del viewport dietro al robot, alla sua stessa altezza (+ offset),
    ricalcolando ad ogni chiamata in base a posizione/orientamento attuali - cosi' la telecamera
    resta sempre dietro anche quando il robot gira (heading_w e' l'angolo di imbardata nel mondo)."""
    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[env_index]
    heading = robot.data.heading_w[env_index].item()
    behind = torch.tensor([-math.cos(heading), -math.sin(heading), 0.0], device=pos.device)
    eye = pos + behind * distance + torch.tensor([0.0, 0.0, height], device=pos.device)
    target = pos + torch.tensor([0.0, 0.0, 0.2], device=pos.device)  # guarda un po' sopra il trunk
    env.unwrapped.sim.set_camera_view(eye.tolist(), target.tolist())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # disattiva il ricampionamento automatico del comando: lo pilotiamo noi ad ogni step
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    # disattiva il time_out: l'episodio normalmente finisce (e il robot si resetta) dopo
    # episode_length_s secondi (20s di default) - in una sessione interattiva non lo vogliamo,
    # l'episodio deve durare finche' non chiudiamo noi lo script. Le altre terminazioni
    # (bad_orientation, base_contact - il robot che cade davvero) restano attive.
    env_cfg.episode_length_s = 1.0e9

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if not args_cli.no_forced_spawn:
        terrain = env.unwrapped.scene.terrain
        if terrain.terrain_origins is None:
            print("[WARNING] --spawn_terrain_col/--spawn_terrain_row richiesti ma il terreno non e' un generator: ignorato.")
        else:
            num_rows, num_cols = terrain.terrain_origins.shape[:2]
            col = min(max(args_cli.spawn_terrain_col, 0), num_cols - 1)
            row = min(max(args_cli.spawn_terrain_row, 0), num_rows - 1)
            env_ids = torch.arange(env.unwrapped.num_envs, device=terrain.device)
            # stessa identica assegnazione che fa TerrainImporter internamente (vedi
            # update_env_origins) - solo che qui scegliamo NOI riga/colonna invece di lasciarla al
            # sorteggio iniziale/curriculum, cosi' lo spawn cade sempre sul terreno scelto.
            terrain.terrain_types[env_ids] = col
            terrain.terrain_levels[env_ids] = row
            terrain.env_origins[env_ids] = terrain.terrain_origins[row, col]
            print(f"[INFO] Spawn forzato su terrain_generator[riga={row}, colonna={col}].")
            env.reset()

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Solo l'actor: per l'inferenza il critic non serve mai (e' usato solo in training per il
    # value function) - caricarlo comunque renderebbe questo script fragile ad ogni esperimento
    # che cambia SOLO le osservazioni del critic (es. aggiungere/togliere l'altimetria li'), anche
    # quando l'actor - l'unica cosa che ci serve qui - e' rimasto identico.
    runner.load(resume_path, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Comando "base_velocity" pilotato direttamente: scriviamo tutte e tre le componenti
    # (lin_vel_x, lin_vel_y, ang_vel_z) ad ogni step dai tasti premuti, senza passare per il
    # meccanismo di heading/standing della classe (che restano ai loro default e non toccano
    # vel_command_b da soli, dato che il resampling e' disattivato).
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    keyboard = KeyboardReader()
    print("[INFO] Tastiera pronta: W/S = avanti/indietro, A/D = laterale, Q/E = rotazione, SPAZIO = stop.")

    dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]
    step_counter = 0
    obs = env.get_observations()
    if not args_cli.no_follow_camera:
        _update_chase_camera(env, args_cli.camera_distance, args_cli.camera_height)
    while simulation_app.is_running():
        # time delay for real-time evaluation (stesso snippet di play.py)
        start_time = time.time()

        lin_x, lin_y, yaw = keyboard.read()
        cmd_term.vel_command_b[:, 0] = lin_x * args_cli.max_forward_speed
        cmd_term.vel_command_b[:, 1] = lin_y * args_cli.max_lateral_speed
        cmd_term.vel_command_b[:, 2] = yaw * args_cli.max_yaw_rate

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)

        if not args_cli.no_follow_camera:
            _update_chase_camera(env, args_cli.camera_distance, args_cli.camera_height)

        step_counter += 1
        # Errore ISTANTANEO (comando vs velocita' reale attuale), calcolato qui direttamente -
        # non usiamo cmd_term.metrics["error_vel_xy"/"error_vel_yaw"] perche' quella e' una somma
        # cumulativa normalizzata su resampling_time_range[1], che noi abbiamo forzato a 1e9 per
        # disattivare il ricampionamento automatico: la stessa formula la userebbe come
        # denominatore, rendendo l'incremento per-step trascurabile e il valore sempre ~0.
        error_xy = torch.norm(cmd_term.vel_command_b[0, :2] - robot.data.root_lin_vel_b[0, :2]).item()
        error_yaw = abs((cmd_term.vel_command_b[0, 2] - robot.data.root_ang_vel_b[0, 2]).item())
        #print(f"[velocity tracking] errore xy: {error_xy:.3f} m/s | errore yaw: {error_yaw:.3f} rad/s")

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
