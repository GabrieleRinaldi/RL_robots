# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guida interattivamente un checkpoint addestrato usando un controller Xbox (o compatibile).

Controllo a doppio stick, diretto sulle tre componenti del comando di velocita' - nessuna
"modalita'" da selezionare, ogni asse pilota sempre la sua componente:

    Analogico sinistro, su/giu'  -> velocita' lineare avanti/indietro (lin_vel_x), fino a
        --max_forward_speed.
    Analogico sinistro, sx/dx    -> velocita' laterale/strafe (lin_vel_y), fino a
        --max_lateral_speed.
    Analogico destro, sx/dx      -> velocita' angolare di imbardata (ang_vel_z), fino a
        --max_yaw_rate.

Nessun input -> tutte e tre le componenti a zero, robot fermo.

La telecamera del viewport segue automaticamente il robot da dietro (distanza --camera_distance,
altezza --camera_height), ruotando insieme a lui cosi' resta sempre alle sue spalle - utile per
vedere dove va e come si muovono le zampe. Disattivabile con --no_follow_camera.

Questo script NON lascia che il comando "base_velocity" si ricampioni da solo (lo facciamo
scrivendo direttamente nel suo stato interno ad ogni step): per questo forziamo
resampling_time_range a un valore enorme prima di creare l'ambiente, cosi' il timer di
ricampionamento automatico non scatta mai durante una sessione interattiva.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Guida un checkpoint RSL-RL con un controller Xbox.")
parser.add_argument("--num_envs", type=int, default=1, help="Numero di ambienti (di norma 1 per il teleop).")
parser.add_argument("--task", type=str, default=None, help="Nome del task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--max_forward_speed",
    type=float,
    default=1.5,
    help="Velocita' avanti/indietro (m/s) a stick sinistro (su/giu') a fondo corsa. Limite di training: 1.0 m/s.",
)
parser.add_argument(
    "--max_lateral_speed",
    type=float,
    default=1.0,
    help=(
        "Velocita' laterale/strafe (m/s) a stick sinistro (sx/dx) a fondo corsa. ATTENZIONE: il"
        " limite di training e' 0.4 m/s (limit_ranges.lin_vel_y in b1_training_env_cfg.py) - a 1.0"
        " si chiede alla policy una velocita' laterale 2.5x superiore a qualunque cosa abbia mai"
        " visto in training, quindi il comportamento a fondo corsa puo' essere instabile/fuori"
        " distribuzione, non e' un bug dello script."
    ),
)
parser.add_argument(
    "--max_yaw_rate",
    type=float,
    default=1.5,
    help="Velocita' angolare (rad/s) a stick destro (sx/dx) a fondo corsa. Limite di training: 1.0 rad/s.",
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
    default=11,
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
parser.add_argument("--deadzone", type=float, default=0.15, help="Zona morta degli stick (frazione, 0-1).")
parser.add_argument(
    "--gamepad_source",
    type=str,
    default="network",
    choices=["local", "network"],
    help=(
        "'local': legge un controller collegato FISICAMENTE a questa macchina (via carb.input). "
        "'network': riceve lo stato del controller via UDP da un altro PC (es. il tuo PC Windows "
        "con il controller collegato, usando gamepad_client_windows.py) - utile quando ti colleghi "
        "al server via NoMachine e il controller resta sul tuo PC locale."
    ),
)
parser.add_argument(
    "--gamepad_port", type=int, default=5555, help="Porta UDP in ascolto (usata solo con --gamepad_source network)."
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
import socket
import struct
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


class _GamepadReaderBase:
    """Stato condiviso: stick sinistro (x/y) + stick destro (solo x) - mappati 1:1 sulle tre
    componenti del comando di velocita' (lin_vel_x, lin_vel_y, ang_vel_z).

    Le sottoclassi si occupano SOLO di popolare questo stato (una legge da carb.input, l'altra
    da un socket UDP) - il resto dello script (in `main()`) non deve sapere da dove arriva il
    segnale.
    """

    def __init__(self, deadzone: float):
        self.deadzone = deadzone
        self._left_x = 0.0  # laterale (destra positivo)
        self._left_y = 0.0  # avanti (su positivo)
        self._right_x = 0.0  # imbardata (destra positivo)

    def _apply_deadzone(self, value: float) -> float:
        return value if abs(value) >= self.deadzone else 0.0

    def _set_left_stick(self, x: float, y: float):
        # segno invertito sull'asse laterale: fisicamente "stick a destra" deve mandare il robot
        # a destra (lin_vel_y negativo nella convenzione del robot), non a sinistra.
        self._left_x = -self._apply_deadzone(x)
        self._left_y = self._apply_deadzone(y)

    def _set_right_stick_x(self, x: float):
        self._right_x = self._apply_deadzone(x)

    def read(self) -> tuple[float, float, float]:
        """Legge lo stato corrente del controller.

        Returns:
            (frazione_lin_vel_x, frazione_lin_vel_y, frazione_ang_vel_z), ciascuna in [-1, 1].
        """
        return self._left_y, self._left_x, self._right_x


class LocalCarbGamepad(_GamepadReaderBase):
    """Legge un controller collegato FISICAMENTE a questa macchina, via carb.input.

    Non riusiamo `isaaclab.devices.gamepad.Se2Gamepad` perche' il suo meccanismo di callback
    per bottoni extra (`add_callback`) non passa il valore dell'evento alla funzione chiamata.
    Qui scriviamo direttamente il nostro `_on_event`, che ha accesso completo a `event.value`.
    """

    def __init__(self, deadzone: float = 0.15):
        super().__init__(deadzone)
        # disattiva il controllo camera di default di Omniverse via gamepad, altrimenti lo
        # stick muoverebbe anche la visuale oltre a pilotare il robot
        carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        # weakref: cosi' l'oggetto puo' essere distrutto normalmente (stesso pattern di Se2Gamepad)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )
        # ogni asse manda due eventi separati (es. "su" e "giu'"): teniamo i valori RAW (senza
        # deadzone, applicata solo in _apply_deadzone) e usiamo la differenza per il segno.
        self._raw = {
            "left_up": 0.0, "left_down": 0.0, "left_left": 0.0, "left_right": 0.0,
            "right_left": 0.0, "right_right": 0.0,
        }

    def __del__(self):
        self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)

    def _on_event(self, event: carb.input.GamepadEvent, *args, **kwargs):
        inp = event.input
        if inp == carb.input.GamepadInput.LEFT_STICK_UP:
            self._raw["left_up"] = event.value
        elif inp == carb.input.GamepadInput.LEFT_STICK_DOWN:
            self._raw["left_down"] = event.value
        elif inp == carb.input.GamepadInput.LEFT_STICK_RIGHT:
            self._raw["left_right"] = event.value
        elif inp == carb.input.GamepadInput.LEFT_STICK_LEFT:
            self._raw["left_left"] = event.value
        elif inp == carb.input.GamepadInput.RIGHT_STICK_LEFT:
            self._raw["right_left"] = event.value
        elif inp == carb.input.GamepadInput.RIGHT_STICK_RIGHT:
            self._raw["right_right"] = event.value
        else:
            return True
        self._set_left_stick(
            self._raw["left_right"] - self._raw["left_left"], self._raw["left_up"] - self._raw["left_down"]
        )
        self._set_right_stick_x(self._raw["right_right"] - self._raw["right_left"])
        return True


class NetworkGamepad(_GamepadReaderBase):
    """Riceve lo stato del controller via UDP da un PC remoto (es. gamepad_client_windows.py).

    Utile quando il controller e' collegato al TUO PC locale (non al server) e ti connetti al
    server via NoMachine/remote desktop: NoMachine non gira i segnali del joystick attraverso la
    sessione remota in modo affidabile, quindi il PC locale legge il controller per conto suo e
    manda i valori qui via rete.

    Formato pacchetto (12 byte, vedi PACKET_FORMAT): forward_x (float, stick sx verticale),
    lateral_y (float, stick sx orizzontale), yaw_z (float, stick dx orizzontale) - stesso ordine
    di vel_command_b (lin_vel_x, lin_vel_y, ang_vel_z). Deve corrispondere ESATTAMENTE a quello
    inviato da gamepad_client_windows.py.
    """

    PACKET_FORMAT = "!fff"
    PACKET_SIZE = struct.calcsize(PACKET_FORMAT)

    def __init__(self, port: int, deadzone: float = 0.15):
        super().__init__(deadzone)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # non-bloccante: se non ci sono pacchetti pronti, .recvfrom() solleva BlockingIOError
        # invece di fermare tutto il loop di simulazione in attesa di dati dalla rete
        self._sock.setblocking(False)
        self._sock.bind(("0.0.0.0", port))
        print(f"[INFO] In ascolto per il controller via rete su UDP porta {port}")

    def poll(self):
        """Da chiamare una volta per step (prima di `read()`): svuota la coda dei pacchetti in
        arrivo e tiene solo l'ULTIMO (i pacchetti piu' vecchi rappresentano solo latenza inutile,
        non ha senso elaborarli se ne e' gia' arrivato uno piu' recente)."""
        latest = None
        while True:
            try:
                data, _ = self._sock.recvfrom(self.PACKET_SIZE)
            except BlockingIOError:
                break
            if len(data) == self.PACKET_SIZE:
                latest = data
        if latest is not None:
            # ATTENZIONE all'ordine: il client (gamepad_client_windows.py) impacchetta
            # (forward_x, lateral_y, yaw_z) - _set_left_stick si aspetta (x=laterale, y=avanti),
            # quindi qui i primi due vanno passati INVERTITI rispetto all'ordine di arrivo.
            forward_x, lateral_y, yaw_z = struct.unpack(self.PACKET_FORMAT, latest)
            self._set_left_stick(lateral_y, forward_x)
            self._set_right_stick_x(yaw_z)


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
    # Default a CPU per la fisica: con num_envs=1 (teleop interattivo) la pipeline fisica GPU di
    # PhysX ha un overhead fisso di lancio/sincronizzazione kernel che non si ammortizza senza
    # migliaia di env paralleli - su CPU quell'overhead sparisce ed e' molto piu' veloce per un
    # solo robot (misurato: env.step() passa da ~70-85ms a ~40-50ms). Il rendering resta comunque
    # su GPU via RTX/DLSS indipendentemente da questo. Passa --device cuda:0 per tornare alla GPU.
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cpu"

    # disattiva il ricampionamento automatico del comando: lo pilotiamo noi ad ogni step
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    # disattiva il time_out: l'episodio normalmente finisce (e il robot si resetta) dopo
    # episode_length_s secondi (20s di default) - in una sessione interattiva col gamepad non lo
    # vogliamo, l'episodio deve durare finche' non chiudiamo noi lo script. Le altre terminazioni
    # (bad_orientation, base_contact - il robot che cade davvero) restano attive: se cade si
    # resetta comunque, che e' il comportamento voluto.
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
    # (lin_vel_x, lin_vel_y, ang_vel_z) ad ogni step dagli stick, senza passare per il
    # meccanismo di heading/standing della classe (che restano ai loro default e non toccano
    # vel_command_b da soli, dato che il resampling e' disattivato).
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    if args_cli.gamepad_source == "network":
        gamepad = NetworkGamepad(port=args_cli.gamepad_port, deadzone=args_cli.deadzone)
    else:
        gamepad = LocalCarbGamepad(deadzone=args_cli.deadzone)
    print("[INFO] Gamepad pronto: stick sinistro = avanti/indietro + laterale, stick destro = rotazione.")

    dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]
    step_counter = 0
    obs = env.get_observations()
    if not args_cli.no_follow_camera:
        _update_chase_camera(env, args_cli.camera_distance, args_cli.camera_height)
    while simulation_app.is_running():
        # time delay for real-time evaluation (stesso snippet di play.py)
        start_time = time.time()

        if args_cli.gamepad_source == "network":
            gamepad.poll()
        lin_x, lin_y, yaw = gamepad.read()
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
