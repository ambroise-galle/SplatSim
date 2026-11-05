"""Serveur du simulateur interactif : physique MuJoCo en temps réel, commandes et états par WebSocket, fichiers du
visualiseur (PlayCanvas) et de la scène (streamed SOG) en HTTP.

  .venv/bin/python simrobot/sim/server.py --scene data_local/simrobot/scenes/HOB-1_5 --port 8765
  puis ouvrir http://localhost:8765

Commandes : la dernière commande reçue est GARDÉE tant que la connexion est ouverte (le navigateur envoie à chaque
changement de touche + un rappel à 10 Hz) ; remise à zéro à la déconnexion. Une remise à zéro sur délai (0,5 s) faisait
avancer / s'arrêter le robot dès que le navigateur ralentissait ses minuteries (onglet masqué, rendu lourd).
Un seul onglet conduit : celui qui a envoyé la dernière commande non nulle. Les rappels « au repos » des autres onglets
sont ignorés ; sinon un second onglet ouvert remettait la commande à zéro dix fois par seconde. Les changements de commande
sont gardés en mémoire pour le diagnostic : GET /debug.

Étiquettes du splat (`terrain_etiquettes.npz`, cf. terrain_labels.py), si la scène en a : les obstacles bas typés
« buisson » ne sont plus des cylindres rigides mais des zones qui freinent le robot, ceux typés « relief » (talus,
mottes de terre) passent dans le champ de hauteur, et l'adhérence des roues suit la classe du sol (`surfaces.json`).
Avec `geometrie_etiquetee.npz` (splat_mesh.py) : troncs inclinés et troncs couchés en capsules MuJoCo.
`--sans-etiquettes` : terrain d'origine (référence) ; `--sans-sol` : obstacles typés mais adhérence uniforme.

Repères : la physique tourne dans le repère monde recentré (z haut) ; tout ce qui part vers le visualiseur est
converti en y haut : p_vue = (x, z, −y), q_vue = q(−90° autour de x) ⊗ q (quaternions envoyés en x, y, z, w).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time

import numpy as np
from aiohttp import WSMsgType, web

sys.path.insert(0, os.path.dirname(__file__))
from rover import WHEELS, Ground, Rover, RoverParams, car_params  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "..", "viewer")
Q_YUP = np.array([np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0.0, 0.0])      # w, x, y, z
IDLE = {"throttle": 0.0, "steer": 0.0, "brake": False, "turbo": False}


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def to_view(pos, quat):
    q = qmul(Q_YUP, np.asarray(quat))
    return [round(pos[0], 4), round(pos[2], 4), round(-pos[1], 4)], [round(float(v), 5) for v in (q[1], q[2], q[3], q[0])]


class Sim:
    def __init__(self, scene_dir, terrain="terrain.npz", sans_obstacles_bas=False, carte_lidar="lidar_occupation.npz",
                 etiquettes=True, sol=True, lidar=True, vehicule="robot"):
        self.scene_dir = scene_dir
        self.meta = json.load(open(f"{scene_dir}/scene.json"))
        self.terrain = dict(np.load(f"{scene_dir}/{terrain}"))
        terrain_lidar = dict(self.terrain)                    # le LiDAR simulé garde le sol d'origine (validé)
        tr, src = self.terrain["troncs"], self.terrain["troncs_source"]
        typ = np.zeros(len(tr), np.int8)                      # 0 rigide, 1 buisson, 2 relief (terrain_labels.py)
        ground = None
        path_e = f"{scene_dir}/terrain_etiquettes.npz"
        if etiquettes and terrain == "terrain.npz" and os.path.exists(path_e):
            E = np.load(path_e)
            assert len(E["obs_type"]) == len(tr), "terrain_etiquettes.npz ne correspond pas à terrain.npz"
            typ = E["obs_type"]
            dH = E["dH"] if "dH" in E else 0.0                # bosses invisibles aplanies, sol abaissé sous la végétation
            self.terrain["H"] = (self.terrain["H"] + E["relief_dh"] + dH).astype(np.float32)
            # plancher du LiDAR simulé : même sol corrigé (plateaux de végétation dense aplanis, sol sous la végétation),
            # sans les reliefs, déjà dans la carte d'occupation (validé : 43,5 → 45 % de rayons justes)
            terrain_lidar["H"] = (terrain_lidar["H"] + dH).astype(np.float32)
            self.terrain["relief"] = E["relief_dh"] > 0
            b = typ == 1
            s_b = E["obs_s"][b] if "obs_s" in E else np.clip(E["obs_mu"][b] / 0.55, 0.3, 3.0)
            veg = E["veg_s"] if "veg_s" in E else None
            ground = Ground(self.terrain, E["sol"] if sol else None, np.column_stack([tr[b, :3], s_b]), veg=veg)
            self.terrain["veg"] = ground.veg > 0.05
            # cellules « obstacle » de l'ancien terrain : affichage seulement, trompeur une fois les obstacles retypés
            self.terrain["obstacle"] = np.zeros_like(self.terrain["obstacle"])
        # troncs inclinés et troncs couchés en capsules (splat_mesh.py) : remplacent les cylindres verticaux concernés
        caps, kinds = [], []
        path_g = f"{scene_dir}/geometrie_etiquetee.npz"
        if etiquettes and terrain == "terrain.npz" and os.path.exists(path_g):
            Gm = np.load(path_g)
            typ = typ.copy()
            for i, b, u in zip(Gm["incl_idx"], Gm["incl_base"], Gm["incl_u"]):
                if typ[i] == 4:                               # tronc sans gaussienne visible : retiré
                    continue
                v = u / u[2]                                  # pas vertical de 1 m le long de l'axe
                caps.append([*(b - 0.1 * v), *(b + 1.9 * v), tr[i, 2]])
                kinds.append(0)
                typ[i] = 3                                    # 3 : remplacé par une capsule
            for c in Gm["couches"]:
                caps.append(list(c))
                kinds.append(1)
            typ[Gm["bas_remplaces"]] = 3
        self.terrain["capsules"] = np.array(caps, np.float32).reshape(-1, 7)
        self.obstacles_vue = {"troncs": tr, "source": src, "type": typ,         # affichage (calque T)
                              "capsules": self.terrain["capsules"], "capsules_type": np.array(kinds, np.int8)}
        keep = typ == 0
        if sans_obstacles_bas:                                # troncs seulement (obstacles bas traversables)
            keep &= src < 2
            self.terrain["obstacle"] = np.zeros_like(self.terrain["obstacle"])
        self.terrain["troncs"], self.terrain["troncs_source"] = tr[keep], src[keep]
        self.params = car_params() if vehicule == "voiture" else RoverParams()
        self.track = None
        if vehicule == "voiture":                             # mode jeu : circuit sur la vraie trajectoire
            from course import build_track

            self.track = build_track(scene_dir, self.terrain)
        # départ par défaut : le plus proche du centre de la scène (les bords sont moins bien reconstruits)
        self.start_idx = int(np.argmin([np.hypot(*d["xyz"][:2]) for d in self.meta["departs"]]))
        self.rover = Rover(self.terrain, self.meta["departs"][self.start_idx], self.params, ground)
        self.cmd = dict(IDLE)
        self.cmd_time = 0.0
        self.driver = None                         # connexion qui conduit (cf. set_cmd)
        self.client_cmd = {}                       # dernière commande reçue de chaque connexion
        self.cmd_log = collections.deque(maxlen=200)
        self.paused = False
        self.flipped_since = None                  # instant où le robot s'est couché (relevage automatique après 2 s)
        self.clients = set()
        self.step_wall = 0.0                       # temps CPU moyen d'un pas (diagnostic)
        # LiDAR simulé (lidar_sim.py) si la scène a sa carte de voxels ; calculé seulement si un onglet l'a demandé
        self.lidar, self.lidar_clients, self.lidar_frac = None, set(), 0.0     # frac : s de calcul / s simulée
        if lidar and (os.path.exists(f"{scene_dir}/lidar_occupation.npz") or os.path.exists(f"{scene_dir}/lidar_voxels.npz")):
            from lidar_sim import Lidar, load_map

            hx, _, hz = self.params.chassis_half
            # carte d'occupation mesurée (lidar_occupancy.py) si présente, sinon l'ancienne carte de voxels
            self.lidar = Lidar(load_map(scene_dir, terrain_lidar, carte_lidar), mount=(hx - 0.05, 0.0, hz + 0.18))
            self.lidar.vmap.cast(np.zeros(3), np.array([[1.0, 0.0, 0.0]]))      # compilation numba au démarrage

    def reset(self, idx=None):
        if idx is not None:
            self.start_idx = idx % len(self.meta["departs"])
        self.rover.reset(self.meta["departs"][self.start_idx])

    def set_cmd(self, ws, cmd):
        """Applique la commande si `ws` conduit, ou prend la main s'il envoie une commande non nulle nouvelle."""
        prev = self.client_cmd.get(ws)
        self.client_cmd[ws] = cmd
        idle = cmd["throttle"] == 0 and cmd["steer"] == 0 and not cmd["brake"]
        if ws is self.driver or (not idle and cmd != prev):
            self.driver = ws
            self.cmd = cmd
            self.cmd_time = time.perf_counter()
        if cmd != prev:
            self.cmd_log.append({"t": round(time.time(), 3), "client": id(ws) % 10000, "applique": ws is self.driver,
                                 **cmd, "vitesse": round(self.rover.state()["speed"], 2)})

    def lidar_scan_ms(self):
        return self.lidar_frac / self.lidar.rate * 1000 if self.lidar else 0.0

    def lidar_msg(self, scan, t):
        """Balayage → message binaire : en-tête 'LID1', lignes, colonnes (uint16), t, origine x y z (physique), temps de
        calcul par tour (float32) ; puis xyz en cm relatifs à l'origine (int16, -32768 = pas de retour), distance en cm
        (uint16, 0 = pas de retour), intensité (uint8)."""
        L = self.lidar
        v = np.isfinite(scan["r"])
        rel = np.where(v[..., None], np.round((scan["pts"] - scan["origin"]) * 100), -32768)
        rng = np.where(v, np.round(scan["r"] * 100), 0)
        head = np.array([L.rings, L.cols], "<u2").tobytes() + np.array(
            [t, *scan["origin"], self.lidar_scan_ms()], "<f4").tobytes()
        return (b"LID1" + head + np.clip(rel, -32768, 32767).astype("<i2").tobytes()
                + np.clip(rng, 0, 65535).astype("<u2").tobytes() + scan["i"].astype(np.uint8).tobytes())

    def drop_client(self, ws):
        self.lidar_clients.discard(ws)
        self.client_cmd.pop(ws, None)
        if ws is self.driver:                                 # plus personne aux commandes : on relâche tout
            self.driver = None
            self.cmd = dict(IDLE)

    def hello(self):
        p = self.params
        return {"type": "hello", "departs": self.meta["departs"], "start": self.start_idx,
                "splat_view_rotation_deg": self.meta.get("splat_view_rotation_deg", [0, 0, 0]),
                "lidar": ({"rings": self.lidar.rings, "cols": self.lidar.cols, "rate": self.lidar.rate,
                           "max_range": self.lidar.max_range} if self.lidar else None),
                "circuit": self.track,
                "rover": {"vehicule": p.vehicule, "chassis_half": p.chassis_half, "wheel_radius": p.wheel_radius,
                          "wheel_half_width": p.wheel_half_width, "wheel_xy": p.wheel_xy(), "mount_z": p.mount_z,
                          "susp_travel": p.susp_travel, "v_max": p.v_max},
                "dt": float(self.rover.model.opt.timestep)}

    def ground_msg(self):
        """Sol sous le robot (HUD) : classe, adhérence moyenne des roues, freinage de la végétation."""
        r = self.rover
        if r.ground is None:
            return None
        x, y = r.data.qpos[:2]
        gi = r.ground_info
        return {"classe": r.ground.name(x, y), "adherence": round(gi["adherence"], 2),
                "frein_veg_N": round(gi["frein_veg_N"])}

    def state_msg(self):
        s = self.rover.state()
        poses = {k: to_view(*v) for k, v in s["poses"].items()}
        return {"type": "state", "t": round(s["t"], 3), "speed": round(s["speed"], 3),
                "susp": [round(v, 4) for v in s["susp"]], "poses": poses,
                "step_us": round(self.step_wall * 1e6, 1), "paused": self.paused,
                "cmd": [self.cmd["throttle"], self.cmd["steer"]], "v_cmd": round(self.rover.v_cmd, 2),
                "clients": len(self.clients), "lidar_ms": round(self.lidar_scan_ms(), 1) if self.lidar_clients else None,
                "sol": self.ground_msg(), "glisse": round(self.rover.rear_slip_deg(), 1),
                "cmd_age_ms": round((time.perf_counter() - self.cmd_time) * 1000)}

    def terrain_msg(self, step=2):
        T = self.terrain
        H = T["H"][::step, ::step]
        cov = T["couvert"][::step, ::step]
        return {"lo": [float(T["lo"][0]), float(T["lo"][1])], "res": float(T["res"]) * step,
                "offset": float(T["res"]) / 2, "shape": list(H.shape),
                "H": np.round(H, 3).ravel().tolist(), "couvert": cov.astype(np.uint8).ravel().tolist(),
                "obstacle": (T["obstacle"][::step, ::step] if "obstacle" in T else np.zeros_like(cov)).astype(np.uint8)
                .ravel().tolist(),
                "relief": (T["relief"][::step, ::step] if "relief" in T else np.zeros_like(cov)).astype(np.uint8)
                .ravel().tolist(),
                "veg": (T["veg"][::step, ::step] if "veg" in T else np.zeros_like(cov)).astype(np.uint8).ravel().tolist(),
                # tous les obstacles d'origine, avec leur type : 0 rigide, 1 végétation (freine), 2 relief (dans le sol),
                # 3 remplacé par une capsule, 4 sans gaussienne visible (retiré)
                "troncs": np.round(self.obstacles_vue["troncs"], 3).tolist(),
                # 0 : tronc (cercles), 1 : tronc (continuité verticale), 2 : obstacle bas (cf. stems.SOURCES)
                "troncs_source": self.obstacles_vue["source"].astype(int).tolist(),
                "troncs_type": self.obstacles_vue["type"].astype(int).tolist(),
                # capsules a(x y z) b(x y z) r : 0 tronc incliné, 1 tronc couché (repère physique)
                "capsules": np.round(self.obstacles_vue["capsules"], 3).tolist(),
                "capsules_type": self.obstacles_vue["capsules_type"].astype(int).tolist()}


async def physics_loop(app):
    sim: Sim = app["sim"]
    dt = float(sim.rover.model.opt.timestep)
    t0 = time.perf_counter()
    sim_t = 0.0
    last_send = 0.0
    while True:
        now = time.perf_counter()
        target = now - t0
        n = int((target - sim_t) / dt)
        if n > 100:                                           # retard (onglet en pause, etc.) : on rattrape sans emballer
            t0 += (n - 100) * dt
            n = 100
        if not sim.paused and n > 0:
            c = sim.cmd
            w0 = time.perf_counter()
            # commande recalculée une fois par lot (~250 Hz), pas à chaque pas de 2 ms : le coût Python dominait
            sim.rover.control(c["throttle"], c["steer"], c["brake"], c["turbo"], dt * n)
            sim.rover.step(n)
            sim.step_wall = 0.9 * sim.step_wall + 0.1 * (time.perf_counter() - w0) / n
            if sim.lidar is not None and sim.lidar_clients:
                d = sim.rover.data
                ci = sim.rover.ids["chassis"]
                w1 = time.perf_counter()
                scan = sim.lidar.advance(dt * n, d.xpos[ci].copy(), d.xmat[ci].reshape(3, 3).copy())
                # temps de calcul LiDAR par seconde simulée (EMA) ; affiché en ms par tour
                sim.lidar_frac = 0.95 * sim.lidar_frac + 0.05 * (time.perf_counter() - w1) / (dt * n)
                if scan is not None:
                    msg = sim.lidar_msg(scan, float(d.time))
                    for ws in list(sim.lidar_clients):
                        try:
                            await ws.send_bytes(msg)
                        except ConnectionResetError:
                            sim.lidar_clients.discard(ws)
            if sim.rover.upright() < 0.3:                      # couché ou sur le toit : relevé après 2 s
                sim.flipped_since = sim.flipped_since or now
                if now - sim.flipped_since > 2.0:
                    sim.rover.right()
                    sim.flipped_since = None
            else:
                sim.flipped_since = None
        sim_t += max(n, 0) * dt
        if now - last_send >= 1 / 60 and sim.clients:
            msg = json.dumps(sim.state_msg())
            for ws in list(sim.clients):
                try:
                    await ws.send_str(msg)
                except ConnectionResetError:
                    sim.clients.discard(ws)
            last_send = now
        await asyncio.sleep(0.002)


async def ws_handler(request):
    sim: Sim = request.app["sim"]
    ws = web.WebSocketResponse(heartbeat=10)
    await ws.prepare(request)
    sim.clients.add(ws)
    await ws.send_str(json.dumps(sim.hello()))
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            if m.get("type") == "cmd":
                sim.set_cmd(ws, {"throttle": float(np.clip(m.get("throttle", 0), -1, 1)),
                                 "steer": float(np.clip(m.get("steer", 0), -1, 1)),
                                 "brake": bool(m.get("brake")), "turbo": bool(m.get("turbo"))})
            elif m.get("type") == "reset":
                sim.reset(m.get("start"))
            elif m.get("type") == "respawn":                # mode jeu : replacé sur le circuit (x, y physique, cap)
                sim.rover.reset({"xyz": [float(m["x"]), float(m["y"]), 0.0], "yaw": float(m["yaw"])})
            elif m.get("type") == "relever":
                sim.rover.right()
            elif m.get("type") == "pause":
                sim.paused = not sim.paused
            elif m.get("type") == "lidar":
                if m.get("on") and sim.lidar is not None:
                    sim.lidar_clients.add(ws)
                else:
                    sim.lidar_clients.discard(ws)
            elif m.get("type") == "terrain":
                await ws.send_str(json.dumps({"type": "terrain", **sim.terrain_msg()}))
    finally:
        sim.clients.discard(ws)
        sim.drop_client(ws)
    return ws


def make_app(scene_dir, terrain="terrain.npz", sans_obstacles_bas=False, carte_lidar="lidar_occupation.npz",
             etiquettes=True, sol=True, jeu=False):
    app = web.Application()
    app["sim"] = Sim(scene_dir, terrain, sans_obstacles_bas, carte_lidar, etiquettes, sol, lidar=not jeu,
                     vehicule="voiture" if jeu else "robot")

    async def index(_):
        return web.FileResponse(os.path.join(VIEWER, "course.html" if jeu else "index.html"))

    async def debug(_):
        sim = app["sim"]
        st = sim.rover.state()
        return web.json_response({"clients": len(sim.clients), "commande": sim.cmd, "v_cmd": sim.rover.v_cmd,
                                  "vitesse": st["speed"], "position": st["poses"]["chassis"][0],
                                  "changements": list(sim.cmd_log)})

    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/debug", debug)
    app.router.add_static("/vendor/playcanvas/", os.path.join(VIEWER, "node_modules", "playcanvas"))
    app.router.add_static("/app/", VIEWER)
    app.router.add_static("/scene/", scene_dir)

    async def start_bg(app):
        app["physics"] = asyncio.create_task(physics_loop(app))

    async def stop_bg(app):
        app["physics"].cancel()

    app.on_startup.append(start_bg)
    app.on_cleanup.append(stop_bg)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--terrain", default="terrain.npz", help="fichier de terrain de la scène (ex. terrain_cellules.npz)")
    ap.add_argument("--sans-obstacles-bas", action="store_true",
                    help="troncs seulement : souches, troncs couchés et buissons denses deviennent traversables")
    ap.add_argument("--carte-lidar", default="lidar_occupation.npz",
                    help="carte du LiDAR simulé : lidar_occupation.npz (Hesai + comblement Livox / Velodyne) ou "
                         "lidar_occupation_hesai.npz (Hesai seul, ~30 %% plus rapide)")
    ap.add_argument("--sans-etiquettes", action="store_true",
                    help="ignorer terrain_etiquettes.npz : obstacles bas tous rigides, adhérence uniforme (référence)")
    ap.add_argument("--sans-sol", action="store_true", help="obstacles typés, mais adhérence uniforme")
    ap.add_argument("--jeu", action="store_true",
                    help="démo de course : petite voiture qui dérape, circuit sur la vraie trajectoire (course.html)")
    a = ap.parse_args()
    web.run_app(make_app(a.scene, a.terrain, a.sans_obstacles_bas, a.carte_lidar, not a.sans_etiquettes,
                         not a.sans_sol, a.jeu), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
