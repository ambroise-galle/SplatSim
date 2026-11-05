"""Robot à roues MuJoCo : 4 roues motrices, 4 suspensions indépendantes (glissière verticale ressort + amortisseur),
direction Ackermann sur l'essieu avant. Repère physique : x avant, y gauche, z haut (mètres, kg, s).

Gabarit (HÖB-1 complète, suivi automatique du vrai sentier, 32 tronçons) :
- 0,83 m de large, garde au sol 0,19 m, 12 N·m : 18–19 tronçons franchis (troncs en bordure, châssis posé) ;
- 0,76 m, garde 0,27 m, 25 N·m, lest bas, braquage limité par la vitesse : 23 tronçons, 2 retournements ;
- 0,66 m : 24 tronçons mais 4 retournements (basculement en escaladant les talus de bord de sentier).

Le terrain est un champ de hauteur MuJoCo (hfield) construit depuis `terrain.npz` (cf. terrain.py).

Sol étiqueté (`Ground`, si la scène a `terrain_etiquettes.npz`) : adhérence de chaque roue et résistance au roulement
selon la classe du sol sous la roue (`surfaces.json`), freinage du châssis dans les buissons. MuJoCo combine les
frottements de deux géométries par le MAXIMUM : le terrain reçoit un frottement quasi nul, et celui de chaque roue est
réécrit à chaque pas.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

WHEELS = ("fl", "fr", "rl", "rr")


@dataclass
class RoverParams:
    chassis_half: tuple = (0.42, 0.20, 0.09)   # demi-dimensions du châssis (m) — gabarit de l'ANYmal (~0,6 m de large)
    chassis_mass: float = 20.0                 # coque
    ballast_mass: float = 25.0                 # batteries, au ras du fond du châssis : centre de gravité bas
    a_lat_max: float = 2.5                     # accélération latérale max (m/s²) : braquage limité selon la vitesse
    wheel_radius: float = 0.20
    wheel_half_width: float = 0.05
    wheel_mass: float = 3.0
    knuckle_mass: float = 1.0
    wheelbase: float = 0.66                    # entraxe avant / arrière
    track: float = 0.66                        # voie gauche / droite (0,76 m hors tout) ; 0,56 : plus de retournements
    mount_z: float = -0.16                     # centre de roue sous le centre du châssis au repos (garde au sol 0,27 m)
    susp_travel: float = 0.10                  # débattement ± (m)
    susp_k: float = 4000.0                     # raideur par roue (N/m)
    susp_zeta: float = 0.45                    # amortissement relatif
    steer_max: float = 0.55                    # braquage max de la roue intérieure (rad)
    v_max: float = 2.0                         # vitesse max (m/s), ×2 en mode turbo
    accel: float = 2.0                         # m/s²
    wheel_kv: float = 8.0                      # gain de l'asservissement en vitesse de roue (N·m·s/rad)
    wheel_torque: float = 25.0                 # couple max par roue (N·m) ; à 12 N·m le robot calait sur racines et marches
    friction: float = 1.0                      # adhérence uniforme (sans sol étiqueté)
    vehicule: str = "robot"
    # transmission : "servo4" = 4 roues asservies en vitesse (robot) ; "propulsion" = roues arrière en couple (voiture)
    drive: str = "servo4"
    power_w: float = 0.0                       # propulsion : puissance max (W, toutes roues motrices)
    brake_torque: float = 0.0                  # propulsion : frein de service par roue (N·m)
    handbrake_torque: float = 0.0              # propulsion : frein à main, roues arrière (N·m)
    steer_rate: float = 2.5                    # vitesse de braquage de la commande (1/s)
    aero_cda: float = 0.0                      # traînée aérodynamique (m²) : F = ½ ρ CdA v²

    def wheel_xy(self):
        hx, hy = self.wheelbase / 2, self.track / 2
        return {"fl": (hx, hy), "fr": (hx, -hy), "rl": (-hx, hy), "rr": (-hx, -hy)}

    def corner_load(self):
        m = self.chassis_mass + self.ballast_mass + 4 * self.knuckle_mass + 4 * self.wheel_mass
        return m * 9.81 / 4

    def susp_damping(self):
        m_sprung = (self.chassis_mass + self.ballast_mass) / 4
        return 2 * self.susp_zeta * np.sqrt(self.susp_k * m_sprung)


class Ground:
    """Sol étiqueté : grilles d'adhérence et de roulement (classe ADE20K du sol par cellule de 10 cm) et freinage de la
    végétation : `veg` (grille de freinage relatif s, végétation en nappe) + `buissons` (x, y, r, s : obstacles bas
    typés végétation, peints dans la grille). `sol` None : adhérence uniforme, seul le freinage s'applique."""

    def __init__(self, terrain, sol=None, buissons=np.zeros((0, 4)), table=None, veg=None):
        table = table or json.load(open(os.path.join(os.path.dirname(__file__), "surfaces.json")))
        self.table = table
        self.lo, self.res = np.asarray(terrain["lo"], float), float(terrain["res"])
        self.nx, self.ny = terrain["H"].shape
        self.sol = sol
        if sol is not None:
            mu = np.full(256, table["defaut"]["adherence"], np.float32)
            crr = np.full(256, table["defaut"]["roulement"], np.float32)
            self.noms = {}
            for k, v in table["classes"].items():
                mu[int(k)], crr[int(k)] = v["adherence"], v["roulement"]
                self.noms[int(k)] = v["nom"].split(" (")[0]
            self.mu, self.crr = mu[sol], crr[sol]
        b = table["buisson"]
        self.F0, self.F1 = b["F0_N"], b["F1_N_par_m_s"]
        self.veg = (np.zeros((self.nx, self.ny), np.float32) if veg is None   # freinage relatif s, 0 hors végétation
                    else np.asarray(veg, np.float32).copy())
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        for x, y, r, sv in buissons:
            cx, cy, rr = (x - self.lo[0]) / self.res, (y - self.lo[1]) / self.res, (r + 0.1) / self.res
            i0, i1 = int(max(0, cx - rr - 1)), int(min(self.nx, cx + rr + 2))
            j0, j1 = int(max(0, cy - rr - 1)), int(min(self.ny, cy + rr + 2))
            m_ = (gx[i0:i1, j0:j1] + 0.5 - cx) ** 2 + (gy[i0:i1, j0:j1] + 0.5 - cy) ** 2 <= rr ** 2
            sub = self.veg[i0:i1, j0:j1]
            sub[m_] = np.maximum(sub[m_], sv)

    def cell(self, xy):
        i = np.clip(((xy[..., 0] - self.lo[0]) / self.res).astype(np.int64), 0, self.nx - 1)
        j = np.clip(((xy[..., 1] - self.lo[1]) / self.res).astype(np.int64), 0, self.ny - 1)
        return i, j

    def name(self, x, y):
        if self.sol is None:
            return None
        i, j = self.cell(np.array([x, y]))
        c = int(self.sol[i, j])
        return self.noms.get(c, f"classe {c}")


def car_params():
    """Petite voiture tout-terrain (buggy ~1/4) : propulsion, moteurs en couple limités en puissance, frein à main.
    Gabarit 1,0 × 0,61 m (passe sur le sentier), 18 kg, garde au sol 0,15 m, roues de 13 cm (≈ buggy thermique 1/5,
    2 kW). Charge par roue ~44 N : sur terre (μ 0,55), une roue arrière décroche au-delà de ~3 N·m ; avec 6 N·m et
    2 kW, l'arrière glisse en accélération ou au frein à main, comme en rallye sur terre."""
    return RoverParams(vehicule="voiture", drive="propulsion",
                       chassis_half=(0.36, 0.17, 0.06), chassis_mass=9.0, ballast_mass=4.0, a_lat_max=0.0,
                       wheel_radius=0.13, wheel_half_width=0.045, wheel_mass=1.0, knuckle_mass=0.25,
                       wheelbase=0.62, track=0.52, mount_z=-0.08, susp_travel=0.07, susp_k=1500.0, susp_zeta=0.5,
                       steer_max=0.6, v_max=11.0, accel=0.0, wheel_torque=6.0, power_w=2000.0, brake_torque=5.0,
                       handbrake_torque=12.0, steer_rate=4.0, aero_cda=0.12)


def build_xml(terrain, start_xyz, start_yaw, p: RoverParams = RoverParams(), terrain_friction=None):
    H, lo, res = terrain["H"], terrain["lo"], float(terrain["res"])
    nx, ny = H.shape
    zmin, zmax = float(H.min()), float(H.max())
    rx, ry = (nx - 1) * res / 2, (ny - 1) * res / 2            # sommets du hfield = centres de cellule
    cx, cy = lo[0] + res / 2 + rx, lo[1] + res / 2 + ry
    # ressort pré-chargé : la suspension est à mi-course au repos (q = 0 ; q > 0 = compression)
    springref = -p.corner_load() / p.susp_k
    damping = p.susp_damping()
    hx, hy, hz = p.chassis_half
    z0 = start_xyz[2] + p.wheel_radius - p.mount_z + 0.05
    qz = (np.cos(start_yaw / 2), 0, 0, np.sin(start_yaw / 2))
    corners = ""
    for w, (x, y) in p.wheel_xy().items():
        steer = (f'<joint name="steer_{w}" type="hinge" axis="0 0 1" range="-{p.steer_max * 1.3} {p.steer_max * 1.3}" '
                 f'damping="5" armature="0.02"/>') if w[0] == "f" else ""
        corners += f"""
      <body name="knuckle_{w}" pos="{x} {y} {p.mount_z}">
        <joint name="susp_{w}" type="slide" axis="0 0 1" range="-{p.susp_travel} {p.susp_travel}" limited="true"
               stiffness="{p.susp_k}" springref="{springref}" damping="{damping}" armature="0.05"/>
        {steer}
        <geom type="sphere" size="0.03" mass="{p.knuckle_mass}" contype="0" conaffinity="0" rgba=".6 .6 .6 1"/>
        <body name="wheel_{w}">
          <joint name="spin_{w}" type="hinge" axis="0 1 0" damping="0.02" armature="0.01"/>
          <geom name="wheel_{w}" type="cylinder" size="{p.wheel_radius} {p.wheel_half_width}" zaxis="0 1 0"
                mass="{p.wheel_mass}" friction="{p.friction} 0.005 0.0001" condim="3" contype="2" conaffinity="1" rgba=".12 .12 .12 1"/>
        </body>
      </body>"""
    # obstacles (troncs, souches) : cylindres statiques de 2 m, frottement réduit (pas d'escalade de paroi)
    trunks = "".join(
        f'\n    <geom type="cylinder" size="{r:.3f} 1.0" pos="{x:.3f} {y:.3f} {z + 0.9:.3f}" friction="0.1 0.005 0.0001" '
        f'contype="1" conaffinity="2" rgba=".9 .6 .1 .5"/>' for x, y, r, z in terrain.get("troncs", np.zeros((0, 4))))
    # troncs inclinés et troncs couchés (splat_mesh.py) : capsules quelconques, même frottement réduit
    trunks += "".join(
        f'\n    <geom type="capsule" fromto="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} {c[3]:.3f} {c[4]:.3f} {c[5]:.3f}" '
        f'size="{c[6]:.3f}" friction="0.1 0.005 0.0001" contype="1" conaffinity="2" rgba=".9 .6 .1 .5"/>'
        for c in terrain.get("capsules", np.zeros((0, 7))))
    if p.drive == "propulsion":                       # moteurs en couple sur les roues arrière (commande = couple)
        acts = "".join(f'<motor name="drive_{w}" joint="spin_{w}" ctrllimited="true" '
                       f'ctrlrange="-{p.wheel_torque} {p.wheel_torque}"/>' for w in ("rl", "rr"))
    else:
        acts = "".join(
            f'<velocity name="drive_{w}" joint="spin_{w}" kv="{p.wheel_kv}" forcelimited="true" '
            f'forcerange="-{p.wheel_torque} {p.wheel_torque}"/>' for w in WHEELS)
    acts += "".join(f'<position name="steer_{w}" joint="steer_{w}" kp="3000" kv="60" forcelimited="true" '
                    f'forcerange="-300 300"/>' for w in ("fl", "fr"))
    return f"""<mujoco model="rover">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="{'elliptic' if p.drive == 'propulsion' else 'pyramidal'}"/>
  <asset>
    <!-- collisions : robot (contype 2, conaffinity 1) contre terrain (1, 2) seulement ; aucune collision interne
         au robot (sinon la roue braquée touche le châssis, « petit-fils » non exclu par MuJoCo) -->
    <hfield name="terrain" nrow="{ny}" ncol="{nx}" size="{rx} {ry} {max(zmax - zmin, 1e-3)} 0.5"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain" pos="{cx} {cy} {zmin}" friction="{p.friction if terrain_friction is None else terrain_friction} 0.005 0.0001"
          contype="1" conaffinity="2"/>
{trunks}
    <body name="chassis" pos="{start_xyz[0]} {start_xyz[1]} {z0}" quat="{qz[0]} {qz[1]} {qz[2]} {qz[3]}">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="{hx} {hy} {hz}" mass="{p.chassis_mass}" rgba=".85 .45 .1 1"
            contype="2" conaffinity="1"/>
      <geom name="lest" type="box" size="{hx * 0.8} {hy * 0.8} 0.02" pos="0 0 {-hz + 0.025}" mass="{p.ballast_mass}"
            contype="0" conaffinity="0" rgba=".2 .2 .2 1"/>
      <site name="capteurs" pos="{hx - 0.05} 0 {hz + 0.15}"/>{corners}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


class Rover:
    """Modèle + données MuJoCo et commande conducteur (accélérateur, direction) → consignes de roues."""

    def __init__(self, terrain, start, p: RoverParams = RoverParams(), ground: Ground | None = None):
        import mujoco

        self.mj = mujoco
        self.p = p
        self.terrain = terrain
        self.ground = ground
        tf = ground.table["terrain_friction"] if ground is not None and ground.sol is not None else None
        self.model = mujoco.MjModel.from_xml_string(build_xml(terrain, start["xyz"], start["yaw"], p, tf))
        H = terrain["H"]
        zmin, zmax = float(H.min()), float(H.max())
        self.model.hfield_data[:] = ((H - zmin) / max(zmax - zmin, 1e-3)).T.ravel()   # ligne MuJoCo = y
        self.data = mujoco.MjData(self.model)
        self.v_cmd = 0.0
        self.steer_cmd = 0.0
        self.ids = {
            "chassis": self.model.body("chassis").id,
            **{f"knuckle_{w}": self.model.body(f"knuckle_{w}").id for w in WHEELS},
            **{f"wheel_{w}": self.model.body(f"wheel_{w}").id for w in WHEELS},
        }
        self.susp_adr = [self.model.jnt_qposadr[self.model.joint(f"susp_{w}").id] for w in WHEELS]
        driven = ("rl", "rr") if p.drive == "propulsion" else WHEELS
        self.act = {a: self.model.actuator(a).id for a in [f"drive_{w}" for w in driven] + ["steer_fl", "steer_fr"]}
        self.wheel_brake = np.zeros(4)                 # couple de frein par roue (fl, fr, rl, rr), appliqué à chaque pas
        self.wheel_bodies = np.array([self.ids[f"wheel_{w}"] for w in WHEELS])
        self.wheel_geoms = np.array([self.model.geom(f"wheel_{w}").id for w in WHEELS])
        self.spin_dof = np.array([self.model.jnt_dofadr[self.model.joint(f"spin_{w}").id] for w in WHEELS])
        self.susp_adr = np.array(self.susp_adr)
        self.ground_info = {"adherence": p.friction, "roulement_Nm": 0.0, "frein_veg_N": 0.0}
        self.reset(start)

    def spawn_height(self, x, y, yaw):
        """Hauteur du centre du châssis pour poser le robot AU-DESSUS du point le plus haut de son empreinte : posé à
        la hauteur du seul point de départ, il apparaît enfoncé dans une bosse voisine et MuJoCo l'éjecte en le
        retournant (départ 6 de HÖB-1 complète)."""
        from terrain import height_at

        hx, hy = self.p.wheelbase / 2 + self.p.wheel_radius + 0.05, self.p.track / 2 + self.p.wheel_half_width + 0.05
        u, v = np.meshgrid(np.linspace(-hx, hx, 9), np.linspace(-hy, hy, 7))
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.stack([x + c * u.ravel() - s * v.ravel(), y + s * u.ravel() + c * v.ravel()], 1)
        return float(height_at(self.terrain, pts).max()) + self.p.wheel_radius - self.p.mount_z + 0.03

    def reset(self, start):
        mj = self.mj
        mj.mj_resetData(self.model, self.data)
        x, y, z = start["xyz"]
        yaw = start["yaw"]
        self.data.qpos[:3] = [x, y, self.spawn_height(x, y, yaw)]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.v_cmd = 0.0
        self.steer_cmd = 0.0
        self.wheel_brake = np.zeros(4)
        mj.mj_forward(self.model, self.data)

    def upright(self):
        """Cosinus de l'inclinaison du châssis (1 = droit, < 0 = sur le toit)."""
        q = self.data.qpos[3:7]
        return float(1 - 2 * (q[1] ** 2 + q[2] ** 2))

    def right(self):
        """Relève le robot sur place : même position horizontale et même cap, posé au-dessus de son empreinte,
        vitesses et suspensions remises à zéro (le temps simulé continue)."""
        q = self.data.qpos[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        x, y = self.data.qpos[:2]
        self.data.qpos[:3] = [x, y, self.spawn_height(x, y, yaw)]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qpos[7:] = 0
        self.data.qvel[:] = 0
        self.v_cmd = 0.0
        self.mj.mj_forward(self.model, self.data)

    def rear_slip_deg(self):
        """Angle de dérive à l'essieu arrière (°) : nul en roulement sans glissement, quel que soit le virage."""
        R = self.data.xmat[self.ids["chassis"]].reshape(3, 3)
        v = self.data.qvel[:3]
        vx, vy = float(v @ R[:, 0]), float(v @ R[:, 1])
        vy_r = vy - float(self.data.qvel[5]) * self.p.wheelbase / 2
        return float(np.degrees(np.arctan2(vy_r, max(abs(vx), 0.5))))

    def forward_speed(self):
        R = self.data.xmat[self.ids["chassis"]].reshape(3, 3)
        return float(self.data.qvel[:3] @ R[:, 0])

    def control_car(self, throttle, steer, handbrake, turbo, dt):
        """Voiture : accélérateur = couple des roues arrière (limité par la puissance et la vitesse max), recul sous
        0,3 m/s ; accélérateur négatif en marche avant = frein de service ; frein à main = blocage des roues arrière.
        Braquage Ackermann, réduit avec la vitesse (contre-braquage possible)."""
        p, d = self.p, self.data
        v = self.forward_speed()
        self.steer_cmd += np.clip(steer - self.steer_cmd, -p.steer_rate * dt, p.steer_rate * dt)
        delta = self.steer_cmd * p.steer_max / (1 + abs(v) / 6.0)
        if abs(delta) < 1e-3:
            self.data.ctrl[self.act["steer_fl"]] = self.data.ctrl[self.act["steer_fr"]] = 0.0
        else:
            R = p.wheelbase / np.tan(delta)
            for w, (x, y) in p.wheel_xy().items():
                if w[0] == "f":
                    self.data.ctrl[self.act[f"steer_{w}"]] = np.arctan((x + p.wheelbase / 2) / (R - y))
        vmax = p.v_max * (1.4 if turbo else 1.0)
        brake = np.zeros(4)
        tq = 0.0
        if throttle > 0 or (throttle < 0 and v < 0.3):
            tq = throttle * p.wheel_torque * (1.0 if throttle > 0 else 0.5)
        elif throttle < 0:
            brake[:] = -throttle * p.brake_torque
        # antipatinage : la roue ne tourne pas plus de (1 + κ) fois plus vite que le sol défile (+ 0,3 m/s au départ).
        # Sans lui, une roue qui patine (frottement de Coulomb) n'a plus d'adhérence latérale et la voiture part en
        # tête-à-queue en ligne droite. κ = 0,3, ou 0,6 en mode turbo (glisse plus facile).
        kappa = 0.6 if turbo else 0.3
        for k, w in enumerate(("rl", "rr")):
            om = float(d.qvel[self.spin_dof[2 + k]])
            lim = min(p.wheel_torque, p.power_w / 2 / max(abs(om), 1e-3))
            if tq * om > 0:                                # au-delà de la vitesse max : couple ramené à zéro
                lim *= float(np.clip(1 - (abs(om) * p.wheel_radius - vmax) / (0.15 * vmax), 0, 1))
                om_lim = (abs(v) * (1 + kappa) + 0.3) / p.wheel_radius
                lim *= float(np.clip((om_lim - abs(om)) / (0.3 * om_lim), 0, 1))
            d.ctrl[self.act[f"drive_{w}"]] = float(np.clip(tq, -lim, lim))
        if handbrake:
            brake[2:] = p.handbrake_torque
        self.wheel_brake = brake
        self.v_cmd = v

    def control(self, throttle, steer, brake, turbo, dt):
        """throttle, steer ∈ [-1, 1] (steer > 0 = gauche)."""
        if self.p.drive == "propulsion":
            return self.control_car(throttle, steer, brake, turbo, dt)
        p = self.p
        v_target = 0.0 if brake else throttle * p.v_max * (2.0 if turbo else 1.0)
        dv = np.clip(v_target - self.v_cmd, -p.accel * dt * (3 if brake else 1), p.accel * dt * (3 if brake else 1))
        self.v_cmd += dv
        self.steer_cmd += np.clip(steer - self.steer_cmd, -2.5 * dt, 2.5 * dt)
        # braquage limité par la vitesse (a_lat = v² tan δ / L ≤ a_lat_max), sinon le robot se couche en virage
        v2 = max(self.v_cmd ** 2, 1e-6)
        delta_max = min(p.steer_max, np.arctan(p.a_lat_max * p.wheelbase / v2))
        delta = self.steer_cmd * delta_max
        ctrl = {}
        if abs(delta) < 1e-3:
            for w in WHEELS:
                ctrl[f"drive_{w}"] = self.v_cmd / p.wheel_radius
            ctrl["steer_fl"] = ctrl["steer_fr"] = 0.0
        else:
            # centre instantané de rotation sur l'axe arrière, à R à gauche (R < 0 : à droite)
            R = p.wheelbase / np.tan(delta)
            for w, (x, y) in p.wheel_xy().items():
                xr = x + p.wheelbase / 2                         # abscisse depuis l'essieu arrière
                ctrl[f"drive_{w}"] = self.v_cmd * np.hypot(xr, R - y) / abs(R) / p.wheel_radius
                if w[0] == "f":
                    ctrl[f"steer_{w}"] = np.arctan(xr / (R - y))
        for name, val in ctrl.items():
            self.data.ctrl[self.act[name]] = val

    def ground_forces(self):
        """Sol étiqueté, avant chaque pas : frottement de chaque roue selon la classe sous elle ; couple de résistance
        au roulement crr·N·r (N : effort de la suspension) ; freinage horizontal du châssis dans les buissons."""
        g, d, p = self.ground, self.data, self.p
        tau = np.zeros(4)
        if g.sol is not None:
            i, j = g.cell(d.xpos[self.wheel_bodies, :2])
            mu, crr = g.mu[i, j], g.crr[i, j]
            self.model.geom_friction[self.wheel_geoms, 0] = mu
            N = np.maximum(0.0, p.corner_load() + p.susp_k * d.qpos[self.susp_adr])
            tau = crr * N * p.wheel_radius
            self.ground_info["adherence"] = float(mu.mean())
            self.ground_info["roulement_Nm"] = float(tau.mean())
        d.qfrc_applied[self.spin_dof] = -(tau + self.wheel_brake) * np.tanh(d.qvel[self.spin_dof] / 0.3)
        c = self.ids["chassis"]
        pts = np.vstack([d.xpos[c, :2], d.xpos[self.wheel_bodies, :2]])
        i, j = g.cell(pts)
        s = float(g.veg[i, j].mean())
        F = 0.0
        f = np.zeros(3)
        v3 = d.qvel[:3]                                        # vitesse linéaire du joint libre (repère monde)
        if s > 0:
            sp = float(np.hypot(v3[0], v3[1]))
            F = (g.F0 + g.F1 * sp) * s * np.tanh(sp / 0.05)
            f[:2] = -F * v3[:2] / max(sp, 1e-6)
        if p.aero_cda > 0:                                     # ½ ρ CdA |v| v, ρ = 1,2 kg/m³
            f += -0.6 * p.aero_cda * float(np.linalg.norm(v3)) * v3
        d.xfrc_applied[c, :3] = f
        self.ground_info["frein_veg_N"] = float(F)

    def step(self, n=1):
        for _ in range(n):
            if self.ground is not None:
                self.ground_forces()
            elif self.wheel_brake.any():
                self.data.qfrc_applied[self.spin_dof] = -self.wheel_brake * np.tanh(self.data.qvel[self.spin_dof] / 0.3)
            self.mj.mj_step(self.model, self.data)

    def state(self):
        """Poses (repère physique) du châssis, des moyeux et des roues, débattements, vitesse."""
        d = self.data
        out = {k: (d.xpos[i].tolist(), d.xquat[i].tolist()) for k, i in self.ids.items()}
        susp = [float(d.qpos[a]) for a in self.susp_adr]
        v = d.qvel[:3]
        R = d.xmat[self.ids["chassis"]].reshape(3, 3)
        return {"poses": out, "susp": susp, "speed": float(v @ R[:, 0]), "t": float(d.time)}
