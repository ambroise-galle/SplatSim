// Vue 3e personne du simulateur : splat GaussGym en streaming (moteur PlayCanvas) + robot à roues piloté par la
// physique MuJoCo du serveur (WebSocket). Repère d'affichage = y haut ; le serveur envoie déjà des poses converties.
import * as pc from 'playcanvas';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const canvas = $('app');

// ------------------------------------------------------------------ application
const deviceTypes = params.get('gfx') === 'webgl2' ? [pc.DEVICETYPE_WEBGL2] : [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2];
const device = await pc.createGraphicsDevice(canvas, { deviceTypes, antialias: false });
// écran Retina : rendu à la moitié de la densité par défaut (les splats y perdent peu, la cadence y gagne beaucoup)
device.maxPixelRatio = parseFloat(params.get('dpr') || Math.max(1, (window.devicePixelRatio || 1) * 0.5));
$('gfx').textContent = device.isWebGPU ? 'WebGPU' : 'WebGL2';

const opts = new pc.AppOptions();
opts.graphicsDevice = device;
opts.componentSystems = [pc.RenderComponentSystem, pc.CameraComponentSystem, pc.LightComponentSystem, pc.GSplatComponentSystem];
opts.resourceHandlers = [pc.TextureHandler, pc.ContainerHandler, pc.GSplatHandler];
const app = new pc.AppBase(canvas);
app.init(opts);
app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
app.setCanvasResolution(pc.RESOLUTION_AUTO);
window.addEventListener('resize', () => app.resizeCanvas());
app.start();

app.scene.ambientLight = new pc.Color(0.5, 0.5, 0.52);
const sun = new pc.Entity('soleil');
sun.addComponent('light', { type: 'directional', color: new pc.Color(1, 0.96, 0.88), intensity: 1.1 });
sun.setLocalEulerAngles(55, 35, 0);
app.root.addChild(sun);

const camera = new pc.Entity('camera');
camera.addComponent('camera', {
    clearColor: new pc.Color(0.66, 0.73, 0.8), fov: 65, nearClip: 0.2, farClip: 400, toneMapping: pc.TONEMAP_LINEAR
});
app.root.addChild(camera);

// ------------------------------------------------------------------ splat (streamed SOG + budget global)
let budget = parseFloat(params.get('budget') || '1.0');
// format compact : moins de mémoire GPU (Mac M2 8 Go : la mission complète faisait swapper le navigateur)
if (params.get('compact') !== '0') app.scene.gsplat.dataFormat = pc.GSPLATDATA_COMPACT;
const applyBudget = () => {
    app.scene.gsplat.splatBudget = Math.round(budget * 1e6);
    $('budget').textContent = `${budget.toFixed(1)} M`;
};
applyBudget();
const changeBudget = (d) => { budget = Math.min(8, Math.max(0.5, budget + d)); applyBudget(); };
for (const [id, d] of [['budget-moins', -0.5], ['budget-plus', +0.5]]) {
    $(id).addEventListener('mousedown', (e) => e.preventDefault());     // le focus reste sur la vue (touches de conduite)
    $(id).addEventListener('click', () => changeBudget(d));
}
const sceneUrl = params.get('scene') || '/scene/lod/lod-meta.json';
const splatAsset = new pc.Asset('scene', 'gsplat', { url: sceneUrl });
// Le moteur affiche le PLY tel quel. La rotation de l'entité splat vient du serveur (`splat_view_rotation_deg` de
// scene.json : [0,0,0] pour un export en repère monde, [180,0,0] pour l'ancien export en repère mission z bas).
// Forçable par ?splatrot=ax,ay,az (degrés). Contrôle : calque du terrain physique (touche T) posé sur le sol du splat.
let splatEntity = null;
let splatRot = params.get('splatrot') ? params.get('splatrot').split(',').map(Number) : null;
const applySplatRot = () => { if (splatEntity && splatRot) splatEntity.setLocalEulerAngles(splatRot[0], splatRot[1], splatRot[2]); };
splatAsset.on('load', () => {
    splatEntity = new pc.Entity('splat');
    splatEntity.addComponent('gsplat', { asset: splatAsset });
    applySplatRot();
    app.root.addChild(splatEntity);
});
splatAsset.on('error', (err) => showMsg(`Chargement du splat impossible : ${err}`));
app.assets.add(splatAsset);
app.assets.load(splatAsset);

// Splat recoloré par classe sémantique (touche E) : mêmes gaussiennes, couleur = classe ADE20K reportée depuis la
// segmentation de rendus (simrobot/tools/splat_labels.py lift + export, niveaux de détail lod_sem/ par build_lod.sh).
const semUrl = params.get('semscene') || '/scene/lod_sem/lod-meta.json';
let semAsset = null, semOn = false;
function setSem(on) {
    if (!splatEntity) return;
    if (on && !semAsset) {
        semAsset = new pc.Asset('scene-sem', 'gsplat', { url: semUrl });
        semAsset.on('error', () => {
            semOn = false; splatEntity.gsplat.asset = splatAsset; $('semleg').style.display = 'none';
            toast('pas de splat étiqueté dans cette scène (lod_sem/ : simrobot/tools/splat_labels.py)');
        });
        app.assets.add(semAsset);
        fetch(semUrl.replace(/[^/]*$/, 'legende.json')).then((r) => r.json()).then((L) => {
            $('semleg').innerHTML = '<small>classes ADE20K (part des gaussiennes)</small><br>' + L.map((c) =>
                `<i style="background:rgb(${c.rgb.join(',')})"></i>${c.nom} <small>${(100 * c.part).toFixed(1)} %</small>`).join('<br>');
        }).catch(() => {});
    }
    semOn = on;
    splatEntity.gsplat.asset = on ? semAsset : splatAsset;
    $('semleg').style.display = on ? 'block' : 'none';
    toast(on ? 'splat : classes sémantiques' : 'splat : couleurs');
}

// ------------------------------------------------------------------ robot
const material = (r, g, b, gloss = 0.35) => {
    const m = new pc.StandardMaterial();
    m.diffuse = new pc.Color(r, g, b);
    m.gloss = gloss;
    m.metalness = 0;
    m.useMetalness = true;
    m.update();
    return m;
};
const M = {
    body: material(0.86, 0.46, 0.1), dark: material(0.08, 0.08, 0.09, 0.2), rim: material(0.55, 0.57, 0.6, 0.6),
    spring: material(0.95, 0.8, 0.15, 0.5), arm: material(0.25, 0.26, 0.28), sensor: material(0.12, 0.13, 0.15, 0.7),
    lamp: material(1, 1, 0.85, 0.9)
};
const prim = (parent, type, mat, scale, pos = [0, 0, 0], euler = [0, 0, 0]) => {
    const e = new pc.Entity();
    e.addComponent('render', { type, material: mat });
    e.setLocalScale(...scale);
    e.setLocalPosition(...pos);
    e.setLocalEulerAngles(...euler);
    parent.addChild(e);
    return e;
};

let robot = null;   // construit à la réception de « hello » (dimensions envoyées par le serveur)
function buildRobot(R) {
    const [hx, hy, hz] = R.chassis_half;
    const r = R.wheel_radius, hw = R.wheel_half_width;
    // repère local du châssis = repère corps physique : x avant, y gauche, z haut
    const chassis = new pc.Entity('chassis');
    app.root.addChild(chassis);
    prim(chassis, 'box', M.body, [2 * hx, 2 * hy, 2 * hz]);
    prim(chassis, 'box', M.dark, [2 * hx * 0.6, 2 * hy * 0.8, 0.02], [-0.05, 0, hz + 0.01]);
    prim(chassis, 'box', M.lamp, [0.02, 0.08, 0.04], [hx + 0.005, hy * 0.6, 0.03]);
    prim(chassis, 'box', M.lamp, [0.02, 0.08, 0.04], [hx + 0.005, -hy * 0.6, 0.03]);
    // mât capteurs : caméra + LiDAR (repère du site « capteurs » de rover.py)
    prim(chassis, 'cylinder', M.arm, [0.03, 0.15, 0.03], [hx - 0.05, 0, hz + 0.075], [90, 0, 0]);
    prim(chassis, 'cylinder', M.sensor, [0.11, 0.07, 0.11], [hx - 0.05, 0, hz + 0.18], [90, 0, 0]);
    prim(chassis, 'box', M.sensor, [0.05, 0.12, 0.06], [hx - 0.02, 0, hz + 0.12]);

    const corners = {};
    for (const [w, [x, y]] of Object.entries(R.wheel_xy)) {
        const side = Math.sign(y);
        const spring = prim(chassis, 'cylinder', M.spring, [0.05, 0.1, 0.05], [x, side * (hy + 0.03), 0], [90, 0, 0]);
        const arm = prim(chassis, 'box', M.arm, [0.05, 0.1, 0.03], [x, side * (hy + 0.03), 0]);
        const wheel = new pc.Entity(`roue_${w}`);
        app.root.addChild(wheel);
        // repère local de la roue : axe de rotation = y
        prim(wheel, 'cylinder', M.dark, [2 * r, 2 * hw, 2 * r]);
        prim(wheel, 'cylinder', M.rim, [1.2 * r, 2 * hw + 0.004, 1.2 * r]);
        prim(wheel, 'box', M.body, [0.05, 2 * hw + 0.01, 0.9 * r]);     // repère visuel de rotation
        corners[w] = { x, y, side, spring, arm, wheel };
    }
    return { chassis, corners, R };
}

function applyState(s) {
    if (!robot) return;
    const set = (ent, [p, q]) => {
        ent.setPosition(p[0], p[1], p[2]);
        ent.setRotation(new pc.Quat(q[0], q[1], q[2], q[3]));
    };
    set(robot.chassis, s.poses.chassis);
    const { R } = robot;
    const hy = R.chassis_half[1];
    const names = ['fl', 'fr', 'rl', 'rr'];
    names.forEach((w, i) => {
        const c = robot.corners[w];
        set(c.wheel, s.poses[`wheel_${w}`]);
        // amortisseur : du haut du flanc du châssis jusqu'au niveau du moyeu (repère châssis)
        const zHub = R.mount_z + s.susp[i];
        const zTop = R.chassis_half[2] * 0.9;
        c.spring.setLocalPosition(c.x, c.side * (hy + 0.03), (zTop + zHub) / 2);
        c.spring.setLocalScale(0.05, Math.max(0.02, zTop - zHub), 0.05);
        // bras : du flanc au moyeu, à la hauteur du moyeu
        const yIn = hy, yOut = Math.abs(c.y) - R.wheel_half_width;
        c.arm.setLocalPosition(c.x, c.side * (yIn + yOut) / 2, zHub);
        c.arm.setLocalScale(0.05, Math.max(0.01, yOut - yIn), 0.03);
    });
}

// ------------------------------------------------------------------ interpolation des poses
// Les états arrivent à ~60 Hz mais par paquets quand le navigateur est chargé : on affiche l'état interpolé à
// (temps simulé estimé − 60 ms), sinon le robot avance par à-coups à l'écran.
const states = [];
let simOffset = null;                                 // temps simulé − horloge locale (s)
const RENDER_DELAY = 0.06;
function pushState(m) {
    const now = performance.now() / 1000;
    const off = m.t - now;
    simOffset = simOffset === null || Math.abs(off - simOffset) > 0.5 ? off : simOffset + 0.05 * (off - simOffset);
    if (states.length && m.t < states[states.length - 1].t) states.length = 0;   // replacement / remise à zéro
    states.push(m);
    if (states.length > 30) states.shift();
}
const qa = new pc.Quat(), qb = new pc.Quat(), qo = new pc.Quat();
function interpolatedState() {
    if (!states.length) return null;
    const tr = performance.now() / 1000 + simOffset - RENDER_DELAY;
    let i = states.length - 1;
    while (i > 0 && states[i - 1].t > tr) i--;
    if (i === 0 || states[i].t <= tr) return states[Math.min(i, states.length - 1)];
    const a = states[i - 1], b = states[i];
    const f = Math.min(1, Math.max(0, (tr - a.t) / Math.max(1e-6, b.t - a.t)));
    const poses = {};
    for (const k of Object.keys(b.poses)) {
        const [pa, xa] = a.poses[k], [pb, xb] = b.poses[k];
        qa.set(xa[0], xa[1], xa[2], xa[3]);
        qb.set(xb[0], xb[1], xb[2], xb[3]);
        qo.slerp(qa, qb, f);
        poses[k] = [[0, 1, 2].map((j) => pa[j] + f * (pb[j] - pa[j])), [qo.x, qo.y, qo.z, qo.w]];
    }
    return { poses, susp: a.susp.map((v, j) => v + f * (b.susp[j] - v)) };
}

// ------------------------------------------------------------------ caméra (poursuite / orbite libre / capteur)
const CAM_MODES = ['poursuite', 'orbite libre', 'capteur'];
let camMode = 0, camYaw = 0, camPitch = 0.12, camDist = 2.8, camAbsYaw = 0, camInit = false;
const camPos = new pc.Vec3();
let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.focus(); });
window.addEventListener('mouseup', () => { dragging = false; });
window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (camMode === 1) camAbsYaw -= dx * 0.006; else camYaw -= dx * 0.006;
    camPitch = Math.min(1.45, Math.max(-0.15, camPitch + dy * 0.005));
});
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    camDist = Math.min(25, Math.max(1.2, camDist * (e.deltaY > 0 ? 1.1 : 0.9)));
}, { passive: false });

const tmpV = new pc.Vec3(), tgt = new pc.Vec3(), desired = new pc.Vec3();
const SENSOR_OFFSET = new pc.Vec3(), tmpS = new pc.Vec3(), tgtS = new pc.Vec3();
function sensorPose(ent) {                       // pose du capteur avant (mât) : repère châssis x avant, z haut
    const ch = robot.chassis;
    const [hx, , hz] = robot.R.chassis_half;
    SENSOR_OFFSET.set(hx + 0.02, 0, hz + 0.12);
    ch.getWorldTransform().transformPoint(SENSOR_OFFSET, tmpS);
    ent.setPosition(tmpS);
    tgtS.copy(tmpS).add(ch.right);                // +x local du châssis = avant
    ent.lookAt(tgtS, ch.forward.clone().mulScalar(-1));    // +z local du châssis (haut) = −forward PlayCanvas
}
function updateCamera(dt) {
    if (!robot) return;
    const ch = robot.chassis;
    const fwd = ch.right;            // +x local du châssis = avant
    if (pip.enabled) sensorPose(pip);
    if (CAM_MODES[camMode] === 'capteur') {
        sensorPose(camera);
        return;
    }
    tgt.copy(ch.getPosition());
    tgt.y += 0.25;
    let yaw;
    if (camMode === 0) yaw = Math.atan2(fwd.x, fwd.z) + camYaw;
    else yaw = camAbsYaw;
    const c = Math.cos(camPitch);
    desired.set(tgt.x - Math.sin(yaw) * camDist * c, tgt.y + Math.sin(camPitch) * camDist, tgt.z - Math.cos(yaw) * camDist * c);
    if (!camInit) { camPos.copy(desired); camInit = true; }
    camPos.lerp(camPos, desired, 1 - Math.exp(-dt * 5));
    camera.setPosition(camPos);
    camera.lookAt(tgt);
}

// ------------------------------------------------------------------ terrain physique (contrôle d'alignement)
// Couche dessinée APRÈS le splat : sinon le sol opaque du splat recouvre le calque dès qu'il est bien aligné.
const debugLayer = new pc.Layer({ name: 'Calque' });
app.scene.layers.push(debugLayer);
camera.camera.layers = [...camera.camera.layers, debugLayer.id];

// ------------------------------------------------------------------ caméra embarquée en incrustation (touche V)
// Seconde caméra rendue dans un coin du canevas (rect normalisé, origine en bas à gauche), à la pose du mode
// « capteur » (mât avant du robot) ; sans le calque de débogage : c'est ce que verrait le capteur. Image 3:2, champ
// vertical 70° (~93° en horizontal ; la caméra avant de l'ANYmal fait 98° en vertical).
const PIP = { aspect: 1.5, margin: 12 };
const pip = new pc.Entity('camera_embarquee');
pip.addComponent('camera', {
    clearColor: new pc.Color(0.66, 0.73, 0.8), fov: 70, nearClip: 0.05, farClip: 400, toneMapping: pc.TONEMAP_LINEAR,
    priority: 1, layers: [pc.LAYERID_WORLD, pc.LAYERID_SKYBOX]
});
app.root.addChild(pip);
let pipOn = params.get('pip') !== '0';
function layoutPip() {
    const W = window.innerWidth, H = window.innerHeight;
    const wPx = Math.round(W < 700 ? 0.46 * W : Math.min(0.34 * W, 640));
    const hPx = Math.round(Math.min(wPx / PIP.aspect, 0.5 * H));
    const x = (W - wPx - PIP.margin) / W, y = PIP.margin / H, w = wPx / W, h = hPx / H;
    pip.camera.rect = new pc.Vec4(x, y, w, h);
    pip.camera.scissorRect = new pc.Vec4(x, y, w, h);
    Object.assign($('pipframe').style, { right: `${PIP.margin}px`, bottom: `${PIP.margin}px`, width: `${wPx}px`, height: `${hPx}px` });
}
function applyPip() {
    const on = pipOn && !clean;
    pip.enabled = on;
    $('pipframe').style.display = on ? 'block' : 'none';
}
layoutPip();
window.addEventListener('resize', layoutPip);
let terrainEntity = null;
function buildTerrain(T) {
    const [nx, ny] = T.shape;
    const pos = new Float32Array(nx * ny * 3), col = new Uint8Array(nx * ny * 4);
    for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
            const k = i * ny + j;
            const x = T.lo[0] + T.offset + i * T.res, y = T.lo[1] + T.offset + j * T.res;
            pos.set([x, T.H[k] + 0.01, -y], 3 * k);         // physique (x, y, z) → vue (x, z, −y)
            col.set(T.relief && T.relief[k] ? [150, 95, 45, 255] : T.veg && T.veg[k] ? [20, 120, 40, 255]
                : T.obstacle && T.obstacle[k] ? [235, 50, 50, 255]
                : T.couvert[k] ? [60, 230, 90, 255] : [250, 175, 155, 255], 4 * k);
        }
    }
    const idx = [];
    for (let i = 0; i < nx - 1; i++) {
        for (let j = 0; j < ny - 1; j++) {
            const a = i * ny + j, b = a + ny;
            idx.push(a, a + 1, b, b, a + 1, b + 1);
        }
    }
    const mesh = new pc.Mesh(device);
    mesh.setPositions(pos);
    mesh.setNormals(pc.calculateNormals(Array.from(pos), idx));
    mesh.setColors32(col);
    mesh.setIndices(idx);
    mesh.update();
    const m = new pc.StandardMaterial();
    m.diffuseVertexColor = true;
    m.diffuse = new pc.Color(1, 1, 1);
    m.opacity = 0.35;
    m.blendType = pc.BLEND_NORMAL;
    m.depthWrite = false;
    m.cull = pc.CULLFACE_NONE;
    m.update();
    terrainEntity = new pc.Entity('terrain_physique');
    terrainEntity.addComponent('render', { meshInstances: [new pc.MeshInstance(mesh, m)], layers: [debugLayer.id] });
    app.root.addChild(terrainEntity);
    // obstacles de collision (cylindres de 2 m) : physique (x, y, r, z_base) → vue (x, z, −y) ; troncs en orange,
    // obstacles bas rigides (souches, troncs couchés vus par le LiDAR ET le splat) en rouge. Avec les étiquettes du
    // splat (troncs_type) : buissons traversables qui freinent en vert, reliefs passés dans le sol (marron) non dessinés
    const obsMat = (r, g, b) => {
        const m = new pc.StandardMaterial();
        m.diffuse = new pc.Color(r, g, b);
        m.opacity = 0.45;
        m.blendType = pc.BLEND_NORMAL;
        m.depthWrite = false;
        m.update();
        return m;
    };
    const mt = [obsMat(1, 0.6, 0.1), obsMat(1, 0.6, 0.1), obsMat(0.95, 0.15, 0.15)];
    const mBuisson = obsMat(0.35, 0.85, 0.25);
    (T.troncs || []).forEach(([x, y, r, z], i) => {
        const src = (T.troncs_source || [])[i] || 0;
        const typ = (T.troncs_type || [])[i] || 0;
        if (typ >= 2) return;                        // relief : dans le sol ; 3 : remplacé par une capsule ; 4 : retiré
        const hc = src === 2 ? (typ === 1 ? 0.7 : 1.0) : 2.0;   // affichage seulement (la physique garde 2 m)
        prim(terrainEntity, 'cylinder', typ === 1 ? mBuisson : mt[src], [2 * r, hc, 2 * r], [x, z + hc / 2 - 0.1, -y])
            .render.layers = [debugLayer.id];
    });
    // capsules (gaussiennes étiquetées, splat_mesh.py) : troncs inclinés en orange, troncs couchés en jaune
    const mCouche = obsMat(1, 0.9, 0.2), up = new pc.Vec3(0, 1, 0);
    (T.capsules || []).forEach(([ax, ay, az, bx, by, bz, r], i) => {
        const d = new pc.Vec3(bx - ax, bz - az, -(by - ay));             // physique → vue (x, z, −y)
        const L = d.length();
        d.mulScalar(1 / L);
        const axis = new pc.Vec3().cross(up, d);
        const q = new pc.Quat();
        if (axis.length() > 1e-6) q.setFromAxisAngle(axis.normalize(), Math.acos(Math.max(-1, Math.min(1, up.dot(d)))) * 180 / Math.PI);
        const e = prim(terrainEntity, 'cylinder', (T.capsules_type || [])[i] === 1 ? mCouche : mt[0], [2 * r, L, 2 * r],
            [(ax + bx) / 2, (az + bz) / 2, -(ay + by) / 2]);
        e.setLocalRotation(q);
        e.render.layers = [debugLayer.id];
    });
}

// ------------------------------------------------------------------ trajectoires et poses du jeu de données (touche J)
// `trajets.json` de la scène (export_trajets.py), repère physique : trajet DLIO de l'ANYmal coloré du bleu (début) au
// rouge (fin), caméras GaussGym en pyramides (avant jaune, gauche verte, droite magenta), départs en jalons blancs.
const TRAJ_MODES = ['masqués', 'trajet ANYmal + départs', 'trajet + poses caméra'];
let trajMode = 0, trajData = null, trajLoading = false;
const traj = {};                                      // entités par contenu : trajet, cameras
const lineMat = new pc.StandardMaterial();
lineMat.useLighting = false;
lineMat.diffuse = new pc.Color(0, 0, 0);
lineMat.emissive = new pc.Color(1, 1, 1);
lineMat.emissiveVertexColor = true;
lineMat.depthWrite = false;
lineMat.update();
function lineEntity(name, segs, cols) {           // segs : [x0,y0,z0,x1,y1,z1,…] (vue), cols : [r,g,b,a,…] par sommet
    const mesh = new pc.Mesh(device);
    mesh.setPositions(new Float32Array(segs));
    mesh.setColors32(new Uint8Array(cols));
    mesh.update(pc.PRIMITIVE_LINES);
    const e = new pc.Entity(name);
    e.addComponent('render', { meshInstances: [new pc.MeshInstance(mesh, lineMat)], layers: [debugLayer.id] });
    app.root.addChild(e);
    return e;
}
const V = (p) => [p[0], p[2], -p[1]];                 // physique → vue
const hueRGB = (h) => {                               // h ∈ [0, 1] : bleu → cyan → vert → jaune → rouge
    const c = (x) => Math.round(255 * Math.max(0, Math.min(1, x)));
    return [c(1.5 - Math.abs(4 * h - 3)), c(1.5 - Math.abs(4 * h - 2)), c(1.5 - Math.abs(4 * h - 1)), 255];
};
function buildTrajets(T) {
    // trajet DLIO (ligne brisée) + jalons verticaux de 1 m aux départs
    const s = [], c = [];
    const P = T.dlio.p, n = P.length;
    for (let i = 0; i + 1 < n; i++) {
        s.push(...V(P[i]), ...V(P[i + 1]));
        c.push(...hueRGB(i / n), ...hueRGB((i + 1) / n));
    }
    for (const d of departs) {
        const [x, y, z] = d.xyz;
        s.push(...V([x, y, z]), ...V([x, y, z + 1]), ...V([x, y, z + 1]), ...V([x + 0.4 * Math.cos(d.yaw), y + 0.4 * Math.sin(d.yaw), z + 1]));
        c.push(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255);
    }
    traj.trajet = lineEntity('trajet_anymal', s, c);
    // caméras : une pyramide de 0,3 m dès que la caméra a bougé de 1 m ou tourné de 25° (sinon un tube illisible)
    const COL = { hdr_front: [255, 220, 40, 255], hdr_left: [70, 230, 90, 255], hdr_right: [240, 80, 220, 255] };
    const sc = [], cc = [];
    const D = 0.3;
    for (const [name, cam] of Object.entries(T.cameras)) {
        const col = COL[name] || [255, 255, 255, 255];
        const ty = Math.tan((cam.fov_y_deg * Math.PI) / 360), tx = ty * cam.aspect;
        let last = null;
        for (let i = 0; i < cam.p.length; i++) {
            const p = cam.p[i], f = cam.f[i], u = cam.u[i];
            if (last && Math.hypot(p[0] - last.p[0], p[1] - last.p[1], p[2] - last.p[2]) < 1.0 &&
                f[0] * last.f[0] + f[1] * last.f[1] + f[2] * last.f[2] > Math.cos((25 * Math.PI) / 180)) continue;
            last = { p, f };
            const r = [f[1] * u[2] - f[2] * u[1], f[2] * u[0] - f[0] * u[2], f[0] * u[1] - f[1] * u[0]];   // f × u
            const corner = (a, b) => [0, 1, 2].map((k) => p[k] + D * (f[k] + a * tx * r[k] + b * ty * u[k]));
            const q = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
            for (let k = 0; k < 4; k++) {
                sc.push(...V(p), ...V(q[k]), ...V(q[k]), ...V(q[(k + 1) % 4]));
                cc.push(...col, ...col, ...col, ...col);
            }
            sc.push(...V(q[2]), ...V([0, 1, 2].map((k) => p[k] + D * (f[k] + 1.4 * ty * u[k]))));    // repère « haut »
            cc.push(...col, ...col);
        }
    }
    traj.cameras = lineEntity('poses_cameras', sc, cc);
}
async function setTrajMode(m) {
    trajMode = m;
    if (trajMode > 0 && !trajData && !trajLoading) {
        trajLoading = true;
        try {
            const r = await fetch('/scene/trajets.json');
            if (!r.ok) throw new Error(`${r.status}`);
            trajData = await r.json();
            buildTrajets(trajData);
        } catch (err) {
            toast(`trajets.json absent de la scène (simrobot/sim/export_trajets.py) : ${err.message}`);
            trajMode = 0;
        }
        trajLoading = false;
    }
    if (traj.trajet) traj.trajet.enabled = trajMode >= 1;
    if (traj.cameras) traj.cameras.enabled = trajMode >= 2;
    toast(`jeu de données : ${TRAJ_MODES[trajMode]}`);
}

// ------------------------------------------------------------------ LiDAR simulé (touche K)
// Le serveur lance les rayons (lidar_sim.py) et envoie chaque tour en binaire (cf. Sim.lidar_msg). Affichage : un petit
// carré face à la caméra par point (les points WebGPU font 1 pixel), coloré par hauteur ou par intensité, dans le
// calque de débogage ; image de distance (lignes × colonnes) au-dessus de la caméra embarquée.
const LIDAR_MODES = ['masqué', 'points colorés par hauteur', 'points colorés par intensité'];
let lidarMode = 0, lidarInfo = null, lidarCap = 0, lidarMesh = null, lidarPos = null, lidarCol = null, lidarLast = null;
const ptMat = new pc.StandardMaterial();
ptMat.useLighting = false;
ptMat.diffuse = new pc.Color(0, 0, 0);
ptMat.emissive = new pc.Color(1, 1, 1);
ptMat.emissiveVertexColor = true;
ptMat.cull = pc.CULLFACE_NONE;
ptMat.update();
function ensureLidarMesh(n) {
    if (lidarCap >= n) return;
    lidarCap = n;
    const idx = new Uint32Array(6 * n);
    for (let k = 0; k < n; k++) idx.set([4 * k, 4 * k + 1, 4 * k + 2, 4 * k, 4 * k + 2, 4 * k + 3], 6 * k);
    lidarPos = new Float32Array(12 * n);
    lidarCol = new Uint8Array(16 * n);
    lidarMesh = new pc.Mesh(device);
    lidarMesh.setPositions(lidarPos);
    lidarMesh.setColors32(lidarCol);
    lidarMesh.setIndices(idx);
    lidarMesh.update();
    const mi = new pc.MeshInstance(lidarMesh, ptMat);
    mi.cull = false;
    if (traj.lidar) traj.lidar.destroy();
    traj.lidar = new pc.Entity('lidar_points');
    traj.lidar.addComponent('render', { meshInstances: [mi], layers: [debugLayer.id] });
    app.root.addChild(traj.lidar);
}
const camR = new pc.Vec3(), camU = new pc.Vec3();
function drawLidarPoints(L) {
    const { n, xyz, inten, o } = L;
    ensureLidarMesh(n);
    camR.copy(camera.right);
    camU.copy(camera.up);
    const cp = camera.getPosition();
    let m = 0;
    for (let k = 0; k < n; k++) {
        const x = xyz[3 * k];
        if (x === -32768) continue;
        // physique (cm relatifs à l'origine) → vue (x, z, −y)
        const px = o[0] + x / 100, py = o[1] + xyz[3 * k + 1] / 100, pz = o[2] + xyz[3 * k + 2] / 100;
        const vx = px, vy = pz, vz = -py;
        const s = Math.min(0.12, Math.max(0.008, 0.0028 * Math.hypot(vx - cp.x, vy - cp.y, vz - cp.z)));
        const rx = camR.x * s, ry = camR.y * s, rz = camR.z * s, ux = camU.x * s, uy = camU.y * s, uz = camU.z * s;
        const b = 12 * m;
        lidarPos[b] = vx - rx - ux; lidarPos[b + 1] = vy - ry - uy; lidarPos[b + 2] = vz - rz - uz;
        lidarPos[b + 3] = vx + rx - ux; lidarPos[b + 4] = vy + ry - uy; lidarPos[b + 5] = vz + rz - uz;
        lidarPos[b + 6] = vx + rx + ux; lidarPos[b + 7] = vy + ry + uy; lidarPos[b + 8] = vz + rz + uz;
        lidarPos[b + 9] = vx - rx + ux; lidarPos[b + 10] = vy - ry + uy; lidarPos[b + 11] = vz - rz + uz;
        const c = lidarMode === 2 ? hueRGB(Math.min(1, inten[k] / 200)) : hueRGB(Math.min(1, Math.max(0, (pz - o[2] + 1.2) / 6)));
        for (let q = 0; q < 4; q++) lidarCol.set(c, 16 * m + 4 * q);
        m++;
    }
    lidarMesh.setPositions(lidarPos);
    lidarMesh.setColors32(lidarCol);
    lidarMesh.update(pc.PRIMITIVE_TRIANGLES, false);
    lidarMesh.primitive[0].count = 6 * m;
    traj.lidar.enabled = true;
}
function drawRangeImage(L) {
    const cv = $('lidarimg');
    if (cv.width !== L.cols || cv.height !== L.rings) { cv.width = L.cols; cv.height = L.rings; }
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(L.cols, L.rings);
    const maxR = 30, half = L.cols >> 1;
    for (let ring = 0; ring < L.rings; ring++) {
        for (let c = 0; c < L.cols; c++) {
            // colonne 0 = avant, au centre de l'image ; droite du robot à droite (azimut horaire)
            const r = L.rng[ring * L.cols + c];
            const col = r === 0 ? [0, 0, 0, 255] : hueRGB(Math.max(0, 1 - r / 100 / maxR));
            img.data.set(col, 4 * (ring * L.cols + ((c + half) % L.cols)));
        }
    }
    ctx.putImageData(img, 0, 0);
}
function onLidar(buf) {
    if (!lidarMode) return;
    const dv = new DataView(buf);
    const rings = dv.getUint16(4, true), cols = dv.getUint16(6, true), n = rings * cols;
    const f = new Float32Array(buf.slice(8, 28));
    let off = 28;
    const xyz = new Int16Array(buf.slice(off, off + 6 * n)); off += 6 * n;
    const rng = new Uint16Array(buf.slice(off, off + 2 * n)); off += 2 * n;
    const inten = new Uint8Array(buf, off, n);
    lidarLast = { rings, cols, n, xyz, rng, inten, o: [f[1], f[2], f[3]], t: f[0], ms: f[4] };
    drawLidarPoints(lidarLast);
    drawRangeImage(lidarLast);
    $('lidarms').textContent = `${f[4].toFixed(1)} ms/tour · ${rings}×${cols}`;
}
function layoutLidarImg() {
    const fr = $('pipframe').getBoundingClientRect();
    const w = fr.width || Math.min(0.34 * window.innerWidth, 640);
    const rings = lidarInfo ? lidarInfo.rings : 32, cols = lidarInfo ? lidarInfo.cols : 1024;
    const h = Math.max(56, Math.round((w * rings) / cols * 4));            // lignes étirées ×4 : lisible
    const pipShown = $('pipframe').style.display !== 'none';
    Object.assign($('lidarbox').style, {
        right: `${PIP.margin}px`, width: `${w}px`, height: `${h}px`,
        bottom: `${PIP.margin + (pipShown ? fr.height + 6 : 0)}px`
    });
}
function applyLidar() {
    const on = lidarMode > 0 && !clean;
    $('lidarbox').style.display = on ? 'block' : 'none';
    $('lidarrow').style.display = lidarMode > 0 ? '' : 'none';
    if (traj.lidar) traj.lidar.enabled = lidarMode > 0;
    if (on) layoutLidarImg();
}
function setLidarMode(mode) {
    if (!lidarInfo) { toast('pas de LiDAR pour cette scène (lidar_voxels.npz : simrobot/sim/lidar_voxels.py)'); return; }
    lidarMode = mode;
    send({ type: 'lidar', on: lidarMode > 0 });
    if (lidarMode > 0 && lidarLast) drawLidarPoints(lidarLast);
    applyLidar();
    toast(`LiDAR : ${LIDAR_MODES[lidarMode]}`);
}
window.addEventListener('resize', () => { if (lidarMode) layoutLidarImg(); });

// ------------------------------------------------------------------ vue nette (touche O) : ni HUD, ni aide, ni calques
let clean = false;
function setClean(v) {
    clean = v;
    document.querySelectorAll('.panel').forEach((el) => el.classList.toggle('cache', v));
    debugLayer.enabled = !v;
    applyPip();
    applyLidar();
}

// ------------------------------------------------------------------ réseau
let ws = null, departs = [], startIdx = 0;
function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
        $('dot').classList.add('on');
        $('conn').textContent = 'connecté';
        if (lidarMode) send({ type: 'lidar', on: true });          // reconnexion : le serveur a oublié la demande
    };
    ws.onclose = () => {
        $('dot').classList.remove('on');
        $('conn').textContent = 'déconnecté, nouvel essai…';
        setTimeout(connect, 1000);
    };
    ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') { onLidar(ev.data); return; }
        const m = JSON.parse(ev.data);
        if (m.type === 'hello') {
            departs = m.departs;
            startIdx = m.start;
            lidarInfo = m.lidar || null;
            if (!params.get('splatrot')) { splatRot = m.splat_view_rotation_deg || [0, 0, 0]; applySplatRot(); }
            if (!robot) {
                robot = buildRobot(m.rover);
                buildSuspBars();
            }
        } else if (m.type === 'state') {
            pushState(m);
            updateHud(m);
        } else if (m.type === 'terrain') {
            if (!terrainEntity) buildTerrain(m);
        }
    };
}
connect();
const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

// ------------------------------------------------------------------ commandes (clavier + manette)
// Touche → action par le code physique OU par le caractère. Le code suit la position QWERTY (le Z d'un clavier AZERTY
// donne « KeyW »), sauf quand la saisie est relayée par logiciel : certains navigateurs intégrés envoient
// « KeyZ » pour Z. Avec le code seul, Z (avancer) et Q (gauche) ne faisaient rien. ZQSD, WASD et flèches marchent.
const DRIVE_KEYS = {
    avant: [['KeyW', 'ArrowUp'], ['z', 'w', 'arrowup']],
    arriere: [['KeyS', 'ArrowDown'], ['s', 'arrowdown']],
    gauche: [['KeyA', 'ArrowLeft'], ['q', 'a', 'arrowleft']],
    droite: [['KeyD', 'ArrowRight'], ['d', 'arrowright']],
    frein: [['Space'], [' ']],
    turbo: [['ShiftLeft', 'ShiftRight'], ['shift']]
};
const lower = (e) => (e.key || '').toLowerCase();
function driveAction(e) {
    for (const [a, [codes]] of Object.entries(DRIVE_KEYS)) if (codes.includes(e.code)) return a;
    for (const [a, [, chars]] of Object.entries(DRIVE_KEYS)) if (chars.includes(lower(e))) return a;
    return null;
}
const is = (e, code, ...chars) => e.code === code || chars.includes(lower(e));
const pressed = new Map();                            // touche enfoncée (code, sinon caractère) → action de conduite
const keyId = (e) => e.code || lower(e);
window.addEventListener('keydown', (e) => {
    const a = driveAction(e);
    if (a) {
        e.preventDefault();
        // une répétition d'une touche inconnue = touche gardée enfoncée pendant une perte de focus : on la reprend
        if (!pressed.has(keyId(e))) { pressed.set(keyId(e), a); sendCmd(); }
        return;
    }
    if (e.repeat || e.metaKey || e.ctrlKey) return;
    if (is(e, 'KeyR', 'r')) { send({ type: 'reset' }); camInit = false; }
    else if (is(e, 'KeyN', 'n')) { startIdx = (startIdx + 1) % Math.max(1, departs.length); send({ type: 'reset', start: startIdx }); camInit = false; }
    else if (is(e, 'KeyC', 'c')) { camMode = (camMode + 1) % CAM_MODES.length; $('cammode').textContent = CAM_MODES[camMode]; camInit = false; }
    else if (is(e, 'KeyT', 't')) {
        if (!terrainEntity) send({ type: 'terrain' });
        else terrainEntity.enabled = !terrainEntity.enabled;
    }
    else if (is(e, 'KeyL', 'l')) app.scene.gsplat.colorizeLod = !app.scene.gsplat.colorizeLod;
    else if (is(e, 'KeyE', 'e')) setSem(!semOn);
    // budget : − et = sont directs sur un clavier Mac AZERTY comme QWERTY ([ ] y demandent Option + Maj)
    else if (is(e, 'BracketRight', ']', '+', '=')) changeBudget(+0.5);
    else if (is(e, 'BracketLeft', '[', '-', '_')) changeBudget(-0.5);
    else if (is(e, 'KeyP', 'p')) send({ type: 'pause' });
    else if (is(e, 'KeyF', 'f')) send({ type: 'relever' });
    else if (is(e, 'KeyH', 'h')) $('help').style.display = $('help').style.display === 'none' ? '' : 'none';
    else if (is(e, 'KeyJ', 'j')) setTrajMode((trajMode + 1) % TRAJ_MODES.length);
    // vue nette : lettre O, ou touche 0 (« à » sans Maj sur un clavier AZERTY)
    else if (is(e, 'KeyO', 'o', '0', 'à') || e.code === 'Digit0') setClean(!clean);
    else if (is(e, 'KeyV', 'v')) { pipOn = !pipOn; applyPip(); applyLidar(); }
    else if (is(e, 'KeyK', 'k')) setLidarMode((lidarMode + 1) % LIDAR_MODES.length);
});
window.addEventListener('keyup', (e) => { if (pressed.delete(keyId(e))) sendCmd(); });
const releaseAll = () => { pressed.clear(); sendCmd(); };
window.addEventListener('blur', releaseAll);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });

const k = (a) => ([...pressed.values()].includes(a) ? 1 : 0);
// commande envoyée à chaque changement de touche + rappel à 10 Hz (manette) ; le serveur garde la dernière reçue
function sendCmd() {
    let throttle = k('avant') - k('arriere');
    let steer = k('gauche') - k('droite');
    let brake = !!k('frein'), turbo = !!k('turbo');
    const gp = navigator.getGamepads ? [...navigator.getGamepads()].find((g) => g) : null;
    if (gp) {
        const dead = (v) => (Math.abs(v) < 0.08 ? 0 : v);
        const t = (gp.buttons[7]?.value || 0) - (gp.buttons[6]?.value || 0);
        if (Math.abs(t) > 0.02) throttle = t;
        if (dead(gp.axes[0])) steer = -dead(gp.axes[0]);
        brake = brake || !!gp.buttons[0]?.pressed;
        turbo = turbo || !!gp.buttons[5]?.pressed;
    }
    send({ type: 'cmd', throttle, steer, brake, turbo });
}
setInterval(sendCmd, 100);

// ------------------------------------------------------------------ affichage tête haute
function buildSuspBars() {
    const box = $('susp');
    box.innerHTML = '';
    for (const w of ['AvG', 'AvD', 'ArG', 'ArD']) {
        box.insertAdjacentHTML('beforeend',
            `<div><div class="bar"><div class="fill"></div><div class="mid"></div></div><div class="lab">${w}</div></div>`);
    }
}
function updateHud(s) {
    $('speed').textContent = (Math.abs(s.speed) * 3.6).toFixed(1);
    $('tsim').textContent = `${s.t.toFixed(1)} s${s.paused ? ' (pause)' : ''}`;
    $('step').textContent = `${s.step_us.toFixed(0)} µs`;
    $('cmdage').textContent = `${s.cmd_age_ms} ms`;
    // commande appliquée par le serveur (contrôle immédiat que les touches arrivent) ; autre onglet = autre pilote possible
    if (s.cmd) {
        const [t, st] = s.cmd;
        const dir = t > 0 ? '↑ avance' : t < 0 ? '↓ recule' : 'arrêt';
        const tourne = st > 0 ? ' · ← gauche' : st < 0 ? ' · → droite' : '';
        $('cmdrecue').textContent = `${dir}${tourne}${s.clients > 1 ? ` (${s.clients} onglets)` : ''}`;
    }
    // sol sous le robot (étiquettes du splat) : classe, adhérence moyenne des roues, freinage de la végétation
    $('solrow').style.display = s.sol ? '' : 'none';
    if (s.sol) {
        $('sol').textContent = `${s.sol.classe ?? 'uniforme'} · μ ${s.sol.adherence.toFixed(2)}`
            + (s.sol.frein_veg_N > 0 ? ` · buisson ${s.sol.frein_veg_N} N` : '');
    }
    if (!robot) return;
    const travel = robot.R.susp_travel;
    document.querySelectorAll('#susp .fill').forEach((el, i) => {
        const f = Math.max(-1, Math.min(1, s.susp[i] / travel));   // > 0 = compression (barre vers le haut)
        el.style.top = f > 0 ? `${50 - 50 * f}%` : '50%';
        el.style.bottom = f > 0 ? '50%' : `${50 + 50 * f}%`;
    });
}
if (window.innerWidth <= 900 || window.innerHeight <= 600) $('help').style.display = 'none';   // petite fenêtre
function showMsg(t) { $('msg').style.display = 'block'; $('msg').textContent = t; }
let toastTimer = null;
function toast(t) {                                    // message bref en bas de l'écran (masqué en vue nette)
    $('toast').textContent = t;
    $('toast').style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').style.display = 'none'; }, 1800);
}

let acc = 0, frames = 0;
app.on('update', (dt) => {
    const st = interpolatedState();
    if (st) applyState(st);
    updateCamera(dt);
    acc += dt; frames++;
    if (acc > 0.5) { $('fps').textContent = `${(frames / acc).toFixed(0)} i/s`; acc = 0; frames = 0; }
});
applyPip();
canvas.focus();
// accès de diagnostic depuis la console du navigateur
window.__sim = { app, pc, traj, pip, get terrain() { return terrainEntity; }, get robot() { return robot; }, get splat() { return splatEntity; } };
