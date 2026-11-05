// Démo de course (server.py --jeu) : petite voiture MuJoCo qui dérape, dans le splat GaussGym en streaming. Circuit = vraie
// trajectoire de l'ANYmal (course.py) : chevrons dans les virages, portes tous les 10 m, chrono, record, carte.
// Une seule caméra : poursuite (3e personne) ou capot (1re personne), touche C.
import * as pc from 'playcanvas';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const canvas = $('app');

// ------------------------------------------------------------------ application
const deviceTypes = params.get('gfx') === 'webgl2' ? [pc.DEVICETYPE_WEBGL2] : [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2];
const device = await pc.createGraphicsDevice(canvas, { deviceTypes, antialias: false });
device.maxPixelRatio = parseFloat(params.get('dpr') || Math.max(1, (window.devicePixelRatio || 1) * 0.5));
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
app.scene.ambientLight = new pc.Color(0.55, 0.55, 0.57);
const sun = new pc.Entity('soleil');
sun.addComponent('light', { type: 'directional', color: new pc.Color(1, 0.96, 0.88), intensity: 1.1 });
sun.setLocalEulerAngles(55, 35, 0);
app.root.addChild(sun);
const camera = new pc.Entity('camera');
camera.addComponent('camera', {
    clearColor: new pc.Color(0.66, 0.73, 0.8), fov: 68, nearClip: 0.05, farClip: 400, toneMapping: pc.TONEMAP_LINEAR
});
app.root.addChild(camera);

// ------------------------------------------------------------------ splat (streamed SOG)
app.scene.gsplat.dataFormat = pc.GSPLATDATA_COMPACT;
app.scene.gsplat.splatBudget = Math.round(parseFloat(params.get('budget') || '3') * 1e6);
const splatAsset = new pc.Asset('scene', 'gsplat', { url: params.get('scene') || '/scene/lod/lod-meta.json' });
let splatEntity = null, splatRot = null;
splatAsset.on('load', () => {
    splatEntity = new pc.Entity('splat');
    splatEntity.addComponent('gsplat', { asset: splatAsset });
    if (splatRot) splatEntity.setLocalEulerAngles(...splatRot);
    app.root.addChild(splatEntity);
});
app.assets.add(splatAsset);
app.assets.load(splatAsset);

// ------------------------------------------------------------------ matériaux, primitives
const material = (r, g, b, { gloss = 0.4, emissive = null, opacity = 1 } = {}) => {
    const m = new pc.StandardMaterial();
    m.diffuse = new pc.Color(r, g, b);
    m.gloss = gloss;
    m.metalness = 0;
    m.useMetalness = true;
    if (emissive) m.emissive = new pc.Color(...emissive);
    if (opacity < 1) { m.opacity = opacity; m.blendType = pc.BLEND_NORMAL; m.depthWrite = false; }
    m.update();
    return m;
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
const M = {
    rouge: material(0.85, 0.08, 0.06, { gloss: 0.75 }), blanc: material(0.92, 0.92, 0.9, { gloss: 0.6 }),
    noir: material(0.05, 0.05, 0.06, { gloss: 0.3 }), vitre: material(0.1, 0.14, 0.18, { gloss: 0.9 }),
    jante: material(0.75, 0.76, 0.78, { gloss: 0.8 }), phare: material(1, 1, 0.9, { emissive: [1, 1, 0.85] }),
    feu: material(0.8, 0.05, 0.05, { emissive: [0.9, 0.05, 0.05] }),
    jaune: material(1, 0.82, 0.1, { emissive: [0.35, 0.28, 0.02] }), poteau: material(0.3, 0.3, 0.32),
    damierB: material(0.95, 0.95, 0.95, { emissive: [0.3, 0.3, 0.3] }), damierN: material(0.04, 0.04, 0.04),
    porte: material(1, 0.5, 0.1, { emissive: [0.5, 0.2, 0.02] }), porteProchaine: material(0.2, 0.95, 1, { emissive: [0.1, 0.6, 0.8] }),
    porteFaite: material(0.4, 0.4, 0.4, { opacity: 0.5 })
};

// ------------------------------------------------------------------ voiture (repère physique du châssis : x avant, y gauche, z haut)
let car = null;
function buildCar(R) {
    const [hx, hy, hz] = R.chassis_half;
    const r = R.wheel_radius, hw = R.wheel_half_width;
    const chassis = new pc.Entity('voiture');
    app.root.addChild(chassis);
    prim(chassis, 'box', M.rouge, [2 * hx, 2 * hy, 2 * hz]);                                  // caisse
    prim(chassis, 'box', M.rouge, [0.22, 2 * hy * 0.92, 0.05], [hx - 0.06, 0, hz + 0.01], [0, 12, 0]);   // capot
    prim(chassis, 'box', M.vitre, [0.3, 2 * hy * 0.78, 0.11], [-0.04, 0, hz + 0.06]);            // habitacle
    prim(chassis, 'box', M.blanc, [0.3, 2 * hy * 0.8, 0.012], [-0.04, 0, hz + 0.12]);            // toit
    prim(chassis, 'box', M.blanc, [2 * hx * 0.98, 0.012, 0.04], [0, hy + 0.001, 0.0]);          // bandes latérales
    prim(chassis, 'box', M.blanc, [2 * hx * 0.98, 0.012, 0.04], [0, -hy - 0.001, 0.0]);
    prim(chassis, 'box', M.noir, [0.06, 2 * hy * 1.1, 0.012], [-hx + 0.02, 0, hz + 0.14]);      // aileron
    prim(chassis, 'box', M.noir, [0.02, 0.012, 0.1], [-hx + 0.03, hy * 0.6, hz + 0.08]);
    prim(chassis, 'box', M.noir, [0.02, 0.012, 0.1], [-hx + 0.03, -hy * 0.6, hz + 0.08]);
    prim(chassis, 'box', M.phare, [0.01, 0.05, 0.03], [hx + 0.004, hy * 0.65, 0.015]);
    prim(chassis, 'box', M.phare, [0.01, 0.05, 0.03], [hx + 0.004, -hy * 0.65, 0.015]);
    prim(chassis, 'box', M.feu, [0.01, 0.06, 0.025], [-hx - 0.004, hy * 0.65, 0.02]);
    prim(chassis, 'box', M.feu, [0.01, 0.06, 0.025], [-hx - 0.004, -hy * 0.65, 0.02]);
    prim(chassis, 'box', M.noir, [2 * hx * 0.9, 2 * hy * 0.9, 0.02], [0, 0, -hz - 0.005]);     // fond plat
    const wheels = {};
    for (const [w, [x, y]] of Object.entries(R.wheel_xy)) {
        const wheel = new pc.Entity(`roue_${w}`);
        app.root.addChild(wheel);
        prim(wheel, 'cylinder', M.noir, [2 * r, 2 * hw, 2 * r]);
        prim(wheel, 'cylinder', M.jante, [1.15 * r, 2 * hw + 0.004, 1.15 * r]);
        prim(wheel, 'box', M.noir, [0.03, 2 * hw + 0.008, 0.9 * r]);
        const bras = prim(chassis, 'box', M.noir, [0.03, 0.1, 0.02], [x, Math.sign(y) * (hy + 0.04), 0]);
        wheels[w] = { wheel, bras, x, y };
    }
    return { chassis, wheels, R };
}
function applyState(s) {
    if (!car) return;
    const set = (ent, [p, q]) => { ent.setPosition(p[0], p[1], p[2]); ent.setRotation(new pc.Quat(q[0], q[1], q[2], q[3])); };
    set(car.chassis, s.poses.chassis);
    const { R } = car;
    ['fl', 'fr', 'rl', 'rr'].forEach((w, i) => {
        const c = car.wheels[w];
        set(c.wheel, s.poses[`wheel_${w}`]);
        const zHub = R.mount_z + s.susp[i], yIn = R.chassis_half[1], yOut = Math.abs(c.y) - R.wheel_half_width;
        c.bras.setLocalPosition(c.x, Math.sign(c.y) * (yIn + yOut) / 2, zHub);
        c.bras.setLocalScale(0.03, Math.max(0.01, yOut - yIn), 0.02);
    });
}

// ------------------------------------------------------------------ interpolation des états (60 Hz, affichés à −60 ms)
const states = [];
let simOffset = null;
function pushState(m) {
    const now = performance.now() / 1000, off = m.t - now;
    simOffset = simOffset === null || Math.abs(off - simOffset) > 0.5 ? off : simOffset + 0.05 * (off - simOffset);
    if (states.length && m.t < states[states.length - 1].t) states.length = 0;
    states.push(m);
    if (states.length > 30) states.shift();
}
const qa = new pc.Quat(), qb = new pc.Quat(), qo = new pc.Quat();
function interpolatedState() {
    if (!states.length) return null;
    const tr = performance.now() / 1000 + simOffset - 0.06;
    let i = states.length - 1;
    while (i > 0 && states[i - 1].t > tr) i--;
    if (i === 0 || states[i].t <= tr) return states[Math.min(i, states.length - 1)];
    const a = states[i - 1], b = states[i];
    const f = Math.min(1, Math.max(0, (tr - a.t) / Math.max(1e-6, b.t - a.t)));
    const poses = {};
    for (const k of Object.keys(b.poses)) {
        const [pa, xa] = a.poses[k], [pb, xb] = b.poses[k];
        qa.set(...xa); qb.set(...xb); qo.slerp(qa, qb, f);
        poses[k] = [[0, 1, 2].map((j) => pa[j] + f * (pb[j] - pa[j])), [qo.x, qo.y, qo.z, qo.w]];
    }
    return { poses, susp: a.susp.map((v, j) => v + f * (b.susp[j] - v)) };
}

// ------------------------------------------------------------------ circuit : chevrons, portes, arche départ / arrivée
// physique (x, y, z) → vue (x, z, −y) ; cap physique θ → direction vue (cos θ, 0, −sin θ)
const V = (x, y, z) => new pc.Vec3(x, z, -y);
let track = null;
const gates = [];
function yawToEulerY(theta) { return (theta * 180) / Math.PI - 90; }      // local +Z de l'entité vers −tangente
function chevron(parent, gauche) {
    // flèche « < » (virage à gauche) ou « > » vue par le pilote qui arrive (il regarde l'entité selon −Z local)
    const s = gauche ? -1 : 1;
    for (const dx of [-0.09, 0.09]) {
        prim(parent, 'box', M.noir, [0.15, 0.035, 0.012], [dx + s * 0.0, 0.045, 0.012], [0, 0, -s * 38]);
        prim(parent, 'box', M.noir, [0.15, 0.035, 0.012], [dx + s * 0.0, -0.045, 0.012], [0, 0, s * 38]);
    }
}
function buildTrack(T) {
    track = T;
    track.P = T.pts.map(([x, y, z]) => [x, y, z]);
    const root = new pc.Entity('circuit');
    app.root.addChild(root);
    for (const c of T.chevrons) {
        const e = new pc.Entity();
        root.addChild(e);
        e.setPosition(V(c.x, c.y, c.z + 0.42));
        e.setEulerAngles(0, yawToEulerY(c.yaw), 0);
        prim(e, 'box', M.jaune, [0.42, 0.24, 0.02]);
        chevron(e, c.gauche);
        prim(e, 'cylinder', M.poteau, [0.025, 0.42, 0.025], [0, -0.33, -0.015]);
    }
    const n = T.pts.length;
    const arche = (i, damier) => {
        const [x, y, z] = T.pts[i], th = T.yaw[i];
        const e = new pc.Entity();
        root.addChild(e);
        e.setPosition(V(x, y, z));
        e.setEulerAngles(0, yawToEulerY(th), 0);
        // banderole à 1,9 m : la caméra de poursuite (~1,2 m) passe dessous
        for (const sx of [-1.15, 1.15]) prim(e, 'cylinder', M.blanc, [0.06, 2.05, 0.06], [sx, 1.02, 0]);
        for (let k = 0; k < 12; k++) {
            for (let j = 0; j < 2; j++) {
                prim(e, 'box', (k + j) % 2 === damier ? M.damierB : M.damierN, [0.19, 0.12, 0.03],
                    [-1.045 + k * 0.19, 1.9 + j * 0.12, 0]);
            }
        }
    };
    arche(T.depart_i, 0);
    arche(n - 2, 1);
    for (const g of T.portes) {
        const [x, y, z] = T.pts[g.i], th = T.yaw[g.i];
        const nx = -Math.sin(th), ny = Math.cos(th);
        const cones = [];
        for (const off of [-1.0, 1.0]) {
            cones.push(prim(root, 'cone', M.porte, [0.16, 0.3, 0.16], [0, 0, 0]));
            cones[cones.length - 1].setPosition(V(x + off * nx, y + off * ny, z + 0.15));
        }
        gates.push({ ...g, cones });
    }
    refreshGates();
    drawMinimapBase();
    $('porte').textContent = `0/${gates.length}`;
}

// ------------------------------------------------------------------ course : compte à rebours, progression, chrono
const LS_KEY = 'simrobot-course-record';
let record = null;
try { record = parseFloat(localStorage.getItem(LS_KEY)) || null; } catch (e) { record = null; }
const fmt = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`;
const showRecord = () => { $('record').textContent = record ? fmt(record) : '–'; };
showRecord();
let phase = 'attente', countdownEnd = 0, tStart = 0, tNow = 0, progK = 0, gateIdx = 0, finalTime = 0;
let driftPts = 0, combo = 0, wrongWay = 0;
function refreshGates() {
    gates.forEach((g, k) => {
        const m = k < gateIdx ? M.porteFaite : k === gateIdx ? M.porteProchaine : M.porte;
        g.cones.forEach((c) => { c.render.material = m; });
    });
}
function poseAt(i) {
    const [x, y] = track.P[i];
    return { x, y, yaw: track.yaw[i] };
}
function restart() {
    if (!track) return;
    send({ type: 'respawn', ...poseAt(track.depart_i) });
    phase = 'compte';
    countdownEnd = performance.now() + 3200;
    progK = track.depart_i; gateIdx = 0; driftPts = 0; combo = 0;
    refreshGates();
    $('porte').textContent = `0/${gates.length}`;
    $('temps').textContent = fmt(0);
    camInit = false;
}
function respawn() {
    if (!track || phase === 'attente') return;
    const i = gateIdx > 0 ? gates[gateIdx - 1].i : track.depart_i;
    send({ type: 'respawn', ...poseAt(i) });
    progK = Math.max(i, track.depart_i);
    combo = 0;
    camInit = false;
}
function updateRace(st) {
    if (!track || !car || !st) return;
    const p = car.chassis.getPosition();
    const px = p.x, py = -p.z;                                         // vue → physique
    const P = track.P, n = P.length;
    let best = progK, bd = 1e9;
    for (let i = Math.max(0, progK - 10); i < Math.min(n, progK + 60); i++) {
        const d = (P[i][0] - px) ** 2 + (P[i][1] - py) ** 2;
        if (d < bd) { bd = d; best = i; }
    }
    if (bd < 9 && best > progK) progK = best;                          // à moins de 3 m du circuit
    if (phase === 'course') {
        tNow = st.t - tStart;
        $('temps').textContent = fmt(tNow);
        while (gateIdx < gates.length && progK >= gates[gateIdx].i) {
            gateIdx++;
            refreshGates();
            $('porte').textContent = `${gateIdx}/${gates.length}`;
        }
        if (progK >= n - 3) {
            phase = 'arrivee';
            finalTime = tNow;
            const nouveau = !record || finalTime < record;
            if (nouveau) { record = finalTime; try { localStorage.setItem(LS_KEY, String(record)); } catch (e) { /* stockage indisponible */ } }
            showRecord();
            big(`${fmt(finalTime)}`, `${nouveau ? 'Nouveau record !' : `record ${fmt(record)}`} · dérive ${Math.round(driftPts)} pts · Entrée pour recommencer`);
        }
        // sens interdit
        const fwd = car.chassis.right;
        const th = track.yaw[Math.min(progK, n - 1)];
        const dot = fwd.x * Math.cos(th) + -fwd.z * Math.sin(th);
        wrongWay = st.speed > 1.5 && dot < -0.3 ? wrongWay + 1 : 0;
        if (wrongWay === 40) toast('MAUVAIS SENS');
    }
    // flèche : direction du circuit 6 m devant, relative au cap de la voiture
    const ahead = P[Math.min(n - 1, progK + 12)];
    const fwd = car.chassis.right;
    const carYaw = Math.atan2(-fwd.z, fwd.x);
    const tYaw = Math.atan2(ahead[1] - py, ahead[0] - px);
    let rel = tYaw - carYaw;
    rel = Math.atan2(Math.sin(rel), Math.cos(rel));
    $('fleche').querySelector('svg').style.transform = `rotate(${(-rel * 180) / Math.PI}deg)`;
    drawMinimap(px, py, carYaw);
}
function big(t, sub = '') {
    $('grand').style.display = t ? 'block' : 'none';
    $('grand').innerHTML = t ? `${t}${sub ? `<small>${sub}</small>` : ''}` : '';
}
let toastTimer = null;
function toast(t) {
    $('toast').textContent = t;
    $('toast').style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { $('toast').style.display = 'none'; }, 1500);
}

// ------------------------------------------------------------------ carte
const mini = $('mini'), mctx = mini.getContext('2d');
let mapT = null, mapBase = null;
function drawMinimapBase() {
    const P = track.P;
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const [x, y] of P) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const sc = 340 / Math.max(x1 - x0, y1 - y0);
    mapT = (x, y) => [20 + (x - x0) * sc + (340 - (x1 - x0) * sc) / 2, 360 - (y - y0) * sc - (340 - (y1 - y0) * sc) / 2];
    mctx.clearRect(0, 0, 380, 380);
    mctx.lineWidth = 7; mctx.strokeStyle = 'rgba(255,255,255,0.25)'; mctx.lineJoin = 'round';
    mctx.beginPath();
    P.forEach(([x, y], i) => { const [u, v] = mapT(x, y); if (i) mctx.lineTo(u, v); else mctx.moveTo(u, v); });
    mctx.stroke();
    mapBase = mctx.getImageData(0, 0, 380, 380);
}
function drawMinimap(px, py, yaw) {
    if (!mapBase || $('carte').style.display === 'none') return;
    mctx.putImageData(mapBase, 0, 0);
    const P = track.P;
    mctx.lineWidth = 7; mctx.strokeStyle = '#ffcf3a';
    mctx.beginPath();
    for (let i = track.depart_i; i <= progK; i++) { const [u, v] = mapT(P[i][0], P[i][1]); if (i > track.depart_i) mctx.lineTo(u, v); else mctx.moveTo(u, v); }
    mctx.stroke();
    if (gateIdx < gates.length) {
        const [u, v] = mapT(...P[gates[gateIdx].i]);
        mctx.fillStyle = '#33e6ff'; mctx.beginPath(); mctx.arc(u, v, 7, 0, 7); mctx.fill();
    }
    const [u, v] = mapT(px, py);
    mctx.save(); mctx.translate(u, v); mctx.rotate(-yaw);
    mctx.fillStyle = '#ff5a3c'; mctx.beginPath(); mctx.moveTo(14, 0); mctx.lineTo(-9, 8); mctx.lineTo(-9, -8); mctx.closePath(); mctx.fill();
    mctx.restore();
}
$('carte').style.display = window.innerWidth > 760 ? 'block' : 'none';
if (window.innerWidth > 760) $('fleche').style.top = '226px';

// ------------------------------------------------------------------ caméra : poursuite (3e personne) / capot (1re personne)
let camMode = 0, camInit = false, camYaw = 0;
const camPos = new pc.Vec3(), tgt = new pc.Vec3(), desired = new pc.Vec3(), tmp = new pc.Vec3(), loc = new pc.Vec3();
function updateCamera(dt) {
    if (!car) return;
    const ch = car.chassis, fwd = ch.right;                            // +x local du châssis = avant
    if (camMode === 1) {                                               // capot : au-dessus du pare-brise
        const [hx, , hz] = car.R.chassis_half;
        loc.set(hx * 0.15, 0, hz + 0.2);
        ch.getWorldTransform().transformPoint(loc, tmp);
        camera.setPosition(tmp);
        loc.set(hx * 0.15 + 3, 0, hz + 0.08);
        ch.getWorldTransform().transformPoint(loc, tgt);
        camera.lookAt(tgt, ch.forward.clone().mulScalar(-1));
        camera.camera.fov = 78;
        return;
    }
    camera.camera.fov = 68;
    const yaw = Math.atan2(fwd.x, fwd.z);
    if (!camInit) camYaw = yaw;
    let d = yaw - camYaw;
    d = Math.atan2(Math.sin(d), Math.cos(d));
    camYaw += d * (1 - Math.exp(-dt * 4));                             // retard : on voit la voiture glisser en dérive
    const p = ch.getPosition();
    desired.set(p.x - Math.sin(camYaw) * 2.3, p.y + 0.95, p.z - Math.cos(camYaw) * 2.3);
    if (!camInit) { camPos.copy(desired); camInit = true; }
    camPos.lerp(camPos, desired, 1 - Math.exp(-dt * 8));
    camera.setPosition(camPos);
    tgt.set(p.x + Math.sin(camYaw) * 1.5, p.y + 0.25, p.z + Math.cos(camYaw) * 1.5);
    camera.lookAt(tgt);
}

// ------------------------------------------------------------------ réseau
let ws = null;
function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => { $('dot').classList.add('on'); $('conn').textContent = 'connecté'; };
    ws.onclose = () => { $('dot').classList.remove('on'); $('conn').textContent = 'déconnecté…'; setTimeout(connect, 1000); };
    ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        const m = JSON.parse(ev.data);
        if (m.type === 'hello') {
            splatRot = m.splat_view_rotation_deg || [0, 0, 0];
            if (splatEntity) splatEntity.setLocalEulerAngles(...splatRot);
            if (!car) car = buildCar(m.rover);
            if (!track && m.circuit) { buildTrack(m.circuit); restart(); }
        } else if (m.type === 'state') {
            pushState(m);
            hud(m);
        }
    };
}
connect();
const send = (o) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); };

// ------------------------------------------------------------------ commandes : touche par code OU caractère (AZERTY, QWERTY)
const KEYS = {
    avant: [['KeyW', 'ArrowUp'], ['z', 'w', 'arrowup']], arriere: [['KeyS', 'ArrowDown'], ['s', 'arrowdown']],
    gauche: [['KeyA', 'ArrowLeft'], ['q', 'a', 'arrowleft']], droite: [['KeyD', 'ArrowRight'], ['d', 'arrowright']],
    frein: [['Space'], [' ']], turbo: [['ShiftLeft', 'ShiftRight'], ['shift']]
};
const lower = (e) => (e.key || '').toLowerCase();
const action = (e) => {
    for (const [a, [codes]] of Object.entries(KEYS)) if (codes.includes(e.code)) return a;
    for (const [a, [, chars]] of Object.entries(KEYS)) if (chars.includes(lower(e))) return a;
    return null;
};
const is = (e, code, ...chars) => e.code === code || chars.includes(lower(e));
const pressed = new Map();
const keyId = (e) => e.code || lower(e);
const k = (a) => ([...pressed.values()].includes(a) ? 1 : 0);
function sendCmd() {
    if (phase === 'compte' || phase === 'attente') { send({ type: 'cmd', throttle: 0, steer: 0, brake: true, turbo: false }); return; }
    let throttle = k('avant') - k('arriere'), steer = k('gauche') - k('droite');
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
window.addEventListener('keydown', (e) => {
    const a = action(e);
    if (a) {
        e.preventDefault();
        if (!pressed.has(keyId(e))) { pressed.set(keyId(e), a); sendCmd(); }
        return;
    }
    if (e.repeat || e.metaKey || e.ctrlKey) return;
    if (is(e, 'KeyC', 'c')) { camMode = 1 - camMode; camInit = false; toast(camMode ? 'caméra capot' : 'caméra poursuite'); }
    else if (is(e, 'KeyR', 'r')) respawn();
    else if (is(e, 'Enter', 'enter') || is(e, 'KeyN', 'n')) { big(''); restart(); }
    else if (is(e, 'KeyM', 'm')) $('carte').style.display = $('carte').style.display === 'none' ? 'block' : 'none';
    else if (is(e, 'KeyH', 'h')) $('aide').style.display = $('aide').style.display === 'none' ? '' : 'none';
});
window.addEventListener('keyup', (e) => { if (pressed.delete(keyId(e))) sendCmd(); });
window.addEventListener('blur', () => { pressed.clear(); sendCmd(); });
canvas.addEventListener('mousedown', () => canvas.focus());

// ------------------------------------------------------------------ affichage tête haute
let lastT = null;
function hud(s) {
    $('vitesse').textContent = Math.round(Math.abs(s.speed) * 3.6);
    if (s.sol) $('sol').textContent = `${s.sol.classe ?? ''} · adhérence ${s.sol.adherence.toFixed(2)}${s.sol.frein_veg_N > 0 ? ' · broussailles' : ''}`;
    const dt = lastT === null ? 0 : Math.max(0, Math.min(0.1, s.t - lastT));
    lastT = s.t;
    const drifting = phase === 'course' && Math.abs(s.glisse) > 12 && Math.abs(s.speed) > 2.5;
    if (drifting) {                                                   // points de dérive : glisse (°) × vitesse (m/s)
        const gain = (Math.abs(s.glisse) * Math.abs(s.speed) * dt) / 4;
        combo += gain;
        driftPts += gain;
        $('combo').textContent = `+${Math.round(combo)}`;
    } else {
        combo = 0;
    }
    $('derive').style.display = drifting ? 'block' : 'none';
    $('points').textContent = Math.round(driftPts);
    if (phase === 'compte') {
        const left = (countdownEnd - performance.now()) / 1000;
        if (left > 0.2) big(String(Math.ceil(left - 0.2)));
        else { phase = 'course'; tStart = s.t; big('GO !'); setTimeout(() => { if (phase === 'course') big(''); }, 700); }
    }
}

let acc = 0, frames = 0;
app.on('update', (dt) => {
    const st = interpolatedState();
    if (st) applyState(st);
    updateCamera(dt);
    if (states.length) updateRace(states[states.length - 1]);
    acc += dt; frames++;
    if (acc > 0.5) { $('fps').textContent = `${(frames / acc).toFixed(0)} i/s`; acc = 0; frames = 0; }
});
canvas.focus();
window.__course = { app, pc, get car() { return car; }, get track() { return track; } };
