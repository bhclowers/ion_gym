"""Read-only three.js viewer for single-FA geometry.

SCOPE: geometry only.  Fields and trajectories stay viz_core's — this
component's one job is to draw the document's parametric metal, with the
stored fraction SOLID and everything derived (mirror images, the r-z
revolution, the planar ghost slab) as visibly non-editable GHOSTS.

Division of labor: Python resolved every convention already
(`outline.render_primitives` — rotation sign measured against raster2d,
cutouts assigned to holes or reported, extrude-axis parity with the
builder).  The JS below is a dumb renderer of resolved polygons; it
carries NO route checks — the viewport list and out-of-plane semantics
arrive as data on the scene (policy-as-data, same rule as the session).

Viewports come from the policy: shapes3d gets xy/xz/yz ortho + orbiting
perspective (the multi-axis rule, natively); rz gets the r–x cross
section (the stored truth) + a perspective view whose revolved body is a
courtesy ghost with a 270-degree cutaway; planar gets xy + perspective
with the display-only ghost slab.  Anything the viewer cannot draw
faithfully is written into the legend, never silently dropped.

three.js is pinned at r168 (same pin and reason as the Slice 1 probe:
r169 changed how TransformControls joins the scene; this file has no
gizmo yet but keeps ONE pin for the epic).
"""
from __future__ import annotations

import panel as pn
import param
from panel.custom import JSComponent

from ion_gym.edit.policy import EditRefusal
from ion_gym.edit.session import EditSession
from ion_gym.edit.outline import render_primitives

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 620

# ---------------------------------------------------------------------
# three.js supply: a LOCAL COPY ships with the package, so the editor works with
# no internet connection.  The importmap points at data: URLs built from
# ion_gym/edit/vendor/three/* — no CDN at render, no static-route server
# wiring, and every serving mode (ion-gym dashboard, ion-gym edit,
# `panel serve <demo>`) behaves identically.  VENDOR_MANIFEST.json is
# the version + integrity authority (the release gate checks the
# hashes); updating is a deliberate, separate step
# (release tooling ASKS about updates, never applies them).
# If those files are missing, fall back to the CDN of the SAME
# version with a loud warning — a broken install should degrade to
# online-only, visibly, not to a blank canvas.
# ---------------------------------------------------------------------
import base64 as _b64
import json as _json
import warnings as _warnings
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "three"


def _vendor_importmap():
    manifest = _VENDOR_DIR / "VENDOR_MANIFEST.json"
    files = {
        "three": _VENDOR_DIR / "three.module.min.js",
        "three/addons/controls/OrbitControls.js":
            _VENDOR_DIR / "jsm/controls/OrbitControls.js",
        "three/addons/controls/TransformControls.js":
            _VENDOR_DIR / "jsm/controls/TransformControls.js",
        # fat-line family (trajectory width; same r168 pin —
        # WebGL ignores LineBasicMaterial.linewidth, Line2 does not)
        "three/addons/lines/Line2.js":
            _VENDOR_DIR / "jsm/lines/Line2.js",
        "three/addons/lines/LineSegments2.js":
            _VENDOR_DIR / "jsm/lines/LineSegments2.js",
        "three/addons/lines/LineGeometry.js":
            _VENDOR_DIR / "jsm/lines/LineGeometry.js",
        "three/addons/lines/LineSegmentsGeometry.js":
            _VENDOR_DIR / "jsm/lines/LineSegmentsGeometry.js",
        "three/addons/lines/LineMaterial.js":
            _VENDOR_DIR / "jsm/lines/LineMaterial.js",
    }
    if not manifest.is_file() or not all(f.is_file()
                                        for f in files.values()):
        return None, "0.168.0"
    ver = _json.loads(manifest.read_text()).get("version", "unknown")
    imports = {}
    for spec, path in files.items():
        b64 = _b64.b64encode(path.read_bytes()).decode()
        imports[spec] = f"data:text/javascript;base64,{b64}"
    return {"imports": imports}, ver


_IMPORTMAP, THREE_VERSION = _vendor_importmap()
if _IMPORTMAP is None:
    _warnings.warn(
        "ion_gym.edit.viewer: vendored three.js files are MISSING under "
        f"{_VENDOR_DIR} — falling back to the jsdelivr CDN, so the "
        "editor will NOT work offline. Re-vendor with "
        "`python internal/tools/vendor_three.py --update "
        + THREE_VERSION + "`.", stacklevel=2)
    _cdn = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"
    _IMPORTMAP = {"imports": {
        "three": f"{_cdn}/build/three.module.js",
        "three/addons/": f"{_cdn}/examples/jsm/",
    }}


class EditorViewer(JSComponent):
    """Renders a payload of {scene: editor_scene(), prims:
    render_primitives()}.  Read-only; no params flow JS -> Python yet."""

    payload = param.Dict(default={}, doc="scene + primitives, resolved "
                                        "by the Python session")
    selected = param.Integer(default=-1, doc="electrode index to "
                             "highlight while edited; -1 = none")
    selected_shape = param.Integer(default=-1, doc="raw shape "
                                   "index the gizmo attaches to")
    gizmo_enabled = param.Boolean(default=False, doc="attach the "
                                  "translate gizmo to the "
                                  "selected shape")
    picked = param.Dict(default={}, doc="JS -> Python: "
                        "{el, sh, seq} from a canvas click")
    drag = param.Dict(default={}, doc="JS -> Python: "
                      "{el, sh, dx, dy, dz, seq} proposed by a "
                      "gizmo drag; Python decides")
    canvas_w = param.Integer(default=940, doc="canvas width px")
    canvas_h = param.Integer(default=560, doc="canvas height px")
    doc_id = param.String(default="", doc="document identity (file "
                          "path) keying the JS-side camera memo: the "
                          "perspective pose survives the post-Apply "
                          "viewer rebuild for the SAME document and "
                          "resets on a document switch (PI "
                          "2026-08-26: editing must not move the view)")

    _importmap = _IMPORTMAP

    _esm = """
    import * as THREE from 'three';
    import { OrbitControls }
      from 'three/addons/controls/OrbitControls.js';
    import { TransformControls }
      from 'three/addons/controls/TransformControls.js';
    import { Line2 } from 'three/addons/lines/Line2.js';
    import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
    import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

    const GHOST_OPACITY = 0.18;
    const GRID_OPACITY  = 0.55;   // is_grid electrodes (meshes/screens)
    const MARGIN        = 1.10;   // ortho framing margin
    const LATHE_SEG     = 48;
    const LATHE_SWEEP   = 1.5 * Math.PI;   // 270 deg cutaway, view only

    // CAMERA MEMO (PI 2026-08-26: editing an electrode must not
    // move the 3-D view — a reset camera makes it hard to see whether
    // the change was accepted). Each commit swaps in a NEW component
    // instance, but this ESM module is a page-lifetime singleton, so
    // the perspective pose banked here survives the swap. Keyed by
    // model.doc_id: the SAME document restores its pose; a document
    // switch gets fresh default framing (a stale pose on a different-
    // scale deck could look at nothing). Only the perspective view
    // has user navigation (orthos are fixed), so only it is banked.
    const CAM_MEMO = {};

    function rgb(c) { return new THREE.Color(c[0]/255, c[1]/255, c[2]/255); }

    function shapeOf(outline, holes) {
      const s = new THREE.Shape();
      outline.forEach((p, i) => i ? s.lineTo(p[0], p[1])
                                  : s.moveTo(p[0], p[1]));
      s.closePath();
      (holes || []).forEach(h => {
        const path = new THREE.Path();
        h.forEach((p, i) => i ? path.lineTo(p[0], p[1])
                              : path.moveTo(p[0], p[1]));
        path.closePath();
        s.holes.push(path);
      });
      return s;
    }

    export function render({ model }) {
      const pay = model.payload || {};
      const scene2 = pay.scene || {}, prims = pay.prims || {};
      // stretch policy (PI 2026-08-25): canvas_w is only the
      // pre-layout FALLBACK; a ResizeObserver below re-fits the
      // canvas to its container on every width change.
      const MIN_W = 480;
      let W = Math.max(MIN_W, model.canvas_w || 940);
      // VERTICAL STACK (PI 2026-08-25): perspective on top,
      // every other view below it, each full width — horizontal
      // packing was what pushed the perspective off-screen.
      const PERSP_H = model.canvas_h || 440;
      const ORTHO_H = 260;
      // H must exist BEFORE the renderer (renderer.setSize uses
      // it): v419 shipped a temporal-dead-zone crash ('Cannot
      // access H before initialization') by computing it after the
      // views. Row plan from the policy's viewport list, filtered
      // to the known set; the views loop below still reports any
      // unknown name it skips.
      const KNOWN_VPS = ['persp', 'xy', 'xz', 'yz', 'rx'];
      const _vpn = ((((model.payload || {}).scene || {})
        .policy || {}).viewports || ['persp'])
        .filter(nm => KNOWN_VPS.includes(nm));
      const H = _vpn.reduce((s, nm) => s +
        (nm === 'persp' ? PERSP_H : ORTHO_H), 0) || PERSP_H;

      const wrap = document.createElement('div');
      wrap.style.cssText =
        'position:relative;font-family:monospace;width:100%';
      const legend = document.createElement('div');
      legend.style.cssText =
        'font-size:12px;line-height:1.5;padding:6px 2px';
      const notes = [];   // JS-side honesty channel -> legend

      try {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf4f4f2);  // light (PI 2026-08-25: dark hid electrodes)
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(W, H);
        renderer.setScissorTest(true);
        wrap.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.75));
        const sun = new THREE.DirectionalLight(0xffffff, 1.1);
        sun.position.set(1, 2, 3);
        scene.add(sun);

        const oop = scene2.out_of_plane || {};
        const gStored = new THREE.Group();
        const gGhost = new THREE.Group();

        // QUARTER CUTAWAY (PI 2026-09-06: "extends across the entire
        // assembly"): Python sends prims.quarter_cut = {x_mm, y_mm}
        // (the drawn metal's bbox centre) and BOTH material factories
        // carry two clipping planes with clipIntersection, so the
        // +x/+y quadrant is removed from EVERY electrode mesh — r-z
        // annuli, native 3-D rods, ghosts and mirror images alike —
        // with no per-route polygon surgery (the concave C-ring
        // construction this supersedes was r-z-only). Plane semantics:
        // a point is clipped where normal.p + constant < 0, so
        // (-1,0,0)/cx clips x > cx; clipIntersection limits the cut to
        // where BOTH hold — exactly the quadrant. Trajectory and
        // contour materials carry no planes: the beam stays whole.
        const qcut = prims.quarter_cut || null;
        const clipPlanes = qcut ? [
          new THREE.Plane(new THREE.Vector3(-1, 0, 0), qcut.x_mm),
          new THREE.Plane(new THREE.Vector3(0, -1, 0), qcut.y_mm)] : null;
        if (qcut) renderer.localClippingEnabled = true;
        const clipProps = qcut
          ? { clippingPlanes: clipPlanes, clipIntersection: true } : {};
        const solidMat = (c, isGrid) => new THREE.MeshStandardMaterial({
          color: rgb(c), roughness: 0.75, metalness: 0.1,
          side: THREE.DoubleSide,
          transparent: !!isGrid, opacity: isGrid ? GRID_OPACITY : 1.0,
          ...clipProps });
        const ghostMat = (c) => new THREE.MeshStandardMaterial({
          color: rgb(c), roughness: 0.9, metalness: 0.0,
          side: THREE.DoubleSide, transparent: true,
          opacity: GHOST_OPACITY, depthWrite: false, ...clipProps });

        for (const el of (prims.electrodes || [])) {
          const sm = solidMat(el.color, el.is_grid);
          const gm = ghostMat(el.color);
          const addSolid = (entry, mat, intoGhost) => {
            const shp = shapeOf(entry.outline, entry.holes);
            let mesh;
            if (entry.extrude) {
              const d = entry.extrude.hi_mm - entry.extrude.lo_mm;
              mesh = new THREE.Mesh(new THREE.ExtrudeGeometry(shp,
                { depth: d, bevelEnabled: false }), mat);
              mesh.position.z = entry.extrude.lo_mm;
            } else if (oop.kind === 'declared_depth') {
              // symmetric fallback: centre the slab on z=0 over the
              // declared depth (extrude3d.py, which this cited, was
              // deleted 2026-08-30 as an orphan)
              const d = oop.depth_mm;
              mesh = new THREE.Mesh(new THREE.ExtrudeGeometry(shp,
                { depth: d, bevelEnabled: false }), mat);
              mesh.position.z = -d / 2;
            } else if (oop.kind === 'declared_axial') {
              const [lo, hi] = oop.axial_extent_mm;
              mesh = new THREE.Mesh(new THREE.ExtrudeGeometry(shp,
                { depth: hi - lo, bevelEnabled: false }), mat);
              mesh.position.z = lo;
            } else {
              // flat cross-section IS the stored truth ('ghost' planar
              // and 'revolved' rz both draw it solid at z = 0)
              mesh = new THREE.Mesh(new THREE.ShapeGeometry(shp), mat);
              if (oop.kind === 'ghost') {
                const e = oop.extent_mm;   // display-only, z undeclared
                const slab = new THREE.Mesh(new THREE.ExtrudeGeometry(
                  shp, { depth: e, bevelEnabled: false }), gm);
                slab.position.z = -e / 2;
                gGhost.add(slab);
              }
              if (oop.kind === 'revolved') {
                if ((entry.holes || []).length)
                  notes.push(`${el.name}: hole(s) shown in the r-x ` +
                             `section only — the lathe ghost cannot ` +
                             `carry holes`);
                const pts = entry.outline.map(
                  p => new THREE.Vector2(p[1], p[0]));   // (r, axial)
                pts.push(pts[0].clone());
                const lat = new THREE.Mesh(new THREE.LatheGeometry(
                  pts, LATHE_SEG, 0, LATHE_SWEEP), gm);
                lat.rotation.z = -Math.PI / 2;   // lathe axis -> world x
                gGhost.add(lat);
              }
            }
            mesh.userData.el = el.index;
            mesh.userData.sh = (entry.shape_index !== undefined
                                ? entry.shape_index : -1);
            mesh.userData.ghost = intoGhost;
            mesh.userData.hasExtrude = !!entry.extrude;
            mesh.userData.z0 = entry.extrude
              ? entry.extrude.lo_mm : 0;
            // outline lows: for drag-time clamps (Python-derived,
            // matching the session's _shape_lo semantics)
            let lx = Infinity, ly = Infinity;
            for (const q of entry.outline) {
              if (q[0] < lx) lx = q[0];
              if (q[1] < ly) ly = q[1];
            }
            mesh.userData.lo_x = lx; mesh.userData.lo_y = ly;
            (intoGhost ? gGhost : gStored).add(mesh);
          };
          el.solids.forEach(s => addSolid(s, sm, false));
          el.ghosts.forEach(g => addSolid(g, gm, true));
        }

        // mirror images: every non-empty combination of declared mirror
        // planes, as ghosts (the stored fraction stays the only solid).
        const mirrors = Object.entries(
            (scene2.symmetry || {}).planes || {})
          .filter(([, v]) => v === 'mirror').map(([a]) => a);
        const combos = [];
        for (let m = 1; m < (1 << mirrors.length); m++)
          combos.push(mirrors.filter((_, i) => m & (1 << i)));
        for (const combo of combos) {
          const img = gStored.clone(true);
          img.traverse(o => {
            if (o.isMesh) o.material = ghostMat(
              [o.material.color.r*255, o.material.color.g*255,
               o.material.color.b*255]);
          });
          const holder = new THREE.Group();
          holder.add(img);
          for (const a of combo) holder.scale[a] *= -1;
          gGhost.add(holder);
        }
        if (mirrors.length)
          notes.push(`mirror image(s) in ${mirrors.join(', ')} drawn as ` +
                     `ghosts — the stored fraction is the only solid`);
        if (oop.kind === 'ghost')
          notes.push(`z UNDECLARED: translucent slab is a display ` +
                     `placeholder (${oop.extent_mm.toFixed(3)} mm = 10% ` +
                     `of the smaller in-plane dimension), not geometry`);
        if (oop.kind === 'revolved')
          notes.push('revolved body is a courtesy ghost with a 270\\u00b0 ' +
                     'cutaway; the r-x section is the stored truth');
        if (qcut)
          notes.push('QUARTER CUTAWAY: the +x/+y quadrant of every ' +
                     'electrode is clipped about (' +
                     qcut.x_mm.toFixed(2) + ', ' + qcut.y_mm.toFixed(2) +
                     ') mm \\u2014 display only; the flown geometry is ' +
                     'uncut');

        // SELECTED-ELECTRODE HIGHLIGHT (PI 2026-08-25): emissive
        // tint on every mesh of the selected electrode, including its
        // mirror ghosts, driven LIVE by the synced param — no rebuild.
        const HILITE = new THREE.Color(0x2266ff);
        const applyHighlight = () => {
          const sel = model.selected;
          [gStored, gGhost].forEach(g => g.traverse(o => {
            if (!o.isMesh || o.userData.el === undefined) return;
            const on = (o.userData.el === sel);
            o.material.emissive = on ? HILITE
                                     : new THREE.Color(0x000000);
            o.material.emissiveIntensity = on ? 0.55 : 0.0;
          }));
        };
        model.on('selected', applyHighlight);
        applyHighlight();

        scene.add(gStored); scene.add(gGhost);

        // DC CONTOURS (Slice 5): Python computed the polylines from
        // the committed document; layer 2 shows them in the 2-D
        // (ortho) views only, per the ruling.  Color runs blue->red
        // with level.
        const cont = pay.contours || null;
        if (cont && cont.refused) {
          notes.push('contours: ' + cont.refused);
        } else if (cont) {
          const zEps = 0.001 * Math.max(1e-6,
            (prims.bbox2d ? (prims.bbox2d[2] - prims.bbox2d[0]) : 1));
          const lo = cont.vmin, span = Math.max(cont.vmax - lo, 1e-12);
          for (const pl of cont.polylines) {
            const s = (pl.level - lo) / span;
            const col = new THREE.Color(
              0.13 + 0.67 * s, 0.33 - 0.13 * s, 0.80 - 0.67 * s);
            const pts = pl.pts.map(
              q => new THREE.Vector3(q[0], q[1], zEps));
            const geo = new THREE.BufferGeometry().setFromPoints(pts);
            const line = new THREE.Line(geo,
              new THREE.LineBasicMaterial({ color: col }));
            line.layers.set(2);
            scene.add(line);
          }
          notes.push('contours: ' + cont.polylines.length +
            ' line(s), ' + cont.levels.length + ' levels [' +
            cont.vmin.toFixed(3) + ' .. ' + cont.vmax.toFixed(3) +
            '] ' + cont.unit + ' — ' + cont.note +
            ' (field build ' + cont.build_s + ' s)');
        }

        // FLIGHT TRAJECTORIES (L-197): world-frame polylines banked by
        // the last-flight slot. Base layer 0 — visible in the
        // perspective AND every ortho view (contours stay layer 2 by
        // their 2-D-views ruling). STYLE comes from the payload (PI
        // 2026-08-26: scheme + width + opacity are user controls);
        // drawn as Line2 fat lines because WebGL ignores
        // LineBasicMaterial.linewidth on essentially every platform —
        // a plain width knob would be a dead control. LineMaterial
        // width is in PIXELS against a resolution uniform, so every
        // trajectory material is tracked and re-fed the canvas size on
        // each relayout.
        const trajMats = [];
        const fl = pay.trajs || null;
        if (fl && fl.paths && fl.paths.length) {
          const st = fl.style || {};
          const colorBy = st.color_by || 'ion';
          const widthPx = Math.max(0.5, st.width_px || 2.0);
          const alpha = Math.min(1.0, Math.max(0.05,
            st.opacity === undefined ? 0.85 : st.opacity));
          const solid = new THREE.Color(st.color || '#c83c3c');
          const nP = fl.paths.length;
          // COLORMAP LUTs (PI 2026-08-26: "add a few more color
          // schemes too (maybe plasma or similar)") — 8-anchor linear
          // interpolations of the matplotlib maps; one `ramp(t)` serves
          // the per-ion ramp AND every parameter ramp (speed/time/KE),
          // so "colormap" means the same thing in every mode.
          const LUTS = {
            'plasma': [[13,8,135],[84,2,163],[139,10,165],[185,50,137],
                       [219,92,104],[244,136,73],[254,188,43],[240,249,33]],
            'viridis': [[68,1,84],[70,50,127],[54,92,141],[39,127,142],
                        [31,161,135],[74,194,109],[159,218,58],[253,231,37]],
          };
          const cmap = st.cmap || 'cool-warm';
          const ramp = (t) => {
            t = Math.min(1, Math.max(0, t));
            const lut = LUTS[cmap];
            if (!lut)            // cool-warm analytic (the original)
              return [0.85 - 0.65 * t, 0.25 + 0.35 * t, 0.25 + 0.60 * t];
            const s = t * (lut.length - 1), i = Math.floor(s),
                  f = s - i, a = lut[i], b = lut[Math.min(i + 1,
                                                          lut.length - 1)];
            return [(a[0] + f * (b[0] - a[0])) / 255,
                    (a[1] + f * (b[1] - a[1])) / 255,
                    (a[2] + f * (b[2] - a[2])) / 255];
          };
          fl.paths.forEach((tp, ti) => {
            const s = nP > 1 ? ti / (nP - 1) : 0.0;
            const rc = ramp(s);
            const col = colorBy === 'solid'
              ? solid : new THREE.Color(rc[0], rc[1], rc[2]);
            const flat = [];
            for (const q of tp.pts) flat.push(q[0], q[1], q[2]);
            const geo = new LineGeometry();
            geo.setPositions(flat);
            let mat;
            if ((colorBy === 'speed' || colorBy === 'time'
                 || colorBy === 'ke') && tp.vals) {
              // per-vertex ramp over the normalized parameter (0..1
              // from Python; the ABSOLUTE range with units is in the
              // legend, so the ramp never floats unanchored).
              const vcols = [];
              for (const v of tp.vals) {
                const c = ramp(v);
                vcols.push(c[0], c[1], c[2]);
              }
              geo.setColors(vcols);
              mat = new LineMaterial({
                color: 0xffffff, vertexColors: true,
                linewidth: widthPx, worldUnits: false,
                transparent: alpha < 1.0, opacity: alpha });
            } else {
              mat = new LineMaterial({
                color: col, linewidth: widthPx, worldUnits: false,
                transparent: alpha < 1.0, opacity: alpha });
            }
            mat.resolution.set(W, H);
            trajMats.push(mat);
            scene.add(new Line2(geo, mat));
          });
          if (fl.detections && fl.detections.length) {
            // DETECTION MARKERS (option B): small green points at each
            // registered crossing — pass-through, so paths continue.
            const dg = new THREE.BufferGeometry();
            const dp = [];
            for (const d of fl.detections) dp.push(d[0], d[1], d[2]);
            dg.setAttribute('position',
              new THREE.Float32BufferAttribute(dp, 3));
            scene.add(new THREE.Points(dg, new THREE.PointsMaterial({
              color: 0x1a7f37, size: 4, sizeAttenuation: false })));
          }
          notes.push('trajectories: ' + fl.legend);
        } else if (fl && fl.legend) {
          notes.push('trajectories: ' + fl.legend);
        }

        // framing
        const box = new THREE.Box3().setFromObject(scene);
        const ctr = box.getCenter(new THREE.Vector3());
        const dim = box.getSize(new THREE.Vector3());
        const R = Math.max(dim.x, dim.y, dim.z, 1e-6);

        // grid + axes on layer 1: perspective viewport only
        const grid = new THREE.GridHelper(2 * R, 20, 0xb0b0bc, 0xdadade);
        // GridHelper lies in xz; rotate into the plane of the two
        // largest extents (data-driven, no route check)
        const order = [['x', dim.x], ['y', dim.y], ['z', dim.z]]
          .sort((a, b) => b[1] - a[1]).map(d => d[0]);
        if (!order.slice(0, 2).includes('y')) grid.rotation.x = Math.PI/2;
        else if (!order.slice(0, 2).includes('z'))
          grid.rotation.x = Math.PI / 2;
        grid.position.copy(ctr);
        grid.layers.set(1);
        scene.add(grid);
        const axes = new THREE.AxesHelper(0.6 * R);
        axes.position.copy(ctr); axes.layers.set(1);
        scene.add(axes);

        // viewports from the POLICY (data), laid out 1 / 1x2 / 2x2
        const defs = {
          persp: { label: 'perspective (drag to orbit)', kind: 'persp' },
          xy: { label: 'xy  (x\\u2192, y\\u2191)', dir: [0, 0, 1],
                up: [0, 1, 0], ax: 'x', ay: 'y' },
          xz: { label: 'xz  (x\\u2192, z\\u2191)', dir: [0, -1, 0],
                up: [0, 0, 1], ax: 'x', ay: 'z' },
          yz: { label: 'yz  (z\\u2192, y\\u2191)', dir: [-1, 0, 0],
                up: [0, 1, 0], ax: 'z', ay: 'y' },
          rx: { label: 'r\\u2013x  (x axial \\u2192, r \\u2191)',
                dir: [0, 0, 1], up: [0, 1, 0], ax: 'x', ay: 'y' },
        };
        const names = ((scene2.policy || {}).viewports) || ['persp'];
        const usable = names.filter(nm => {
          if (defs[nm]) return true;
          notes.push(`unknown viewport ${nm} — skipped, reported`);
          return false;
        });
        const views = [];
        let orbit = null;
        usable.forEach((nm, i) => {
          const d = defs[nm];
          let cam;
          if (d.kind === 'persp') {
            cam = new THREE.PerspectiveCamera(45, 1, R/100, R*40);
            cam.position.set(ctr.x + 1.0*R, ctr.y + 0.8*R,
                             ctr.z + 1.4*R);
            cam.lookAt(ctr);
            cam.layers.enable(1);
            orbit = new OrbitControls(cam, renderer.domElement);
            orbit.target.copy(ctr);
            // camera memo: restore this document's banked pose so a
            // post-Apply rebuild shows the SAME view (see CAM_MEMO).
            // Backed by localStorage (PI 2026-09-07: a hard refresh of
            // the /flight tab reset the perspective): the in-module
            // memo dies with the page, so on a miss the pose is read
            // back from localStorage under a namespaced key, and every
            // completed orbit interaction writes through. Storage can
            // refuse (privacy modes, quota) — that is REPORTED once via
            // console.warn and the memo degrades to page-lifetime,
            // never silently pretended.
            const memoKey = model.doc_id || '';
            const lsKey = memoKey ? ('ion_gym.cam.' + memoKey) : '';
            const _lsGet = () => {
              if (!lsKey) return null;
              try {
                const raw = window.localStorage.getItem(lsKey);
                return raw ? JSON.parse(raw) : null;
              } catch (e) {
                console.warn('ion_gym camera memo: localStorage read ' +
                             'refused (' + e + ') — pose is page-lifetime ' +
                             'only this session');
                return null;
              }
            };
            const _lsPut = (pose) => {
              if (!lsKey) return;
              try {
                window.localStorage.setItem(lsKey, JSON.stringify(pose));
              } catch (e) {
                console.warn('ion_gym camera memo: localStorage write ' +
                             'refused (' + e + ') — pose is page-lifetime ' +
                             'only this session');
              }
            };
            const saved = memoKey ? (CAM_MEMO[memoKey] || _lsGet()) : null;
            if (saved) {
              cam.position.fromArray(saved.pos);
              orbit.target.fromArray(saved.tgt);
              cam.zoom = saved.zoom;
              cam.updateProjectionMatrix();
              orbit.update();
            }
            if (memoKey) {
              orbit.addEventListener('change', () => {
                CAM_MEMO[memoKey] = { pos: cam.position.toArray(),
                                      tgt: orbit.target.toArray(),
                                      zoom: cam.zoom };
              });
              // 'end' fires once per completed interaction — the write
              // cadence localStorage wants (per-frame 'change' writes
              // would serialize JSON at pointer-move rate for nothing).
              orbit.addEventListener('end', () => {
                if (CAM_MEMO[memoKey]) _lsPut(CAM_MEMO[memoKey]);
              });
            }
          } else {
            cam = new THREE.OrthographicCamera(-1, 1, 1, -1,
                                               -R * 20, R * 20);
            cam.position.set(ctr.x + d.dir[0] * R * 3,
                             ctr.y + d.dir[1] * R * 3,
                             ctr.z + d.dir[2] * R * 3);
            cam.up.set(...d.up);
            cam.lookAt(ctr);
            cam.layers.enable(2);   // contours: 2-D views only
          }
          const lab = document.createElement('div');
          lab.textContent = d.label;
          lab.style.cssText =
            'position:absolute;color:#55555e;font-size:11px;' +
            'pointer-events:none';
          wrap.appendChild(lab);
          views.push({ cam, d, lab,
                       rh: d.kind === 'persp' ? PERSP_H : ORTHO_H,
                       rect: { x: 0, y: 0, w: 1, h: 1 } });
        });

        let perspRect = null;
        const relayout = (newW) => {
          W = Math.max(MIN_W, Math.floor(newW));
          renderer.setSize(W, H);
          for (const m of trajMats) m.resolution.set(W, H);
          let yAcc = 0;
          for (const v of views) {
            v.rect = { x: 0, y: yAcc, w: W, h: v.rh };
            yAcc += v.rh;
            if (v.d.kind === 'persp') {
              v.cam.aspect = W / v.rh;
              perspRect = v.rect;
            } else {
              const hw = Math.max(dim[v.d.ax] / 2, 1e-3) * MARGIN;
              const hh = Math.max(dim[v.d.ay] / 2, 1e-3) * MARGIN;
              const s = Math.max(hw / (W / 2), hh / (v.rh / 2));
              v.cam.left = -s * W / 2; v.cam.right = s * W / 2;
              v.cam.top = s * v.rh / 2;
              v.cam.bottom = -s * v.rh / 2;
            }
            v.cam.updateProjectionMatrix();
            v.lab.style.left = (v.rect.x + 6) + 'px';
            v.lab.style.top = (v.rect.y + 4) + 'px';
          }
        };
        relayout(W);
        // track the container: the observer fires once on attach
        // (correcting the fallback width) and on every resize after
        const ro = new ResizeObserver((entries) => {
          const cw = Math.floor(entries[0].contentRect.width);
          if (cw >= MIN_W && Math.abs(cw - W) > 1) relayout(cw);
        });
        ro.observe(wrap);

        // ---- Slice 4: viewport-aware picking + translate gizmo
        const raycaster = new THREE.Raycaster();
        const viewAt = (ev) => {
          const b = renderer.domElement.getBoundingClientRect();
          const px = ev.clientX - b.left, py = ev.clientY - b.top;
          for (const v of views) {
            const r = v.rect;
            if (px >= r.x && px < r.x + r.w
                && py >= r.y && py < r.y + r.h) {
              return { v, ndc: new THREE.Vector2(
                ((px - r.x) / r.w) * 2 - 1,
                -(((py - r.y) / r.h) * 2 - 1)) };
            }
          }
          return null;
        };

        const gizmo = new TransformControls(
          views.length ? views[0].cam : camDummy(),
          renderer.domElement);
        gizmo.setMode('translate');
        gizmo.translationSnap = scene2.pitch_mm || null;
        // r168: the controls object itself joins the scene
        scene.add(gizmo);
        gizmo.addEventListener('dragging-changed', (e) => {
          if (orbit) orbit.enabled = !e.value;
          if (!e.value && gizmo.object) {
            const o = gizmo.object;
            model.drag = { el: o.userData.el, sh: o.userData.sh,
                           dx: o.position.x, dy: o.position.y,
                           dz: o.position.z - o.userData.z0,
                           seq: ((model.drag||{}).seq || 0) + 1 };
          }
        });
        function camDummy() {
          return new THREE.PerspectiveCamera();
        }
        const findMesh = (elI, shI) => {
          let hit = null;
          gStored.traverse(o => {
            if (!hit && o.isMesh && o.userData.el === elI
                && o.userData.sh === shI) hit = o;
          });
          return hit;
        };
        const attachGizmo = () => {
          const want = model.gizmo_enabled
            && model.selected >= 0 && model.selected_shape >= 0;
          const mesh = want
            ? findMesh(model.selected, model.selected_shape)
            : null;
          if (!mesh) { gizmo.detach(); return; }
          gizmo.attach(mesh);
          gizmo.showZ = !!mesh.userData.hasExtrude;
          // drag-time clamps from POLICY data: position minima so
          // the shape's low edge cannot cross a declared boundary
          gizmo.minX = -Infinity; gizmo.minY = -Infinity;
          gizmo.minZ = -Infinity;
          for (const c of (scene2.clamps || [])) {
            if (c.target === 'shape') {
              if (c.axis === 'x')
                gizmo.minX = c.min - mesh.userData.lo_x;
              if (c.axis === 'y')
                gizmo.minY = c.min - mesh.userData.lo_y;
            } else if (c.target === 'extrude'
                       && mesh.userData.hasExtrude) {
              gizmo.minZ = c.min;  // position.z IS the slab lo
            }
          }
        };
        model.on('selected', attachGizmo);
        model.on('selected_shape', attachGizmo);
        model.on('gizmo_enabled', attachGizmo);
        attachGizmo();

        renderer.domElement.addEventListener('pointerdown', (e) => {
          if (gizmo.dragging) return;
          const at = viewAt(e);
          if (!at) return;
          // the gizmo answers to whichever viewport the pointer
          // is in, so drags work in ortho views too
          gizmo.camera = at.v.cam;
          raycaster.setFromCamera(at.ndc, at.v.cam);
          const hits = raycaster.intersectObjects(
            gStored.children, true);
          for (const h of hits) {
            const u = (h.object || {}).userData || {};
            if (u.el !== undefined && !u.ghost) {
              model.picked = { el: u.el, sh: u.sh,
                seq: ((model.picked||{}).seq || 0) + 1 };
              break;
            }
          }
        }, true);

        // orbit only inside the perspective viewport
        renderer.domElement.addEventListener('pointerdown', (e) => {
          if (!orbit || !perspRect) return;
          const b = renderer.domElement.getBoundingClientRect();
          const x = e.clientX - b.left, y = e.clientY - b.top;
          orbit.enabled = (x >= perspRect.x && x < perspRect.x+perspRect.w
                        && y >= perspRect.y && y < perspRect.y+perspRect.h);
        }, true);

        renderer.setAnimationLoop(() => {
          for (const v of views) {
            const gy = H - v.rect.y - v.rect.h;   // GL origin bottom-left
            renderer.setViewport(v.rect.x, gy, v.rect.w, v.rect.h);
            renderer.setScissor(v.rect.x, gy, v.rect.w, v.rect.h);
            renderer.render(scene, v.cam);
          }
        });
      } catch (err) {
        console.error('EditorViewer:', err);
        notes.push('VIEWER ERROR (see JS console): ' + err.message);
      }

      // legend: document facts + every report, dropped nothing
      const planes = Object.entries((scene2.symmetry||{}).planes||{})
        .map(([a, v]) => `${a}:${v}`).join(' ');
      const lines = [
        `<b>${scene2.document||'?'}</b> — route <b>${scene2.route||'?'}` +
        `</b>, pitch ${scene2.pitch_mm} mm, symmetry ${planes}`,
      ];
      const els = (prims.electrodes || []);
      const chips = els.slice(0, 30).map(e => {
        const c = `rgb(${e.color[0]},${e.color[1]},${e.color[2]})`;
        const tag = e.editable ? '' : ' [not editable]';
        const gr = e.is_grid ? ' [grid]' : '';
        return `<span style="border-left:10px solid ${c};` +
               `padding-left:4px;margin-right:10px">${e.name}${gr}${tag}` +
               `</span>`;
      });
      if (els.length > 30) chips.push(`… ${els.length - 30} more`);
      lines.push(chips.join(''));
      for (const e of els)
        for (const r of (e.reported || []))
          lines.push(`&#9888; ${e.name}: ${r}`);
      for (const nnote of notes) lines.push(`&#8505; ${nnote}`);
      legend.innerHTML = lines.join('<br>');
      wrap.appendChild(legend);
      return wrap;
    }
    """


def viewer_for_session(session: EditSession,
                       width: int = DEFAULT_WIDTH,
                       height: int = DEFAULT_HEIGHT,
                       canvas_w: int = 940, canvas_h: int = 560,
                       selected: int = -1,
                       selected_shape: int = -1,
                       gizmo_enabled: bool = False,
                       contours: dict | None = None,
                       doc_id: str = "") -> EditorViewer:
    """Compose the payload from the two Python authorities and mount
    it.  STRETCH POLICY: the component stretches
    to its container width; `canvas_w` is only the pre-layout
    fallback and `width` is ignored for sizing (kept for signature
    stability).  Height stays declared via canvas_h.  `doc_id` keys
    the JS camera memo (empty = memo off, default framing)."""
    payload = {"scene": session.editor_scene(),
               "prims": render_primitives(session)}
    if contours is not None:
        payload["contours"] = contours
    return EditorViewer(
        payload=payload,
        sizing_mode="stretch_width",
        canvas_w=canvas_w, canvas_h=canvas_h, selected=selected,
        selected_shape=selected_shape,
        gizmo_enabled=gizmo_enabled, doc_id=doc_id)


def view_document(path, width: int = DEFAULT_WIDTH,
                  height: int = DEFAULT_HEIGHT):
    """UI-path entry: a viewer for an in-scope document, or the REFUSAL,
    named, as a pane — a refusal is a normal outcome here, not a
    traceback.  Genuine malfunctions still raise."""
    try:
        session = EditSession.load(path)
    except EditRefusal as e:
        return pn.pane.Markdown(
            f"### Viewer refuses this document\n**{e}**\n\n"
            f"(Editor scope: parametric single-FA specs on the planar, "
            f"r-z and shapes3d routes — L-193.)",
            width=width)
    return viewer_for_session(session, width=width, height=height)
