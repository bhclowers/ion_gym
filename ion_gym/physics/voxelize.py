"""
ion_gym.voxelize
----------------
Rung-2 geometry path: labeled occupancy grids produced from MESHES, so the
CadQuery/STL front-end and the analytic-primitive front-end collapse onto
ONE voxelizer (the universal-interface principle). The solver,
tracer, and all views consume this grid, never the source solid.

Entry points
  tessellate_scene(scene, sections)  GeomScene primitives -> per-electrode
      watertight meshes (+ the a-priori sagitta bound). Lets the SAME GeomScene
      that drive the analytic emitter drive the mesh path, so the two can be
      diffed against the node-exact ground truth banked in Rung 1.
  export_stls(scene, dir, stem)      one STL per electrode on the
      quad_N.stl convention (N = electrode(N) = .paN = grid label).
  voxelize_meshes / voxelize_stls    meshes -> labeled grid (int16),
      stored/folded frame, matching rasterize3d exactly.

Containment engine: scanline ray parity.
  One +x ray per (y,z) node row (~ny*nz rays, not nx*ny*nz point queries).
  Each ray is intersected against the triangle soup (2-D edge-function
  point-in-triangle on the (y,z) projection, barycentric x at the crossing);
  a node is inside iff the number of crossings below its x is odd. Disjoint
  watertight bodies in one mesh (an electrode's two rods) compose correctly
  by parity. Rays are jittered by an irrational sub-nano-gu offset because
  this geometry is grid-aligned EVERYWHERE (faces exactly on node planes,
  vertices on the axis) — the classic ray-through-edge degeneracies are not
  edge cases here, they are the common case. Jitter misclassifies only
  nodes within the jitter of a surface, and those are exactly the nodes the
  eps promotion (below) re-includes.

Surface convention (the shared geometry<->solver contract):
  nodes ON a metal surface are marked electrode (surface-inclusive;
  pinned in Rung 1). Two mesh effects fight that:
    (a) flat faces exactly on node planes: strict parity is a coin flip;
    (b) curved faces: the tessellation lies INSIDE the ideal surface by up
        to the facet sagitta r*(1 - cos(pi/sections)), so the outermost
        node ring on round electrodes is systematically lost.
  voxelize therefore promotes outside nodes within `surface_eps` (grid
  units) of the mesh surface, computed EXACTLY (point-triangle distance)
  but only on the one-node boundary shell around the parity-inside set.
  Default eps = the tessellation sagitta (reported). This inflation is a
  convention choice at the mesh->grid boundary; what it costs in field and
  trajectory terms is precisely Rung 2's measurement, not something to
  assume. The error band on curved surfaces is +-sagitta by construction:
  facet vertices lie ON the ideal surface, facet centers sagitta inside.
"""

import numpy as np
import trimesh

from ion_gym.physics.scene3d import GeomScene, Cylinder, Box3D

_JIT = (np.sqrt(5) - 1) * 1e-7      # irrational jitter, gu-scale


# ---------------------------------------------------------- tessellation
def _cyl_mesh(p: Cylinder, sections: int) -> trimesh.Trimesh:
    """z-axis cylinder spanning [z-L, z] (pinned convention). L=0
    (ideal zero-thickness disc) gets a token 1e-4 gu-scale thickness so the
    mesh is a closed solid; the parity test then sees a thin slab exactly
    like a surface-inclusive rasterizer sees a one-node-plane electrode."""
    L = max(p.length, 1e-4)
    m = trimesh.creation.cylinder(radius=p.r, height=L, sections=sections)
    m.apply_translation([p.cx, p.cy, p.z - L / 2.0])
    return m


def _box_mesh(p: Box3D) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=[p.x2 - p.x1, p.y2 - p.y1,
                                      max(p.z2 - p.z1, 1e-4)])
    m.apply_translation([(p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2,
                         (p.z1 + p.z2) / 2])
    return m


def _prim_mesh(p, sections):
    return _cyl_mesh(p, sections) if isinstance(p, Cylinder) else _box_mesh(p)


def sagitta_gu(scene: GeomScene, sections: int) -> float:
    """Worst-case chord error of the tessellation over all cylinders, in
    grid units — the a-priori boundary-quantization bound."""
    sc = scene.in_gu()
    r = 0.0
    for e in sc.electrodes:
        for sh in e.shapes:
            for p in sh.within + sh.notin:
                if isinstance(p, Cylinder):
                    r = max(r, p.r)
    return r * (1.0 - np.cos(np.pi / sections))


def tessellate_scene(scene: GeomScene, sections=64):
    """GeomScene -> {index -> watertight trimesh} in GRID UNITS, full physical
    frame (within minus notin per shape, union across shapes)."""
    sc = scene.check().in_gu()
    meshes = {}
    for e in sc.electrodes:
        bodies = []
        for sh in e.shapes:
            w = [_prim_mesh(p, sections) for p in sh.within]
            body = w[0] if len(w) == 1 else trimesh.boolean.union(w)
            if sh.notin:
                cut = [_prim_mesh(p, sections) for p in sh.notin]
                cut = cut[0] if len(cut) == 1 else trimesh.boolean.union(cut)
                body = trimesh.boolean.difference([body, cut])
            if not body.is_watertight:
                body.fill_holes()
            assert body.is_watertight, f"electrode {e.index} not watertight"
            bodies.append(body)
        meshes[e.index] = (bodies[0] if len(bodies) == 1
                           else trimesh.util.concatenate(bodies))
    return meshes


def export_stls(scene: GeomScene, out_dir, stem="quad", sections=64):
    """One STL per electrode: {stem}_{N}.stl, in mm (importers typically take units to
    confirm on its docs), full physical frame. Returns ({N: path}, sagitta
    of this tessellation in gu)."""
    from pathlib import Path
    sc = scene.check()
    meshes = tessellate_scene(sc, sections=sections)
    mm = sc.grid.mm_per_gu
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for idx, mesh in sorted(meshes.items()):
        m = mesh.copy()
        m.apply_scale(mm)                       # gu -> mm
        p = out / f"{stem}_{idx}.stl"
        m.export(p)
        written[idx] = str(p)
    return written, sagitta_gu(sc, sections)


# ------------------------------------------------- scanline ray parity
def _ray_crossings(tris, ys, zs):
    """For +x rays at (y, z) in `ys, zs` (flat arrays, jitter already
    applied), return list-of-arrays of crossing x per ray.
    tris: (T, 3, 3) float64 in gu."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    # 2-D projection onto (y, z)
    ay, az = v0[:, 1], v0[:, 2]
    by, bz = v1[:, 1], v1[:, 2]
    cy, cz = v2[:, 1], v2[:, 2]
    area = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
    keep = np.abs(area) > 1e-12
    ax_, ay, az = v0[keep, 0], ay[keep], az[keep]
    bx_, by, bz = v1[keep, 0], by[keep], bz[keep]
    cx_, cy, cz = v2[keep, 0], cy[keep], cz[keep]
    area = area[keep]
    R = len(ys)
    out = [None] * R
    CH = max(1, int(4e7 // max(len(area), 1)))     # ~40M pair budget
    for s in range(0, R, CH):
        y = ys[s:s + CH, None]
        z = zs[s:s + CH, None]
        # barycentric via edge functions (signed sub-areas / total)
        w0 = ((by - y) * (cz - z) - (bz - z) * (cy - y)) / area
        w1 = ((cy - y) * (az - z) - (cz - z) * (ay - y)) / area
        w2 = 1.0 - w0 - w1
        hit = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        xh = w0 * ax_ + w1 * bx_ + w2 * cx_
        for r in range(hit.shape[0]):
            out[s + r] = np.sort(xh[r][hit[r]])
    return out


def _parity_inside(mesh, nx, ny, nz):
    """(nx,ny,nz) bool: node strictly inside by ray parity (jittered)."""
    tris = mesh.triangles.astype(np.float64)
    yy, zz = np.meshgrid(np.arange(ny, dtype=float),
                         np.arange(nz, dtype=float), indexing="ij")
    ys = yy.ravel() + _JIT
    zs = zz.ravel() + _JIT * np.sqrt(2)
    cross = _ray_crossings(tris, ys, zs)
    inside = np.zeros((nx, ny, nz), bool)
    xn = np.arange(nx, dtype=float)
    for r, xs in enumerate(cross):
        if xs is None or len(xs) == 0:
            continue
        j, k = divmod(r, nz)
        inside[:, j, k] = (np.searchsorted(xs, xn, side="left") % 2) == 1
    return inside


def _shell_promote(mesh, inside, eps_gu):
    """Promote outside nodes within eps_gu of the mesh surface (exact
    point-triangle distance), evaluated ONLY on the 1-node shell around the
    parity-inside set plus nodes flagged by proximity to any surface plane
    would be overkill — the shell is where the convention bites."""
    from scipy import ndimage
    if eps_gu <= 0:
        return inside
    out = inside.copy()
    # iterate to fixpoint: a promoted node extends the shell (nodes lying in
    # the sagitta band can be reachable only THROUGH other promoted nodes —
    # e.g. the quad's r0 node at a rod end-plane rim, whose 6-neighbors are
    # all in the band too). Typically converges in 2 passes.
    for _ in range(8):
        shell = ndimage.binary_dilation(out, iterations=1) & ~out
        if not out.any():
            lo = np.maximum(np.floor(mesh.bounds[0] - eps_gu - 1),
                            0).astype(int)
            hi = np.minimum(np.ceil(mesh.bounds[1] + eps_gu + 1),
                            np.array(inside.shape) - 1).astype(int)
            shell = np.zeros_like(inside)
            shell[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
        pts = np.argwhere(shell).astype(np.float64)
        if len(pts) == 0:
            break
        d = _dist_to_tris(mesh.triangles.astype(np.float64), pts)
        keep = d <= eps_gu + 1e-9
        if not keep.any():
            break
        ii = np.argwhere(shell)[keep]
        out[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return out


def _dist_to_tris(tris, pts, chunk=800):
    """Exact unsigned point-to-triangle-soup distance (numpy, chunked)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = (n * n).sum(1)
    nn[nn == 0] = 1.0
    best = np.full(len(pts), np.inf)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]                       # (P,3)
        ap = p[:, None, :] - a[None, :, :]         # (P,T,3)
        # project into triangle plane, barycentric clamp
        d1 = (ap * ab[None]).sum(-1)
        d2 = (ap * ac[None]).sum(-1)
        aa = (ab * ab).sum(1)[None]
        bb = (ac * ac).sum(1)[None]
        abac = (ab * ac).sum(1)[None]
        den = aa * bb - abac * abac
        den = np.where(np.abs(den) < 1e-30, 1.0, den)
        v = (bb * d1 - abac * d2) / den
        w = (aa * d2 - abac * d1) / den
        v = np.clip(v, 0, 1)
        w = np.clip(w, 0, 1)
        over = v + w > 1
        scale = np.where(over, v + w, 1.0)
        v, w = v / scale, w / scale
        q = (a[None] + v[..., None] * ab[None] + w[..., None] * ac[None])
        dv = p[:, None, :] - q
        d2min = (dv * dv).sum(-1).min(1)
        best[s:s + chunk] = np.sqrt(d2min)
    return best


# --------------------------------------------------------------- facade
def voxelize_meshes(meshes, grid, surface_eps_gu=0.0):
    """{index -> mesh (gu, full frame)} -> (labeled grid int16, eps used).
    Stored/folded frame (nodes at integer gu, origin on the mirror axes)."""
    g = grid
    lab = np.zeros((g.nx, g.ny, g.nz), np.int16)
    for idx, mesh in sorted(meshes.items()):
        inside = _parity_inside(mesh, g.nx, g.ny, g.nz)
        inside = _shell_promote(mesh, inside, surface_eps_gu)
        clash = inside & (lab != 0) & (lab != idx)
        if clash.any():
            i, j, k = [int(v[0]) for v in np.where(clash)]
            raise ValueError(f"electrode {idx} overlaps {int(lab[i, j, k])}"
                             f" at node ({i},{j},{k}) (+{int(clash.sum())})")
        lab[inside] = idx
    return lab


def voxelize_stls(paths, grid, surface_eps_gu=0.0):
    """{index -> STL path (mm, full frame)} -> labeled grid. STLs are
    converted mm -> gu with the grid's spacing; no primitive provenance, so
    the caller chooses eps (the STL's own tessellation sets the floor)."""
    meshes = {}
    for i, p in sorted(paths.items()):
        m = trimesh.load(p, force="mesh", process=True)
        m.apply_scale(1.0 / grid.mm_per_gu)
        meshes[i] = m
    return voxelize_meshes(meshes, grid, surface_eps_gu=surface_eps_gu)
