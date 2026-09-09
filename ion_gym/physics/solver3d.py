"""
ion_gym.solver3d
----------------
Native 3-D finite-difference Laplace solver (planar symmetry), the
producer-side milestone that makes the gym self-sufficient: solve fields
for GeomScene geometries with no external field solver in the loop.

Conventions inherited from the validated consume side (do not drift):
  * Outer open edges: linear ghost-value convention phi_ghost = phi_edge
    (empirically matched to refine at 1e-6 relative on the einzel;
    solver2d edge_ghost='ghost_linear').
  * Mirror planes (opt-in per axis, LOW side): potential EVEN across the
    plane -> ghost(-1) = phi(+1).
  * Electrode nodes are Dirichlet; fast-adjust bases solve one electrode
    at 10 kV, the rest at 0 (superposition downstream is phi = sum V_i *
    pa_i / 1e4, banked).

surface=fractional == Shortley-Weller unequal-leg stencils. For a vacuum
node whose neighbour in direction d is metal, the ideal surface lies at
theta*h into the gap (0 < theta <= 1); the 1-D second difference on legs
(theta_m, theta_p) is
    phi_xx ~ 2 [ V_m/(th_m(th_m+th_p)) + V_p/(th_p(th_m+th_p))
                 - phi_P/(th_m th_p) ] / h^2.
KEY SIMPLIFICATION: the surface potential V_d equals the Dirichlet value
of the metal node behind it (in a basis solve, 10 kV or 0), so the kernel
needs only the theta arrays — the neighbour gather is unchanged. Regular
legs have theta = 1 and reduce to the standard 7-point star. h^2 cancels.
Thetas are computed by bisection on the EXACT GeomScene inside() test
(geometry.py analytic solids — the same surfaces proven byte-identical to
an independent field calculation, in Rung 1), clamped at theta >= 1e-3.

Solver: numba red-black SOR, omega = 2/(1 + sin(pi/(N_max+1))), converged
when the sweep max |delta| < tol (default 1e-6 V).
"""

import numpy as np
from numba import njit, prange

from ion_gym.physics.scene3d import GeomScene, Cylinder

EPS = 1e-9


# ------------------------------------------------------------ inside test
def inside_scene_pts(scene: GeomScene, pts):
    """Exact solid test for points (N,3) in gu, full analytic frame; union
    over electrodes of (within minus strict-interior notin) — the same
    semantics the masks were proven against (notin_inside: points ON a
    hole surface remain metal)."""
    sc = scene.in_gu()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    total = np.zeros(len(pts), bool)

    def pmask(p, strict):
        e = -EPS if strict else EPS
        if isinstance(p, Cylinder):
            return (((x - p.cx) ** 2 + (y - p.cy) ** 2 <= p.r ** 2 + e)
                    & (z >= p.z - p.length - e) & (z <= p.z + e))
        return ((x >= p.x1 - e) & (x <= p.x2 + e)
                & (y >= p.y1 - e) & (y <= p.y2 + e)
                & (z >= p.z1 - e) & (z <= p.z2 + e))

    for el in sc.electrodes:
        for sh in el.shapes:
            w = np.zeros(len(pts), bool)
            for p in sh.within:
                w |= pmask(p, strict=False)
            for p in sh.notin:
                w &= ~pmask(p, strict=True)
            total |= w
    return total


def fractions_from_scene(scene: GeomScene, metal, mirror=(False, False, False),
                         iters=45):
    """theta arrays (6, nx, ny, nz) float32, order (-x,+x,-y,+y,-z,+z).
    theta[d] < 1 only at VACUUM nodes whose d-neighbour is metal: the
    fraction of h from the node to the exact surface, by bisection on
    inside_scene_pts. Mirror-low axes reflect geometry, so a node at index
    0 looking across the mirror sees the mirrored solid — handled by
    reflecting the probe coordinate."""
    nx, ny, nz = metal.shape
    th = np.ones((6, nx, ny, nz), np.float32)
    dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)]
    vac = ~metal
    for d, (dx, dy, dz) in enumerate(dirs):
        nb = np.zeros_like(metal)
        src = metal
        # neighbour-is-metal mask, honouring mirror reflection at index 0
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        ax = 0 if dx else (1 if dy else 2)
        s = (dx + dy + dz)
        if s < 0:
            sl_a[ax] = slice(1, None)
            sl_b[ax] = slice(0, -1)
            nb[tuple(sl_a)] = src[tuple(sl_b)]
            if mirror[ax]:
                sl0 = [slice(None)] * 3
                sl1 = [slice(None)] * 3
                sl0[ax] = 0
                sl1[ax] = 1
                nb[tuple(sl0)] = src[tuple(sl1)]
        else:
            sl_a[ax] = slice(0, -1)
            sl_b[ax] = slice(1, None)
            nb[tuple(sl_a)] = src[tuple(sl_b)]
        pairs = np.argwhere(vac & nb)
        if len(pairs) == 0:
            continue
        p0 = pairs.astype(np.float64)
        dvec = np.array([dx, dy, dz], float)
        lo = np.zeros(len(p0))
        hi = np.ones(len(p0))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            probe = p0 + mid[:, None] * dvec
            if mirror[ax] and s < 0:
                probe[:, ax] = np.abs(probe[:, ax])   # reflect across plane
            ins = inside_scene_pts(scene, probe)
            hi = np.where(ins, mid, hi)
            lo = np.where(ins, lo, mid)
        theta = np.clip(0.5 * (lo + hi), 1e-3, 1.0).astype(np.float32)
        th[d][pairs[:, 0], pairs[:, 1], pairs[:, 2]] = theta
    return th


# ------------------------------------------------------------------ SOR
@njit(cache=True, parallel=True, nogil=True)
def _sweep(phi, fixed, th, mirror_x, mirror_y, mirror_z, omega, colour,
           ghost_linear_stencil):
    nx, ny, nz = phi.shape
    rowmax = np.zeros(nx)          # per-row maxima: prange-safe reduction
    for i in prange(nx):
        dmax = 0.0
        for j in range(ny):
            k0 = (i + j + colour) & 1
            for k in range(k0, nz, 2):
                if fixed[i, j, k]:
                    continue
                # neighbour values with mirror / linear-ghost edges
                # ghosts: EVEN reflection at mirror planes AND at open
                # outer edges — pinned empirically in 3-D: the
                # refined array satisfies ghost=inner to machine zero at
                # regular boundary nodes, while the 2-D-banked ghost=edge
                # leaves 30-40 V residuals. (2-D einzel keeps its own
                # validated convention; the two differ by dimension.)
                # Edge ghosts reflect to the first interior node. Guard the
                # DEGENERATE case (a size-1 axis): there is no interior to
                # reflect to, so the ghost is the node itself — this makes a
                # size-1 axis a true translational-invariant (2-D) limit
                # rather than reading out of range. (A size-1 z-axis at
                # nz=1 previously read phi[i,j,1] out of bounds, breaking
                # x/y symmetry of the whole solve.)
                vxm = phi[i - 1, j, k] if i > 0 else (
                    phi[1, j, k] if nx > 1 else phi[i, j, k])
                vxp = phi[i + 1, j, k] if i < nx - 1 else (
                    phi[nx - 2, j, k] if nx > 1 else phi[i, j, k])
                vym = phi[i, j - 1, k] if j > 0 else (
                    phi[i, 1, k] if ny > 1 else phi[i, j, k])
                vyp = phi[i, j + 1, k] if j < ny - 1 else (
                    phi[i, ny - 2, k] if ny > 1 else phi[i, j, k])
                vzm = phi[i, j, k - 1] if k > 0 else (
                    phi[i, j, 1] if nz > 1 else phi[i, j, k])
                vzp = phi[i, j, k + 1] if k < nz - 1 else (
                    phi[i, j, nz - 2] if nz > 1 else phi[i, j, k])

                txm = th[0, i, j, k]; txp = th[1, i, j, k]
                tym = th[2, i, j, k]; typ = th[3, i, j, k]
                tzm = th[4, i, j, k]; tzp = th[5, i, j, k]

                if ghost_linear_stencil:
                    # surface=fractional == linear ghost-value:
                    # weight 1/theta on the short-leg neighbour (whose node
                    # value IS the surface potential), 1 elsewhere. a(theta)=1/theta, c=1 across the full theta range.
                    wxm = 1.0 / txm; wxp = 1.0 / txp
                    wym = 1.0 / tym; wyp = 1.0 / typ
                    wzm = 1.0 / tzm; wzp = 1.0 / tzp
                    num = (vxm * wxm + vxp * wxp + vym * wym + vyp * wyp
                           + vzm * wzm + vzp * wzp)
                    den = wxm + wxp + wym + wyp + wzm + wzp
                else:
                    num = (vxm / (txm * (txm + txp))
                           + vxp / (txp * (txm + txp))
                           + vym / (tym * (tym + typ))
                           + vyp / (typ * (tym + typ))
                           + vzm / (tzm * (tzm + tzp))
                           + vzp / (tzp * (tzm + tzp)))
                    den = (1.0 / (txm * txp) + 1.0 / (tym * typ)
                           + 1.0 / (tzm * tzp))
                new = phi[i, j, k] + omega * (num / den - phi[i, j, k])
                d = abs(new - phi[i, j, k])
                if d > dmax:
                    dmax = d
                phi[i, j, k] = new
        rowmax[i] = dmax
    return rowmax.max()


def optimal_omega(shape, spacing=None):
    """Optimal SOR relaxation factor from Young's theory, for the 7-point
    Laplacian on a (possibly anisotropic) rectangular grid.

        rho_Jacobi = sum_a w_a cos(pi/(n_a+1)) / sum_a w_a,   w_a = 1/h_a^2
        omega_opt  = 2 / (1 + sqrt(1 - rho^2))

    This is geometry-general: it uses ALL grid dimensions (and, if given, the
    per-axis spacings), so a long thin domain gets its correct omega instead of
    the value for a cube of side max(shape). Reduces EXACTLY to the previous
    max-dimension formula when the grid is cubic and isotropic, so cubic solves
    are unchanged. Mirror/Neumann faces and interior electrodes perturb the
    true optimum slightly; this analytic estimate stays near-optimal and only
    ever costs a few extra sweeps, never correctness (SOR converges to the same
    solution for any 0<omega<2)."""
    # A dimension with < 2 interior-coupled nodes (e.g. nz=1 for a 2-D
    # problem embedded in the 3-D solver) has NO coupling along that axis and
    # must be excluded — clamping it to 2 injected a spurious cos(pi/3)=0.5
    # term that pulled rho (and omega) far too low, making 2-D solves crawl.
    if spacing is None:
        dims = [(int(n), 1.0) for n in shape if int(n) >= 2]
    else:
        dims = [(int(n), 1.0 / (float(s) * float(s)))
                for n, s in zip(shape, spacing) if int(n) >= 2]
    if not dims:
        return 1.0
    num = sum(wi * np.cos(np.pi / (ni + 1)) for ni, wi in dims)
    rho = num / sum(wi for _, wi in dims)
    rho = min(rho, 1.0 - 1e-15)
    return 2.0 / (1.0 + np.sqrt(1.0 - rho * rho))


class SolveNotConverged(RuntimeError):
    """The iteration hit its sweep cap without meeting tol.

    This exists because solve3d used to just `return phi, max_sweeps, d` when it
    ran out of sweeps -- handing back an UNCONVERGED field, with no exception and
    no warning, indistinguishable from a converged one.  On the einzel geometry
    (520 x 80, long and thin -- precisely where SOR's convergence collapses) it
    burned all 60,000 sweeps and returned a field with a 3.2% error while
    reporting delta = 1.96e-02 against a tol of 1e-04 that nobody checked.  The
    `tol` argument was decorative.

    A solver that cannot meet its tolerance must SAY SO (a loud failure
    costs an hour, a quiet wrong number costs a paper).  Multigrid exists for
    exactly this geometry; the answer is to use it, not to accept SOR's stall.
    """


def solve3d(fixed, val, theta=None, mirror=(False, False, False),
            tol=1e-6, max_sweeps=60000, omega=None, phi0=None,
            verbose=False, stencil="ghost_linear"):
    """stencil='ghost_linear' (default): the linear ghost-value fractional
    surface stencil -- a missing neighbour is replaced by a linear
    extrapolation through the surface fraction theta.
    'sw': classic Shortley-Weller (higher-order; use for native-quality
    solves).

    Solve Laplace: fixed (bool Dirichlet mask), val (potentials at fixed
    nodes), theta (6,nx,ny,nz) or None for pure node-Dirichlet. Returns
    (phi, sweeps, final_delta)."""
    nx, ny, nz = fixed.shape
    phi = (np.array(phi0, np.float64) if phi0 is not None
           else np.zeros((nx, ny, nz)))
    phi[fixed] = val[fixed]
    th = (np.ones((6, nx, ny, nz), np.float32) if theta is None
          else theta.astype(np.float32))
    if omega is None:
        omega = optimal_omega((nx, ny, nz))
    mx, my, mz = mirror
    # KNOWN STENCILS: no branch handles a retired
    # spelling. An unrecognised stencil is REFUSED and quoted back.
    if stencil not in ("ghost_linear", "sw"):
        raise ValueError(
            f"unknown stencil {stencil!r}; known stencils are "
            f"'ghost_linear' (linear ghost-value fractional surface) and "
            f"'sw' (Shortley-Weller).")
    sim = stencil == "ghost_linear"
    for s in range(max_sweeps):
        d1 = _sweep(phi, fixed, th, mx, my, mz, omega, 0, sim)
        d2 = _sweep(phi, fixed, th, mx, my, mz, omega, 1, sim)
        d = max(d1, d2)
        if verbose and s % 500 == 0:
            print(f"  sweep {s}: max|d| {d:.3e}")
        if d < tol:
            return phi, s + 1, d
    raise SolveNotConverged(
        f"SOR hit its cap of {max_sweeps} sweeps on a {nx}x{ny}x{nz} grid "
        f"without reaching tol: final delta {d:.3e} vs tol {tol:.3e} "
        f"({d/max(tol, 1e-30):.0f}x too loose). This USED TO RETURN THE "
        f"UNCONVERGED FIELD SILENTLY. SOR stalls on long thin domains; solve "
        f"this geometry with multigrid3d.solve_bases_mg, or raise max_sweeps "
        f"if you have reason to think it will actually converge.")


def solve_bases(masks, scene=None, mirror=(False, False, False),
                v_basis=1e4, tol=1e-6, verbose=False, stencil="ghost_linear",
                omega=None, only=None):
    """Fast-adjust basis solves: {index: mask} -> {index: phi} with
    electrode i at v_basis and all other metal at 0. If scene is given,
    Shortley-Weller thetas are computed once from its exact solids."""
    metal = np.zeros(next(iter(masks.values())).shape, bool)
    for m in masks.values():
        metal |= m
    th = (fractions_from_scene(scene, metal, mirror=mirror)
          if scene is not None else None)
    out = {}
    for i, m in sorted(masks.items()):
        if only is not None and i not in only:
            continue          # metal union above already includes EVERY
                              # electrode as a boundary; skipping a basis
                              # solve never changes another basis's field
        val = np.zeros(metal.shape)
        val[m] = v_basis
        phi, sw, d = solve3d(metal, val, theta=th, mirror=mirror, tol=tol,
                             stencil=stencil, omega=omega)
        if verbose:
            print(f"basis {i}: {sw} sweeps, final delta {d:.2e} V")
        out[i] = phi
    return out


def fractions_from_pa_surf(path, shape, metal):
    """theta (6,nx,ny,nz) decoded from the imported surface-fraction sidecar —
    use when the goal is matching an external solve bit-for-bit-in-convention
    (its 1/1024-quantized fractions, amplified by the 1/theta ghost weight,
    otherwise dominate the residual near the smallest gaps).

    Format (pinned empirically, quad_monolithic article): 'SIMEXP' file,
    chunks tagged XOFF/YOFF/ZOFF; records (u64 flat node index, f64
    fraction), 0xFFFFFFFF-prefixed terminator record; fraction = distance
    (in gu) from the record's node to the surface along the POSITIVE axis
    of the chunk, one record per metal-vacuum gap (stored at the low node
    of the gap, vacuum or metal). Node flat order: x fastest
    (i + nx*j + nx*ny*k). Empty ZOFF == all z-surfaces on node planes.
    """
    import re as _re
    nx, ny, nz = shape
    th = np.ones((6, nx, ny, nz), np.float32)
    raw = open(path, "rb").read()
    tags = [(m.start(), m.group().decode())
            for m in _re.finditer(rb"XOFF|YOFF|ZOFF", raw)]
    axmap = {"XOFF": 0, "YOFF": 1, "ZOFF": 2}
    dpos = {0: 1, 1: 3, 2: 5}
    dneg = {0: 0, 1: 2, 2: 4}
    for n, (pos, tag) in enumerate(tags):
        end = tags[n + 1][0] if n + 1 < len(tags) else len(raw)
        a = np.frombuffer(raw[pos + 4:pos + 4 + ((end - pos - 4)//16)*16],
                          dtype=[("idx", "<u8"), ("f", "<f8")])
        a = a[a["idx"] < nx * ny * nz]
        ax = axmap[tag]
        i = (a["idx"] % nx).astype(int)
        j = ((a["idx"] // nx) % ny).astype(int)
        k = (a["idx"] // (nx * ny)).astype(int)
        for ii, jj, kk, f in zip(i, j, k, a["f"]):
            if not metal[ii, jj, kk]:
                th[dpos[ax], ii, jj, kk] = max(f, 1e-3)
            else:
                nb = [ii, jj, kk]
                nb[ax] += 1
                if nb[ax] < shape[ax] and not metal[nb[0], nb[1], nb[2]]:
                    th[dneg[ax], nb[0], nb[1], nb[2]] = max(1.0 - f, 1e-3)
    return th


@njit(cache=True, parallel=True, nogil=True)
def _residual_kernel(u, fixed, th):
    nx, ny, nz = u.shape
    r = np.zeros_like(u)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if fixed[i, j, k]:
                    continue
                vxm = u[i-1, j, k] if i > 0 else u[1, j, k]
                vxp = u[i+1, j, k] if i < nx-1 else u[nx-2, j, k]
                vym = u[i, j-1, k] if j > 0 else u[i, 1, k]
                vyp = u[i, j+1, k] if j < ny-1 else u[i, ny-2, k]
                vzm = u[i, j, k-1] if k > 0 else u[i, j, 1]
                vzp = u[i, j, k+1] if k < nz-1 else u[i, j, nz-2]
                w0 = 1/th[0, i, j, k]; w1 = 1/th[1, i, j, k]
                w2 = 1/th[2, i, j, k]; w3 = 1/th[3, i, j, k]
                w4 = 1/th[4, i, j, k]; w5 = 1/th[5, i, j, k]
                num = vxm*w0 + vxp*w1 + vym*w2 + vyp*w3 + vzm*w4 + vzp*w5
                r[i, j, k] = num/(w0+w1+w2+w3+w4+w5) - u[i, j, k]
    return r


@njit(cache=True, parallel=True, nogil=True)
def _err_sweep(e, fixed, th, rhs, omega, colour):
    nx, ny, nz = e.shape
    rowmax = np.zeros(nx)
    for i in prange(nx):
        dmax = 0.0
        for j in range(ny):
            k0 = (i + j + colour) & 1
            for k in range(k0, nz, 2):
                if fixed[i, j, k]:
                    continue
                vxm = e[i-1, j, k] if i > 0 else e[1, j, k]
                vxp = e[i+1, j, k] if i < nx-1 else e[nx-2, j, k]
                vym = e[i, j-1, k] if j > 0 else e[i, 1, k]
                vyp = e[i, j+1, k] if j < ny-1 else e[i, ny-2, k]
                vzm = e[i, j, k-1] if k > 0 else e[i, j, 1]
                vzp = e[i, j, k+1] if k < nz-1 else e[i, j, nz-2]
                w0 = 1/th[0, i, j, k]; w1 = 1/th[1, i, j, k]
                w2 = 1/th[2, i, j, k]; w3 = 1/th[3, i, j, k]
                w4 = 1/th[4, i, j, k]; w5 = 1/th[5, i, j, k]
                num = vxm*w0 + vxp*w1 + vym*w2 + vyp*w3 + vzm*w4 + vzp*w5
                new = e[i, j, k] + omega*((num/(w0+w1+w2+w3+w4+w5)
                                           + rhs[i, j, k]) - e[i, j, k])
                d = abs(new - e[i, j, k])
                if d > dmax:
                    dmax = d
                e[i, j, k] = new
        rowmax[i] = dmax
    return rowmax.max()


def residual_field(u, fixed, theta):
    """Residual of u under the pinned ghost_linear stencil with mirror ghosts —
    the convergence audit: an accelerated solve leaves its objective-level
    residual here; a converged native solve leaves ~tol."""
    return _residual_kernel(u, fixed, theta.astype(np.float32))


def attribute_difference(ref, phi, fixed, theta, tol=1e-7,
                         max_sweeps=30000):
    """Solve the error equation for the difference (ref - phi) induced
    purely by ref's non-zero residual: (I - M) diff = -r  =>  sweep with
    rhs = -r. Returns (explained_diff, unexplained = (ref-phi) -
    explained). If unexplained ~ uV, the entire disagreement is ref's
    convergence floor — i.e. both solutions solve the SAME discretization
    and ours is the more converged one."""
    r = residual_field(ref, fixed, theta)
    e = np.zeros_like(ref)
    th32 = theta.astype(np.float32)
    om = 2.0/(1.0 + np.sin(np.pi/(max(ref.shape)+1)))
    for _ in range(max_sweeps):
        d = max(_err_sweep(e, fixed, th32, -r, om, 0),
                _err_sweep(e, fixed, th32, -r, om, 1))
        if d < tol:
            break
    return e, (ref - phi) - e


def fractions_from_mesh(meshes, metal, mirror=(False, False, False),
                        slack=0.05):
    """theta (6,nx,ny,nz) from watertight electrode meshes (trimesh, GRID
    UNITS, full analytic frame): for each vacuum node with a metal
    d-neighbour, cast a ray along d (embree) and take the first surface
    hit within the gap. This is the mesh-native leg path — CadQuery /
    STL geometry becomes first-class in the native solver.

    Notes pinned to the occupancy contract (voxelize.py):
      * masks are surface-inclusive with eps = tessellation sagitta, so a
        promoted metal node can sit up to eps OUTSIDE the facets; a ray
        toward it may hit slightly beyond the gap or not at all -> theta
        = 1 (Dirichlet at the full leg; <= eps geometry error, exactly
        the contract's bound).
      * mirror axes need NO reflection: full-frame meshes contain the
        real negative-side geometry, and fold solves are only legitimate
        for symmetric scenes (certified per-problem by the S1-c engine),
        where the two are identical.
      * theta clamped to [1e-3, 1].
    """
    import trimesh
    mesh = trimesh.util.concatenate(list(meshes.values()))
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError:
        ray = mesh.ray            # embree absent: slower, same intersections
    else:
        ray = RayMeshIntersector(mesh)
    nx, ny, nz = metal.shape
    th = np.ones((6, nx, ny, nz), np.float32)
    vac = ~metal
    dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)]
    for d, dv in enumerate(dirs):
        ax = 0 if dv[0] else (1 if dv[1] else 2)
        s = sum(dv)
        nb = np.zeros_like(metal)
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        if s < 0:
            sl_a[ax] = slice(1, None)
            sl_b[ax] = slice(0, -1)
            nb[tuple(sl_a)] = metal[tuple(sl_b)]
            if mirror[ax]:
                sl0 = [slice(None)] * 3
                sl1 = [slice(None)] * 3
                sl0[ax] = 0
                sl1[ax] = 1
                nb[tuple(sl0)] = metal[tuple(sl1)]
        else:
            sl_a[ax] = slice(0, -1)
            sl_b[ax] = slice(1, None)
            nb[tuple(sl_a)] = metal[tuple(sl_b)]
        pairs = np.argwhere(vac & nb)
        if len(pairs) == 0:
            continue
        origins = pairs.astype(np.float64)
        dvec = np.tile(np.array(dv, float), (len(pairs), 1))
        loc, iray, _ = ray.intersects_location(origins, dvec,
                                               multiple_hits=False)
        if len(iray) == 0:
            continue
        t = np.abs(loc[:, ax] - origins[iray, ax])
        good = t <= 1.0 + slack
        theta = np.clip(t[good], 1e-3, 1.0).astype(np.float32)
        pg = pairs[iray[good]]
        th[d][pg[:, 0], pg[:, 1], pg[:, 2]] = theta
    return th
