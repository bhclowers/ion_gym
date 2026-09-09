"""
ion_gym.multigrid3d
-------------------
Cascadic (nested-iteration) multigrid ACCELERATOR for solver3d.

Why cascadic, not a full V-cycle correction scheme: the final answer must be
byte-defensible as the fixed point of the already-validated solver. Cascadic
MG never touches the fine-grid operator — it solves the SAME Dirichlet
problem on a hierarchy of 2x-coarsened grids (coarsest solved outright, each
level's solution trilinearly prolongated up as the next level's initial
guess) and then hands phi0 to the validated `solve3d`, which converges with
its own stencil (fractional thetas, mirror/edge ghosts) to the same tol as a
cold solve. Convergence criterion, stencil, and boundary conventions are
therefore UNCHANGED; the hierarchy only supplies a good starting point, so
the result is identical-by-construction and the speedup is pure.

Coarsening is vertex-centred injection (coarse node = fine node at even
indices), which preserves the low-side mirror plane at index 0 and the
the edge-ghost convention at open faces. Dirichlet masks/values are
injected; a block-OR fallback keeps thin electrodes (< 2 fine nodes) present
on coarse levels — their values come from the strongest fixed node in the
2^3 block, which can only mis-shape the GUESS near sub-grid metal, never the
converged answer. Coarse levels use the plain 7-point stencil (theta=1):
fractional-leg detail is sub-grid there by definition and is restored by the
fine-level solve.

General to any grid/geometry — nothing example-specific.
"""

import numpy as np
import threading


# ---- cooperative solve interrupt (a long field solve can be cancelled
# without killing the process; the heavy inner sweeps are numba, which
# defers Python signals, so we check a flag at every V-cycle and between
# bases instead of relying on Ctrl-C reaching the njit loop) ----
class SolveInterrupted(Exception):
    """Raised by the solver when a stop was requested mid-solve."""


_STOP = threading.Event()


def request_stop():
    """Ask any in-flight solve to stop at the next cycle boundary."""
    _STOP.set()


def clear_stop():
    """Re-arm the solver after a stop (call before the next solve)."""
    _STOP.clear()

from ion_gym.physics.solver3d import solve3d, optimal_omega, _sweep


# ------------------------------------------------------------- coarsening
def _coarsen_mask(fixed, val):
    """Injection + thin-feature rescue. Coarse node c=(i,j,k) maps to fine
    f=(2i,2j,2k). fixed_c = fixed_f at the injection point, OR (if the
    injection point is vacuum) any-fixed within the fine 2x2x2 block whose
    low corner is f — with the value of the largest-|val| fixed node in the
    block, so sub-grid metal stays represented in the coarse guess."""
    nx, ny, nz = fixed.shape
    cx, cy, cz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
    fc = np.zeros((cx, cy, cz), bool)
    vc = np.zeros((cx, cy, cz))

    # injection
    fi = fixed[::2, ::2, ::2]
    vi = val[::2, ::2, ::2]
    fc[:fi.shape[0], :fi.shape[1], :fi.shape[2]] = fi
    vc[:vi.shape[0], :vi.shape[1], :vi.shape[2]] = vi

    # block-OR rescue for thin metal missed by injection
    # build block-any and block-max|val| by padding to even and pooling
    px = nx + (nx & 1)
    py = ny + (ny & 1)
    pz = nz + (nz & 1)
    fpad = np.zeros((px, py, pz), bool)
    fpad[:nx, :ny, :nz] = fixed
    vpad = np.zeros((px, py, pz))
    vpad[:nx, :ny, :nz] = np.where(fixed, val, 0.0)
    fblk = fpad.reshape(px // 2, 2, py // 2, 2, pz // 2, 2)
    vblk = vpad.reshape(px // 2, 2, py // 2, 2, pz // 2, 2)
    any_fixed = fblk.any(axis=(1, 3, 5))
    # value of the max-|val| fixed node in each block
    vmag = np.where(fblk, np.abs(vblk), -1.0).reshape(px // 2, 2, py // 2, 2,
                                                      pz // 2, 2)
    flatv = vblk.reshape(px // 2, py // 2, pz // 2, 8)
    flatm = vmag.transpose(0, 2, 4, 1, 3, 5).reshape(px // 2, py // 2,
                                                     pz // 2, 8)
    # careful: transpose both consistently
    flatv = vblk.transpose(0, 2, 4, 1, 3, 5).reshape(px // 2, py // 2,
                                                     pz // 2, 8)
    idx = np.argmax(flatm, axis=-1)
    vany = np.take_along_axis(flatv, idx[..., None], axis=-1)[..., 0]

    bx, by, bz = any_fixed.shape
    sub = (slice(0, min(cx, bx)), slice(0, min(cy, by)), slice(0, min(cz, bz)))
    rescue = any_fixed[sub] & ~fc[sub]
    fc_sub = fc[sub]
    vc_sub = vc[sub]
    fc_sub[rescue] = True
    vc_sub[rescue] = vany[sub][rescue]
    fc[sub] = fc_sub
    vc[sub] = vc_sub
    return fc, vc


def _prolongate(phic, shape_f):
    """Trilinear prolongation from the coarse grid (vertex-centred, coarse
    node c at fine 2c) to the fine grid shape. Pure numpy; clamps at the
    high edge when the fine axis is even (last fine node extrapolates by
    nearest)."""
    nx, ny, nz = shape_f
    # fine coordinates in coarse index units
    xf = np.minimum(np.arange(nx) / 2.0, phic.shape[0] - 1)
    yf = np.minimum(np.arange(ny) / 2.0, phic.shape[1] - 1)
    zf = np.minimum(np.arange(nz) / 2.0, phic.shape[2] - 1)
    x0 = np.floor(xf).astype(int)
    x1 = np.minimum(x0 + 1, phic.shape[0] - 1)
    y0 = np.floor(yf).astype(int)
    y1 = np.minimum(y0 + 1, phic.shape[1] - 1)
    z0 = np.floor(zf).astype(int)
    z1 = np.minimum(z0 + 1, phic.shape[2] - 1)
    tx = (xf - x0)[:, None, None]
    ty = (yf - y0)[None, :, None]
    tz = (zf - z0)[None, None, :]
    c000 = phic[np.ix_(x0, y0, z0)]
    c100 = phic[np.ix_(x1, y0, z0)]
    c010 = phic[np.ix_(x0, y1, z0)]
    c110 = phic[np.ix_(x1, y1, z0)]
    c001 = phic[np.ix_(x0, y0, z1)]
    c101 = phic[np.ix_(x1, y0, z1)]
    c011 = phic[np.ix_(x0, y1, z1)]
    c111 = phic[np.ix_(x1, y1, z1)]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


# ------------------------------------------------------------- driver
def solve3d_mg(fixed, val, theta=None, mirror=(False, False, False),
               tol=1e-6, max_sweeps=60000, verbose=False, stencil="ghost_linear",
               min_coarse=9, max_levels=8, level_tol_factor=30.0):
    """Cascadic-MG accelerated solve. Same signature contract as solve3d;
    returns (phi, fine_sweeps, final_delta). The FINE level is solved by the
    validated solve3d (same stencil/thetas/mirror/tol) from the prolongated
    guess, so the converged field is the identical fixed point of the plain
    solver — the hierarchy only accelerates.

    min_coarse: stop coarsening when any axis would drop below this.
    level_tol_factor: intermediate levels converge to tol*factor (they only
    seed the next level; over-solving them wastes sweeps)."""
    # build the hierarchy of (fixed, val), fine -> coarse
    levels = [(np.asarray(fixed, bool), np.asarray(val, np.float64))]
    while len(levels) < max_levels:
        f, v = levels[-1]
        if min(f.shape) <= min_coarse or min((s + 1) // 2
                                             for s in f.shape) < 3:
            break
        levels.append(_coarsen_mask(f, v))

    if verbose:
        print(f"[mg] {len(levels)} levels:",
              " <- ".join(str(f.shape) for f, _ in levels))

    # coarsest: cold solve with the plain 7-point stencil
    fc, vc = levels[-1]
    phic, swc, _ = solve3d(fc, vc, theta=None, mirror=mirror,
                           tol=tol, max_sweeps=max_sweeps, stencil=stencil)
    total_coarse = swc * np.prod(fc.shape)

    # walk up: prolongate, then relax to the loose level tol
    for lev in range(len(levels) - 2, 0, -1):
        f, v = levels[lev]
        phi0 = _prolongate(phic, f.shape)
        phi0[f] = v[f]
        phic, sw, _ = solve3d(f, v, theta=None, mirror=mirror,
                              tol=tol * level_tol_factor,
                              max_sweeps=max_sweeps, phi0=phi0,
                              stencil=stencil)
        total_coarse += sw * np.prod(f.shape)
        if verbose:
            print(f"[mg]   level {lev} {f.shape}: {sw} sweeps")

    # fine level: the VALIDATED solver, real thetas, full tolerance
    f, v = levels[0]
    phi0 = _prolongate(phic, f.shape) if len(levels) > 1 else None
    if phi0 is not None:
        phi0[f] = v[f]
    phi, sw, d = solve3d(f, v, theta=theta, mirror=mirror, tol=tol,
                         max_sweeps=max_sweeps, phi0=phi0, stencil=stencil,
                         verbose=verbose)
    if verbose:
        eq = total_coarse / np.prod(f.shape)
        print(f"[mg] fine: {sw} sweeps (+{eq:.0f} fine-equivalent coarse)")
    return phi, sw, d


def solve_bases_mg(masks, scene=None, mirror=(False, False, False),
                   v_basis=1e4, tol=1e-6, verbose=False, stencil="ghost_linear",
                   theta=None, only=None):
    """Drop-in accelerated counterpart of solver3d.solve_bases.

    Interruptible: raises SolveInterrupted between bases if request_stop()
    has been called (or on KeyboardInterrupt), so a long solve can be
    cancelled without killing the process.  Progress ('electrode i of N')
    prints when verbose."""
    from ion_gym.physics.solver3d import fractions_from_scene
    metal = np.zeros(next(iter(masks.values())).shape, bool)
    for m in masks.values():
        metal |= m
    th = theta
    if th is None and scene is not None:
        th = fractions_from_scene(scene, metal, mirror=mirror)
    out = {}
    items = sorted(masks.items())
    n = len(items)
    try:
        for j, (i, m) in enumerate(items, start=1):
            if only is not None and i not in only:
                continue          # boundaries (metal union) already include
                                  # EVERY electrode; skipping a basis solve
                                  # never changes another basis's field
            if _STOP.is_set():
                raise SolveInterrupted(
                    f"stopped after {j-1} of {n} electrodes")
            import time as _time
            _t0 = _time.time()
            print(f"  [{_time.strftime('%H:%M:%S')}] solving electrode "
                  f"{j} of {n} (basis {i}) ...", flush=True)
            val = np.zeros(metal.shape)
            val[m] = v_basis
            phi, sw, d = solve3d_vmg(metal, val, theta=th, mirror=mirror,
                                     tol=tol, stencil=stencil,
                                     verbose=verbose)
            print(f"  electrode {j} of {n}: {sw} sweeps, "
                  f"delta {d:.2e} V ({_time.time()-_t0:.0f}s)", flush=True)
            if d >= tol and not np.isnan(d):
                # defense in depth: solve3d_vmg now raises on every
                # non-converged path, so reaching here with d >= tol
                # means a future edit broke that contract — refuse to
                # bank an uncertified basis rather than cache it.
                from ion_gym.physics.solver3d import SolveNotConverged
                raise SolveNotConverged(
                    f"basis {i}: solver returned delta {d:.3e} V >= tol "
                    f"{tol:.1e} without raising — convergence contract "
                    f"broken upstream; refusing to cache.")
            out[i] = phi
    except KeyboardInterrupt:
        raise SolveInterrupted(f"interrupted during electrode solve "
                               f"(completed {len(out)} of {n})")
    return out


# ===================================================================
# TRUE V-CYCLE (coarse-grid correction) — the O(N) path.
# Fine smoothing & the final convergence check use the VALIDATED _sweep
# (real thetas, mirror/edge ghosts); coarse levels solve the error
# equation with the plain 7-point stencil. The V-cycle only steers the
# iterate; the answer is certified by the validated sweep's own
# max|delta| < tol.
# ===================================================================
from numba import njit, prange


@njit(cache=True, parallel=True, nogil=True)
def _residual(phi, fixed, th, ghost_linear_stencil, out):
    """r = num - den*phi with EXACTLY the weights _sweep uses (so the
    correction targets the same operator the smoother relaxes). r=0 at
    fixed nodes. Ghosts: even reflection (mirror/edge) as in _sweep."""
    nx, ny, nz = phi.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if fixed[i, j, k]:
                    out[i, j, k] = 0.0
                    continue
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
                txm = th[0, i, j, k]
                txp = th[1, i, j, k]
                tym = th[2, i, j, k]
                typ = th[3, i, j, k]
                tzm = th[4, i, j, k]
                tzp = th[5, i, j, k]
                if ghost_linear_stencil:
                    wxm = 1.0 / txm
                    wxp = 1.0 / txp
                    wym = 1.0 / tym
                    wyp = 1.0 / typ
                    wzm = 1.0 / tzm
                    wzp = 1.0 / tzp
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
                out[i, j, k] = num - den * phi[i, j, k]


@njit(cache=True, parallel=True, nogil=True)
def _sweep_rhs(e, fixed, f, omega, colour):
    """RB-GS sweep for the coarse ERROR equation  sum(nbr) - 6 e = f
    (plain 7-point; e=0 at fixed nodes; even-reflection ghosts). Returns
    max update."""
    nx, ny, nz = e.shape
    rowmax = np.zeros(nx)
    for i in prange(nx):
        dmax = 0.0
        for j in range(ny):
            k0 = (i + j + colour) & 1
            for k in range(k0, nz, 2):
                if fixed[i, j, k]:
                    continue
                vxm = e[i - 1, j, k] if i > 0 else (
                    e[1, j, k] if nx > 1 else e[i, j, k])
                vxp = e[i + 1, j, k] if i < nx - 1 else (
                    e[nx - 2, j, k] if nx > 1 else e[i, j, k])
                vym = e[i, j - 1, k] if j > 0 else (
                    e[i, 1, k] if ny > 1 else e[i, j, k])
                vyp = e[i, j + 1, k] if j < ny - 1 else (
                    e[i, ny - 2, k] if ny > 1 else e[i, j, k])
                vzm = e[i, j, k - 1] if k > 0 else (
                    e[i, j, 1] if nz > 1 else e[i, j, k])
                vzp = e[i, j, k + 1] if k < nz - 1 else (
                    e[i, j, nz - 2] if nz > 1 else e[i, j, k])
                new_val = (vxm + vxp + vym + vyp + vzm + vzp
                           - f[i, j, k]) / 6.0
                new = e[i, j, k] + omega * (new_val - e[i, j, k])
                d = abs(new - e[i, j, k])
                if d > dmax:
                    dmax = d
                e[i, j, k] = new
        rowmax[i] = dmax
    return rowmax.max()


@njit(cache=True, parallel=True, nogil=True)
def _residual_rhs(e, fixed, f, out):
    """Residual of the coarse error equation: r = (sum nbr - 6 e) - f."""
    nx, ny, nz = e.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if fixed[i, j, k]:
                    out[i, j, k] = 0.0
                    continue
                vxm = e[i - 1, j, k] if i > 0 else (
                    e[1, j, k] if nx > 1 else e[i, j, k])
                vxp = e[i + 1, j, k] if i < nx - 1 else (
                    e[nx - 2, j, k] if nx > 1 else e[i, j, k])
                vym = e[i, j - 1, k] if j > 0 else (
                    e[i, 1, k] if ny > 1 else e[i, j, k])
                vyp = e[i, j + 1, k] if j < ny - 1 else (
                    e[i, ny - 2, k] if ny > 1 else e[i, j, k])
                vzm = e[i, j, k - 1] if k > 0 else (
                    e[i, j, 1] if nz > 1 else e[i, j, k])
                vzp = e[i, j, k + 1] if k < nz - 1 else (
                    e[i, j, nz - 2] if nz > 1 else e[i, j, k])
                out[i, j, k] = (vxm + vxp + vym + vyp + vzm + vzp
                                - 6.0 * e[i, j, k]) - f[i, j, k]


@njit(cache=True, parallel=True, nogil=True)
def _restrict_fw_njit(fine, out):
    """Full-weighting restriction, fused njit (see _restrict_fw docstring).
    Edge handling = clamp (equivalent to edge-replicated pad)."""
    nx, ny, nz = fine.shape
    cx, cy, cz = out.shape
    for ic in prange(cx):
        i = 2 * ic
        for jc in range(cy):
            j = 2 * jc
            for kc in range(cz):
                k = 2 * kc
                acc = 0.0
                for di in range(-1, 2):
                    ii = i + di
                    if ii < 0:
                        ii = 0
                    if ii > nx - 1:
                        ii = nx - 1
                    wi = 1.0 if di == 0 else 0.5
                    for dj in range(-1, 2):
                        jj = j + dj
                        if jj < 0:
                            jj = 0
                        if jj > ny - 1:
                            jj = ny - 1
                        wj = 1.0 if dj == 0 else 0.5
                        for dk in range(-1, 2):
                            kk = k + dk
                            if kk < 0:
                                kk = 0
                            if kk > nz - 1:
                                kk = nz - 1
                            wk = 1.0 if dk == 0 else 0.5
                            acc += wi * wj * wk * fine[ii, jj, kk]
                out[ic, jc, kc] = acc / 8.0


@njit(cache=True, parallel=True, nogil=True)
def _prolong_add_njit(coarse, out, fixed):
    """Trilinear prolongation of `coarse` ADDED into `out` (skip fixed),
    fused njit — replaces _prolongate + corr[fixed]=0 + phi+=corr."""
    nx, ny, nz = out.shape
    cx, cy, cz = coarse.shape
    for i in prange(nx):
        xf = i * 0.5
        if xf > cx - 1:
            xf = cx - 1.0
        x0 = int(xf)
        if x0 > cx - 2:
            x0 = cx - 2 if cx > 1 else 0
        x1 = x0 + 1 if cx > 1 else x0
        tx = xf - x0
        for j in range(ny):
            yf = j * 0.5
            if yf > cy - 1:
                yf = cy - 1.0
            y0 = int(yf)
            if y0 > cy - 2:
                y0 = cy - 2 if cy > 1 else 0
            y1 = y0 + 1 if cy > 1 else y0
            ty = yf - y0
            for k in range(nz):
                if fixed[i, j, k]:
                    continue
                zf = k * 0.5
                if zf > cz - 1:
                    zf = cz - 1.0
                z0 = int(zf)
                if z0 > cz - 2:
                    z0 = cz - 2 if cz > 1 else 0
                z1 = z0 + 1 if cz > 1 else z0
                tz = zf - z0
                c00 = coarse[x0, y0, z0] * (1 - tx) + coarse[x1, y0, z0] * tx
                c10 = coarse[x0, y1, z0] * (1 - tx) + coarse[x1, y1, z0] * tx
                c01 = coarse[x0, y0, z1] * (1 - tx) + coarse[x1, y0, z1] * tx
                c11 = coarse[x0, y1, z1] * (1 - tx) + coarse[x1, y1, z1] * tx
                c0 = c00 * (1 - ty) + c10 * ty
                c1 = c01 * (1 - ty) + c11 * ty
                out[i, j, k] += c0 * (1 - tz) + c1 * tz


def _restrict(fine):
    """Injection restriction to the (n+1)//2 coarse grid (masks only)."""
    return np.ascontiguousarray(fine[::2, ::2, ::2])


def _coarsen_fixed_any(fixed):
    """Coarse Dirichlet mask for the ERROR equation: a coarse node is fixed
    if ANY fine node in its 2^3 block is. Thin electrodes (boards a few
    nodes thick) vanish under plain injection at deep levels, leaving the
    coarse error un-pinned exactly where the fine error is zero — the
    correction then injects long-range junk and the cycle AMPLIFIES. Block-
    ANY keeps all metal represented; over-constraining the error (e=0 on a
    slightly fattened mask) only weakens the correction locally, never
    destabilizes."""
    nx, ny, nz = fixed.shape
    px, py, pz = nx + (nx & 1), ny + (ny & 1), nz + (nz & 1)
    fp = np.zeros((px, py, pz), bool)
    fp[:nx, :ny, :nz] = fixed
    blk = fp.reshape(px // 2, 2, py // 2, 2, pz // 2, 2).any(axis=(1, 3, 5))
    cx, cy, cz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
    out = np.zeros((cx, cy, cz), bool)
    out[:min(cx, blk.shape[0]), :min(cy, blk.shape[1]),
        :min(cz, blk.shape[2])] = blk[:cx, :cy, :cz]
    # injection points are always included by block-ANY (they're in the block)
    return out


def _restrict_fw(fine):
    """FULL-WEIGHTING restriction (27-point, the adjoint of trilinear
    prolongation, normalized to transfer smooth fields exactly). REQUIRED
    for residuals: after smoothing the residual is concentrated in
    one-node-thick layers at Dirichlet surfaces, and plain injection of
    such a spiky source over-drives the coarse problem by ~2x (a nodal
    delta on the h_c=2h grid has twice the Green's-function response),
    which flips and amplifies the correction — the divergence seen in the
    first V-cycle attempt. Full weighting represents thin layers
    integrally and removes the factor exactly. Boundary: edge-replicated
    pad (consistent with the even-reflection ghost convention)."""
    p = np.pad(fine, 1, mode="edge")
    out = None
    for di, wi in ((-1, .5), (0, 1.), (1, .5)):
        for dj, wj in ((-1, .5), (0, 1.), (1, .5)):
            for dk, wk in ((-1, .5), (0, 1.), (1, .5)):
                s = p[1 + di:p.shape[0] - 1 + di,
                      1 + dj:p.shape[1] - 1 + dj,
                      1 + dk:p.shape[2] - 1 + dk][::2, ::2, ::2]
                term = (wi * wj * wk) * s
                out = term if out is None else out + term
    return np.ascontiguousarray(out / 8.0)


def _solve_err(fixed, f, rel_tol=0.02, max_cycles=30, nu1=2, nu2=2,
               min_coarse=9, _depth=0):
    """SOLVE the error equation  L e = f  (plain 7-point, e=0 at fixed) to a
    RELATIVE tolerance: iterate GS-RB smoothing + recursively-SOLVED coarse
    correction until this level's own sweep delta < rel_tol * scale(f).

    Design point (from the two-grid experiment on the tetramer): the
    two-grid iteration with an exact coarse solve contracts at ~0.4/cycle;
    the earlier divergence came from UNDER-SOLVED intermediate levels (a
    fixed nu of sweeps is a correction, not a solve, and its error
    amplifies up the hierarchy). Solving every level to a tolerance is the
    proven-stable structure; each level is 8x cheaper, so cost stays
    geometric. Bottom level: SOR with the anisotropic optimal omega (near-2
    omega is correct for a SOLVE, wrong for a smoother)."""
    e = np.zeros_like(f)
    scale = max(np.abs(f).max(), 1e-300)
    tol = rel_tol * scale / 6.0          # delta ~ residual/6 for this stencil
    if min(e.shape) <= min_coarse:
        om = optimal_omega(e.shape)
        for _ in range(20000):
            d1 = _sweep_rhs(e, fixed, f, om, 0)
            d2 = _sweep_rhs(e, fixed, f, om, 1)
            if max(d1, d2) < 0.02 * tol:  # bottom solved tight
                break
        return e
    fc = _coarsen_fixed_any(fixed)
    r = np.empty_like(e)
    for _ in range(max_cycles):
        for _s in range(nu1):
            _sweep_rhs(e, fixed, f, 1.0, 0)
            _sweep_rhs(e, fixed, f, 1.0, 1)
        _residual_rhs(e, fixed, f, r)
        cx, cy, cz = fc.shape
        rc = np.empty((cx, cy, cz))
        _restrict_fw_njit(r, rc)
        rc *= 4.0
        ec = _solve_err(fc, -rc, rel_tol, max_cycles, nu1, nu2,
                        min_coarse, _depth + 1)
        _prolong_add_njit(ec, e, fixed)
        d = 0.0
        for _s in range(nu2):
            d1 = _sweep_rhs(e, fixed, f, 1.0, 0)
            d2 = _sweep_rhs(e, fixed, f, 1.0, 1)
            d = max(d1, d2)
        if d < tol:
            break
    return e


def solve3d_vmg(fixed, val, theta=None, mirror=(False, False, False),
                tol=1e-6, max_cycles=60, nu1=3, nu2=3, verbose=False,
                stencil="ghost_linear", phi0=None,
                accept_stagnation=False):
    """V-cycle multigrid solve of Laplace with the validated fine stencil.

    Each cycle: nu1+nu2 validated `_sweep` pairs on the fine grid (real
    thetas, real ghosts) + a plain-stencil error V-cycle in between. The
    CONVERGENCE CRITERION is the validated sweep's own max|delta| < tol —
    identical to solve3d — so the accepted answer is a fixed point of the
    validated solver regardless of what the correction did.

    NOTE mirror caveat: coarse error levels use even-reflection ghosts at
    ALL open faces (same as _sweep), so mirror planes are consistent."""
    fixed = np.asarray(fixed, bool)
    nx, ny, nz = fixed.shape
    phi = (np.array(phi0, np.float64) if phi0 is not None
           else np.zeros((nx, ny, nz)))
    phi[fixed] = val[fixed]
    th = (np.ones((6, nx, ny, nz), np.float32) if theta is None
          else theta.astype(np.float32))
    # KNOWN STENCILS: no branch handles a retired
    # spelling. An unrecognised stencil is REFUSED and quoted back.
    if stencil not in ("ghost_linear", "sw"):
        raise ValueError(
            f"unknown stencil {stencil!r}; known stencils are "
            f"'ghost_linear' (linear ghost-value fractional surface) and "
            f"'sw' (Shortley-Weller).")
    sim = stencil == "ghost_linear"
    om = 1.0            # GS-RB smoother (see _vcycle_err); NOT near-2 SOR
    mx, my, mz = mirror
    # VERIFIED per-problem mirror symmetries (symmetry is opt-in and proven,
    # never inferred): axis a is symmetric iff fixed, val (and theta, when
    # supplied) are exactly invariant under np.flip along a. When proven, the
    # exact solution is symmetric, so projecting phi onto the symmetric
    # subspace each cycle removes ONLY error — in particular the asymmetric
    # mode that misaligned coarse grids inject (pair-coarsening cannot be
    # mirror-equivariant for every size/parity), which converges too slowly
    # to grind out otherwise. This is what lets the V-cycle meet structural-
    # symmetry gates (e.g. SLIM mid-gap Ey) that plain SOR meets by
    # construction.
    sym_axes = []
    if theta is None:
        for a in range(3):
            if fixed.shape[a] < 3:
                continue
            if (np.array_equal(fixed, np.flip(fixed, axis=a))
                    and np.array_equal(val, np.flip(val, axis=a))):
                sym_axes.append(a)
    r = np.empty_like(phi)
    sweeps = 0
    prev_cycle_d = np.inf
    for cyc in range(max_cycles):
        if _STOP.is_set():
            raise SolveInterrupted(
                f"stopped mid-solve at cycle {cyc} (delta "
                f"{prev_cycle_d:.2e} V)")
        phi_c0 = phi.copy()          # cycle-start snapshot for the criterion
        # pre-smooth with the VALIDATED kernel
        for _ in range(nu1):
            _sweep(phi, fixed, th, mx, my, mz, om, 0, sim)
            _sweep(phi, fixed, th, mx, my, mz, om, 1, sim)
            sweeps += 1
        # cooperative stop BETWEEN the smooths (Stop once took
        # ~15 s to register on a big grid — the per-CYCLE check meant a whole
        # cycle's sweeps + coarse solve had to finish first; this halves the
        # worst-case latency within a cycle).
        if _STOP.is_set():
            raise SolveInterrupted(
                f"stopped mid-cycle {cyc} (delta {prev_cycle_d:.2e} V)")
        # coarse-grid correction: coarse error equation SOLVED to rel tol
        _residual(phi, fixed, th, sim, r)
        r[fixed] = 0.0
        fcm = _coarsen_fixed_any(fixed)
        rc = np.empty(fcm.shape)
        _restrict_fw_njit(r, rc)
        rc *= 4.0
        ec = _solve_err(fcm, -rc, rel_tol=0.05, nu1=nu1, nu2=nu2)
        _prolong_add_njit(ec, phi, fixed)
        # post-smooth with the validated kernel
        d = 0.0
        for _ in range(nu2):
            d1 = _sweep(phi, fixed, th, mx, my, mz, om, 0, sim)
            d2 = _sweep(phi, fixed, th, mx, my, mz, om, 1, sim)
            d = max(d1, d2)
            sweeps += 1
        # project onto the verified symmetric subspace (removes only error)
        for a in sym_axes:
            np.add(phi, np.flip(phi, axis=a), out=phi)
            phi *= 0.5
        # CONVERGENCE: the FULL-CYCLE phi change. The per-sweep GS delta is
        # blind to smooth error (GS barely moves smooth modes per sweep, so
        # it collapses below tol while volts of smooth error remain — this
        # was the original criterion and it stopped cycles far too early,
        # leaving behind coarse-correction artifacts such as the SLIM
        # mid-gap asymmetry). The cycle delta INCLUDES the coarse
        # correction, which is precisely an estimate of the remaining
        # smooth error; with a healthy cycle convergence factor rho_c ~ 0.1
        # the remaining error is ~ cycle_d * rho_c/(1-rho_c) ~ cycle_d/9,
        # i.e. this criterion is strictly tighter than SOR's per-sweep one.
        cycle_d = float(np.abs(phi - phi_c0).max())
        if verbose:
            print(f"[vmg] cycle {cyc}: cycle delta {cycle_d:.3e} "
                  f"(sweep {d:.3e})")
        if cycle_d < tol:
            return phi, sweeps, cycle_d
        if cycle_d >= prev_cycle_d:
            # stagnated (precision floor of the float32 transfer operators)
            if accept_stagnation:
                # EXPLICIT opt-in (exploratory use): the caller stated in
                # code that a stalled iterate is acceptable; the achieved
                # delta still returns so it can be quoted.
                return phi, sweeps, cycle_d
            from ion_gym.physics.solver3d import SolveNotConverged
            raise SolveNotConverged(
                f"V-cycle STAGNATED at cycle {cyc}: delta {cycle_d:.3e} V "
                f"stopped improving (previous {prev_cycle_d:.3e}) before "
                f"reaching tol {tol:.1e}. A stalled iterate is not a "
                f"converged field (2026-09-05: external review found this "
                f"path returned silently, contradicting V01's never-silent "
                f"contract). Loosen tol toward the achieved delta, refine "
                f"the float32 transfer floor, or pass "
                f"accept_stagnation=True to knowingly take the iterate.")
        prev_cycle_d = cycle_d
    from ion_gym.physics.solver3d import SolveNotConverged
    raise SolveNotConverged(
        f"V-cycle exhausted max_cycles={max_cycles} at delta "
        f"{cycle_d:.3e} V without reaching tol {tol:.1e}. An "
        f"iteration-capped iterate is not a converged field; raise "
        f"max_cycles or loosen tol.")
