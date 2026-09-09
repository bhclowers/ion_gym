"""
ion_gym.solver2d
----------------
2-D finite-difference Laplace solver on a uniform grid, supporting BOTH
symmetries 2-D field grids use:

    symmetry="planar"      : phi_zz + phi_uu = 0
    symmetry="cylindrical" : phi_zz + phi_rr + (1/r) phi_r = 0   (u = r >= 0)

Performance-fork ancestor of the solve in anamorphic_lens_sim.py; structured
so assembly can later be jitted and the solved array memmapped without
changing the interface.

Grid: phi shape (nz, nu); Z axis 0, U axis 1. Cylindrical grids start at the
axis: U[0] == 0.

Stencils (spacing h both directions):
  planar interior : centre -4, all four neighbours 1
  cyl interior    : centre -4, z-nbrs 1, r+ (1 + h/2r), r- (1 - h/2r)
  cyl axis (r=0)  : L'Hopital: lap = phi_zz + 2 phi_rr, phi_r(0)=0
                    -> centre -6, z-nbrs 1, first off-axis node 4
Open edges: Neumann mirror — a missing neighbour's coefficient folds onto the
opposite (existing) neighbour.
"""

import numpy as np
from scipy.sparse import coo_matrix


def assemble_laplace(fixed, h, symmetry="planar",
                     neumann_edges=("z0", "z1", "u0", "u1"),
                     edge_ghost="ghost_linear"):
    """edge_ghost: discrete treatment of the missing neighbour at open edges.
      'ghost_linear'     : phi_ghost = phi_edge  (fold coef onto the CENTRE node).
                 Empirically matched to an external solve at 1e-6 relative on the
                 einzel basis cross-check -- use for all external comparisons.
      'mirror' : phi_ghost = phi_inner (fold onto the OPPOSITE neighbour) --
                 the standard second-order symmetric Neumann. This, NOT
                 ghost_linear, is the even-reflection BC a symmetry FOLD
                 plane requires: on a discretely symmetric problem
                 phi(plane-1) == phi(plane+1) != phi(plane), so
                 ghost_linear at a fold plane halves the transverse
                 second difference — an O(h^2 * phi_perp'') boundary
                 error that scales with field CURVATURE at the plane
                 (measured 38 V on a real MRT median plane;
                 invisible on curvature-flat planes, which
                 is how it survived certification).
    The two differ at the edge at O(h); interior stencils are identical.

    edge_ghost may be a single kind (applied to every open edge — the
    historical behaviour, byte-identical) or a mapping {edge: kind}
    naming a kind PER open edge, e.g. {"u0": "mirror"} for a solve
    folded about the low-u plane. Unnamed edges default to
    'ghost_linear'; unknown edge names or kinds refuse loudly."""
    _KINDS = ("ghost_linear", "mirror")
    _EDGES = ("z0", "z1", "u0", "u1")
    if isinstance(edge_ghost, str):
        if edge_ghost not in _KINDS:
            raise ValueError(f"edge_ghost must be one of {_KINDS} or a "
                             f"{{edge: kind}} mapping, got {edge_ghost!r}")
        ghost_of = dict.fromkeys(_EDGES, edge_ghost)
    else:
        bad_e = [e for e in edge_ghost if e not in _EDGES]
        bad_k = [k for k in edge_ghost.values() if k not in _KINDS]
        if bad_e or bad_k:
            raise ValueError(
                f"edge_ghost mapping: unknown edge(s) {bad_e} / "
                f"kind(s) {bad_k}; edges are {_EDGES}, kinds {_KINDS}")
        ghost_of = dict.fromkeys(_EDGES, "ghost_linear")
        ghost_of.update(edge_ghost)
    nz, nu = fixed.shape
    if symmetry == "cylindrical" and "u0" in neumann_edges:
        neumann_edges = tuple(e for e in neumann_edges if e != "u0")

    idx = np.arange(nz * nu).reshape(nz, nu)
    rows, cols, data = [], [], []

    fn = idx[fixed]
    rows.append(fn); cols.append(fn); data.append(np.ones(fn.size))

    fi, fj = np.where(~fixed)
    k = idx[fi, fj]
    axis = (fj == 0) if symmetry == "cylindrical" else np.zeros(k.size, bool)

    # per-node neighbour coefficients
    if symmetry == "cylindrical":
        r = fj * h
        safe_r = np.where(fj == 0, 1.0, r)
        c_up = np.where(axis, 4.0, 1.0 + h / (2.0 * safe_r))   # u+ (r+)
        c_dn = np.where(axis, 0.0, 1.0 - h / (2.0 * safe_r))   # u- (r-)
        centre = np.where(axis, -6.0, -4.0)
    else:
        c_up = np.ones(k.size); c_dn = np.ones(k.size)
        centre = np.full(k.size, -4.0)
    c_zm = np.ones(k.size); c_zp = np.ones(k.size)

    rows.append(k); cols.append(k); data.append(centre)

    def link(coef, exists, tgt_i, tgt_j, mirror_i, mirror_j, edge):
        """Add coef to (tgt) where neighbour exists; fold onto (mirror) where
        it does not (Neumann), erroring if the edge isn't declared open."""
        e = exists & (coef != 0.0)
        if e.any():
            rows.append(k[e]); cols.append(idx[tgt_i[e], tgt_j[e]]); data.append(coef[e])
        m = (~exists) & (coef != 0.0)
        if m.any():
            if edge not in neumann_edges:
                raise ValueError(f"free node with missing neighbour on closed edge {edge}")
            if ghost_of[edge] == "mirror":
                rows.append(k[m]); cols.append(idx[mirror_i[m], mirror_j[m]]); data.append(coef[m])
            else:
                rows.append(k[m]); cols.append(k[m]); data.append(coef[m])

    link(c_zm, fi > 0,      fi - 1, fj, fi + 1, fj, "z0")
    link(c_zp, fi < nz - 1, fi + 1, fj, fi - 1, fj, "z1")
    link(c_dn, fj > 0,      fi, fj - 1, fi, fj + 1, "u0")
    link(c_up, fj < nu - 1, fi, fj + 1, fi, fj - 1, "u1")

    N = nz * nu
    A = coo_matrix((np.concatenate(data),
                    (np.concatenate(rows).astype(int), np.concatenate(cols).astype(int))),
                   shape=(N, N)).tocsr()
    return A, fn


def laplace_factor(fixed, h, symmetry="planar",
                   neumann_edges=("z0", "z1", "u0", "u1"),
                   edge_ghost="ghost_linear", dtype=np.float64):
    """FACTOR-ONCE path:
    assemble and LU-factorize the Laplace system for a given metal mask
    ONCE, returning solve_rhs(val) that back-substitutes per Dirichlet
    value set. Every electrode basis of one geometry shares this matrix
    (identical stencil + fixed mask; only the boundary VALUES differ),
    so k bases cost one factorization + k triangular solves instead of
    k full solves. dtype=float32 is the declared reduced-precision
    option (screen maps only; certified paths stay float64)."""
    from scipy.sparse.linalg import splu
    A, fn = assemble_laplace(fixed, h, symmetry, neumann_edges,
                             edge_ghost)
    nz, nu = fixed.shape
    lu = splu(A.astype(dtype).tocsc())

    def solve_rhs(val):
        b = np.zeros(nz * nu, dtype=dtype)
        b[fn] = np.asarray(val, dtype)[fixed]
        return lu.solve(b).reshape(nz, nu).astype(np.float64)
    return solve_rhs


def solve_laplace(fixed, val, h, symmetry="planar",
                  neumann_edges=("z0", "z1", "u0", "u1"),
                  edge_ghost="ghost_linear"):
    """Single-RHS wrapper over the factored path (same system as
    always; behaviour unchanged for existing callers)."""
    return laplace_factor(fixed, h, symmetry, neumann_edges,
                          edge_ghost)(val)


# ---------------------------------------------------------------------------
def cylinder_einzel(voltages, bore_r, thick, egap, lead, r_max, h):
    """Coaxial cylinder-electrode stack (node-centred einzel). Metal fills
    bore_r..r_max in each electrode's z-band. Returns Z, R, fixed, val, bands."""
    n = len(voltages)
    Lz = 2 * lead + n * thick + (n - 1) * egap
    # THE counting function: the stack length is DERIVED from the
    # declared lead/thick/egap, and if those do not compose to an
    # integer number of cells at this pitch the grid would silently
    # cover a different stack — refuse with the conforming neighbours
    # instead of rounding the discrepancy away.
    from ion_gym.io.lattice import gu_nodes
    nz = gu_nodes(Lz, h, axis="z", what="einzel stack length")
    nr = gu_nodes(r_max, h, axis="r", what="radial extent")
    Z = np.linspace(0.0, Lz, nz)
    R = np.linspace(0.0, r_max, nr)
    fixed = np.zeros((nz, nr), bool)
    val = np.zeros((nz, nr))
    z, bands = lead, []
    for V in voltages:
        zi = (Z >= z - 1e-9) & (Z <= z + thick + 1e-9)
        m = np.outer(zi, R >= bore_r - 1e-9)
        fixed |= m; val[m] = V
        bands.append((z, z + thick))
        z += thick + egap
    fixed[0, :] = True;  val[0, :] = voltages[0]
    fixed[-1, :] = True; val[-1, :] = voltages[-1]
    return Z, R, fixed, val, bands
