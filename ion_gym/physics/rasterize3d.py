"""
ion_gym.rasterize3d
-------------------
Node rasterizer: GeomScene -> labeled voxel occupancy grid (int16, 0 = vacuum,
N = electrode index). This is the universal interface — the
solver, the tracer, and every view consume this grid, never the source solid.

Inclusion rule (the shared geometry<->solver contract, ONE specification):
a node at integer coordinates (i, j, k) in the stored/folded frame is inside a
primitive if it is strictly inside OR on the ideal surface (the
surface-inclusive convention marks
on-surface points as electrode). Implemented as <= with EPS = 1e-9 gu so that
authored surfaces passing exactly through node planes behave identically here
and in the rasterizer. The fitter (fit_geometry.py) only ever emits
parameters whose surfaces either (a) pass exactly through node coordinates or
(b) keep a finite margin from every node — so no tie is ever left to a
convention difference. Fractional-surface sub-cell corrections are a FIELD
property of an accelerated solve, not an occupancy property; node labels here are
the complete occupancy contract.

Shapes are within-minus-notin per shape; shapes and electrodes union.
Overlapping claims by two different electrode indices raise (a real geometry
error — a permissive rasterizer would silently let the later electrode win).
"""

import numpy as np

from ion_gym.physics.scene3d import GeomScene, Cylinder, Box3D

EPS = 1e-9


def _prim_mask(p, X, Y, Z):
    if isinstance(p, Cylinder):
        zlo, zhi = p.z - p.length, p.z
        return (((X - p.cx) ** 2 + (Y - p.cy) ** 2 <= p.r ** 2 + EPS)
                & (Z >= zlo - EPS) & (Z <= zhi + EPS))
    if isinstance(p, Box3D):
        return ((X >= p.x1 - EPS) & (X <= p.x2 + EPS)
                & (Y >= p.y1 - EPS) & (Y <= p.y2 + EPS)
                & (Z >= p.z1 - EPS) & (Z <= p.z2 + EPS))
    raise TypeError(f"unknown primitive {type(p)}")


def rasterize(scene: GeomScene) -> np.ndarray:
    """Labeled occupancy grid, shape (nx, ny, nz) int16, stored-frame nodes."""
    sc = scene.check().in_gu()
    g = sc.grid
    X, Y, Z = np.meshgrid(np.arange(g.nx, dtype=float),
                          np.arange(g.ny, dtype=float),
                          np.arange(g.nz, dtype=float), indexing="ij")
    lab = np.zeros((g.nx, g.ny, g.nz), np.int16)
    for e in sc.electrodes:
        m = np.zeros_like(lab, bool)
        for sh in e.shapes:
            w = np.zeros_like(m)
            for p in sh.within:
                w |= _prim_mask(p, X, Y, Z)
            for p in sh.notin:
                w &= ~_prim_mask(p, X, Y, Z)
            m |= w
        clash = m & (lab != 0) & (lab != e.index)
        if clash.any():
            i, j, k = [int(v[0]) for v in np.where(clash)]
            raise ValueError(f"electrode {e.index} ({e.name}) overlaps "
                             f"electrode {int(lab[i, j, k])} at node "
                             f"({i},{j},{k}) (+{int(clash.sum())} more)")
        # An authored electrode owning ZERO stored-frame nodes is a dangling
        # Dirichlet basis: the solve proceeds, the electrode simply is not
        # there, and the field is silently wrong.  This is the "rasterise
        # EMPTY and still solve" failure named in GeomScene.check(), enforced
        # HERE because only the assembled mask can measure it -- and enforced
        # UNCONDITIONALLY, because overhang='allow' relaxes fit, never
        # existence.  Common causes: the whole electrode lies outside the
        # stored box (in a mirrored frame, entirely on the reflected side --
        # author the stored-quadrant copy too), or notin solids consumed it.
        if not m.any():
            g_ = sc.grid
            raise ValueError(
                f"electrode {e.index} ({e.name}) rasterises EMPTY: no node "
                f"of the {g_.nx}x{g_.ny}x{g_.nz} stored grid "
                f"(mirror {g_.mirror!r}, overhang {g_.overhang!r}) falls "
                f"inside it. A Dirichlet electrode with zero nodes would "
                f"solve silently as if absent — fix the geometry or the "
                f"grid; if it lives only on the mirrored side of a fold "
                f"plane, author its stored-quadrant image.")
        lab[m] = e.index
    return lab


# masks_from_pa_bases REMOVED: it read .paN
# fast-adjust basis arrays through the deleted field-array reader, and both
# the reader and the arrays are gone from the tree. Native geometry
# rasterizes from the spec -- see this module's own shape routes and
# notebooks/08_shapes_io.ipynb.
