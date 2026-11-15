"""
birth_domain.py -- an ion born outside the solved domain is reported.

There is no field outside the solved domain, and every kernel ends such an
ion at its first step as having left the domain; that used to be recorded
with no word of why (an x0 of 10 m on a 20 mm deck read as an ordinary
loss). Each route's per-ion birth check calls refuse_birth_outside before
its born-in-metal test, with the node coordinates it already computes.

BirthOutsideDomain is a raster2d.BirthInMetal, so every route's per-ion
handler already catches it: ion_envelope.birth_splat records it as fate 1
(left the domain) with born_outside_domain=True and a warning, and the
other ions fly. (It lives here, not in raster2d, because raster2d is solve
code -- physics.solve_code keys cached fields on it -- and a birth check
changes no field.)
"""
from ion_gym.physics.raster2d import BirthInMetal

# a birth this close to the last node (in nodes) is on it, not past it
DOMAIN_EDGE_TOL = 1e-9


class BirthOutsideDomain(BirthInMetal):
    """An ion whose birth lies outside the solved domain."""


def refuse_birth_outside(shape, grid_pos, axes, h_mm, who, pos_txt):
    """Raise BirthOutsideDomain when a birth at node coordinates
    `grid_pos` (one per axis of `shape`, named by `axes`) lies outside
    nodes 0 .. n - 1. `who` prefixes the message (e.g. "ion 7 ")."""
    for g, n, ax in zip(grid_pos, shape, axes):
        if not (-DOMAIN_EDGE_TOL <= g <= n - 1 + DOMAIN_EDGE_TOL):
            raise BirthOutsideDomain(
                f"{who}birth at {pos_txt} lies outside the solved domain: "
                f"{ax} is at node {g:.3f}, and the domain spans nodes 0 to "
                f"{n - 1} (pitch {h_mm} mm). There is no field there, so "
                f"the ion leaves the domain at once: move the source "
                f"inside it")
