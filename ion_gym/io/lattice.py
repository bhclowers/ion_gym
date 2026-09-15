"""
ion_gym.io.lattice — THE mm->gu boundary.

One counting function, one owner. Every route (planar, r-z, stl2d,
3-D stl3d/scene3d/shapes3d, and their display previews) converts a
declared mm extent to a cell/node count HERE, and nowhere else. The
per-route `floor(mm/h) + 1` / `round(mm/h)` conversions this module
retires silently absorbed any non-integer remainder — a declared
20.0 mm at 0.35 mm/gu quietly became a real 19.95, which displaced
the a quarter deck's symmetry plane by 0.86 cell in z, and put
mirrored probe decks 0.907 mm off.

TWO KINDS OF LENGTH, TWO RULES:

1. LATTICE quantities — domain extents, node counts, the grid anchor,
   and the position of every symmetry plane — are integer gu, exactly,
   on every route and axis, mirrored or not. Extents are DERIVED
   (width_mm := cells * mm_per_gu); a spec declaring an extent that is
   not exactly count x pitch is REFUSED here, naming the axis, the
   extent, the pitch, and the two nearest conforming extents. Lattice
   errors are SYSTEMATIC: they displace symmetry planes and do not
   converge away with pitch.

2. METAL edges stay in mm and rasterize (measured hardware: the
   gaps are DATA). Rasterization error is O(h) and pitch-convergence
   studies own it; this module never touches shape coordinates.

WHY io, not physics: GeometrySpec (io.sim_spec) owns the declared
extents and mm_per_gu, and spec validation must run the same check the
builders run — physics stands on io (the declared dependency
direction), so the one owner lives here.
"""
from __future__ import annotations

import math

# Conformance tolerance, in CELLS. A declared extent within this many
# cells of an integer count is that count (IEEE noise: a conforming
# 6.2 mm / 0.05 evaluates to 123.99999999999999). Matches the 1e-9
# node-lattice tolerance the origin_mm check uses.
# A real misdeclaration is orders of magnitude larger (the smallest in
# a tree-wide scan was 0.02 cell); a tolerance that could mask
# one would change the answer and therefore is not a tolerance.
LATTICE_TOL_CELLS = 1e-9


def on_lattice(value_mm: float, mm_per_gu: float) -> bool:
    """Is `value_mm` an exact integer number of cells at this pitch
    (within LATTICE_TOL_CELLS)?"""
    if mm_per_gu <= 0.0:
        raise ValueError(f"mm_per_gu must be > 0, got {mm_per_gu!r}")
    q = float(value_mm) / float(mm_per_gu)
    return abs(q - round(q)) <= LATTICE_TOL_CELLS


def nearest_conforming(value_mm: float, mm_per_gu: float):
    """The two conforming extents bracketing `value_mm`:
    (floor_cells * pitch, (floor_cells + 1) * pitch), floats in mm."""
    h = float(mm_per_gu)
    lo_cells = math.floor(float(value_mm) / h + LATTICE_TOL_CELLS)
    return lo_cells * h, (lo_cells + 1) * h


def conformance_error(value_mm: float, mm_per_gu: float, *,
                      axis: str, what: str = "domain extent"):
    """The A7 refusal text for a non-conforming lattice quantity, or
    None when it conforms. Shared by the counting functions (which
    raise it) and spec validation (which lists it), so the loader and
    the builders can never disagree about what conforms."""
    if mm_per_gu <= 0.0:
        return (f"{axis}: mm_per_gu must be > 0, got {mm_per_gu!r}")
    v = float(value_mm)
    h = float(mm_per_gu)
    q = v / h
    if abs(q - round(q)) <= LATTICE_TOL_CELLS:
        return None
    lo, hi = nearest_conforming(v, h)
    lo_c, hi_c = int(round(lo / h)), int(round(hi / h))
    return (
        f"{axis}-axis {what} {v:g} mm is not an integer number of cells "
        f"at mm_per_gu={h:g} ({q:.6g} cells). Lattice quantities are "
        f"integer gu, exactly, on every route and axis (charter A7): "
        f"the extent is count x pitch, and the remainder of a physical "
        f"envelope is absorbed at the OUTER WALLS, never silently or at "
        f"a symmetry plane. The two nearest conforming {what}s are "
        f"{round(lo, 9):g} mm ({lo_c} cells) and {round(hi, 9):g} mm "
        f"({hi_c} cells). Metal edges stay in mm and rasterize (the "
        f"measured-hardware carve-out) — only the lattice must conform.")


def gu_cells(extent_mm: float, mm_per_gu: float, *, axis: str,
             what: str = "domain extent") -> int:
    """THE counting function: exact cell count of a conforming mm
    extent. REFUSES a non-conforming extent, naming the axis, the
    extent, the pitch, and the two nearest conforming extents (A7).
    An extent of exactly 0.0 is 0 cells (the 2-D depth convention)."""
    err = conformance_error(extent_mm, mm_per_gu, axis=axis, what=what)
    if err is not None:
        raise ValueError(err)
    n = int(round(float(extent_mm) / float(mm_per_gu)))
    if n < 0:
        raise ValueError(f"{axis}-axis {what} {extent_mm!r} mm is "
                         f"negative — an extent cannot be")
    return n


def gu_nodes(extent_mm: float, mm_per_gu: float, *, axis: str,
             what: str = "domain extent") -> int:
    """Node count of the node-centred inclusive span [0, extent]:
    cells + 1. The node-centred routes (planar, r-z, the 3-D voxel
    builders) use this; the stl2d voxel-count convention uses
    gu_cells. Both go through the ONE conformance check."""
    return gu_cells(extent_mm, mm_per_gu, axis=axis, what=what) + 1


def derived_extent_mm(cells: int, mm_per_gu: float) -> float:
    """extent := count x pitch — the A7 derivation direction, for
    generators that size a domain by counts and write the mm extent."""
    if cells < 0:
        raise ValueError(f"cells must be >= 0, got {cells!r}")
    return int(cells) * float(mm_per_gu)


def cover_extent_mm(extent_mm: float, mm_per_gu: float, *,
                    mirrored: bool = False) -> float:
    """THE shared cover-up snap: the smallest conforming extent that is
    at least `extent_mm`. For every generator that sizes a domain from
    PHYSICAL geometry — mesh bounds, rail pitches, guard widths, board
    thicknesses — rather than from a cell count.

    WHY THIS IS ONE FUNCTION AND NOT A THREE-LINE IDIOM. The same defect
    was found independently in four generators (including
    build_stl.einzel3d_spec), each time by tripping over it rather than
    by looking: a domain computed as a sum of arbitrary floats is almost
    never an integer number of cells, so the spec is refused by the very
    loader its own package ships. Copying the idiom a fifth time would
    make a fifth place for it to be got wrong.

    COVER-UP, NEVER TRIM. Callers size a domain to CONTAIN something —
    a mesh AABB, a guard flush against a wall, a rod bundle. Rounding the
    extent down moves a wall inside the metal it was sized to hold;
    rounding up adds vacuum at the outer wall, which is exactly where A7
    says the remainder belongs.

    `mirrored=True` forces an EVEN cell count. An axis that folds about
    the DOMAIN MIDLINE (the default when no plane_mm is declared) puts
    its plane at extent/2, so an odd count lands the plane at cell N.5
    and the A7 plane check refuses it. This is the part that is easy to
    miss: an extent can be perfectly integer and still be wrong on a
    mirrored axis, and it hides whenever the count happens to come out
    even (measured: W = 8.25 mm = 55 cells, plane at 27.5, invisible in
    the first configuration tested because that one's H landed on 36).

    An extent already conforming to within LATTICE_TOL_CELLS keeps its
    count rather than gaining a spurious cell from float noise.
    """
    h = float(mm_per_gu)
    if h <= 0.0:
        raise ValueError(f"mm_per_gu must be > 0, got {mm_per_gu!r}")
    v = float(extent_mm)
    if v < 0.0:
        raise ValueError(
            f"extent must be >= 0 to be covered, got {extent_mm!r} mm — a "
            f"negative extent is a caller bug, not something to round")
    raw = v / h
    n = (int(round(raw)) if abs(raw - round(raw)) <= LATTICE_TOL_CELLS
         else math.ceil(raw))
    step = 2 if mirrored else 1
    if n % step:
        n += step - (n % step)
    return derived_extent_mm(n, h)
