"""
ion_gym.scene3d
---------------
Declarative 3-D scene schema (producer side).

A GeomScene is the single source of truth for a geometry: the grid spec, the
electrodes (integer index + name + voltage/waveform ref + solid shapes), and
optional per-plane symmetry declarations. Everything downstream consumes the
GeomScene: rasterize3d renders it to the
labeled voxel occupancy grid (the universal interface), and
the coming CadQuery/STL path will target the same schema.

ONE integer threads through everything: the
electrode index N is simultaneously the declared electrode(N) number, the
fast-adjust basis index (.paN), the STL suffix (quad_N.stl), and the label
the rasterizer stamps into the occupancy grid.

Units: a GeomScene carries `units` = "gu" (grid units) or "mm". For Rung 1 the
canonical frame is GRID UNITS in the *stored/folded* array frame (origin at
node (0,0,0), the mirror axes), because that is the frame in which the
node-exact audits against node-centred grids are unambiguous. mm <-> gu conversion is
exact via grid.mm_per_gu.

Primitive conventions (pinned; each was validated or is flagged):
  * Cylinder is z-axis aligned and SPANS [z - length, z] — the pinned cylinder
    z-extent convention pinned by the quad instrument geometry (detector face at
    local z = 474 gu = 94.8 mm; internal validation Gate 0). length == 0 is a
    zero-thickness ideal disc (legal).
  * Box3D spans the closed box [x1,x2]x[y1,y2]x[z1,z2].
  * A Shape is within-minus-notin: the union of `within` primitives minus the
    union of `notin` primitives (fill-within-notin semantics).
  * Surface-inclusive: a node exactly on an ideal surface is ELECTRODE
    (the surface-inclusive rule; rasterize3d matches it with a tiny epsilon).
"""

from __future__ import annotations

import json
import math

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Union


# ------------------------------------------------------------------ grid
@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int
    mm_per_gu: float = 1.0
    symmetry: str = "planar"          # "planar" | "cylindrical"
    mirror: str = ""                  # subset of "xyz", e.g. "xy"
    surface: str = "fractional"       # refine surface handling to request
    overhang: str = "refuse"          # "refuse" | "allow" -- see check().
    # overhang="allow" DECLARES that primitives deliberately protrude past
    # the stored grid box and that only the in-box portion is meant to be
    # rasterised (the Rung-1 authoring pattern: infinite analytic constructs made
    # finite-with-overhang, node-identical inside the stored region; a rod
    # straddling a mirror fold authored in the full physical frame).  It is a
    # declaration, not a mode switch: rasterization is IDENTICAL either way.
    # The default refuses, because an UNdeclared protrusion is almost always
    # a mis-set grid, and a clipped/empty array otherwise solves silently.

    @property
    def mirror_x(self): return "x" in self.mirror
    @property
    def mirror_y(self): return "y" in self.mirror
    @property
    def mirror_z(self): return "z" in self.mirror


# ------------------------------------------------------------- primitives
@dataclass
class Cylinder:
    """z-axis cylinder, spans [z - length, z] (pinned convention)."""
    cx: float
    cy: float
    z: float          # HIGH-z face
    r: float
    length: float
    kind: str = "cylinder"


@dataclass
class Box3D:
    x1: float; y1: float; z1: float
    x2: float; y2: float; z2: float
    kind: str = "box3d"

    def __post_init__(self):
        if self.x2 < self.x1 or self.y2 < self.y1 or self.z2 < self.z1:
            raise ValueError("Box3D requires x1<=x2, y1<=y2, z1<=z2")


Primitive = Union[Cylinder, Box3D]
_PRIM_KINDS = {"cylinder": Cylinder, "box3d": Box3D}


@dataclass
class Shape:
    within: List[Primitive] = field(default_factory=list)
    notin: List[Primitive] = field(default_factory=list)


@dataclass
class Electrode:
    index: int                    # = declared electrode(N) = _N.stl = voxel label
    name: str
    shapes: List[Shape] = field(default_factory=list)
    voltage: Union[float, str] = 0.0   # static volts, or a waveform reference name

    def __post_init__(self):
        if self.index < 1:
            raise ValueError("electrode index must be >= 1 (0 is vacuum)")


@dataclass
class GeomScene:
    grid: GridSpec
    electrodes: List[Electrode] = field(default_factory=list)
    units: str = "gu"             # "gu" | "mm"
    name: str = "scene"
    notes: str = ""

    # ---------------------------------------------------------- unit views
    def in_gu(self) -> "GeomScene":
        """Return an equivalent GeomScene with all primitive coordinates in grid
        units (the canonical audit frame). Exact scaling by 1/mm_per_gu."""
        if self.units == "gu":
            return self
        s = 1.0 / self.grid.mm_per_gu
        return _scaled(self, s, "gu")

    def in_mm(self) -> "GeomScene":
        if self.units == "mm":
            return self
        return _scaled(self, self.grid.mm_per_gu, "mm")

    # ------------------------------------------------------- resolution
    def extent_mm(self):
        """(lo, hi) mm AABB of ALL primitives -- the geometry's own box.

        This is the DECLARED extent: it comes from the electrodes, not from
        the grid. That distinction is the whole point. nx/ny/nz and
        mm_per_gu together also imply a box, and when the two disagree the
        geometry is silently clipped or floated -- see check().
        """
        sc = self.in_mm()
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for e in sc.electrodes:
            for sh in e.shapes:
                for p in list(sh.within) + list(sh.notin):
                    if isinstance(p, Box3D):
                        plo = np.array([p.x1, p.y1, p.z1], float)
                        phi = np.array([p.x2, p.y2, p.z2], float)
                    else:                                   # Cylinder (z-axis)
                        plo = np.array([p.cx - p.r, p.cy - p.r,
                                        p.z - p.length], float)
                        phi = np.array([p.cx + p.r, p.cy + p.r, p.z], float)
                    lo = np.minimum(lo, plo)
                    hi = np.maximum(hi, phi)
        if not np.all(np.isfinite(lo)):
            raise ValueError("scene has no primitives — extent undefined")
        return lo, hi

    def grid_extent_mm(self):
        """The box the GRID covers: nodes 0..n-1 at mm_per_gu, origin 0."""
        g = self.grid
        return (np.zeros(3),
                np.array([(g.nx - 1), (g.ny - 1), (g.nz - 1)], float)
                * g.mm_per_gu)

    def at_resolution(self, mm_per_gu, *, margin_mm=None):
        """Re-grid to a new pitch.  THE DIMENSIONS DO NOT CHANGE -- only the
        voxel size does.

        THE MISSING KNOB.  Before this, a loaded GeomScene carried nx/ny/nz AND
        mm_per_gu -- two sources of truth for one quantity -- so changing the
        solve resolution meant hand-editing three ints and a float in step.

        DOMAIN SEMANTICS (the part that bit us):
          margin_mm=None (default) -- PRESERVE the existing domain.  The grid
            box is re-cut into smaller/larger cells; the geometry does not
            move.  This is what "change the resolution" must mean.  Deriving
            the box afresh from the METAL each time silently threw away the
            vacuum margin and pushed conductors onto the open boundary.
          margin_mm=<float> -- (re)declare the domain as metal +/- margin.
            Use this ONCE, when first gridding a scene that has none.
        """
        sc = self.in_mm()
        h = float(mm_per_gu)
        if h <= 0:
            raise ValueError(f"mm_per_gu must be > 0, got {h}")
        mlo, mhi = sc.extent_mm()

        if margin_mm is None:
            glo, ghi = sc.grid_extent_mm()
            # With overhang DECLARED, the domain to preserve is the declared
            # grid box itself -- the geometry protrudes by design, so
            # containment is not the test.  Only degeneracy still refuses.
            if self.grid.overhang == "allow":
                covers = np.all(ghi > glo)
            else:
                covers = (np.all(mlo >= glo - 1e-9)
                          and np.all(mhi <= ghi + 1e-9)
                          and np.all(ghi > glo))
            if not covers:
                raise ValueError(
                    "this scene has no valid domain to preserve (its grid "
                    "does not contain its geometry). Declare one once with "
                    "at_resolution(h, margin_mm=<mm>), then later pitch "
                    "changes will preserve it.")
            lo, hi, shift = glo, ghi, np.zeros(3)
        else:
            lo = mlo - float(margin_mm)
            hi = mhi + float(margin_mm)
            shift = -lo
            lo, hi = np.zeros(3), hi - lo

        shifted = _translated(sc, shift) if np.any(shift) else sc
        # CEIL: nodes span [0, (n-1)*h] and must COVER the domain.
        n = [int(math.ceil((hi[i] - lo[i]) / h - 1e-9)) + 1 for i in range(3)]
        g = GridSpec(nx=n[0], ny=n[1], nz=n[2], mm_per_gu=h,
                     symmetry=self.grid.symmetry, mirror=self.grid.mirror,
                     surface=self.grid.surface,
                     overhang=self.grid.overhang)
        out = GeomScene(grid=g, electrodes=shifted.electrodes, units="mm",
                    name=self.name, notes=self.notes)
        return out.check()

    # ---------------------------------------------------------- validation
    def check(self):
        idx = [e.index for e in self.electrodes]
        if len(set(idx)) != len(idx):
            raise ValueError(f"duplicate electrode indices: {sorted(idx)}")
        if self.grid.symmetry not in ("planar", "cylindrical"):
            raise ValueError(f"bad symmetry {self.grid.symmetry!r}")
        if any(c not in "xyz" for c in self.grid.mirror):
            raise ValueError(f"bad mirror string {self.grid.mirror!r}")
        if self.grid.mm_per_gu <= 0:
            raise ValueError(f"mm_per_gu must be > 0, "
                             f"got {self.grid.mm_per_gu}")
        if self.grid.overhang not in ("refuse", "allow"):
            # An unrecognised policy value is REFUSED, never defaulted --
            # "refus" silently becoming "refuse" is how a declaration rots.
            raise ValueError(f"bad overhang policy {self.grid.overhang!r} "
                             f"(must be 'refuse' or 'allow')")
        # ---- the guard that was missing -------------------------------
        # rasterize3d/voxelize index the grid from arange(n) with the ORIGIN
        # AT 0. Geometry outside [0, (n-1)*mm_per_gu] is therefore not
        # "somewhere else" -- it is NOT RASTERISED. The array comes back
        # empty or clipped, the solve succeeds, and the field is silently
        # wrong. Refuse instead.
        #
        # UNLESS the scene DECLARES the protrusion (overhang="allow"): the
        # legitimate pattern is finite-with-overhang authoring in the full
        # physical frame -- infinite analytic constructs made finite past the
        # array edge, rods straddling a mirror fold plane -- where clipping
        # to the stored region is the DEFINITION of correct, node-identical
        # inside the box.  The declaration is on the GridSpec (serialised,
        # travels with the artifact), so this branch is authored intent, not
        # a caught error -- which is why it is silent here (an
        # enumerated, legitimate silence).  The invariant the relaxation must
        # not lose -- an electrode rasterising EMPTY and still solving -- is
        # enforced unconditionally where the mask exists, in
        # rasterize3d.rasterize().
        if self.electrodes and self.grid.overhang == "refuse":
            lo, hi = self.extent_mm()
            glo, ghi = self.grid_extent_mm()
            tol = 1e-9
            out = []
            for i, ax in enumerate("xyz"):
                if lo[i] < glo[i] - tol:
                    out.append(f"{ax}: geometry starts at {lo[i]:.3f} mm, "
                               f"before the grid origin 0")
                if hi[i] > ghi[i] + tol:
                    out.append(f"{ax}: geometry reaches {hi[i]:.3f} mm, "
                               f"past the grid edge {ghi[i]:.3f} mm "
                               f"(n={[self.grid.nx, self.grid.ny, self.grid.nz][i]}"
                               f" x {self.grid.mm_per_gu} mm/gu)")
            if out:
                raise ValueError(
                    "scene geometry does not fit its grid — it would "
                    "rasterise CLIPPED or EMPTY and still solve: "
                    + "; ".join(out)
                    + ". Use GeomScene.at_resolution(h) to derive the grid from "
                      "the geometry instead of setting nx/ny/nz by hand; or, "
                      "if the protrusion is DELIBERATE (full-physical-frame "
                      "authoring, finite-with-overhang primitives), declare "
                      "it with GridSpec(overhang='allow') — only the in-box "
                      "portion is rasterised, and every electrode must still "
                      "own at least one stored-frame node.")
        return self

    # --------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        """The GeomScene as a plain dict, so it can be EMBEDDED in another
        document (a SimSpec) rather than referenced by a file path. A path is
        local to the machine that wrote it; a dict travels."""
        return asdict(self)

    def to_json(self, path=None) -> str:
        txt = json.dumps(self.to_dict(), indent=2)
        if path:
            open(path, "w").write(txt)
        return txt

    @staticmethod
    def from_dict(d: dict) -> "GeomScene":
        return GeomScene.from_json(json.dumps(d))

    @staticmethod
    def from_json(src) -> "GeomScene":
        if isinstance(src, dict):
            d = src
        else:
            d = json.loads(open(src).read()
                           if not src.lstrip().startswith("{") else src)
        def prim(p):
            return _PRIM_KINDS[p["kind"]](**{k: v for k, v in p.items()
                                             if k != "kind"})
        els = [Electrode(index=e["index"], name=e["name"],
                         voltage=e["voltage"],
                         shapes=[Shape(within=[prim(p) for p in s["within"]],
                                       notin=[prim(p) for p in s["notin"]])
                                 for s in e["shapes"]])
               for e in d["electrodes"]]
        return GeomScene(grid=GridSpec(**d["grid"]), electrodes=els,
                     units=d.get("units", "gu"), name=d.get("name", "scene"),
                     notes=d.get("notes", "")).check()


def _scale_prim(p: Primitive, s: float) -> Primitive:
    if isinstance(p, Cylinder):
        return Cylinder(p.cx * s, p.cy * s, p.z * s, p.r * s, p.length * s)
    return Box3D(p.x1 * s, p.y1 * s, p.z1 * s, p.x2 * s, p.y2 * s, p.z2 * s)


def _translate_prim(p: Primitive, d) -> Primitive:
    if isinstance(p, Box3D):
        return Box3D(p.x1 + d[0], p.y1 + d[1], p.z1 + d[2],
                     p.x2 + d[0], p.y2 + d[1], p.z2 + d[2])
    return Cylinder(cx=p.cx + d[0], cy=p.cy + d[1], z=p.z + d[2],
                    r=p.r, length=p.length)


def _translated(sc: GeomScene, d) -> GeomScene:
    """Rigid shift of every primitive (mm).  Used to put a centred geometry
    onto the grid's origin-at-zero convention exactly once, in ONE place,
    instead of by hand at every call site."""
    els = [Electrode(index=e.index, name=e.name, voltage=e.voltage,
                     shapes=[Shape(
                         within=[_translate_prim(p, d) for p in sh.within],
                         notin=[_translate_prim(p, d) for p in sh.notin])
                         for sh in e.shapes])
           for e in sc.electrodes]
    return GeomScene(grid=sc.grid, electrodes=els, units=sc.units,
                 name=sc.name, notes=sc.notes)


def _scaled(sc: GeomScene, s: float, units: str) -> GeomScene:
    els = [Electrode(index=e.index, name=e.name, voltage=e.voltage,
                     shapes=[Shape(within=[_scale_prim(p, s) for p in sh.within],
                                   notin=[_scale_prim(p, s) for p in sh.notin])
                             for sh in e.shapes])
           for e in sc.electrodes]
    return GeomScene(grid=sc.grid, electrodes=els, units=units,
                 name=sc.name, notes=sc.notes)
