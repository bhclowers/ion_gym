"""assembly.py — DECLARATIVE multi-FA placement.

The concern: a coaxial multi-FA instrument (a TOF
source + reflectron + detector; the oa-TOF stack) should be placeable
from the JSON, not glued together by a Python special case per example.
But it should also stay EFFICIENT — a coaxial assembly is physically one
continuous domain, so it should solve as ONE grid, not as stitched
sub-solves.

This module serves both from ONE artifact:
  * an AssemblySpec is a readable list of PLACED UNITS (each a native
    sub-geometry + an offset), serializable to/from plain JSON — the
    clean multi-FA example format; and
  * `flatten_to_simspec()` folds a coaxial assembly into a single
    ordinary SimSpec (every unit's electrodes shifted by its offset,
    names prefixed by unit), which goes through the normal build_rz /
    build_planar solve unchanged — the efficiency.

So the SAME JSON demonstrates multi-FA placement AND solves in one shot.
No new solver, no per-example glue: the placement is data, the flatten is
one general function.

SCOPE: coaxial / shared-frame assemblies (units share coords, pitch, and
one placement axis — the case the TOF and the oa-TOF stack are). A
genuinely non-coaxial assembly (rotated units, mismatched resolutions,
or a domain too large for one grid) needs separate placed grids with a
seam handoff. That is now BUILT, in physics/staged_flight.py (Region +
Pose + fly_staged): each region solves in its OWN grid at its OWN
resolution and the ion is handed across the seam through a full rigid
pose (translation + rotation), so an orthogonal-acceleration stage (the
oa pusher, 90 deg to the beam) is expressible. flatten_to_simspec stays
the fast SINGLE-SOLVE path for the genuinely coaxial sub-chain and still
raises rather than silently mis-placing a unit it cannot fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ion_gym.io.lattice import cover_extent_mm
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SymmetrySpec, SourceSpec,
                                 CollisionSpec, IntegrationSpec, ViewSpec)


@dataclass
class PlacedUnit:
    """One native sub-optic placed in the assembly frame. `offset_mm` is
    added to every electrode coordinate (axial first, then radial/other);
    `electrodes` is the unit's own native electrode list (same schema as
    a GeometrySpec's). The unit is authored at its OWN origin and placed
    here — so a reflectron generator, a source, and a detector each stay
    independently defined and reusable."""
    name: str
    offset_mm: List[float]
    electrodes: List[ElectrodeSpec]

    def to_dict(self):
        d = {"name": self.name,
             "offset_mm": [float(v) for v in self.offset_mm],
             "electrodes": [e.to_dict() for e in self.electrodes]}
        # PASSTHROUGH: foreign keys survive at their position;
        # known wins (an extra never overrides a real emitted key).
        for k, v in (getattr(self, "_extras", None) or {}).items():
            if k not in d:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d):
        obj = cls(name=d["name"],
                  offset_mm=[float(v) for v in d["offset_mm"]],
                  electrodes=[ElectrodeSpec.from_dict(e)
                              for e in d.get("electrodes", [])])
        _extra = {k: v for k, v in d.items()
                  if k not in ("name", "offset_mm", "electrodes")}
        if _extra:
            import warnings
            warnings.warn(
                f"PlacedUnit: unknown key(s) {sorted(_extra)} retained in "
                f"the saved file but NOT interpreted", stacklevel=3)
            obj._extras = _extra
        return obj


@dataclass
class AssemblySpec:
    """A coaxial multi-FA assembly declared as placed units. Carries the
    shared frame (coords, mm_per_gu, radial/other extent) plus the run
    settings (source/collisions/integration/view) that the flattened
    SimSpec needs. `axis` names the placement axis ('x' for r-z axial)."""
    units: List[PlacedUnit]
    coords: str = "rz"
    mm_per_gu: float = 1.0
    axis: str = "x"
    other_extent_mm: float = 20.0          # radial (r-z) / transverse span
    name: str = "assembly"
    source: SourceSpec = field(default_factory=SourceSpec)
    collisions: CollisionSpec = field(default_factory=CollisionSpec)
    integration: IntegrationSpec = field(default_factory=IntegrationSpec)
    view: Optional[ViewSpec] = None

    def to_dict(self):
        d = {"kind": "assembly", "name": self.name, "coords": self.coords,
             "mm_per_gu": self.mm_per_gu, "axis": self.axis,
             "other_extent_mm": self.other_extent_mm,
             "units": [u.to_dict() for u in self.units],
             "source": self.source.to_dict(),
             "collisions": self.collisions.to_dict(),
             "integration": self.integration.to_dict()}
        if self.view is not None:
            d["view"] = self.view.to_dict()
        # PASSTHROUGH: foreign document keys survive; known wins.
        for k, v in (getattr(self, "_extras", None) or {}).items():
            if k not in d:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("kind") != "assembly":
            raise ValueError(
                f"not an assembly spec (kind={d.get('kind')!r}); load a "
                f"single-FA spec with SimSpec.from_dict instead")
        _known = {"kind", "units", "coords", "mm_per_gu", "axis",
                  "other_extent_mm",
                  "width_mm", "height_mm", "source", "collisions",
                  "integration", "bounds", "view", "name", "notes",
                  "schema_version", "_display_name"}
        _extra = sorted(k for k in d if k not in _known)
        if _extra:
            import warnings
            warnings.warn(
                f"AssemblySpec: unknown top-level key(s) {_extra} "
                f"retained in the saved file but NOT interpreted — "
                f"misspelled or foreign parameter?",
                stacklevel=3)
        obj = cls(
            units=[PlacedUnit.from_dict(u) for u in d["units"]],
            coords=d.get("coords", "rz"), mm_per_gu=d.get("mm_per_gu", 1.0),
            axis=d.get("axis", "x"),
            other_extent_mm=d.get("other_extent_mm", 20.0),
            name=d.get("name", "assembly"),
            source=(SourceSpec.from_dict(d["source"]) if "source" in d
                    else SourceSpec()),
            collisions=(CollisionSpec.from_dict(d["collisions"])
                        if "collisions" in d else CollisionSpec()),
            integration=(IntegrationSpec.from_dict(d["integration"])
                         if "integration" in d else IntegrationSpec()),
            view=(ViewSpec.from_dict(d["view"]) if "view" in d else None))
        if _extra:
            obj._extras = {k: d[k] for k in _extra}   # retained foreign keys
        return obj


# axis name -> which shape param the offset shifts (r-z: axial=x, radial=y)
_AXIS_PARAM = {"x": "x_mm", "y": "y_mm"}


def _shift_electrode(el: ElectrodeSpec, dax: float, drad: float,
                     axis: str) -> ElectrodeSpec:
    """Copy an electrode with its shapes shifted by (dax along `axis`,
    drad along the other in-plane axis). Only inline-shape electrodes are
    placeable; an STL-backed unit in a coaxial flatten is refused (its
    offset belongs in the STL frame, not here)."""
    if not el.shapes:
        raise ValueError(
            f"unit electrode {el.name!r} has no inline shapes; STL-backed "
            f"units are not supported in a coaxial flatten")
    ax_p = _AXIS_PARAM[axis]
    other_p = "y_mm" if ax_p == "x_mm" else "x_mm"
    new_shapes = []
    for s in el.shapes:
        p = dict(s.params)
        if ax_p in p:
            p[ax_p] = p[ax_p] + dax
        if other_p in p:
            p[other_p] = p[other_p] + drad
        new_shapes.append(ShapeSpec(s.type, p))
    return ElectrodeSpec(name=el.name, dc=el.dc, rf_groups=list(el.group_names()),
                         is_grid=el.is_grid, shapes=new_shapes)


def flatten_to_simspec(asm: AssemblySpec) -> SimSpec:
    """Fold a coaxial assembly into ONE SimSpec: every unit's electrodes
    shifted by its offset and renamed `<unit>_<electrode>`, on a single
    grid spanning the full placement axis. The result is an ordinary spec
    — the normal build path solves it in one shot (the efficiency), while
    the assembly JSON stays the readable multi-FA description (the clean
    example). Names are prefixed so the impact-plane / stats grouping and
    the electrode table stay per-unit legible."""
    if len(asm.axis) != 1 or asm.axis not in _AXIS_PARAM:
        raise ValueError(f"placement axis {asm.axis!r} must be one of "
                         f"{sorted(_AXIS_PARAM)} for a coaxial flatten")
    electrodes = []
    max_axial = 0.0
    for u in asm.units:
        off = u.offset_mm
        dax = float(off[0])
        drad = float(off[1]) if len(off) > 1 else 0.0
        for el in u.electrodes:
            shifted = _shift_electrode(el, dax, drad, asm.axis)
            shifted.name = f"{u.name}_{el.name}"
            electrodes.append(shifted)
            for s in shifted.shapes:
                p = s.params
                hi = p.get(_AXIS_PARAM[asm.axis], 0.0) + p.get("width_mm", 0.0)
                max_axial = max(max_axial, hi)
    # A7: both extents are LATTICE quantities. max_axial is accumulated
    # from shifted stage placements and other_extent_mm is a declared
    # span -- neither is counted in cells, so the assembled spec could be
    # refused by the loader it is handed to. Covered up through the
    # shared helper; the added slop is vacuum at the outer wall, which is
    # where A7 puts it. `coords` here is the ROUTE (xyz / rz), and this
    # assembler declares no mirror plane, so no axis needs an even count.
    _sym = SymmetrySpec(coords=asm.coords)
    geo = GeometrySpec(
        width_mm=cover_extent_mm(max_axial, asm.mm_per_gu),
        height_mm=cover_extent_mm(asm.other_extent_mm, asm.mm_per_gu),
        mm_per_gu=asm.mm_per_gu,
        symmetry=_sym, electrodes=electrodes)
    return SimSpec(geometry=geo, source=asm.source,
                   collisions=asm.collisions, integration=asm.integration,
                   view=asm.view or ViewSpec(mode="2d", planes=["xy"]),
                   name=asm.name)
