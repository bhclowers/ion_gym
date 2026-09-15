"""
stl_channels.py — channel-bundled STL export + flyable-spec emission for
CadQuery authoring notebooks.

THE BUNDLING DOCTRINE. An ElectrodeSpec is one ELECTRICAL CHANNEL — one
basis, one voltage/drive assignment. A CAD assembly, by contrast, is a bag
of SOLIDS, and a travelling-wave device has many solids per channel (a TW
phase owns rungs on both tracks plus its curved segments). Exporting one
STL per SOLID therefore fractures a channel across files, and wiring those
files back up is exactly the per-configuration hand-work this module
retires: register every solid under its CHANNEL name, export ONE STL per
channel (the voxelizer's parity containment composes disjoint watertight
bodies in one mesh correctly — see ion_gym.physics.voxelize), and emit a
SimSpec whose electrodes reference those files. The complement of
cadquery_emit's one-file-per-electrode rule: two conductors in one STL are
a defect only when they are on DIFFERENT channels; same-channel solids
BELONG together.

Nothing in this module knows any particular instrument. Channel names,
drive wiring, voltages, pitch, and margins are all caller data; the module
owns only the mechanics (grouping, compound export, frame bookkeeping,
manifest, schema emission through the real sim_spec dataclasses).

FRAME. Exported STLs stay pristine CAD exports — never translated bytes.
Placement into the build frame is DECLARED on the deck as
geometry.frame_offset_mm and applied at the single mesh-ingest point
(stl_resolve.load_mesh). emit_simspec computes that declaration from the
registry's AABB and the caller's vacuum margin, and sizes the domain to an
exact integer number of grid units (lattice quantities are integer in gu).

CadQuery is an OPTIONAL, AUTHOR-TIME dependency: it is imported lazily
inside the functions that touch solids, so importing ion_gym.cad (or this
module) never requires it and has no side effects.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ion_gym.io.sim_spec import (
    BoundsSpec, CollisionSpec, DCGroupSpec, ElectrodeSpec, GeometrySpec,
    IntegrationSpec, RFGroupSpec, SimSpec, SourceSpec, SymmetrySpec,
)

__all__ = [
    "ChannelRegistry", "emit_simspec", "write_spec_json",
    "sanitize_channel",
]

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _require_cadquery():
    try:
        import cadquery as cq
        return cq
    except ImportError as e:
        raise ImportError(
            "stl_channels needs the 'cadquery' package for solid handling "
            "(pip install cadquery). It is an author-time dependency; "
            "ion_gym itself never imports it at runtime.") from e


def sanitize_channel(name: str) -> str:
    """Filename-safe form of a channel name. Refuses names that sanitize
    to nothing — a channel that cannot name its own file is a caller bug,
    not something to paper over with a generated token."""
    out = _SAFE.sub("_", str(name)).strip("_")
    if not out:
        raise ValueError(
            f"channel name {name!r} sanitizes to an empty filename stem")
    return out


class ChannelRegistry:
    """Ordered CHANNEL -> [solids] registry for a CadQuery build.

    add() during generation replaces the sequential cadDict pattern
    (part-number -> [solid, label]); the channel key IS the electrical
    identity every downstream artifact (STL, ElectrodeSpec, basis) keys on.
    Insertion order of channels is preserved and becomes the electrode
    order of the emitted spec.
    """

    def __init__(self):
        self._parts: "Dict[str, List[tuple]]" = {}

    # ------------------------------------------------------------ intake
    def add(self, channel: str, solid, part_label: str = ""):
        """Register one CadQuery Workplane/Shape under its channel."""
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError(
                f"channel must be a non-empty string, got {channel!r}")
        if not (hasattr(solid, "vals") or hasattr(solid, "wrapped")):
            raise TypeError(
                f"channel {channel!r}: expected a CadQuery Workplane or "
                f"Shape, got {type(solid).__name__}")
        self._parts.setdefault(channel, []).append((part_label, solid))

    # ----------------------------------------------------------- queries
    def channels(self) -> tuple:
        return tuple(self._parts)

    def part_count(self, channel: str) -> int:
        return len(self._require(channel))

    def summary(self) -> str:
        lines = [f"{len(self._parts)} channel(s), "
                 f"{sum(len(v) for v in self._parts.values())} part(s) total"]
        for ch, parts in self._parts.items():
            labels = ", ".join(lbl for lbl, _ in parts if lbl)
            lines.append(f"  {ch:12s} {len(parts):2d} part(s)"
                         + (f": {labels}" if labels else ""))
        return "\n".join(lines)

    def _require(self, channel: str) -> list:
        if channel not in self._parts:
            raise KeyError(
                f"unknown channel {channel!r}; registered: "
                f"{list(self._parts)}")
        return self._parts[channel]

    def _shapes(self, channel: str) -> list:
        """Flatten a channel's registered objects to cq Shapes."""
        _require_cadquery()
        shapes = []
        for lbl, obj in self._require(channel):
            if hasattr(obj, "vals"):            # Workplane
                vals = [v for v in obj.vals() if hasattr(v, "wrapped")]
                if not vals:
                    raise ValueError(
                        f"channel {channel!r} part {lbl!r}: Workplane "
                        f"holds no solids — nothing to export")
                shapes.extend(vals)
            else:                               # bare Shape
                shapes.append(obj)
        return shapes

    # ------------------------------------------------------------- frame
    def aabb(self):
        """(lo, hi) mm over ALL channels, CAD frame. Refuses when empty."""
        if not self._parts:
            raise ValueError("registry is empty: no channels registered")
        los, his = [], []
        for ch in self._parts:
            for s in self._shapes(ch):
                bb = s.BoundingBox()
                los.append([bb.xmin, bb.ymin, bb.zmin])
                his.append([bb.xmax, bb.ymax, bb.zmax])
        return (np.min(np.asarray(los, float), axis=0),
                np.max(np.asarray(his, float), axis=0))

    # ------------------------------------------------------------ export
    def export_stls(self, out_dir, stem: str, *, tol_mm: float = 0.02,
                    ang_tol_rad: float = 0.1) -> Dict[str, str]:
        """ONE STL per channel: '{stem}_{channel}.stl' in out_dir.

        tol_mm is the linear tessellation deflection (facet sagitta bound)
        handed to the exporter; the voxelizer's surface-eps promotion is
        sized against exactly this quantity, so keep it well below the
        solve pitch. Returns {channel: filename}. Any export failure
        re-raises with the channel named — never a half-written set that
        looks complete.
        """
        cq = _require_cadquery()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        names: Dict[str, str] = {}
        for ch in self._parts:
            fn = f"{stem}_{sanitize_channel(ch)}.stl"
            if fn in names.values():
                clash = [c for c, f in names.items() if f == fn]
                raise ValueError(
                    f"channels {clash + [ch]} sanitize to the same file "
                    f"{fn!r} — rename one; two channels in one STL solve "
                    f"as one electrode and every downstream voltage is "
                    f"wrong")
            shapes = self._shapes(ch)
            compound = cq.Compound.makeCompound(shapes)
            try:
                cq.exporters.export(
                    compound, str(out_dir / fn),
                    tolerance=tol_mm, angularTolerance=ang_tol_rad)
            except Exception as e:  # exporter raises OCP-internal types;
                # name the channel and keep the cause rather than letting
                # a bare OCP traceback point at nothing.
                raise RuntimeError(
                    f"STL export failed for channel {ch!r} "
                    f"({len(shapes)} solids) -> {fn}") from e
            names[ch] = fn
        return names

    # ----------------------------------------------------- viz triangles
    def triangles(self, channel: str, *, tol_mm: float = 0.05,
                  ang_tol_rad: float = 0.3) -> list:
        """Channel metal as a list of (3,3) mm triangle arrays (CAD
        frame) — polygon fodder for viz_core.Body/scene_from_bodies, the
        sanctioned schematic path for CAD solids. Coarser default
        tolerance than export: these triangles draw, they never solve."""
        _require_cadquery()
        tris = []
        for s in self._shapes(channel):
            verts, faces = s.tessellate(tol_mm, ang_tol_rad)
            pts = np.asarray([(v.x, v.y, v.z) for v in verts], float)
            for f in faces:
                tris.append(pts[list(f)])
        return tris


def _snap_up_gu(extent_mm: float, h_mm: float) -> float:
    """Smallest integer-gu length >= extent_mm (lattice quantities are
    exactly integer in gu)."""
    n = int(math.ceil(round(extent_mm / h_mm, 9)))
    return n * h_mm


def emit_simspec(registry: ChannelRegistry, stl_files: Mapping[str, str], *,
                 name: str,
                 mm_per_gu: float,
                 margin_mm: float,
                 electrodes: Mapping[str, dict],
                 rf_groups: Sequence[RFGroupSpec] = (),
                 dc_groups: Sequence[DCGroupSpec] = (),
                 symmetry: Optional[SymmetrySpec] = None,
                 stl_dir: str = ".",
                 source: Optional[SourceSpec] = None,
                 collisions: Optional[CollisionSpec] = None,
                 integration: Optional[IntegrationSpec] = None,
                 bounds: Optional[BoundsSpec] = None,
                 notes: str = "") -> SimSpec:
    """Build a validated, flyable SimSpec from a channel registry.

    electrodes maps EVERY channel to its ElectrodeSpec drive fields
    (e.g. {"rf_groups": ["TW1"]}, {"dc": 4.0}, {"color": [r, g, b]}).
    Every registered channel must appear — a grounded channel is declared
    as {"dc": 0.0}, never implied by omission — and every mapped channel
    must exist in the registry. `source` is required: a deck with no ion
    source is not flyable, and this function refuses to guess where ions
    are born.

    Domain: the union AABB of all channels plus `margin_mm` of vacuum on
    every side, each extent snapped UP to an integer number of grid units.
    STL placement is declared as geometry.frame_offset_mm (CAD -> build
    translation = margin - AABB lower corner); the STL bytes are never
    touched. Symmetry defaults to NONE declared — declare planes only
    when the metal AND the drive assignment genuinely admit them (the
    builder verifies declarations against the masks).
    """
    missing = [ch for ch in registry.channels() if ch not in electrodes]
    unknown = [ch for ch in electrodes if ch not in registry.channels()]
    if missing or unknown:
        raise ValueError(
            "electrode wiring does not cover the registry exactly — "
            + (f"unwired channel(s): {missing}; " if missing else "")
            + (f"wired but unregistered: {unknown}; " if unknown else "")
            + "every channel is declared, even grounded ones ({'dc': 0.0})")
    no_file = [ch for ch in registry.channels() if ch not in stl_files]
    if no_file:
        raise ValueError(
            f"stl_files carries no filename for channel(s) {no_file} — "
            f"pass the mapping returned by export_stls()")
    if source is None:
        raise ValueError(
            "source is required: a flyable deck needs an ion source "
            "(pass a SourceSpec); this emitter does not guess birth "
            "positions")
    if not (mm_per_gu > 0):
        raise ValueError(f"mm_per_gu must be > 0, got {mm_per_gu!r}")
    if not (margin_mm >= 0):
        raise ValueError(f"margin_mm must be >= 0, got {margin_mm!r}")

    lo, hi = registry.aabb()
    extent = hi - lo
    dims = [_snap_up_gu(float(e) + 2.0 * margin_mm, mm_per_gu)
            for e in extent]
    frame_offset = [float(round(margin_mm - v, 9)) for v in lo]

    els = []
    for ch in registry.channels():
        fields = dict(electrodes[ch])
        bad = [k for k in fields
               if k not in ElectrodeSpec.__dataclass_fields__]
        if bad:
            raise ValueError(
                f"channel {ch!r}: unknown ElectrodeSpec field(s) {bad}; "
                f"known: {sorted(ElectrodeSpec.__dataclass_fields__)}")
        els.append(ElectrodeSpec(name=ch, stl=stl_files[ch], **fields))

    geom = GeometrySpec(
        width_mm=dims[0], height_mm=dims[1], depth_mm=dims[2],
        mm_per_gu=float(mm_per_gu),
        symmetry=symmetry if symmetry is not None
        else SymmetrySpec(coords="xyz"),
        stl_dir=stl_dir,
        frame_offset_mm=frame_offset,
        electrodes=els,
        rf_groups=list(rf_groups),
        dc_groups=list(dc_groups),
    )
    spec = SimSpec(geometry=geom, name=name, notes=notes, source=source)
    if collisions is not None:
        spec.collisions = collisions
    if integration is not None:
        spec.integration = integration
    if bounds is not None:
        spec.bounds = bounds

    errs = spec.validate()
    if errs:
        raise ValueError(
            "emitted spec does not validate:\n  " + "\n  ".join(errs))
    return spec


def write_spec_json(spec: SimSpec, path, *, stl_dir_path) -> Path:
    """Write the spec JSON with an stl_manifest ({filename: sha256}) over
    the files it references, hashed from the ACTUAL bytes on disk so a
    future load can verify the STLs are the ones this deck was authored
    with (stl_resolve.preflight_stls checks the same digests). Refuses if
    any referenced STL is missing — a manifest over absent files certifies
    nothing.
    """
    stl_dir_path = Path(stl_dir_path)
    manifest, missing = {}, []
    for el in spec.geometry.electrodes:
        if not el.stl:
            continue
        fp = stl_dir_path / el.stl
        if not fp.exists():
            missing.append(el.stl)
            continue
        manifest[el.stl] = hashlib.sha256(fp.read_bytes()).hexdigest()
    if missing:
        raise FileNotFoundError(
            f"cannot write manifest: referenced STL(s) missing from "
            f"{stl_dir_path}: {missing}")
    spec.geometry.stl_manifest = manifest
    path = Path(path)
    path.write_text(spec.to_json(indent=1))
    return path
