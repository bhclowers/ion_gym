"""
ion_gym.sim_spec
----------------
A single, self-contained JSON schema describing a WHOLE simulation, for
any geometry — not just the funnel. One SimSpec answers every setup
question:

  geometry   : electrodes as inline shapes (rect/ellipse/polygon/cutout,
               mm-space, the scene-JSON shape vocabulary) OR a
               reference to STL files (one per electrode/basis). Symmetry
               is declared (planar | cylindrical | none) so the same spec
               drives a 2-D cross-section or a 3-D solve.
  voltages   : per electrode — dc, and rf_groups naming the drive groups
               it belongs to; each drive group (geometry.rf_groups) carries
               amplitude_v, frequency_hz, phase_deg and its waveform. The
               field assembler turns these into A + sum_k s_k(t) B_k, so
               DC-only, single-RF (two-phase), and multi-phase drives are
               all expressible. Adjusting voltages never re-solves the
               bases — it re-weights them (the fast-adjust invariant), so
               the app can sweep voltages and redraw fields live.
  source     : ion births — a distribution spec (point | disc | grid |
               line | file), KE or velocity spread, mass list (Da), charge,
               time-of-birth spread, count.
  collisions : HS on/off + gas, T, P, sigma (the C-1 kernel's params).
               enabled=false gives a vacuum run (reflectron, TOF).
  integration: dt, t_max, record stride, and record_channels (which
               per-step quantities to store).
  view       : default plane(s) (xy | xz | yz) and 2d/3d preference, so a
               spec remembers how it likes to be shown.

Round-trips losslessly (to_json/from_json); forward-compatible; a saved
spec plus any referenced STL files is a complete, portable simulation.

This module is the SCHEMA only — pure data + validation. The builder that
turns a SimSpec into (field model, fly_fn) lives in sim_build.py, so the
schema has no heavy dependencies and can be edited/inspected anywhere.
"""

# The public names (docs/PUBLIC_API.md); everything else is internal.
__all__ = [
    "SimSpec", "GeometrySpec", "ElectrodeSpec", "ShapeSpec", "SymmetrySpec",
    "SourceSpec", "CollisionSpec", "IntegrationSpec", "BoundsSpec",
    "StationSpec", "ViewSpec", "RFGroupSpec", "DCGroupSpec",
    "TransporterSpec", "OpticAxisSpec",
    "MissingPhysicsKey", "SCHEMA_VERSION",
    "set_pitch", "PitchChange",
    "check_recording_capacity", "min_rec_every", "flight_steps",
    "BASE_CHANNELS", "OPTIONAL_CHANNELS", "RADIAL_CHANNELS",
    "build_travelling_wave", "retune_travelling_wave",
    "reverse_travelling_wave", "travelling_wave_groups",
]

import json
import math
from dataclasses import dataclass, field, asdict, fields, MISSING
from pathlib import Path
from typing import List, Optional, Dict, Any


SCHEMA_VERSION = 1


def check_schema_version(d, what="SimSpec"):
    """Refuse a deck whose declared schema_version this version of
    ion_gym does not read.

    A deck written for a later schema may mean something different by
    the same keys, and saving it would relabel it as this version, so it
    is refused rather than loaded on trust. A deck that declares no
    version predates versioning and is read as version 1."""
    if "schema_version" not in d:
        return          # pre-versioning deck: read as version 1
    v = d["schema_version"]
    if type(v) is not int or v != SCHEMA_VERSION:
        raise ValueError(
            f"{what}: schema_version {v!r} is not one this ion_gym reads "
            f"(it reads {SCHEMA_VERSION}). A deck written for a different "
            f"schema can mean something different by the same keys; load it "
            f"with the ion_gym that wrote it, or correct the declared "
            f"version if the content is version {SCHEMA_VERSION}.")

# Builders whose geometry is NOT carried in the spec's electrodes. For these
# the ElectrodeSpec is a VOLTAGE record (name/dc/rf_group/basis) and the
# geometry truth lives in the referenced file:
#   scene3d -> an analytic-CSG scene3d.GeomScene at import_path
# Requiring shapes-or-stl on those electrodes is a category error: it reports
# "no geometry" for a spec whose geometry is simply somewhere else.
EXTERNAL_GEOMETRY_BUILDERS = ("scene3d",)

# RF time-resolution floor (measured on an 8-rail SLIM run):
# integrating an RF drive at 25 steps/period produced numerical
# micro-heating — 10-100 eV phantom samples in a 300 K ensemble that
# the GUI (5 ns = 250 steps/period) did not show. Every stock RF
# example runs at >= 200 steps/period; 100 is the REFUSAL floor and
# the message recommends the corpus convention. Analogous to the
# pressure-coherence guard: a dt that cannot resolve the drive is a
# wrong answer, not a slow one.
RF_MIN_STEPS_PER_PERIOD = 100
RF_RECOMMENDED_STEPS_PER_PERIOD = 250
# stroboscopic-sampling guard: recording at (near-)integer multiples
# of the RF period samples the SAME drive phase every time — the
# velocity ensemble then measures one phase's micromotion, not the
# distribution. 5% commensurability window.
STROBE_WINDOW_FRAC = 0.05
# LONG-FLIGHT ADVISORY. Above this many integrator steps for the whole
# ensemble (steps per ion x ions, an upper bound: an ion that ends early
# takes fewer) the run is announced as long, with an estimate. About five
# times the largest shipped deck (the RF drift cell, 2e9). A warning, not
# a refusal: long trapping runs are legitimate; a flight too long to count
# is refused separately (MAX_FLIGHT_STEPS).
LONG_FLIGHT_ION_STEPS = 1e10
# Integrator cost per ion-step on one core, measured on shipped decks
# (r-z vacuum 32 ns; r-z and planar with gas 87-101 ns; 3-D vacuum
# 108 ns). An order of magnitude for the estimate, not a calibration:
# other machines differ, and worker threads divide the wall time.
FLIGHT_NS_PER_STEP = (30.0, 110.0)

# recordable per-step channels (same set the funnel path validated)
OPTIONAL_CHANNELS = {
    "speed": ("speed", "mm/us"), "ke_ev": ("kinetic energy", "eV"),
    "ke_x": ("KE x", "eV"), "ke_y": ("KE y", "eV"), "ke_z": ("KE z", "eV"),
    "e_field": ("|E| total", "V/mm"), "e_axial": ("E axial", "V/mm"),
    "e_radial": ("E radial", "V/mm"), "e_x": ("Ex", "V/mm"),
    "e_y": ("Ey", "V/mm"), "e_z": ("Ez", "V/mm"),
    "radius": ("radius", "mm"), "n_col": ("collisions", "count"),
    # cumulative arc length actually flown (sum of per-step |dr|, incl.
    # micromotion + thermal random walk); the "excess path" observable
    # for the RF-drift-cell transport gate.
    "path_mm": ("path length", "mm"),
    # exact per-step time integrals (kernel accumulators, no recording
    # cadence dependence -- a stroboscopic rec_every vs an RF period was
    # measured to alias <E_axial> by +11% in a transport pilot):
    "e_axial_tint": ("integral E_axial dt", "V/mm us"),
    "ke_tint": ("integral KE dt", "eV us"),
    # cumulative transporter pass count (3-D route)
    "wrap_passes": ("transporter passes", "count"),
}
BASE_CHANNELS = ["t", "x", "y", "z", "vx", "vy", "vz"]


# --------------------------------------------------------------- geometry
class MissingPhysicsKey(ValueError):
    """A deck leaves out a key that sets the physics and whose default is
    one deck's operating point."""


# ELECTRODE LIMIT: at most 255; a deck with more electrodes is refused
# with a message to contact the developer. The solved-field
# cache stores electrode labels in 8 bits (basis_cache, uint8), so label
# 256 would wrap to 0 silently; the largest deck in the tree has 140.
MAX_ELECTRODES = 255

# THE source distributions (validate and sim_build.generate_births both read
# this; a distribution added in one place and not the other is refused, not
# silently born as a point source).
SOURCE_DISTRIBUTIONS = ("point", "disc", "line", "grid", "box", "gaussian",
                        "file")


def electrode_limit_error(n):
    """The refusal text for a deck with n electrodes, or None when n fits.
    ONE message, used at load, in validate and by the cache store."""
    if n <= MAX_ELECTRODES:
        return None
    return (f"this deck has {n} electrodes; ion_gym supports at most "
            f"{MAX_ELECTRODES} (the solved-field cache stores electrode "
            f"labels in 8 bits, so more would be mislabelled). Please "
            f"contact the ion_gym developer to raise the limit.")


def nonfinite_numbers(obj, path=""):
    """[(deck path, value)] for every NaN or infinite number inside a
    nested dict/list such as a spec's to_dict(). A list item that is a
    dict with a "name" is labelled by it, so the path reads the way the
    deck does: geometry.electrodes[2](lens).dc. Booleans are not numbers
    here."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += nonfinite_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            label = (f"({v['name']})" if isinstance(v, dict) and "name" in v
                     else "")
            out += nonfinite_numbers(v, f"{path}[{i}]{label}")
    elif isinstance(obj, float) and not math.isfinite(obj):
        out.append((path, obj))
    elif (not isinstance(obj, (bool, int, float, str)) and obj is not None
          and hasattr(obj, "dtype") and getattr(obj, "shape", None) == ()
          and not math.isfinite(float(obj))):
        out.append((path, float(obj)))    # a numpy scalar
    return out


_NO_DEFAULT = object()


def _field_default(cls, name):
    for f in fields(cls):
        if f.name == name:
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:
                return f.default_factory()
    return _NO_DEFAULT


# Required keys renamed with a unit change: {new: (old, factor, old unit)},
# so a deck that states only the old key is told how to convert it.
_RENAMED_KEYS = {"dt_ns": ("dt_us", 1000.0, "microseconds")}


def _check_missing(cls, d):
    """MISSING-KEY ACCOUNTING, split between physics and cosmetic keys.
    Each spec class declares two tiers; one rule
    enforces them for every block:
      cls._load_required(d) -- keys that SET THE PHYSICS and whose
          dataclass default is one deck's operating point (556 Da, KE
          0.1-1.9 eV, 500 kHz / 50 V ...). Missing -> REFUSED by name.
      cls._LOAD_ANNOUNCE -- COSMETIC keys (display, recording density).
          Missing -> loaded at the default and ANNOUNCED by name.
    Every other key has a NEUTRAL default (zero, off, unlimited, the
    convention) and loads silently: a feature left out is a feature off,
    the same reading as "no collisions block = vacuum"."""
    req = [k for k in cls._load_required(d) if k not in d]
    if req:
        def _one(k):
            v = _field_default(cls, k)
            return k if v is _NO_DEFAULT else f"{k} (default {v!r})"
        shown = ", ".join(_one(k) for k in req)
        # a required key the deck states under an OLD name: say so, with
        # the conversion (a dt_us deck was refused naming only dt_ns)
        hints = []
        for k in req:
            old, factor, unit = _RENAMED_KEYS.get(k, (None, None, None))
            if old is not None and old in d:
                try:
                    conv = f" ({old} = {d[old]!r} -> {k} = " \
                           f"{float(d[old]) * factor:g})"
                except (TypeError, ValueError):
                    conv = ""
                hints.append(f" The deck has {old!r}, the old name of {k!r} "
                             f"in {unit}: state {k} = {old} x {factor:g}"
                             f"{conv}.")
        raise MissingPhysicsKey(
            f"{cls.__name__}: the deck does not state {shown}. These set "
            f"the physics, and the defaults are one deck's operating "
            f"point, not a neutral value -- state them in the deck."
            + "".join(hints))
    ann = [k for k in cls._LOAD_ANNOUNCE if k not in d]
    if ann:
        import warnings
        warnings.warn(
            f"{cls.__name__}: cosmetic key(s) not in the deck, loaded at "
            f"their defaults: "
            + ", ".join(f"{k}={_field_default(cls, k)!r}" for k in ann),
            stacklevel=4)


def _take_known(cls, d, checked=False):
    """LOUD LOAD: unknown keys in a spec dict are WARNED about, never
    silently dropped — the silent filter is exactly how a legacy
    'dt_us' key ran an example at the 1.0 ns default while its author
    believed 2.0 was in force. Loading still succeeds (foreign metadata
    is tolerated), but loudly and by name. MISSING keys are accounted
    for by _check_missing (refused or announced per the class's tiers);
    checked=True when the caller already ran it on the AUTHORED dict,
    before filling in keys of its own."""
    if not checked:
        _check_missing(cls, d)
    known = {f.name for f in fields(cls)}
    unknown = sorted(k for k in d if k not in known)
    if unknown:
        import warnings
        warnings.warn(
            f"{cls.__name__}: unknown key(s) {unknown} retained in the "
            f"saved file but NOT interpreted — misspelled or foreign "
            f"parameter? Known keys: {sorted(known)}", stacklevel=3)
    obj = cls(**{k: v for k, v in d.items() if k in known})
    if unknown:
        # PASSTHROUGH: foreign keys survive the round-trip at
        # their original position (see _StrictAttrs.__init_subclass__).
        # The warning above is MORE important now, not less: a retained
        # typo looks accepted forever, so every load names it.
        obj._extras = {k: d[k] for k in unknown}
    return obj


# SHAPE PARAM VOCABULARY — the schema authority for
# per-type shape params, PINNED TO MEASURED CONSUMERS, not invented:
# raster2d._rect_mask/_ellipse_mask/_polygon_mask, build_shapes3d
# _shape_volume, edit/outline.shape_outline, edit/editor_panel._FIELDS.
# rotation_deg is RECT-ONLY (no ellipse/polygon consumer exists — a
# rotated ellipse would silently not rotate, which is exactly what the
# warning is for). cutout/group carry geometry in children, so a param
# on the cutout itself (including extrude — subtraction extrude lives on
# the CHILD) is consumed by nothing. Census: 84 deck JSONs,
# 1,876 shapes, zero keys outside this table — the warning is silent on
# every shipped configuration.
SHAPE_PARAM_KEYS = {
    "rect": frozenset({"x_mm", "y_mm", "width_mm", "height_mm",
                       "rotation_deg", "extrude"}),
    "ellipse": frozenset({"cx_mm", "cy_mm", "rx_mm", "ry_mm", "extrude"}),
    "polygon": frozenset({"points_mm", "extrude"}),
    "cutout": frozenset(),
    "group": frozenset(),
}


class _StrictAttrs:
    """Refuse attribute writes to names that are not declared fields.

    The write-side twin of the loud unknown-key reader above
    (the write-side twin): a plain dataclass silently absorbs a stray attribute,
    so `spec.mm_per_gu = 0.025` (the field lives at
    spec.geometry.mm_per_gu) created a dead attribute and the run flew a
    cache hit at the OLD pitch — a wrong number with no symptom. The
    the refusal names what didn't line up, including the nested home of
    the field when one exists.

    Runtime bookkeeping that is deliberately NOT schema (never
    serialized, never a physics input) is declared per class in
    `_RUNTIME_ATTRS` — an explicit, greppable allowlist, not a silent
    absorption. Two exist today: SimSpec._loaded_from (provenance path
    set by from_json so relative resources resolve) and
    SourceSpec._drawn_seed (the seed actually drawn in random mode)."""

    _RUNTIME_ATTRS: frozenset = frozenset({"_extras"})
    # Missing-key tiers (see _check_missing). Default: nothing is
    # refused and nothing announced -- a class opts keys in.
    _LOAD_ANNOUNCE: tuple = ()

    @classmethod
    def _load_required(cls, d):
        return ()

    def __init_subclass__(cls, **kw):
        # SAVE-SIDE PASSTHROUGH (collision
        # semantics as the motivation): every subclass's to_dict is
        # wrapped ONCE, here, to re-emit the foreign keys _take_known
        # retained -- at their original positions, so a deck touched by a
        # NEWER ion_gym (or another tool) survives a round-trip
        # through THIS version's save path instead of being silently
        # stripped (the frame_offset_mm case: a dropped geometry-
        # affecting key re-solves different metal). Wrapping centrally,
        # not per-class, is deliberate: a future spec block gets
        # passthrough automatically, which is the whole version-skew
        # story -- a block someone forgets to wire is a block whose
        # future fields v-now corrupts. KNOWN WINS on save exactly as on
        # load: an extra never overrides a real emitted key.
        super().__init_subclass__(**kw)
        td = cls.__dict__.get("to_dict")
        if td is not None and not getattr(td, "_extras_wrapped", False):
            def _to_dict_with_extras(self, __td=td):
                d = __td(self)
                ex = getattr(self, "_extras", None)
                if ex:
                    for k, v in ex.items():
                        if k not in d:
                            d[k] = v
                return d
            _to_dict_with_extras._extras_wrapped = True
            _to_dict_with_extras.__doc__ = td.__doc__
            cls.to_dict = _to_dict_with_extras

    def __setattr__(self, name, value):
        flds = self.__dataclass_fields__
        if name not in flds and name not in self._RUNTIME_ATTRS:
            hints = []
            for fname in flds:
                sub = getattr(self, fname, None)
                if hasattr(sub, "__dataclass_fields__") and \
                        name in sub.__dataclass_fields__:
                    hints.append(f"{fname}.{name}")
            import difflib
            close = difflib.get_close_matches(name, flds, n=1)
            raise AttributeError(
                f"{type(self).__name__} has no field '{name}'"
                + (f" — did you mean {' or '.join(hints)}?" if hints
                   else (f" — closest field: '{close[0]}'" if close
                         else "")))
        super().__setattr__(name, value)


# The ShapeSpec extrusion convention, as ONE named authority (it was
# previously restated as a private literal in viz_core and implied by
# the edit viewer): the cross-section plane of an extrude along `axis`
# is the two remaining axes in CYCLIC order, and the shape's own
# x/y params read as the (first, second) in-plane coordinate.
#   origin: this module's ShapeSpec extrude docstring (below);
#   consumers: viz_core._CYCLIC (aliased), ion_gym.edit (viewer frame).
EXTRUDE_INPLANE_AXES = {"x": ("y", "z"), "y": ("z", "x"),
                        "z": ("x", "y")}


# ONE SHAPE-OFFSET AUTHORITY, for every shape kind the
# rasterizer draws (raster2d._shape_mask: rect, ellipse, polygon; a
# cutout subtracts its child). It lived in the editor (edit.outline.
# shift_shape, which now delegates here); the assembly flatten shifted
# only x_mm/y_mm, so an ellipse or polygon unit stayed at its own origin.
def shift_shape_dict(shape: dict, dx: float = 0.0, dy: float = 0.0,
                     dz: float = 0.0) -> dict:
    """A COPY of a shape dict (ShapeSpec.to_dict form) translated by
    (dx, dy) in its own plane and dz along its extrude axis. A kind with
    no shift rule is refused by name -- never passed through unmoved."""
    import copy as _copy
    out = _copy.deepcopy(shape)
    t = out.get("type")
    if t == "rect":
        out["x_mm"] = float(out["x_mm"]) + dx
        out["y_mm"] = float(out["y_mm"]) + dy
    elif t == "ellipse":
        out["cx_mm"] = float(out["cx_mm"]) + dx
        out["cy_mm"] = float(out["cy_mm"]) + dy
    elif t == "polygon":
        out["points_mm"] = [[float(a) + dx, float(b) + dy]
                            for a, b in out["points_mm"]]
    elif t == "cutout":
        out["children"] = [shift_shape_dict(k, dx, dy, dz)
                           for k in (out.get("children") or [])]
        return out
    else:
        raise ValueError(f"no shift rule for shape type {t!r} (known: "
                         f"rect, ellipse, polygon, cutout)")
    ex = out.get("extrude")
    if isinstance(ex, dict) and dz:
        ex["lo_mm"] = float(ex["lo_mm"]) + dz
        ex["hi_mm"] = float(ex["hi_mm"]) + dz
    return out


def shape_metal_bounds_mm(shape: dict):
    """In-plane (x_lo, x_hi, y_lo, y_hi) of the METAL a shape adds, or None
    for a cutout (it only removes metal). Rects rotate about their centre
    exactly as raster2d._rect_mask does. A kind with no rule is refused."""
    t = shape.get("type")
    if t == "rect":
        x0, y0 = float(shape["x_mm"]), float(shape["y_mm"])
        w, h = float(shape["width_mm"]), float(shape["height_mm"])
        rot = math.radians(float(shape.get("rotation_deg", 0.0) or 0.0))
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        ca, sa = math.cos(rot), math.sin(rot)
        xs, ys = [], []
        for px, py in ((x0, y0), (x0 + w, y0), (x0, y0 + h),
                       (x0 + w, y0 + h)):
            xs.append(cx + (px - cx) * ca - (py - cy) * sa)
            ys.append(cy + (px - cx) * sa + (py - cy) * ca)
        return min(xs), max(xs), min(ys), max(ys)
    if t == "ellipse":
        cx, cy = float(shape["cx_mm"]), float(shape["cy_mm"])
        rx, ry = float(shape["rx_mm"]), float(shape["ry_mm"])
        return cx - rx, cx + rx, cy - ry, cy + ry
    if t == "polygon":
        pts = [(float(a), float(b)) for a, b in shape["points_mm"]]
        return (min(p[0] for p in pts), max(p[0] for p in pts),
                min(p[1] for p in pts), max(p[1] for p in pts))
    if t == "cutout":
        return None
    raise ValueError(f"no bounds rule for shape type {t!r} (known: rect, "
                     f"ellipse, polygon, cutout)")


@dataclass
class ShapeSpec(_StrictAttrs):
    """A geometry primitive in mm-space. type in
    {rect, ellipse, polygon, cutout, group}. Fields mirror
    the scene-JSON shape vocabulary so scene JSONs translate directly.

    OPTIONAL EXTRUSION: params may
    carry
        "extrude": {"axis": "x"|"y"|"z", "lo_mm": float, "hi_mm": float}
    declaring that this 2-D cross-section occupies [lo_mm, hi_mm] along
    `axis`. The cross-section plane is the two axes perpendicular to
    `axis` in CYCLIC order — x->(y,z), y->(z,x), z->(x,y) — with the
    shape's own x/y params reading as the (first, second) in-plane
    coordinate. Absent descriptor: 2-D builds are byte-identical to the
    pre-descriptor behavior; a 3-D shapes build treats the shape as a
    full-depth slab (build_shapes3d.DEFAULT_EXTRUDE_AXIS). Coordinates
    are the STORED frame; on a declared mirror axis the stored frame is
    the non-negative half with the mirror plane at 0 supplying the
    image (lo_mm < 0 there is refused by the builder)."""
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List["ShapeSpec"] = field(default_factory=list)

    def extrude(self):
        """Validated extrusion descriptor as a NAMED record:
        {"axis", "lo_mm", "hi_mm"}, or None when absent. Malformed
        descriptors refuse with the offending values named — never a
        silent default."""
        legacy = [k for k in ("z0_mm", "z1_mm") if k in self.params]
        if legacy:
            raise ValueError(
                f"shape {self.type!r} carries legacy extrusion key(s) "
                f"{legacy} (older extrude_spec output). These "
                f"were superseded by the extrude descriptor; reading "
                f"them as full-length would be silently wrong. Rewrite "
                f"as extrude={{'axis': 'z', 'lo_mm': ..., 'hi_mm': ...}} "
                f"or regenerate the spec with the current extrude_spec.")
        d = self.params.get("extrude")
        if d is None:
            return None
        if not isinstance(d, dict):
            raise ValueError(
                f"shape {self.type!r}: extrude must be a mapping with "
                f"axis/lo_mm/hi_mm, got {type(d).__name__} {d!r}")
        missing = [k for k in ("axis", "lo_mm", "hi_mm") if k not in d]
        if missing:
            raise ValueError(
                f"shape {self.type!r}: extrude is missing {missing} "
                f"(got keys {sorted(d)})")
        unknown = [k for k in d if k not in ("axis", "lo_mm", "hi_mm")]
        if unknown:
            raise ValueError(
                f"shape {self.type!r}: extrude has unknown keys "
                f"{unknown} (allowed: axis, lo_mm, hi_mm)")
        ax = d["axis"]
        if ax not in ("x", "y", "z"):
            raise ValueError(
                f"shape {self.type!r}: extrude axis must be one of "
                f"'x'|'y'|'z', got {ax!r}")
        lo = float(d["lo_mm"])
        hi = float(d["hi_mm"])
        if not (lo <= hi):
            raise ValueError(
                f"shape {self.type!r}: extrude lo_mm ({lo:g}) must be "
                f"<= hi_mm ({hi:g})")
        return {"axis": ax, "lo_mm": lo, "hi_mm": hi}

    # Size parameters that must be positive. The rasterizers do not refuse
    # a bad one: an ellipse squares its radii, so a negative radius draws
    # the same ellipse as the positive one, and a zero one draws almost
    # nothing; a non-positive rect side draws nothing.
    SIZE_PARAMS = ("width_mm", "height_mm", "rx_mm", "ry_mm")

    def size_errors(self, where):
        """Refusals for non-positive size parameters, for this shape and
        its children (a cutout's inner shape), naming `where`."""
        errs = []
        for k in self.SIZE_PARAMS:
            v = self.params.get(k)
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v <= 0):
                errs.append(f"{where} ({self.type}): {k} = {v!r} must be > 0 "
                            f"-- a non-positive size is drawn as its mirror "
                            f"or as nothing, not refused, by the rasterizer")
        for i, c in enumerate(self.children):
            errs += c.size_errors(f"{where}.children[{i}]")
        return errs

    def to_dict(self):
        d = {"type": self.type, **self.params}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        t = d.pop("type")
        children = [cls.from_dict(c) for c in d.pop("children", [])]
        # Loud load at SHAPE level (the last silent tier): a
        # misspelled OPTIONAL param ('rotation_dg') previously vanished
        # into params and the rasterizer flew the default — the dt_us
        # failure mode, one level down. Unknown keys WARN by name but
        # are still ABSORBED: params round-trips through to_dict by
        # design, so dropping here would create the exact re-serialize
        # data loss this row is about. Loading always succeeds.
        known = SHAPE_PARAM_KEYS.get(t)
        if known is None:
            import warnings
            warnings.warn(
                f"ShapeSpec: shape type {t!r} is not in the schema "
                f"vocabulary {sorted(SHAPE_PARAM_KEYS)} — params "
                f"unchecked; a builder that cannot rasterize it will "
                f"refuse it by name", stacklevel=3)
        else:
            unknown = sorted(k for k in d if k not in known)
            if unknown:
                import warnings
                warnings.warn(
                    f"ShapeSpec ({t}): unknown param key(s) {unknown} "
                    f"— misspelled or legacy? No consumer reads them, "
                    f"so they have NO effect on the build (they are "
                    f"preserved through to_dict). Known keys for "
                    f"{t!r}: {sorted(known)}", stacklevel=3)
        return cls(type=t, params=d, children=children)


@dataclass
class RFGroupSpec(_StrictAttrs):
    """A named DRIVE group: a scalar waveform w(t) that any number of
    electrodes can be assigned to. The electrode voltage from the group is
        V_el(t) = amplitude_v * w(t)      (on top of the electrode's dc)
    and because the field is LINEAR in electrode voltages with precomputed
    fast-adjust bases, this one primitive is the reference-scripting
    equivalent for every drive whose waveform does not depend on ion
    state:
      * DC-only element  -> assigned to no group (rf_group = None)
      * RF-only element  -> dc = 0, assigned to a group
      * RF+DC element    -> dc set, assigned to a group
    Two sin groups 180 apart = a funnel or quad rod-pair drive; N groups
    at stepped phases + cyclic electrode assignment = a travelling wave
    (sinusoidal or the real square-stepped SLIM drive); a table group =
    pulsed extraction / gates / ramps at explicit time breakpoints.

    waveform:
      'sin'    w = sin(2*pi*f*t + phase)                (analytic, exact)
      'square' w = sign(sin(2*pi*f*t + phase))          (stepped TW /
               digital-trap drive: with N phase groups at 360/N steps this
               is the classic half-up/half-down pattern shifting one
               electrode per T/N)
      'table'  w = interp(table_t_us, table_v), 'hold' (zero-order,
               stepped tstep-style switching) or 'linear' (ramps). Values
               are DIMENSIONLESS multipliers of amplitude_v; outside the
               breakpoint range w clamps to the end values.

    CLOCK CONVENTION: w(t) runs on the LAB clock (t = time-of-birth +
    ion's own flight time). Fields are lab-frame objects; a birth stagger
    changes the drive phase an ion is born into — that is the physics.

    pe_mode controls the PE-surface VISUALIZATION only (never dynamics):
      'pseudo'  contribute the adiabatic Dehmelt envelope (valid for the
                FAST drive, e.g. ~MHz confinement RF; square-wave drives
                get the digital-trap harmonic factor pi^2/6)
      'instant' contribute the instantaneous potential w(t_view)*B — the
                right picture for a SLOW travelling wave the ions surf
                (a ~40 kHz TW is NOT adiabatic; ions do not average it)
      None      default by waveform: sin -> 'pseudo',
                square/table -> 'instant'.
    The SIMULATION always integrates the real time-varying field; the
    pseudopotential never enters the dynamics.
    """
    name: str
    frequency_hz: float = 5e5
    amplitude_v: float = 50.0
    phase_deg: float = 0.0
    duty: float = 0.5              # squares only: fraction of the period
                                   # HIGH from phase 0 (11110000 = 0.5,
                                   # 11000000 = 0.25 at stepped phases)
    offset_v: float = 0.0          # DC offset of THIS drive: w = amp*base
                                   # + offset. A unipolar 0..V square (the
                                   # SLIM stepped TW, a 1/0 table) is
                                   # amplitude_v=V/2, offset_v=V/2.
    waveform: str = "sin"          # 'sin' | 'square' | 'table'
    table_t_us: list = field(default_factory=list)
    table_v: list = field(default_factory=list)
    interp: str = "hold"           # tables: 'hold' | 'linear'
    pe_mode: str = None            # None | 'pseudo' | 'instant'

    @classmethod
    def _load_required(cls, d):
        # The drive's size and rate are one deck's operating point
        # (50 V / 500 kHz). A table waveform takes its timing from its
        # breakpoints, so it has no frequency to state.
        if d.get("waveform", "sin") == "table":
            return ("amplitude_v",)
        return ("amplitude_v", "frequency_hz")

    def resolved_pe_mode(self):
        if self.pe_mode in ("pseudo", "instant"):
            return self.pe_mode
        return "pseudo" if self.waveform == "sin" else "instant"

    def validate(self):
        if not (0.0 < float(self.duty) < 1.0):
            raise ValueError(f"RF group {self.name!r}: duty {self.duty} "
                             f"outside (0, 1)")
        if self.waveform not in ("sin", "square", "table"):
            raise ValueError(f"RF group {self.name!r}: unknown waveform "
                             f"{self.waveform!r}")
        if self.waveform == "table":
            t = list(self.table_t_us)
            if len(t) < 1 or len(t) != len(self.table_v):
                raise ValueError(
                    f"RF group {self.name!r}: table needs matching, "
                    f"non-empty table_t_us/table_v")
            if any(b <= a for a, b in zip(t, t[1:])):
                raise ValueError(
                    f"RF group {self.name!r}: table_t_us must be strictly "
                    f"increasing")
            if self.interp not in ("hold", "linear"):
                raise ValueError(f"RF group {self.name!r}: interp must be "
                                 f"'hold' or 'linear'")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


def travelling_wave_groups(n, frequency_hz=5e5, amplitude_v=50.0,
                           phase_step_deg=None, prefix="TW",
                           waveform="sin", duty=0.5, offset_v=0.0):
    """Generate n RF groups at evenly stepped phases — a travelling wave.
    phase_step defaults to +360/n (one full wave across the n groups).
    DIRECTION (measured, Gate C): with w_k = base(om*t + k*step), higher-k
    electrodes cross EARLIER, so +360/n steps march the wave toward
    DECREASING electrode index (-x for cyclic k -> k%n assignment along +x).
    Pass phase_step_deg = -360/n for a wave toward +x."""
    step = phase_step_deg if phase_step_deg is not None else 360.0 / n
    return [RFGroupSpec(name=f"{prefix}{k}", frequency_hz=frequency_hz,
                        amplitude_v=amplitude_v, phase_deg=k * step,
                        waveform=waveform, duty=duty, offset_v=offset_v)
            for k in range(n)]


def reverse_travelling_wave(spec, *, prefix="TW"):
    """Flip the propagation DIRECTION of an existing travelling-wave
    ladder (a TW specified one way sometimes needs to run the other).

    Each TW group drives waveform(w*t + phase_k); the wave's direction
    lives entirely in the SIGN of the phase progression across the
    groups, so negating every phase reverses the travel while leaving
    amplitude, frequency, waveform, duty and offset untouched. Phase is
    drive-side, so the reversed deck is a basis-cache HIT — never a
    re-solve. Applying it twice restores the original ladder exactly.

    Selection matches retune_travelling_wave: every rf_group whose name
    starts with `prefix` (which also catches split ladders named
    TW1_a/TW1_b — both halves flip together, keeping the wave continuous
    across a seam). Mutates `spec` in place and returns the reversed
    group names. Raises (not a silent no-op) if no group matches, so a
    mis-prefixed call is caught rather than quietly doing nothing."""
    g = spec.geometry
    tw = [gr for gr in g.rf_groups if gr.name.startswith(prefix)]
    if not tw:
        raise ValueError(
            "reverse_travelling_wave: no groups with prefix {0!r} — "
            "nothing to reverse (build the ladder first, or pass the "
            "ladder's actual prefix)".format(prefix))
    for gr in tw:
        gr.phase_deg = -float(gr.phase_deg)
    return [gr.name for gr in tw]


def retune_travelling_wave(spec, *, amplitude_v=None, frequency_hz=None,
                           waveform=None, duty=None, offset_v=None,
                           phase_step_deg=None, prefix="TW"):
    """Retune an EXISTING travelling-wave ladder from ONE set of values,
    fanning to every phase group at once (the TW drive
    is one physical wave, so amplitude/frequency should be a single knob,
    not eight separate edits).

    Only the phase DIFFERS between the groups; amplitude, frequency,
    waveform, duty and offset are shared properties of the wave and are
    written to every TW group. The per-group phase is preserved unless
    phase_step_deg is given, in which case the whole ladder is re-stepped
    (group k -> k * step) in creation order — the same convention
    travelling_wave_groups uses, so direction stays consistent.

    Every argument left None is unchanged. Mutates `spec` in place and
    returns the list of retuned group names. Raises (not a silent no-op)
    if no TW ladder is present, so a mis-prefixed call is caught rather
    than quietly doing nothing.
    """
    g = spec.geometry
    tw = [gr for gr in g.rf_groups if gr.name.startswith(prefix)]
    if not tw:
        raise ValueError(
            "retune_travelling_wave: no groups with prefix {0!r} — build "
            "the ladder first with build_travelling_wave".format(prefix))
    # creation order = trailing integer (TW0, TW1, ...): the phase index,
    # so re-stepping matches travelling_wave_groups' k*step convention.
    # A suffix that is not an integer has no phase index: refused by name
    # (it used to sort as 0, silently re-phasing that group).
    def _k(gr):
        tail = gr.name[len(prefix):]
        if not tail.isdigit():
            raise ValueError(
                f"retune_travelling_wave: group {gr.name!r} matches prefix "
                f"{prefix!r} but its suffix {tail!r} is not an integer "
                f"phase index (TW0, TW1, ...)")
        return int(tail)
    tw_sorted = sorted(tw, key=_k)
    step = phase_step_deg
    for k, gr in enumerate(tw_sorted):
        if amplitude_v is not None:
            gr.amplitude_v = float(amplitude_v)
        if frequency_hz is not None:
            gr.frequency_hz = float(frequency_hz)
        if waveform is not None:
            gr.waveform = waveform
        if duty is not None:
            gr.duty = float(duty)
        if offset_v is not None:
            gr.offset_v = float(offset_v)
        if step is not None:
            gr.phase_deg = k * step
    return [gr.name for gr in tw_sorted]


def load_drive_template(target_spec, source_spec):
    """Transplant the VOLTAGE/DRIVE setup (rf_groups + dc_groups) from
    `source_spec` into `target_spec`, touching NOTHING else — geometry
    (electrode shapes), integration, physics/collisions, and the ion
    source are all left exactly as they were (this is a
    template loader for the voltage components only).

    Conflict policy: WIPE the target's existing drive
    groups entirely and replace them with the source's; then CLEAR every
    electrode's drive bindings (rf_groups -> [], dc_group -> None,
    dc_index -> None, dc_weight -> None) so the user re-assigns from
    scratch. This leaves a clean, unambiguous state — no dangling
    references to now-deleted group names, which resolve_dc_groups() and
    the composer would otherwise choke on.

    The DC VALUE on each electrode (`dc`) is a per-electrode property, not
    a group, so it is left as-is; only GROUP MEMBERSHIP is cleared.

    Returns a report dict (moved counts + names + how many electrodes were
    cleared) so the caller can tell the user exactly what happened — a
    silent transplant would hide a destructive change.
    """
    import copy
    tg = target_spec.geometry
    sg = source_spec.geometry
    old_rf = [gr.name for gr in tg.rf_groups]
    old_dc = [gr.name for gr in tg.dc_groups]
    new_rf = [gr.name for gr in sg.rf_groups]
    new_dc = [gr.name for gr in sg.dc_groups]
    tg.rf_groups = [copy.deepcopy(gr) for gr in sg.rf_groups]
    tg.dc_groups = [copy.deepcopy(gr) for gr in sg.dc_groups]
    cleared = 0
    for e in tg.electrodes:
        had = bool(e.rf_groups) or e.dc_group is not None
        e.rf_groups = []
        e.dc_group = None
        e.dc_index = None
        e.dc_weight = None
        if had:
            cleared += 1
    return dict(rf_removed=old_rf, dc_removed=old_dc,
                rf_loaded=new_rf, dc_loaded=new_dc,
                electrodes_cleared=cleared,
                electrodes_total=len(tg.electrodes))


def build_travelling_wave(spec, electrode_names, n_phases, *,
                          frequency_hz=25e3, amplitude_v=50.0,
                          waveform="square", duty=0.5, offset_v=0.0,
                          phase_step_deg=None, prefix="TW",
                          replace_existing=True):
    """One call to lay a travelling wave on an ORDERED list of electrodes
    (wiring
    n_phases stepped-phase drive groups and assigns the electrodes to them
    CYCLICALLY (electrode j -> group j % n_phases), which is what makes a
    wave. So 200 segments still need only n_phases groups.

    electrode_names : ordered electrode names along the wave axis (order IS
                      the wave direction; the cyclic k=j%n_phases assignment
                      steps the phase one electrode per segment).
    n_phases        : number of distinct phases (4 and 8 are common SLIM).
    waveform        : 'square' (classic stepped SLIM TW), 'sin', or 'table'.
    unipolar 0->V   : set amplitude_v=V/2, offset_v=V/2 (SLIM digital drive).

    Mutates `spec` in place and returns the list of created group names.
    Refuses (with a diagnostic) unknown electrode names or n_phases<2 rather
    than silently mis-wiring the wave.
    """
    g = spec.geometry
    by_name = {e.name: e for e in g.electrodes}
    missing = [nm for nm in electrode_names if nm not in by_name]
    if missing:
        raise ValueError(
            "build_travelling_wave: these electrode names are not in the "
            "geometry: {0}".format(", ".join(missing)))
    if int(n_phases) < 2:
        raise ValueError("build_travelling_wave: n_phases must be >= 2 "
                         "(got {0})".format(n_phases))
    n = int(n_phases)
    groups = travelling_wave_groups(
        n, frequency_hz=frequency_hz, amplitude_v=amplitude_v,
        phase_step_deg=phase_step_deg, prefix=prefix,
        waveform=waveform, duty=duty, offset_v=offset_v)
    names = [gr.name for gr in groups]
    if replace_existing:
        # drop any prior groups with this prefix + detach their members,
        # so re-running the builder is idempotent (no orphan groups).
        drop = {gr.name for gr in g.rf_groups if gr.name.startswith(prefix)}
        if drop:
            g.rf_groups = [gr for gr in g.rf_groups if gr.name not in drop]
            for e in g.electrodes:
                if e.rf_groups:
                    e.rf_groups = [x for x in e.rf_groups if x not in drop]
    g.rf_groups = list(g.rf_groups) + groups
    # CYCLIC assignment: electrode j on the ladder -> phase group j % n.
    # additive to any existing rf_groups (an electrode may also carry
    # confinement RF from another group).
    for j, nm in enumerate(electrode_names):
        e = by_name[nm]
        gname = names[j % n]
        e.rf_groups = [x for x in (e.rf_groups or []) if not x.startswith(prefix)]
        e.rf_groups.append(gname)
    return names


@dataclass
class DCGroupSpec(_StrictAttrs):
    """A DC ladder: an ORDERED set of electrodes fed from two ends.

    v_in is applied at the LOWEST dc_index in the group, v_out at the HIGHEST,
    and the members in between are interpolated. This is the resistor-divider
    chain of a drag-field guide (Cao Q3: E3..E35, DC2 in, DC3 out), but nothing
    here knows that: any electrodes, any indices, any number of groups.

    ORDER IS EXPLICIT. Membership carries a NUMBER (ElectrodeSpec.dc_index),
    not a name to be parsed and not a position in a list. "E10" sorting before
    "E9" is exactly the kind of silent mis-ordering that would put a monotonic
    ramp out of order and still solve.

    THE MEMBER dc IS DERIVED, NOT AUTHORED. resolve_dc_groups() recomputes it
    before every use. A group member whose dc is also written by hand is two
    sources of truth for one voltage, and validate() refuses it.

    interp:
      'linear'  -- equal rungs (an ideal divider with identical resistors)
      'weights' -- per-member `dc_weight` in [0,1] positions each tap along
                   the ramp, for an unequal ladder. Set exactly one.

    mode (groups can be additive):
      'set' -- everything above: the group OWNS the member voltage, the
               member dc is derived, not authored. The ladder/divider
               semantics every shipped deck uses; the default, so every
               existing deck is unchanged.
      'add' -- a FLOAT rail: the member dc stays AUTHORED and editable,
               and the group's per-member value (v_in for uniform, the
               interpolated tap for a ladder) is ADDED on top at
               composition time. Physically a supply riding on a common
               float. The addition is applied by SimSpec.dc_effective()
               inside the static-field composers and is NEVER written
               into the member: an additive resolve_dc_groups() would
               compound on every call and leak summed voltages into
               saved decks. Sweeping the float is a basis re-weight,
               never a re-solve.
    """
    name: str = "LADDER"
    v_in: float = 0.0
    v_out: float = 0.0
    interp: str = "linear"          # 'linear' | 'weights'
    mode: str = "set"               # 'set' (group owns dc) | 'add' (float)
    uniform: bool = False           # plain equal-V set: all members at v_in
    # A 'uniform' group is a non-ladder DC group — every
    # member sits at the SAME voltage (v_in); v_out tracks it. It is exactly
    # a ladder with v_in==v_out, surfaced as its own kind so the UI shows one
    # voltage field instead of in/out/number. The ladder is unchanged.

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Through the one key-accounting door: this filtered on
        # __dataclass_fields__, so a misspelled 'v_inn' loaded as v_in =
        # 0.0 with no warning.
        return _take_known(cls, dict(d or {}))


def dc_group_member_values(grp, mem):
    """{member electrode name: the GROUP's per-member voltage}.

    THE one implementation of a DC group's value distribution — uniform,
    linear ladder, weighted ladder — factored out of resolve_dc_groups()
    so that 'set' mode (which WRITES these values into member dc) and
    'add' mode (which ADDS them at composition, via dc_effective) cannot
    drift into two interpolation rules. The refusals are the resolve
    originals, verbatim in meaning: a weighted ladder with an unweighted
    member and an indexed ladder with an unindexed member are refused by
    name, never silently parked at v_in."""
    if grp.uniform:
        return {e.name: float(grp.v_in) for e in mem}
    if grp.interp == "weights":
        _now = [e.name for e in mem if e.dc_weight is None]
        if _now:
            raise ValueError(
                f"dc_group {grp.name!r} (interp='weights'): "
                f"member(s) {_now} have no dc_weight — a "
                f"weighted ladder needs every tap placed")
        return {e.name: float(grp.v_in)
                + (float(grp.v_out) - float(grp.v_in)) * float(e.dc_weight)
                for e in mem}
    idx = [e.dc_index for e in mem]
    _noi = [e.name for e in mem if e.dc_index is None]
    if _noi:
        raise ValueError(
            f"dc_group {grp.name!r} (interp="
            f"{grp.interp!r}): member(s) {_noi} have no "
            f"dc_index — an indexed ladder needs every "
            f"rung numbered")
    lo, hi = min(idx), max(idx)
    span = (hi - lo) or 1
    return {e.name: float(grp.v_in)
            + (float(grp.v_out) - float(grp.v_in)) * (e.dc_index - lo) / span
            for e in mem}


@dataclass
class ElectrodeSpec(_StrictAttrs):
    """A named conductor: geometry (inline shapes OR an STL file) + a
    voltage assignment. DC and AC/RF are INDEPENDENT axes: every
    electrode has a `dc` (possibly 0) AND membership in zero or more
    named drive groups via `rf_groups` (a list of group names). An
    electrode may belong to SEVERAL groups at once (e.g. a segment
    carrying both a confinement RF and a travelling-wave AC) plus a DC
    bias. DC-only, single-RF, RF+DC, multi-AC+DC, and grounded are all
    expressible without a mode flag.

    THE DRIVE MODEL: a group's waveform, frequency,
    amplitude, AND PHASE live on the RFGroupSpec — never on the
    electrode. A rod pair is two groups 180 deg apart with the rods
    assigned one each; a travelling wave is N groups at stepped phases
    with electrodes assigned cyclically. There is no per-electrode phase
    or amplitude: that made two ways to say one thing.

    Back-compat on LOAD only: from_dict folds a legacy singular
    `rf_group` into `rf_groups`. The obsolete per-electrode
    rf_amplitude/rf_frequency_hz/rf_phase_deg are dropped when zero (zero
    asks for nothing the group model does not already do) and REFUSED
    when non-zero, because the group model never applies them. New specs
    never write them."""
    name: str
    shapes: List[ShapeSpec] = field(default_factory=list)
    stl: Optional[str] = None
    is_grid: bool = False
    # voltages: DC always applies; drive-group membership(s) optional
    dc: float = 0.0
    rf_groups: List[str] = field(default_factory=list)
    dc_group: Optional[str] = None     # membership in a DCGroupSpec
    dc_index: Optional[int] = None     # position in that ladder (the NUMBER)
    dc_weight: Optional[float] = None  # 0..1 tap position, interp='weights'
    color: List[int] = field(default_factory=lambda: [200, 60, 60])
    basis: Optional[int] = None
    # DECLARED z extent of THIS
    # electrode's metal, overriding the stage's `geometry.metal_depth_mm`.
    # None (the default) means "use the stage's declaration".
    #
    # NAMED `metal_depth_mm`, NOT `depth_mm`, and the distinction is
    # load-bearing: `geometry.depth_mm` is the 2-D/3-D ROUTE SWITCH ("0
    # for 2-D"), and overloading it for this declaration flipped a planar
    # stage into a full 3-D STL solve at ~120 s PER ELECTRODE — caught by
    # its own adversarial test, not by reading. This field is a
    # STATEMENT ABOUT HARDWARE that no solver ever consults: it exists so
    # drawings can be honest and the assembly clearance check has
    # something to check. Convention: metal spans z in [-d/2, +d/2] in
    # the STAGE frame, before the stage pose is applied.
    metal_depth_mm: Optional[float] = None

    def group_names(self):
        """AC/RF group memberships, order-stable and de-duplicated. The
        ONE authority — every consumer asks here."""
        out = []
        for g in self.rf_groups:
            if g and g not in out:
                out.append(g)
        return out

    def describe(self) -> str:
        """One readable line stating how this conductor is CONSTRUCTED:
        where its metal comes from, and what drives it. Reused by
        SimSpec.describe(); teaching notebooks print it directly."""
        if self.stl:
            geom = f"metal from STL '{self.stl}'"
        elif self.shapes:
            kinds = [s.to_dict().get("type", "?") for s in self.shapes]
            counted = ", ".join(f"{kinds.count(k)}x {k}"
                                for k in dict.fromkeys(kinds))
            geom = f"metal from {len(self.shapes)} inline shape(s) ({counted})"
        else:
            # a conductor with no metal is describable, but say so loudly:
            # validate() is the refusal path; describe() states the truth.
            geom = "NO METAL DECLARED (no shapes, no stl)"
        parts = [f"{self.name}: {geom}"]
        if self.is_grid:
            parts.append("grid (ion-transparent mesh)")
        drive = f"dc {self.dc:+g} V"
        if self.group_names():
            drive += " + drive group(s) " + ", ".join(self.group_names())
        parts.append(drive)
        if self.dc_group is not None:
            tap = (f"weight {self.dc_weight:g}" if self.dc_weight is not None
                   else f"index {self.dc_index}")
            # mode lives on the GROUP, which this per-electrode view
            # cannot see: 'set' derives the dc, 'add' floats on top of it
            parts.append(f"DC group '{self.dc_group}' tap {tap} "
                         "(set mode derives dc; add mode floats it)")
        if self.metal_depth_mm is not None:
            parts.append(f"metal depth {self.metal_depth_mm:g} mm "
                         "(overrides the stage declaration)")
        return " | ".join(parts)

    @property
    def is_rf(self):
        return bool(self.group_names())

    _LOAD_ANNOUNCE = ("color",)          # display only

    def to_dict(self):
        d = asdict(self)
        d["shapes"] = [s.to_dict() for s in self.shapes]
        return d

    @classmethod
    def from_dict(cls, d):
        _check_missing(cls, d)
        d = dict(d)
        d["shapes"] = [ShapeSpec.from_dict(s) for s in d.get("shapes", [])]
        # LEGACY FOLD (clean drive model): singular rf_group -> rf_groups,
        # which means the same thing in the group model.
        groups = list(d.get("rf_groups", []) or [])
        if d.get("rf_group") and d["rf_group"] not in groups:
            groups = [d["rf_group"]] + groups
        d["rf_groups"] = groups
        d.pop("rf_group", None)
        # LEGACY PER-ELECTRODE DRIVE: amplitude, frequency and phase live
        # on the drive group (RFGroupSpec), and nothing reads these keys.
        # A ZERO value asks for nothing the group model does not already
        # do, so it is dropped on purpose (legacy decks carry
        # rf_phase_deg: 0 on every electrode). A NON-ZERO value is a drive
        # the deck asks for and would not get -- two plates at phase 0 and
        # 180 flew in phase, and per-electrode amplitudes flew with no RF
        # at all -- so it is refused, naming the way to say it.
        for key in _LEGACY_ELECTRODE_DRIVE:
            if key not in d:
                continue
            v = d.pop(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    and v == 0:
                continue      # zero: a deliberate no-op, see above
            raise ValueError(
                f"electrode {d.get('name', '?')!r}: per-electrode {key} = "
                f"{v!r} is not part of the drive model and would be ignored. "
                f"Amplitude, frequency and phase belong to a drive group: "
                f"declare geometry.rf_groups entries with amplitude_v, "
                f"frequency_hz and phase_deg (e.g. two groups 180 deg apart "
                f"for a rod pair) and list the group names in this "
                f"electrode's rf_groups.")
        return _take_known(cls, d, checked=True)


# Per-electrode drive keys from before the drive-group model (see
# ElectrodeSpec.from_dict): dropped when zero, refused otherwise.
_LEGACY_ELECTRODE_DRIVE = ("rf_amplitude", "rf_frequency_hz", "rf_phase_deg")


# RESOLUTION ADVISORY THRESHOLD, in CELLS.
# Field sampling between nodes is (tri)linear, so INSIDE one cell the
# interpolated field varies linearly BY CONSTRUCTION: any structure of a
# packet narrower than a cell is the stencil's, not the field's. Measured
# cost of ignoring this: the oa3d campaign ran a packet 0.19 cells wide
# and reported a y-time aberration 5x its converged value, with the wrong
# functional form (a V instead of a quadratic), then optimised against it
# for a long time. Two cells is the advisory line, not a proof of
# sufficiency -- convergence is demonstrated by a pitch ladder, never
# assumed from a threshold.
SOURCE_CELLS_ADVISORY = 2.0

AXES = ("x", "y", "z")
KINDS = ("none", "mirror", "translational")


@dataclass
class SymmetrySpec(_StrictAttrs):
    coords: str = "xyz"                       # 'rz' | 'xyz'
    planes: Dict[str, str] = field(
        default_factory=lambda: {"x": "none", "y": "none", "z": "none"})
    # DECLARED PLANE LOCATION:
    # axis -> physical coordinate (mm, spec frame) of the declared plane.
    # An ABSENT axis means the domain midline -- exactly today's meaning --
    # so every existing spec/JSON behaves byte-identically. This field is
    # the single authority consumers (gates, display, builders) ask for a
    # plane's location; guessing it from a correlate (source y0, argmin of
    # a grid) is the defect class this field retires.
    plane_mm: Dict[str, float] = field(default_factory=dict)

    def kind(self, axis):
        return self.planes.get(axis, "none")

    def plane(self, axis, extent_mm):
        """The declared plane coordinate on `axis`, defaulting to the
        domain midline of `extent_mm` when none is declared. Only
        meaningful for axes whose kind is 'mirror'."""
        v = self.plane_mm.get(axis)
        return 0.5 * float(extent_mm) if v is None else float(v)

    def normalized(self):
        """Fill any missing axes with 'none'; for rz, z is meaningless
        (2-D), so force it to 'none'."""
        p = {a: self.planes.get(a, "none") for a in AXES}
        if self.coords == "rz":
            p["z"] = "none"
        # plane_mm MUST survive normalization: the fold gate asks
        # `axis in sym.plane_mm` on the NORMALIZED spec — dropping it here
        # would silently re-enable folding for every located plane (the
        # exact hidden branch the gate exists to prevent). Locations for
        # axes normalized to 'none' are dropped with the axis.
        return SymmetrySpec(coords=self.coords, planes=p,
                            plane_mm={a: float(v)
                                      for a, v in self.plane_mm.items()
                                      if p.get(a, "none") != "none"})

    def validate(self):
        errs = []
        if self.coords not in ("rz", "xyz"):
            errs.append(f"coords must be 'rz' or 'xyz', got {self.coords!r}")
        for a, k in self.planes.items():
            if a not in AXES:
                errs.append(f"unknown symmetry axis {a!r}")
            if k not in KINDS:
                errs.append(f"axis {a}: kind must be one of {KINDS}, "
                            f"got {k!r}")
        # rz implies y is a radius (y>=0) — a mirror on y is the axis
        # itself, handled by the r-z solver, so we don't allow declaring
        # it as a foldable plane here.
        if self.coords == "rz" and self.planes.get("y") == "mirror":
            errs.append("rz coords: y is radius (axis at y=0) — the r-z "
                        "solver already handles it; don't declare y mirror")
        for a, v in self.plane_mm.items():
            if a not in AXES:
                errs.append(f"plane_mm: unknown axis {a!r}")
            elif self.planes.get(a, "none") == "none":
                errs.append(f"plane_mm[{a!r}] declared but planes[{a!r}] "
                            f"is 'none' — a location without an assertion "
                            f"is meaningless; declare the kind too")
            if not isinstance(v, (int, float)):
                errs.append(f"plane_mm[{a!r}] must be a number, got {v!r}")
        return errs

    def to_dict(self):
        d = {"coords": self.coords, "planes": dict(self.planes)}
        # KEY STABILITY: emit plane_mm ONLY when declared. This dict feeds
        # basis_cache.geometry_key_dict verbatim; emitting an empty {} for
        # every legacy spec would rekey (and cold-invalidate) every cached
        # basis in existence for a no-op. Absent == default == midline.
        if self.plane_mm:
            d["plane_mm"] = {a: float(v) for a, v in self.plane_mm.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls()
        # (_take_known is local now — schema homecoming)
        return _take_known(cls, d)


# --------------------------------------------------------- verification


@dataclass
class GeometrySpec(_StrictAttrs):
    """Domain + symmetry + how electrodes are defined.

    symmetry is now a SymmetrySpec (see ion_gym.symmetry): a coordinate
    choice ('rz' | 'xyz') plus per-axis plane ASSERTIONS
    ({axis: none|mirror|translational}) that the builder VERIFIES against
    the actual masks before using them to shrink the solve. This replaces
    the old coarse enum, which smashed together 'described in r-z' (a
    coordinate) and 'the midplane is a mirror' (an assertion to prove).

    Back-compat: from_dict accepts the old string form
    ('cylindrical'|'planar'|'none') and maps it (cylindrical->rz,
    planar/none->xyz, no declared planes).

    mm_per_gu sets solve resolution. stl_dir (optional) holds per-electrode
    STLs, resolved relative to the spec file so spec+STLs is portable."""
    width_mm: float
    height_mm: float
    depth_mm: float = 0.0          # 0 for 2-D
    # DECLARED z extent of the STAGE's metal, for drawing
    # and the assembly clearance check ONLY. Distinct from `depth_mm`
    # above, which selects the SOLVE ROUTE — see ElectrodeSpec
    # .metal_depth_mm for the incident that makes the split mandatory.
    # 0 = undeclared (legacy planar decks are bit-identical).
    metal_depth_mm: float = 0.0
    # DECLARED axial extent [lo, hi] mm of the bodies for a z-invariant
    # (depth_mm=0) model (configurations come from the
    # json — that's it"). Display draws finite electrode spans from THIS,
    # never from measuring meshes at draw time (the app quad's rod STLs
    # are 1 mm cross-section TOKENS; measuring them drew 1 mm rods).
    # None = unknown -> full-range transport bands, as before.
    axial_extent_mm: Optional[List[float]] = None
    mm_per_gu: float = 0.1
    symmetry: SymmetrySpec = field(default_factory=SymmetrySpec)
    stl_dir: Optional[str] = None
    stl_manifest: Optional[dict] = None   # {stl_filename: sha256}
                                          # for portable STL specs
    # DECLARED build-frame offset:
    # build-frame coordinates = as-uploaded (CAD) coordinates +
    # frame_offset_mm — the DECLARED CAD->build translation, LOAD-BEARING
    # (declare placement in the spec, apply it
    # at the single mesh-ingest point, make it part of geometry identity,
    # and delete the mutation path"). Applied to every electrode mesh at
    # stl_resolve.load_mesh — the one ingest point both STL routes load
    # through — so STL fixtures stay pristine CAD exports and placement
    # is visible, editable deck data instead of hidden mutated bytes
    # (which is what this field USED to annotate: an earlier comment
    # here said "the translated STLs ARE the geometry ... never enters
    # the basis cache key", documenting exactly the pattern whose failure
    # broke a notebook end-to-end). Because it now moves metal, it IS
    # geometry: basis_cache.geometry_key_dict keys it — but ONLY when
    # declared (the plane_mm precedent), and to_dict omits None, so every
    # undeclaring spec's JSON and cache key stay byte-identical.
    # Rigid translation only; validated by stl_resolve.placement_offset
    # (the one interpretation authority). plane_mm on an STL spec remains
    # stated in the BUILD frame.
    frame_offset_mm: Optional[List[float]] = None
    # DECLARED FRAME ORIGIN (any plane that
    # is mirrored is at zero. Period."): [x_lo, y_lo] mm — the physical
    # coordinate of grid node (0, 0), so a deck may state its geometry
    # in a SIGNED frame with a symmetry plane literally at coordinate 0
    # (the r-z axis-at-0 and 3-D shapes-route mirror conventions, now on
    # the planar route). The domain spans [lo, lo + extent] per axis.
    # None = [0, 0]: every pre-existing spec's JSON, grid, and basis
    # cache key are byte-identical (omitted from to_dict, same precedent
    # as frame_offset_mm). All deck-frame quantities (shapes, source,
    # bounds, stations) are stated in this frame; the kernel translation
    # through model.anchor_mm maps them in and records back out.
    origin_mm: Optional[List[float]] = None
    # HOW the drive field is differenced from the solved potential, and the
    # STORED precision of the channel stacks. Both are DECLARED (not
    # inferred from builder/grid — that would be a hidden branch), so a
    # geometry opts into the cheaper path only when it says so.
    #   field_method: "electrode_aware" (default) applies the one-sided
    #     difference into vacuum at metal-surface nodes — ESSENTIAL where
    #     ions ride close to electrode surfaces (the tof source: an ion
    #     born at the surface is mis-accelerated ~1% of gap energy without
    #     it; tof Gate 1). "plain_gradient" is a straight central-
    #     difference everywhere — correct where ions stay off the surfaces
    #     (SLIM's central-gap transport), and cheaper (no metal-node pass).
    #   channel_dtype: "float64" (default) or "float32" for the STORED
    #     channel stacks. The gradient is always computed in float64; only
    #     storage precision changes. float32 halves the channel-pack memory
    #     (matters as SLIM grids grow) with negligible flight effect (the
    #     old slim3d stored float32 and matched an external solve). float32 shifts the
    #     stored values at rounding level, so a frozen anchor that stores
    #     channels must be re-baselined when switching.
    field_method: str = "electrode_aware"    # | "plain_gradient"
    channel_dtype: str = "float64"           # | "float32"
    electrodes: List[ElectrodeSpec] = field(default_factory=list)
    rf_groups: List[RFGroupSpec] = field(default_factory=list)
    dc_groups: List[DCGroupSpec] = field(default_factory=list)

    @property
    def coords(self):
        return self.symmetry.coords

    # (electrode_rf / electrode_drive and their _group_map helper removed:
    # no
    # callers, a docstring describing retired per-electrode RF fields, and
    # both read only an electrode's FIRST group -- wrong for multi-group
    # electrodes. Drives are composed from every group by the builders.)

    def to_dict(self):
        d = asdict(self)
        if d.get("frame_offset_mm") is None:
            d.pop("frame_offset_mm", None)   # identity: omitted (2a.3)
        if d.get("origin_mm") is None:
            d.pop("origin_mm", None)         # legacy frame: omitted
        # omit the field-build options when they hold their defaults, so
        # every pre-existing spec's JSON and basis cache key stay
        # byte-identical (same precedent as frame_offset_mm).
        if d.get("field_method") == "electrode_aware":
            d.pop("field_method", None)
        if d.get("channel_dtype") == "float64":
            d.pop("channel_dtype", None)
        d["electrodes"] = [e.to_dict() for e in self.electrodes]
        d["symmetry"] = self.symmetry.to_dict()
        d["rf_groups"] = [g.to_dict() for g in self.rf_groups]
        d["dc_groups"] = [g.to_dict() for g in self.dc_groups]
        return d

    @classmethod
    def _load_required(cls, d):
        # The pitch sets the solved grid; 0.1 mm/gu is one deck's
        # resolution, not a neutral value.
        return ("mm_per_gu",)

    @classmethod
    def from_dict(cls, d):
        _check_missing(cls, d)
        _lim = electrode_limit_error(len(d.get("electrodes", []) or []))
        if _lim:
            raise ValueError(_lim)          # refused at LOAD
        d = dict(d)
        d["electrodes"] = [ElectrodeSpec.from_dict(e)
                           for e in d.get("electrodes", [])]
        d["rf_groups"] = [RFGroupSpec.from_dict(g)
                          for g in d.get("rf_groups", [])]
        d["dc_groups"] = [DCGroupSpec.from_dict(g)
                          for g in d.get("dc_groups", [])]
        sym = d.get("symmetry")
        if isinstance(sym, str):
            # back-compat: the old coarse enum, through the one mapping
            d["symmetry"] = symmetry_from_string(sym)
        else:
            d["symmetry"] = SymmetrySpec.from_dict(sym)
        return _take_known(cls, d, checked=True)

    def __post_init__(self):
        # The constructor accepts the same old string form as from_dict,
        # through the same mapping: it used to keep the string and fail
        # much later (at to_json), so the two doors disagreed.
        if isinstance(self.symmetry, str):
            self.symmetry = symmetry_from_string(self.symmetry)


# The old string form of geometry.symmetry and the coordinates each means.
# Exact names only: anything else used to fall through to Cartesian
# silently ("rz", a typo, "Cylindrical"), so a deck meant for r-z flew a
# planar solve.
SYMMETRY_STRINGS = {"cylindrical": "rz", "planar": "xyz", "none": "xyz"}


def write_text_atomic(path, text):
    """Write `text` to `path` so a failed write never leaves a truncated
    file: a temp file in the same folder, fsync, then os.replace (the
    field_io._atomic_savez rule). A full disk or a killed process used to
    leave a half-written deck under the good one's name (review F8). The
    file keeps the mode an existing file had, else the umask default."""
    import os
    import tempfile
    path = Path(path)
    folder = path.parent if str(path.parent) else Path(".")
    if path.exists():
        mode = path.stat().st_mode & 0o777
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=folder)
    done = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        done = True
    finally:
        if not done and os.path.exists(tmp):
            os.remove(tmp)


def _whole_number(v):
    """An int, or a float with an integral finite value (JSON 2.0); never
    a bool."""
    return (not isinstance(v, bool)
            and (isinstance(v, int)
                 or (isinstance(v, float) and math.isfinite(v)
                     and v.is_integer())))


def charge_error(value, where):
    """The refusal for a charge state that is not a whole number, else
    None: one rule for every source and beam that states a charge (0 is a
    neutral and is allowed)."""
    if _whole_number(value):
        return None
    return (f"{where} = {value!r}: a charge state is a whole number of "
            f"elementary charges (e.g. 1, -2, or 0 for a neutral); a "
            f"fractional one used to be truncated silently (2.5 flew as 2)")


def symmetry_from_string(s):
    """SymmetrySpec for the old string form, or a refusal naming the
    accepted strings and the mapping form."""
    if s not in SYMMETRY_STRINGS:
        hint = (" ('rz' is a coords value: write {\"coords\": \"rz\"} or "
                "'cylindrical')" if str(s).lower() == "rz" else "")
        raise ValueError(
            f"geometry.symmetry = {s!r} is not a known symmetry. Use one of "
            f"{sorted(SYMMETRY_STRINGS)} or a mapping such as "
            f"{{\"coords\": \"rz\"}} / {{\"coords\": \"xyz\"}}{hint}. An "
            f"unknown string is refused rather than read as Cartesian.")
    return SymmetrySpec(coords=SYMMETRY_STRINGS[s])


# ----------------------------------------------------------------- source
@dataclass
class SourceSpec(_StrictAttrs):
    _RUNTIME_ATTRS = frozenset({"_drawn_seed", "_extras"})
    """Ion births. distribution in {point, disc, line, grid, box,
    gaussian, file}.
      point    : all ions at (x0,y0,z0)
      disc     : uniform disc radius r_mm in the plane normal to `axis`,
                 centred at (x0,y0,z0)
      line     : evenly along `axis` from (x0..) length len_mm
      grid     : n x n lattice on the disc
      gaussian : TRUNCATED Gaussian, per-axis FWHM (fwhm_mm) with
                 MANDATORY truncation half-widths (trunc_mm) — see
                 gaussian_layout() for the refusal contract
      file     : births_file (csv with x,y,z,vx,vy,vz[,tob]) — e.g. an external
    Kinematics: ke_lo..ke_hi (eV) directed along `direction`, OR
    thermal at temperature_k; mass list flown as a mix; charge signed;
    tob_span_us spreads birth times (RF phase sampling)."""
    n_ions: int = 100
    distribution: str = "disc"      # point|disc|line|grid|box|file
    x0_mm: float = 1.0
    y0_mm: float = 0.0
    z0_mm: float = 0.0
    r_mm: float = 4.0
    len_mm: float = 0.0
    box_mm: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0])   # box full extents (mm),
    # centred on (x0,y0,z0); ions placed uniformly at random inside
    # (ionspec-style volume source -- avoids the contrived single point)
    axis: str = "x"                 # disc normal / line direction
    direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    ke_lo: float = 0.1
    ke_hi: float = 1.9
    temperature_k: float = 0.0      # >0 overrides ke_* with thermal speeds
    # ANISOTROPIC beam spreads.  A real
    # gas-cell/OA beam is NOT a single-temperature bath: the drift (z)
    # direction is damped by the cell's DC gradient to dKz < 0.5 eV, while
    # the transverse directions are cooled by the telescopic x5 expansion
    # (Liouville) to ~10 m/s.  Setting one isotropic temperature_k that is
    # right transversely is then ~5x too hot in z -- enough to spread the
    # packet past the detector window -- and vice versa.
    #
    # Declared per LAB AXIS, not about `direction`, because the three axes
    # of a planar MRT mean different physics and must be set separately:
    #   dv_fwhm_ms[0] (x) : turnaround, from the pusher field
    #   dv_fwhm_ms[1] (y) : window-height divergence
    #   dv_fwhm_ms[2] (z) : drift energy spread dKz -> packet z width
    # A beam-frame "transverse" would fold y and z together and hide
    # exactly the distinction this exists to make.  Gaussian, FWHM in m/s,
    # superposed on whatever ke_*/temperature_k already declare.  Defaults
    # to zeros (off), so every existing deck is bit-identical.
    dv_fwhm_ms: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    # GAUSSIAN SPATIAL BEAM. The literature beam is
    # a TRUNCATED Gaussian (compact-MRT paper: distributions "close to
    # Gaussian", collimated by a heated 1 mm aperture); at equal FWHM a
    # Gaussian carries 1.47x the sigma of a uniform box, so the shape
    # is not cosmetic. Per LAB AXIS for the same reason dv_fwhm_ms is:
    # the push axis sets dK, the window axis the y-phase-space term.
    # fwhm_mm: per-axis FWHM in mm; an entry of exactly 0 means NO
    # spread on that axis (a thin sheet is a legitimate beam), but all
    # three zero is a point wearing a Gaussian's name and refuses.
    # trunc_mm: per-axis truncation HALF-WIDTHS in mm — mandatory,
    # because an untruncated Gaussian is not conservative, it is a
    # different beam with wings the real collimator removes. Both None
    # by default so every existing deck is bit-identical;
    # gaussian_layout() refuses (never defaults) anything malformed.
    fwhm_mm: Optional[List[float]] = None
    trunc_mm: Optional[List[float]] = None
    # CONSTANT DRIFT VELOCITY, mm/us, per LAB AXIS.
    # A beam entering an OA travels: the MRT-paper injection carries
    # K_z = 22 eV of drift along the fold direction, and that drift is a
    # DECLARED PROPERTY OF THE SOURCE, not something a downstream packet
    # may assert as an unexplained constant (which is exactly what the
    # fossil beam did — vz = 2.06 with no provenance). Superposed on
    # whatever ke_*/temperature_k/dv_fwhm_ms already declare. Defaults to
    # zeros, so every existing deck is bit-identical.
    v_drift_mm_us: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    # ION MASSES IN Da. The name is historical: this
    # is NOT m/z. Every route reads it as the mass (sim_build.mz_of), with
    # the signed charge state below; m/z = mass / |charge| is derived for
    # display only (ion_envelope.ion_label). Kept as mz_list so every deck,
    # bank and trajectory file stays readable.
    mz_list: List[float] = field(default_factory=lambda: [556.0])
    charge: int = 1
    tob_span_us: float = 2.0
    births_file: Optional[str] = None
    # RUN RANDOMNESS (restoring a decision that got
    # lost): None = RANDOM by default -- fresh entropy per run, drawn once
    # and PRINTED so any run can be pinned (resolve_run_seed). An integer
    # is the SEEDED option: bit-identical reruns / CRN comparisons. Decks
    # that declare "seed": <int> keep their declared reproducibility.
    seed: Optional[int] = None

    # one authority for the Gaussian beam's refusal contract:
    # validate() collects its message, generate_births consumes its
    # numbers, and any API caller gets the same refusal — never a
    # silent default.
    _FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

    def gaussian_layout(self):
        """(sigmas_mm, trunc_mm) per lab axis for distribution
        'gaussian', or raise ValueError naming exactly what did not
        line up. Contract: fwhm_mm and trunc_mm are 3-vectors; FWHM
        entries >= 0 with at least one > 0; every positive-FWHM axis
        carries a truncation half-width >= 1 sigma (below that the
        result is closer to a top hat than a Gaussian and should be
        DECLARED as a box); a truncation on a zero-FWHM axis is a
        claim about a spread that does not exist and refuses."""
        f, t = self.fwhm_mm, self.trunc_mm
        for name, v in (("fwhm_mm", f), ("trunc_mm", t)):
            if v is None:
                raise ValueError(
                    f"source: distribution='gaussian' requires {name} "
                    f"(a 3-vector, mm). There is no default beam width "
                    f"— a guessed width is a wrong beam.")
            if not hasattr(v, "__len__") or len(v) != 3:
                raise ValueError(
                    f"source: {name} must be a 3-vector (x, y, z) in "
                    f"mm, got {v!r}. Padding would assign a width to "
                    f"the wrong axis.")
        f = [float(x) for x in f]
        t = [float(x) for x in t]
        if any(x < 0 for x in f + t):
            raise ValueError(
                f"source: fwhm_mm/trunc_mm entries must be >= 0, got "
                f"fwhm_mm={f}, trunc_mm={t}.")
        if not any(x > 0 for x in f):
            raise ValueError(
                "source: all three fwhm_mm entries are 0 — that is a "
                "point source wearing a Gaussian's name; declare "
                "distribution='point'.")
        sig = [x / self._FWHM_PER_SIGMA for x in f]
        for k, ax in enumerate("xyz"):
            if f[k] > 0 and t[k] < sig[k]:
                raise ValueError(
                    f"source: trunc_mm[{ax}] = {t[k]:g} mm is below 1 "
                    f"sigma ({sig[k]:.4g} mm) of fwhm_mm[{ax}] = "
                    f"{f[k]:g} mm. Truncated that hard the shape is "
                    f"closer to a top hat than a Gaussian — declare it "
                    f"as a box instead.")
            if f[k] == 0 and t[k] != 0:
                raise ValueError(
                    f"source: trunc_mm[{ax}] = {t[k]:g} mm on an axis "
                    f"with fwhm_mm[{ax}] = 0 — a truncation on a "
                    f"spread that does not exist; set it to 0.")
        return sig, t

    # The size of the distribution IN USE is
    # physics; the others are never read.
    _SIZE_KEY = {"disc": "r_mm", "grid": "r_mm", "line": "len_mm",
                 "box": "box_mm"}

    @classmethod
    def _load_required(cls, d):
        """Keys whose defaults are one deck's operating point (the funnel
        ion mass 556 Da, KE 0.1-1.9 eV, a 2 us birth span, x0 = 1 mm, a
        4 mm disc): the deck must state the ones its distribution
        reads. A births file carries positions and velocities itself."""
        req = ["distribution", "n_ions", "mz_list"]
        dist = d.get("distribution")
        if dist == "file":
            return req
        req += ["x0_mm", "y0_mm", "z0_mm", "tob_span_us"]
        if not float(d.get("temperature_k") or 0.0) > 0.0:
            req += ["ke_lo", "ke_hi"]     # thermal births ignore ke_*
        if dist in cls._SIZE_KEY:
            req.append(cls._SIZE_KEY[dist])
        return req

    def value_errors(self):
        """Refusals for source values that cannot mean anything, by deck
        key. Each used to be accepted and then truncated, padded or
        dropped where it was used (review F6): charge 2.5 flew as 2, a
        two-entry box_mm was padded with 0, a negative temperature gave
        the births of T = 0."""
        errs = []
        ce = charge_error(self.charge, "source.charge")
        if ce:
            errs.append(ce)
        if not (_whole_number(self.n_ions) and self.n_ions >= 1):
            errs.append(f"source.n_ions = {self.n_ions!r}: a whole number "
                        f">= 1 of ions per mass")
        if self.seed is not None and not (_whole_number(self.seed)
                                          and self.seed >= 0):
            errs.append(f"source.seed = {self.seed!r}: a whole number >= 0, "
                        f"or null for a fresh seed per run")
        for key in ("temperature_k", "tob_span_us", "r_mm"):
            v = getattr(self, key)
            if isinstance(v, (int, float)) and v < 0:
                errs.append(f"source.{key} = {v!r}: must be >= 0")
        lo, hi = self.ke_lo, self.ke_hi
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) \
                and (lo < 0 or hi < lo):
            errs.append(f"source.ke_lo..ke_hi = {lo!r}..{hi!r} eV: energies "
                        f">= 0 with ke_lo <= ke_hi")
        if self.distribution == "box":
            b = self.box_mm
            if not (isinstance(b, (list, tuple)) and len(b) == 3
                    and all(isinstance(x, (int, float))
                            and not isinstance(x, bool) and x >= 0
                            for x in b)):
                errs.append(f"source.box_mm = {b!r}: three full extents "
                            f">= 0 mm (x, y, z)")
        return errs

    def direction_error(self):
        """The refusal for a source.direction that is not a 3-vector of
        finite numbers, else None. One authority: validate() reports it,
        and zero_direction_note() raises it."""
        d = self.direction
        try:
            dv = [float(v) for v in d]
        except (TypeError, ValueError):
            dv = None                # not numbers: refused just below
        if dv is not None and len(dv) == 3 and all(math.isfinite(v)
                                                   for v in dv):
            return None
        return (f"source.direction must be a 3-vector of finite floats, "
                f"got {d!r} — e.g. [1.0, 0.0, 0.0] for +x (a scalar sign "
                f"is not the contract; `axis` is a separate field)")

    def zero_direction_note(self):
        """WARNING text when a beam energy is declared with a zero
        direction, else None. Such an energy has nowhere to point and is
        IGNORED: ions are born at rest (plus any temperature_k or
        dv_fwhm_ms spread). This is what every such deck has always flown
        (births normalise the direction, and a zero vector left the
        directed term at zero); it is said out loud, never silently.
        A malformed direction raises ValueError (direction_error): there
        is no note to give about a direction that is not a vector."""
        err = self.direction_error()
        if err:
            raise ValueError(err)
        zero = not any(float(v) != 0.0 for v in self.direction)
        if not zero or not (self.ke_lo > 0 or self.ke_hi > 0):
            return None
        return (f"WARNING: source.direction is the zero vector, so "
                f"ke_lo..ke_hi = {self.ke_lo:g}..{self.ke_hi:g} eV has "
                f"nowhere to point and is IGNORED: ions are born at rest. "
                f"Give the beam a direction (e.g. [1.0, 0.0, 0.0] for +x), "
                f"declare ke_lo = ke_hi = 0 to state births at rest, or "
                f"use temperature_k for a thermal spread.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


# ------------------------------------------------------------- collisions
from ion_gym.physics.constants import TORR_PA  # noqa: E402  (101325/760, exact)
from ion_gym.physics.ccs import (  # noqa: E402  (one default each)
    LEGACY_FIXED_SIGMA_M2, SIGMA_MODEL_DEFAULT)

# One fixed text, so Python's warning filter
# shows it once per call site rather than once per deck in a sweep.
_P_PA_DEPRECATED = (
    "collisions.P_pa is deprecated: pressure is authored in Torr "
    "(P_torr). P_pa is derived from P_torr and is no longer saved; "
    "re-save the deck to drop it.")


# Two authored pressures closer than this (relative) are the SAME number
# to float rounding (a Torr <-> Pa round trip), not a disagreement.
_PA_SAME_NUMBER_REL = 1e-12


def _warn_pa_disagrees(p_torr, p_pa):
    """Both pressures authored: Torr wins. A Pa value that disagrees is
    named and ignored, never silently."""
    derived = float(p_torr) * TORR_PA
    if abs(derived - float(p_pa)) > _PA_SAME_NUMBER_REL * abs(derived):
        import warnings
        warnings.warn(
            f"collisions: P_torr = {float(p_torr):.12g} Torr "
            f"({derived:.12g} Pa) and P_pa = {float(p_pa):.12g} Pa disagree. "
            f"Torr is canonical: flying {float(p_torr):g} Torr, P_pa "
            f"ignored.", stacklevel=4)


@dataclass
class CollisionSpec(_StrictAttrs):
    """HS buffer gas. enabled=false -> vacuum (reflectron/TOF).

    PRESSURE IS TORR: Torr is canonical and P_pa is deprecated, with a
    warning. P_torr is the one stored value;
    P_pa is DERIVED from it (the kernels take Pa) and is never written
    to a saved deck. Authoring P_pa is deprecated and warns:
      * P_pa alone        -> converted to P_torr (legacy decks keep
                             flying at their pressure)
      * P_torr and P_pa   -> in a DECK: P_torr wins, and a P_pa that
                             disagrees is named and ignored. In CODE
                             (the constructor, dataclasses.replace) a
                             disagreeing pair is refused: which one was
                             meant cannot be told
      * assigning .P_pa   -> sets P_torr (None re-derives from P_torr)
      * P_torr None       -> the default (1.0 Torr), marked as defaulted
                             so the advisory says so, whether at
                             construction or by assignment
    so the two can never disagree. The old rule was the reverse: a
    present P_pa overwrote P_torr, so a hand-edited P_torr was silently
    ignored.

    P_pa is a field the constructor does not take (init=False), so
    dataclasses.replace re-derives it from P_torr instead of carrying the
    old one along (a replace of P_torr kept the stale Pa, and a replace of
    P_pa was silently overruled by the old Torr; replace(spec, P_pa=...)
    is now refused by dataclasses itself). The deprecated P_pa= keyword
    still works: _collision_init below takes it."""
    enabled: bool = True
    gas: str = "N2"
    T_k: float = 273.0
    P_torr: Optional[float] = None     # None = not authored -> 1.0 Torr
    P_pa: Optional[float] = field(default=None, init=False)   # DERIVED
    sigma_m2: float = LEGACY_FIXED_SIGMA_M2   # used under sigma_model "fixed"
    model: str = "hs"                  # "hs" (hard-sphere) or "sds";
                                       # legacy "hs1" is accepted and
                                       # normalized at ingest (renamed
                                       # at ingest; schema too)
    gas_diam_nm: float = 0.366         # SDS gas diameter (air default)
    flow_mm_us: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    # PER-ION HARD-SPHERE CROSS SECTION. "fixed" flies
    # sigma_m2 for every ion. "ccs_fit" takes each ion's cross section
    # from the N2 drift-tube CCS power-law fit in physics.ccs, subset
    # ccs_fit_class. physics.ccs.ion_sigma_m2 is the one authority every
    # route reads.
    # DEFAULT IS THE FIT: a deck that names no
    # sigma_model flies the CCS fit, because the old fixed 2.27e-18 was a
    # SIMION default (He with a ~200 amu ion) inherited without a reason.
    # The fit is evaluated at the ion's MASS for every charge (a charge
    # other than +/-1 gets an advisory); for a gas other than N2 it falls
    # back, loudly, to sigma_m2 (physics.ccs.fallback_note). A deck that
    # means one sigma for every ion states sigma_model: "fixed". The default lives in
    # physics.ccs (SIGMA_MODEL_DEFAULT), the module that owns the models.
    sigma_model: str = SIGMA_MODEL_DEFAULT
    ccs_fit_class: str = "all"
    # SDS SPECIES VALUES: a measured reduced mobility and/or ion diameter
    # per ion MASS, each K0 tagged with the gas it was measured in, e.g.
    #   [{"mass_amu": 133, "gas": "N2", "K0_cm2_per_Vs": 2.31},
    #    {"mass_amu": 622, "d_nm": 1.1}]
    # A K0 measured in another gas than the deck's is converted, loudly,
    # which needs that gas's diameter ("gas_d_nm"). Masses not listed are
    # estimated from mass (physics.sds). Read only under model 'sds'.
    sds_species: List[dict] = field(default_factory=list)

    # _init_pa: the deprecated P_pa= keyword, handed to __post_init__
    _RUNTIME_ATTRS = frozenset({"_extras", "_ready", "_defaulted",
                                "_init_pa"})
    # Physics keys a gas deck may leave out (they fly their default),
    # recorded in _defaulted so the advisory can SAY so (before this,
    # gas on with no pressure flew 1 Torr N2 at 273 K without a word). An
    # explicit assignment later clears the mark.
    _DEFAULTABLE = ("gas", "T_k", "P_torr")
    # Written to a saved deck only when they differ from these values.
    # sigma_model is NOT here: it is written on every save, so a saved
    # deck states which model it flew whatever the
    # default is later. An older deck that omits it loads with the
    # current default, on purpose.
    _OMIT_WHEN_DEFAULT = (("ccs_fit_class", "all"), ("sds_species", []))
    P_TORR_DEFAULT = 1.0               # the dataclass default

    def __post_init__(self):
        import warnings
        pa = getattr(self, "_init_pa", None)
        self._init_pa = None
        self._defaulted = ({"P_torr"} if (pa is None and self.P_torr is None)
                           else set())
        if pa is not None and self.P_torr is None:
            # legacy Pa-only: converted to Torr, and the kernel keeps the
            # AUTHORED Pa exactly (a Torr round trip can move it by an
            # ulp, which would break bit-identity with every old run)
            warnings.warn(_P_PA_DEPRECATED, FutureWarning, stacklevel=3)
            self.P_torr = float(pa) / TORR_PA
            self.P_pa = float(pa)
        else:
            if self.P_torr is None:
                self.P_torr = self.P_TORR_DEFAULT
            derived = float(self.P_torr) * TORR_PA
            if pa is not None and abs(float(pa) - derived) <= (
                    _PA_SAME_NUMBER_REL * abs(derived)):
                # the SAME pressure to rounding: every deck saved before
                # Torr was canonical stores P_torr back-filled from P_pa, so re-deriving
                # Pa would move the kernel input by an ulp. Keep it.
                self.P_pa = float(pa)
            else:
                if pa is not None:
                    # only a constructor call reaches here with both
                    # (from_dict settles a deck's pair first): refused, as
                    # which was meant cannot be told
                    raise ValueError(
                        f"CollisionSpec: P_torr = {float(self.P_torr):.12g} "
                        f"Torr ({derived:.12g} Pa) and P_pa = "
                        f"{float(pa):.12g} Pa disagree. Pass P_torr only "
                        f"(P_pa is deprecated and derived from it; "
                        f"P_torr = Pa / {TORR_PA:.12g}).")
                self.P_pa = derived
        # legacy model spelling normalized at INGEST (was rewritten inside
        # validate(), which must not mutate what it checks)
        if str(self.model or "").lower() == "hs1":
            self.model = "hs"
        self._ready = True

    def __setattr__(self, name, value):
        if not getattr(self, "_ready", False):
            super().__setattr__(name, value)
            return
        if name in getattr(self, "_defaulted", ()):
            self._defaulted.discard(name)   # stated now, not defaulted
        if name == "P_torr" and value is None:
            # "not stated" = the default, as at construction, and said so;
            # leaving the old Pa in place flew the old pressure while a save
            # wrote null, which reloads as 1 Torr
            super().__setattr__("P_torr", self.P_TORR_DEFAULT)
            super().__setattr__("P_pa", self.P_TORR_DEFAULT * TORR_PA)
            self._defaulted.add("P_torr")
            return
        if name == "P_pa":
            if value is None:          # "null P_pa" = re-derive from Torr
                super().__setattr__("P_pa", float(self.P_torr) * TORR_PA)
                return
            import warnings
            warnings.warn(_P_PA_DEPRECATED, FutureWarning, stacklevel=2)
            self.P_torr = float(value) / TORR_PA
            super().__setattr__("P_pa", float(value))   # authored Pa, exact
            return
        super().__setattr__(name, value)
        if name == "P_torr" and value is not None:
            super().__setattr__("P_pa", float(value) * TORR_PA)

    def validate(self):
        """Refusals for gas values that cannot describe a gas, as messages
        naming the deck key. Empty for vacuum (enabled false).

        A pressure, temperature or cross section that is zero or negative
        does not raise anywhere in the kernels: it makes the collision
        rate zero or negative, so the ion flies a trajectory bit-identical
        to vacuum while the deck says gas is on. The cross section is
        checked only under sigma_model "fixed", the one model that flies
        it; the gas diameter only under the SDS model, which reads it.
        NaN and infinity are refused here too, because callers that build
        a gas without a deck (and so never run SimSpec.validate) reach
        the kernels through resolve_collisions, which applies this."""
        if not self.enabled:
            return []

        def _bad(v):
            return v is not None and not (math.isfinite(v) and v > 0)

        errs = []
        for key, unit in (("P_torr", "Torr"), ("T_k", "K")):
            v = getattr(self, key)
            if _bad(v):
                errs.append(f"collisions.{key} = {v!r} {unit}: gas is "
                            f"enabled, so this must be a finite number > 0 "
                            f"(otherwise the ion flies as if in vacuum); "
                            f"for a vacuum run set collisions.enabled: "
                            f"false")
        if self.sigma_model == "fixed" and _bad(self.sigma_m2):
            errs.append(f"collisions.sigma_m2 = {self.sigma_m2!r} m^2 under "
                        f"sigma_model 'fixed': a cross section must be a "
                        f"finite number > 0 (otherwise there are no "
                        f"collisions)")
        if str(self.model).lower() == "sds" and _bad(self.gas_diam_nm):
            errs.append(f"collisions.gas_diam_nm = {self.gas_diam_nm!r} nm "
                        f"under model 'sds': a gas diameter must be a "
                        f"finite number > 0")
        if self.sds_species:
            from ion_gym.physics.sds import species_errors
            errs.extend(species_errors(self.sds_species, self.gas))
        return errs

    def set_pressure_torr(self, torr):
        """Set BOTH pressure fields from Torr, atomically.

        Assigning P_torr and nulling P_pa does NOT re-derive (the derivation
        lives in __post_init__), so the two fields drift apart and the kernel
        -- which reads P_torr -- silently disagrees with anything reading
        P_pa. Two fields for one quantity is a standing hazard; this is the
        one door through it. (P_pa is now derived on every P_torr
        write, so this is equivalent to assigning P_torr.)"""
        self.P_torr = float(torr)
        return self

    def set_pressure_pa(self, pa):
        """DEPRECATED: pressure is Torr. Converts and warns."""
        self.P_pa = float(pa)          # warns, sets P_torr (see __setattr__)
        return self

    def to_dict(self):
        d = asdict(self)
        d.pop("P_pa", None)            # derived: Torr is the stored value
        for key, default in self._OMIT_WHEN_DEFAULT:
            if d.get(key) == default:
                # omitted ON PURPOSE: the loader restores the default,
                # and pre-existing decks stay byte-identical on save
                d.pop(key)
        if (self.sigma_model != "fixed"
                and d["sigma_m2"] == LEGACY_FIXED_SIGMA_M2):
            # The legacy value under any model but "fixed" is the field
            # default nobody chose: writing it made every saved deck show
            # a cross section the flight never used. Omitted, the loader
            # restores the same number, so the fallback for a gas the fit
            # does not cover flies what it did. A value someone stated is
            # always written (and ccs.stated_sigma_note flags it).
            d.pop("sigma_m2")
        return d

    @classmethod
    def from_dict(cls, d):
        import warnings
        d = dict(d)
        pa = d.pop("P_pa", None)
        if pa is not None:
            warnings.warn(_P_PA_DEPRECATED, FutureWarning, stacklevel=3)
        obj = _take_known(cls, d)
        if pa is not None:
            if d.get("P_torr") is None:
                # legacy Pa-only deck: converted to Torr, and the kernel
                # keeps the AUTHORED Pa exactly (a Torr round trip can move
                # it by an ulp, which would break bit-identity with every
                # old run)
                obj.P_torr = float(pa) / TORR_PA
                object.__setattr__(obj, "P_pa", float(pa))
            elif abs(float(pa) - obj.P_pa) <= _PA_SAME_NUMBER_REL * abs(
                    obj.P_pa):
                # the SAME pressure to rounding (decks saved before Torr
                # was canonical carry both): keep the authored Pa, as above
                object.__setattr__(obj, "P_pa", float(pa))
            else:
                _warn_pa_disagrees(obj.P_torr, pa)    # a deck: Torr wins
        # keys the deck left out (P_torr is already marked by
        # __post_init__ unless a P_pa stood in for it)
        obj._defaulted |= {k for k in ("gas", "T_k") if k not in d}
        return obj


_collision_init = CollisionSpec.__init__


def _collision_init_with_pa(self, *args, P_pa=None, **kwargs):
    """CollisionSpec's constructor, still taking the deprecated P_pa=
    keyword (the dataclass field is init=False, so replace() never passes
    a stale one); __post_init__ converts or refuses it."""
    object.__setattr__(self, "_init_pa", P_pa)
    _collision_init(self, *args, **kwargs)


_collision_init_with_pa.__doc__ = (_collision_init.__doc__ or "") + \
    _collision_init_with_pa.__doc__
CollisionSpec.__init__ = _collision_init_with_pa


# ------------------------------------------------------------ integration
@dataclass
class IntegrationSpec(_StrictAttrs):
    # dt_ns None (JSON null / omitted-as-null) = DERIVE from the
    # spec's drives at load (fastest period / 250, the stock
    # convention).  An explicit number is honored as authored --
    # resolution warnings stay ADVISORIES by design.
    dt_ns: Optional[float] = 1.0
    t_max_us: float = 400.0
    rec_every: int = 80
    # Trajectory record cap (the old hard-coded
    # 100k truncated SILENTLY; now spec-settable and every truncation
    # or predicted shortfall is announced loudly at fly time).
    max_records: int = 100000
    record_channels: List[str] = field(
        default_factory=lambda: ["speed", "ke_ev", "e_field"])

    # The step and the flight time are physics; recording density
    # and the record cap are storage (sizing.record_volume: rec_every is
    # STORAGE ONLY), so they are announced, not refused. dt_ns: null is
    # an AUTHORED value (derive from the drives), so only an absent key
    # is refused.
    _LOAD_ANNOUNCE = ("rec_every", "max_records")
    # _dt_derived: dt_ns holds a step derived from an authored null. The
    # flight needs a number, but a save must write null again, or the deck
    # stops following its drives (a later frequency change would keep the
    # old step).
    # _channels_added: record channels a build added because a derived
    # channel needs them (sim_build: e_axial/e_radial off the r-z route
    # need e_x/e_y/e_z). The flight records them; a save writes the
    # channels the deck asked for.
    _RUNTIME_ATTRS = frozenset({"_extras", "_dt_derived", "_channels_added"})

    @classmethod
    def _load_required(cls, d):
        return ("dt_ns", "t_max_us")

    def set_derived_dt(self, dt_ns):
        """Fill an unset step with the value derived from the drives, and
        remember that it was derived, so to_dict writes null."""
        self.dt_ns = float(dt_ns)
        self._dt_derived = True

    def add_derived_channels(self, names):
        """Record `names` as well, remembered as added by a build, so
        to_dict writes the channels the deck asked for. Returns the
        channels actually added."""
        add = [c for c in names if c not in (self.record_channels or [])]
        if add:
            already = list(getattr(self, "_channels_added", None) or [])
            self.record_channels = list(self.record_channels or []) + add
            self._channels_added = already + add
        return add

    def __setattr__(self, name, value):
        # a DIFFERENT step is a stated one; the same value written back
        # (the app syncs every widget into the spec) is not
        if (name == "dt_ns" and getattr(self, "_dt_derived", False)
                and value != self.dt_ns):
            self._dt_derived = False
        # likewise a different channel list is a stated one
        if (name == "record_channels"
                and getattr(self, "_channels_added", None)
                and value != self.record_channels):
            self._channels_added = []
        super().__setattr__(name, value)

    def to_dict(self):
        d = asdict(self)
        if getattr(self, "_dt_derived", False):
            d["dt_ns"] = None
        added = getattr(self, "_channels_added", None)
        if added:
            d["record_channels"] = [c for c in d["record_channels"]
                                    if c not in added]
        return d

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


# ---------------------------------------------------------------- stations
@dataclass
class StationSpec(_StrictAttrs):
    """A declared PLANE the deck carries as a first-class object: a
    detector face, a slit, or a diagnostic recording plane
    (bounding planes are absolute domain kills and cannot
    represent a detector; a station is a plane at `axis` = `pos_mm`
    with optional WINDOWS over the other axes).

    kind (record/detect was "a distinction
    without a difference"; the meaningful axis is what happens ON
    a hit, so it is explicit):
      "detect" -- a measuring plane; `on_hit` REQUIRED:
          on_hit="pass"  -- transparent: crossings logged post-hoc,
                            the ion continues (the old "record").
          on_hit="splat" -- detector patch: a crossing INSIDE the
                            window ABSORBS the ion at the exact
                            interpolated crossing (fate 6, "station
                            detect" — the detection event); outside
                            passes; empty window = full-plane
                            detector.
      "impact_plane" -- PHYSICAL plate, the other kind that alters
                  flight: the kernel terminates an ion crossing the
                  plane OUTSIDE the window at the exact interpolated
                  crossing (fate 5, "station impact plane"); inside
                  the window it passes. Both crossing directions
                  count (a plate has two faces). An EMPTY window is
                  a wall: every crossing splats. This is the "splat
                  window" bounds cannot express (bounding planes are
                  whole-plane absolute kills). (Renamed from
                  "aperture": an aperture is the
                  OPENING; this kind is the plate.)

    Stations are configuration-agnostic: any axis in the run's
    channel set is legal, windows may cover any subset of the other
    axes, and a station never alters the solve or the kernel -- it is
    applied to recorded trajectories (exact where the crossing is
    bracketed by field-free records; consumers own that check).
    """
    name: str = "station"
    kind: str = "detect"
    on_hit: Optional[str] = None  # detect only: 'pass' | 'splat' (REQUIRED)
    axis: str = "x"
    pos_mm: float = 0.0
    window: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)

    def validate(self, prefix=""):
        errs = []
        if self.kind == "aperture":
            errs.append(f"{prefix}station {self.name!r}: kind "
                        f"'aperture' was RENAMED 'impact_plane' "
                        f"(an aperture is the opening; "
                        f"this kind is the plate). Update the deck.")
        elif self.kind == "record":
            errs.append(f"{prefix}station {self.name!r}: kind "
                        f"'record' was RETIRED ("
                        f"record vs detect was a distinction "
                        f"without a difference). Use kind='detect' "
                        f"with on_hit='pass'.")
        elif self.kind == "detect":
            if self.on_hit not in ("pass", "splat"):
                errs.append(f"{prefix}station {self.name!r}: "
                            f"kind='detect' requires on_hit='pass' "
                            f"(log, ion continues) or "
                            f"on_hit='splat' (absorbing detector, "
                            f"fate 6), got {self.on_hit!r} — the "
                            f"hit behavior is the whole "
                            f"distinction, so it is explicit")
        elif self.kind == "impact_plane":
            if self.on_hit is not None:
                errs.append(f"{prefix}station {self.name!r}: "
                            f"on_hit={self.on_hit!r} on an "
                            f"impact_plane — the plate's behavior "
                            f"is fixed (aperture passes, plate "
                            f"splats); on_hit is detect-only")
        else:
            errs.append(f"{prefix}station {self.name!r}: unknown kind "
                        f"{self.kind!r} (detect|impact_plane)")
        if self.axis not in ("x", "y", "z"):
            errs.append(f"{prefix}station {self.name!r}: unknown axis "
                        f"{self.axis!r}")
        for ax, w in (self.window or {}).items():
            if ax not in ("x", "y", "z") or ax == self.axis:
                errs.append(f"{prefix}station {self.name!r}: window axis "
                            f"{ax!r} invalid for a {self.axis}-plane")
            elif (not hasattr(w, "__len__")) or len(w) != 2                     or not float(w[0]) < float(w[1]):
                errs.append(f"{prefix}station {self.name!r}: window[{ax}] "
                            f"must be [lo, hi] with lo < hi, got {w!r}")
        return errs


# ---------------------------------------------------------------- bounds
@dataclass
class BoundsSpec(_StrictAttrs):
    """Optional bounding/impact planes — a physical modeling primitive (a
    detector plane, an aperture, the edge of the real instrument). Each
    axis has an enable flag and a min/max; an ion crossing an ENABLED
    bound terminates with fate code 3 ('bounding plane'). All OFF by
    default, so behaviour is unchanged unless a bound is turned on. This
    solves the 'ions fly to infinity' problem (e.g. an einzel with no
    downstream wall) without inventing geometry.

    Coordinates are the trajectory's own (x/y/z in mm); for r-z specs x
    is the axis and y the radius, so x_max is a detector-plane distance
    and y_max a radial aperture."""
    x_min_on: bool = False
    x_max_on: bool = False
    y_min_on: bool = False
    y_max_on: bool = False
    z_min_on: bool = False
    z_max_on: bool = False
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = -50.0
    y_max: float = 50.0
    z_min: float = -50.0
    z_max: float = 50.0

    def any_on(self):
        return any([self.x_min_on, self.x_max_on, self.y_min_on,
                    self.y_max_on, self.z_min_on, self.z_max_on])

    def as_tuple(self):
        """Numba-friendly: (flags 6, values 6) for the kernels."""
        return ((self.x_min_on, self.x_max_on, self.y_min_on,
                 self.y_max_on, self.z_min_on, self.z_max_on),
                (self.x_min, self.x_max, self.y_min, self.y_max,
                 self.z_min, self.z_max))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


# ------------------------------------------------------------------- view
@dataclass
class ViewSpec(_StrictAttrs):
    """How the spec likes to be shown. planes: any of xy/xz/yz. mode:
    '2d' (fast, quantitative slices) or '3d' (Scatter3d bundle, capped).
    For 2-D-symmetry geometry only xy is meaningful; 3-D geometry can use
    all three planes or the 3-D bundle."""
    mode: str = "2d"
    planes: List[str] = field(default_factory=lambda: ["xy"])
    equipotential_lines: int = 16
    show_field: bool = False
    color_by: str = "fate"
    decimate: int = 4

    # every view key is display only
    _LOAD_ANNOUNCE = ("mode", "planes", "equipotential_lines",
                      "show_field", "color_by", "decimate")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


# ------------------------------------------------------------ optic axis
# The channels measured RELATIVE TO THE INSTRUMENT AXIS. They are DERIVED
# AFTER FLIGHT from the recorded positions and field components about
# SimSpec.radial_frame(): derived from the route's axis (on r-z it is r;
# in 3-D it is declared), after flight; no kernel computes them.
RADIAL_CHANNELS = ("radius", "e_axial", "e_radial")


@dataclass
class OpticAxisSpec(_StrictAttrs):
    """The straight line the radial channels are measured about: parallel
    to `axis`, through `through_mm` = the two OTHER coordinates in cyclic
    order (axis z -> (x, y); x -> (y, z); y -> (z, x), as
    EXTRUDE_INPLANE_AXES), deck frame, mm.

      radius   = distance of the ion from this line
      e_axial  = field component along +axis (signed; reversing the
                 travel direction does not flip it)
      e_radial = field magnitude perpendicular to the line (>= 0)

    Analysis only: it never touches the solve or a cache key, so a deck
    can change it and re-derive without re-solving or re-flying. An
    oblique axis is not representable here (refused by type), and a
    curved transport path has no single radius -- such decks should not
    record the radial channels."""
    axis: str
    through_mm: List[float]

    def validate(self):
        errs = []
        if self.axis not in ("x", "y", "z"):
            errs.append(f"optic_axis.axis {self.axis!r} must be 'x', 'y' "
                        f"or 'z' (an oblique axis is not supported)")
        t = self.through_mm
        if (not hasattr(t, "__len__") or len(t) != 2
                or not all(isinstance(v, (int, float))
                           and not isinstance(v, bool)
                           and math.isfinite(v) for v in t)):
            errs.append(f"optic_axis.through_mm must be two finite numbers "
                        f"(the coordinates of the line in the other two "
                        f"axes, cyclic order), got {t!r}")
        return errs

    def to_dict(self):
        return {"axis": self.axis,
                "through_mm": [float(v) for v in self.through_mm]}

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, dict(d))


# r-z: the axis is built into the route -- the record frame's x is the
# symmetry axis and (y, z) are Cartesian transverse, r = hypot(y, z)
RZ_OPTIC_AXIS = OpticAxisSpec(axis="x", through_mm=[0.0, 0.0])


# ------------------------------------------------------------ transporter
TRANSPORTER_UNLIMITED_PASSES = 10 ** 9   # "no pass limit" (the kernel's own
                                         # sentinel, tracer3d.fly3d)


@dataclass
class TransporterSpec(_StrictAttrs):
    """The periodic transporter plane pair: an ion
    crossing the ACCEPT plane along `axis` in `direction` is moved to the
    EMIT plane, so a short periodic section stands in for a long device.
    Only the 3-D shapes route implements it; every other route REFUSES a
    spec that carries one. Planes are in the spec frame (mm).

    A TYPED block loaded through the same key
    accounting as every other block: it was a bare dict that to_dict and
    from_dict both dropped, silently. Code that assigns a dict
    (spec.transporter = dict(...), the internal ATD studies) is converted
    through from_dict, so the same refusals apply."""
    axis: str
    accept_mm: float
    emit_mm: float
    direction: int
    max_passes: int = TRANSPORTER_UNLIMITED_PASSES   # neutral: no limit

    @classmethod
    def _load_required(cls, d):
        # the planes and the crossing sense ARE the transporter
        return ("axis", "accept_mm", "emit_mm", "direction")

    def validate(self):
        errs = []
        if self.axis not in ("x", "y", "z"):
            errs.append(f"transporter.axis {self.axis!r} not x/y/z")
        for k in ("accept_mm", "emit_mm"):
            if isinstance(getattr(self, k), bool) or not isinstance(
                    getattr(self, k), (int, float)):
                errs.append(f"transporter.{k} non-numeric: "
                            f"{getattr(self, k)!r}")
        if not errs and float(self.accept_mm) == float(self.emit_mm):
            errs.append("transporter accept_mm == emit_mm (zero-length)")
        if self.direction not in (1, -1):
            errs.append(f"transporter.direction must be +1 or -1, got "
                        f"{self.direction!r}")
        if int(self.max_passes) < 1:
            errs.append("transporter.max_passes must be >= 1")
        return errs

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, dict(d))


# The documents spec_io.sniff tells apart from a single deck, and what
# loads each one (SimSpec.from_dict refuses them by name).
_OTHER_DOCUMENT_LOADERS = {
    "staged_assembly": (
        "a staged assembly (schema ion_gym.assembly/1), not a single deck: "
        "each stage solves in its own grid and the ion is handed across "
        "the seams. Load it with ion_gym.physics.staged_flight."
        "load_assembly(path) or fly it with fly_assembly(path); each "
        "stage's own deck is its 'spec' entry."),
    "assembly": (
        "a coaxial assembly (kind 'assembly'), not a single deck. Load it "
        "with ion_gym.io.spec_io.load_any_spec, which flattens it to one "
        "SimSpec."),
    "scene3d": (
        "a 3-D CSG scene (grid + indexed electrodes), not a deck. Load it "
        "with ion_gym.io.spec_io.load_any_spec, which embeds it in a "
        "SimSpec."),
}


# =================================================================== SimSpec
@dataclass
class SimSpec(_StrictAttrs):
    _RUNTIME_ATTRS = frozenset({"_loaded_from", "_extras"})
    geometry: GeometrySpec
    source: SourceSpec = field(default_factory=SourceSpec)
    # Collisions left out = VACUUM. An
    # explicit CollisionSpec(...) still means gas on.
    collisions: CollisionSpec = field(
        default_factory=lambda: CollisionSpec(enabled=False))
    integration: IntegrationSpec = field(default_factory=IntegrationSpec)
    bounds: "BoundsSpec" = field(default_factory=BoundsSpec)
    # Declared planes (detectors, slits, diagnostic planes) the deck
    # carries as first-class objects; see StationSpec.
    stations: List[StationSpec] = field(default_factory=list)
    # optional periodic transporter plane pair (TransporterSpec). None ->
    # disabled. A dict assigned here is converted through
    # TransporterSpec.from_dict (see __setattr__).
    transporter: Optional[TransporterSpec] = None
    # the axis the radial channels are measured about (OpticAxisSpec).
    # Top level, NOT geometry: it is analysis, and geometry feeds the
    # field-cache keys -- declaring it must never force a re-solve.
    optic_axis: Optional[OpticAxisSpec] = None
    view: ViewSpec = field(default_factory=ViewSpec)
    name: str = "simulation"
    notes: str = ""
    builder: str = ""          # "" = dispatch on geometry ("slim3d" is retired: refuses)
    import_path: str = ""      # scene file for an external-geometry builder
    # builder=="scene3d": the GeomScene is EMBEDDED here, not referenced by path.
    # A path is machine-local — a spec that points at /tmp/foo.json on the
    # machine that made it is a dangling reference everywhere else, and the
    # JSON tab promises a SELF-DESCRIBING spec. The geometry travels with it.
    scene: Optional[dict] = None

    def __setattr__(self, name, value):
        # ONE typed transporter: a dict (constructor or assignment) is
        # loaded through TransporterSpec.from_dict, so its keys are
        # accounted for exactly as in a deck file.
        if name == "transporter" and isinstance(value, dict):
            value = TransporterSpec.from_dict(value)
        if name == "optic_axis" and isinstance(value, dict):
            value = OpticAxisSpec.from_dict(value)
        super().__setattr__(name, value)

    def radial_frame(self):
        """THE axis the radial channels (RADIAL_CHANNELS) are measured
        about, for every route: r-z -> its built-in axis (RZ_OPTIC_AXIS);
        any other route -> the declared optic_axis, or None when none is
        declared (validate() then refuses a requested radial channel)."""
        if self.geometry.symmetry.coords == "rz":
            return RZ_OPTIC_AXIS
        return self.optic_axis

    # ------------------------------------------------------------ channels
    def describe(self) -> str:
        """A sectioned, unit-carrying description of how this spec is
        CONSTRUCTED: the domain and its lattice, every conductor and its
        drive, the ion source's position and kinematics, the gas, the
        integration contract, and the declared planes. Everything printed
        is the declared field the builders consume — display equals
        solver input. Reusable (for a notebook's teaching
        printouts): notebooks call this instead of hand-rolling one."""
        g = self.geometry
        L = [f"SimSpec '{self.name}' — construction"]
        if self.notes:
            L.append(f"  notes: {self.notes}")
        # ---- geometry / lattice
        dom = f"{g.width_mm:g} x {g.height_mm:g}"
        dom += f" x {g.depth_mm:g} mm (3-D solve)" if g.depth_mm else \
               " mm (2-D solve; depth_mm 0)"
        L.append(f"  geometry: domain {dom} at {g.mm_per_gu:g} mm/gu")
        sym = g.symmetry
        planes = {a: k for a, k in (sym.planes or {}).items()
                  if k and k != "none"} if hasattr(sym, "planes") else {}
        L.append(f"    coordinates '{getattr(sym, 'coords', sym)}'"
                 + (f"; declared symmetry planes: {planes}" if planes
                    else "; no symmetry planes declared"))
        if g.axial_extent_mm:
            L.append(f"    declared axial metal extent "
                     f"[{g.axial_extent_mm[0]:g}, {g.axial_extent_mm[1]:g}] mm")
        if g.stl_dir:
            off = g.frame_offset_mm or [0.0, 0.0, 0.0]
            L.append(f"    STL dir '{g.stl_dir}'; declared CAD->build "
                     f"frame offset {off} mm (applied at mesh ingest)")
        # ---- drives
        if g.rf_groups:
            L.append(f"  drive groups ({len(g.rf_groups)}):")
            for gr in g.rf_groups:
                wf = getattr(gr, "waveform", "sin")
                line = (f"    {gr.name}: {wf}, {gr.amplitude_v:g} V"
                        + (f" at {gr.frequency_hz/1e6:g} MHz"
                           if getattr(gr, "frequency_hz", None) else "")
                        + (f", phase {gr.phase_deg:g} deg"
                           if getattr(gr, "phase_deg", 0.0) else ""))
                L.append(line)
        else:
            L.append("  drive groups: none (pure DC device)")
        if g.dc_groups:
            for grp in g.dc_groups:
                L.append(f"  DC ladder '{grp.name}': v_in {grp.v_in:g} V -> "
                         f"v_out {grp.v_out:g} V, interp '{grp.interp}' "
                         "(member dc values derived at build)")
        # ---- electrodes
        L.append(f"  electrodes ({len(g.electrodes)}):")
        for e in g.electrodes:
            L.append("    " + e.describe())
        # ---- source
        s = self.source
        pos = f"({s.x0_mm:g}, {s.y0_mm:g}, {s.z0_mm:g}) mm"
        d = s.distribution
        if d == "point":
            ext = "point"
        elif d in ("disc", "grid"):
            ext = f"{d}, r {s.r_mm:g} mm, normal '{s.axis}'"
        elif d == "line":
            ext = f"line, {s.len_mm:g} mm along '{s.axis}'"
        elif d == "box":
            ext = f"box {list(s.box_mm)} mm full extents"
        elif d == "gaussian":
            ext = (f"truncated Gaussian, FWHM {s.fwhm_mm} mm, "
                   f"truncation half-widths {s.trunc_mm} mm")
        elif d == "file":
            ext = f"external births file '{s.births_file}'"
        else:
            # unknown distribution: describe() reports; validate() refuses
            ext = f"'{d}' distribution (unrecognized here)"
        L.append(f"  source: {s.n_ions} ion(s), {ext}, centred {pos}")
        kin = []
        if s.ke_lo > 0 or s.ke_hi > 0:
            kin.append(f"beam KE {s.ke_lo:g}..{s.ke_hi:g} eV along "
                       f"{list(s.direction)}")
        if s.temperature_k > 0:
            kin.append(f"thermal bath {s.temperature_k:g} K (isotropic "
                       "Maxwell-Boltzmann, superposed)")
        else:
            kin.append("T 0 K (idealized cold beam: no thermal spread, "
                       "no TOF turn-around term)")
        if any(s.dv_fwhm_ms or []):
            kin.append(f"per-axis dv FWHM {list(s.dv_fwhm_ms)} m/s")
        if any(getattr(s, "v_drift_mm_us", None) or []):
            kin.append(f"drift {list(s.v_drift_mm_us)} mm/us")
        L.append("    kinematics: " + "; ".join(kin))
        from ion_gym.physics.ion_envelope import charge_str
        L.append(f"    mass {list(s.mz_list)} Da, {charge_str(s.charge)}; birth "
                 f"times spread over {s.tob_span_us:g} us"
                 + (f"; seed {s.seed}" if getattr(s, "seed", None) is not None
                    else "; unseeded (random draw announced at build)"))
        # ---- collisions
        c = self.collisions
        if c.enabled:
            L.append(f"  collisions: {c.model.upper()} {c.gas} at "
                     f"{c.P_torr:g} Torr, {c.T_k:g} K")
        else:
            L.append("  collisions: disabled (vacuum flight)")
        # ---- integration
        it = self.integration
        if it.dt_ns is None:
            L.append(f"  integration: dt DERIVED at build -> "
                     f"{self.derive_dt_ns():g} ns (fastest drive period / "
                     f"{RF_RECOMMENDED_STEPS_PER_PERIOD}); "
                     f"t_max {it.t_max_us:g} us, record every "
                     f"{it.rec_every} step(s)")
        else:
            L.append(f"  integration: dt {it.dt_ns:g} ns (authored); "
                     f"t_max {it.t_max_us:g} us, record every "
                     f"{it.rec_every} step(s)")
        L.append(f"    recorded channels: {list(it.record_channels)}")
        # ---- bounds / stations / transporter
        b = self.bounds
        on = [f"{ax}_{side} at {getattr(b, f'{ax}_{side}'):g} mm"
              for ax in "xyz" for side in ("min", "max")
              if getattr(b, f"{ax}_{side}_on")]
        L.append("  bounding planes: " + (", ".join(on) if on
                 else "none declared (ions fly to the domain edge)"))
        if self.stations:
            L.append(f"  stations ({len(self.stations)}): "
                     + ", ".join(st.name for st in self.stations))
        if self.transporter is not None:
            L.append(f"  periodic transporter: {self.transporter.to_dict()}")
        _fr = self.radial_frame()
        if _fr is not None:
            L.append(f"  optic axis (radial channels): {_fr.axis} through "
                     f"{list(_fr.through_mm)} mm"
                     + (" (built into r-z)" if _fr is RZ_OPTIC_AXIS
                        else " (declared)"))
        return "\n".join(L)

    def column_names(self):
        # Canonical column order = BASE then OPTIONAL_CHANNELS order, NOT the
        # order the user selected them in. Both build routes construct their
        # column list as [c for c in OPTIONAL_CHANNELS if c in record_channels]
        # and the desync guard asserts against THIS, so the single source of
        # truth for layout is OPTIONAL_CHANNELS' order — a file's column
        # positions must not depend on the sequence of GUI clicks.
        sel = set(self.integration.record_channels)
        opt = [c for c in OPTIONAL_CHANNELS if c in sel]
        return BASE_CHANNELS + opt

    # -------------------------------------------------------- dt policy
    def derive_dt_ns(self) -> float:
        """dt as a DRIVE property: fastest active
        drive period / RF_RECOMMENDED_STEPS_PER_PERIOD; 1.0 ns for
        drive-less (pure DC) specs."""
        fs = [g.frequency_hz for g in self.geometry.rf_groups
              if g.frequency_hz and g.amplitude_v]
        if not fs:
            return 1.0
        return 1e9 / (max(fs) * RF_RECOMMENDED_STEPS_PER_PERIOD)

    # ------------------------------------------------------------ validate
    def resolve_dc_groups(self):
        """Write each ladder member's derived dc. Call BEFORE any build.

        v_in lands on the member with the LOWEST dc_index, v_out on the
        HIGHEST, everything between is interpolated. With interp='linear' the
        taps are equally spaced by INDEX (an ideal equal-resistor divider);
        with 'weights' each member's dc_weight in [0,1] places its own tap.

        Bases are per-electrode, so this is a RE-WEIGHT, never a re-solve:
        sweeping v_in/v_out (i.e. sweeping the axial field) is a cache HIT.
        """
        g = self.geometry
        for grp in g.dc_groups:
            mem = [e for e in g.electrodes if e.dc_group == grp.name]
            if not mem:
                continue
            if getattr(grp, "mode", "set") == "add":
                # ADDITIVE (float) group: the member dc stays AUTHORED —
                # nothing is written here. The group's contribution is
                # applied at composition through dc_effective(), because
                # an additive write here would compound on every resolve
                # call and a save after resolve would leak the summed
                # voltage into the deck.
                continue
            vals = dc_group_member_values(grp, mem)
            if grp.uniform:
                # plain equal-V group: v_out mirrors v_in
                grp.v_out = grp.v_in
            for e in mem:
                e.dc = vals[e.name]
        return self

    def dc_effective(self, el):
        """The static voltage this electrode contributes to the field:
        its authored dc PLUS any additive (mode='add') DC-group value.

        THE one authority the static-field composers consult (planar,
        r-z, and both 3-D assemblers) — displayed == solver input means
        any surface that states an electrode's voltage states THIS
        number, not el.dc alone. For a set-mode member el.dc is already
        the resolved value and the group adds nothing here; for an
        add-mode member el.dc is the authored baseline and the group's
        per-member value (dc_group_member_values) rides on top. Never
        mutates anything, so calling it any number of times is safe."""
        base = float(el.dc or 0.0)
        if el.dc_group is None:
            return base
        grp = next((x for x in self.geometry.dc_groups
                    if x.name == el.dc_group), None)
        if grp is None or getattr(grp, "mode", "set") != "add":
            return base
        mem = [e for e in self.geometry.electrodes
               if e.dc_group == grp.name]
        return base + dc_group_member_values(grp, mem)[el.name]

    def validate(self):
        errs = []
        # NON-FINITE NUMBERS. The JSON loader accepts NaN and Infinity and
        # no flight kernel checks for them, so a NaN voltage, frequency,
        # amplitude or mass flew every ion to t_max and came back as an
        # ordinary result (a timeout reported as 100% transmitted). Every
        # number the spec serialises is checked, by its deck path, so a
        # field added later is covered without keeping a list in step.
        for _path, _val in nonfinite_numbers(self.to_dict()):
            errs.append(
                f"{_path} = {_val!r} is not a finite number. NaN and "
                f"infinity propagate through the field and the flight "
                f"without raising and come back as ordinary-looking "
                f"results; give it a finite value.")
        # DECLARED PLACEMENT: validated by the same authority the
        # mesh ingest uses, so the deck that refuses here is exactly the
        # deck that would have loaded wrong. Declared on a deck with no
        # mesh electrode it is a silent no-op wearing a declaration --
        # parametric shapes place themselves -- so that refuses too.
        if getattr(self.geometry, "frame_offset_mm", None) is not None:
            from ion_gym.io.stl_resolve import placement_offset
            try:
                placement_offset(self)
            except ValueError as e:
                errs.append(str(e))
            if not any(getattr(e, "stl", None)
                       for e in self.geometry.electrodes):
                errs.append(
                    "geometry.frame_offset_mm is declared but no electrode "
                    "carries an stl -- the offset places MESHES; parametric "
                    "shapes declare their own positions, so this "
                    "declaration would silently do nothing")
        # source.direction is a 3-VECTOR (List[float]; `axis` is a
        # separate field). A scalar (+1) built fine, validated clean,
        # and travelled until the FIRST consumer that iterates it —
        # the UI's direction widgets, an IndexError-adjacent crash in
        # the field. Refuse it here, by name, with
        # the fix in the message. (A non-numeric entry used to escape
        # validate() as a raw ValueError: only TypeError was caught.)
        _dir_err = self.source.direction_error()
        if _dir_err:
            errs.append(_dir_err)
        # (A ZERO direction with a declared energy is not refused: the
        # energy has nowhere to point and is ignored, LOUDLY -- see
        # SourceSpec.zero_direction_note, raised as an advisory and by
        # the births generator.)
        for st in self.stations:
            errs += st.validate(prefix="stations: ")
        # A None pressure with collisions ON passed validation and then died
        # inside the numba kernel with a typing wall instead of a diagnostic
        # Refuse it here, by name.
        # The model enum was once UNCHECKED, so a deck saying
        # model="none" validated clean and then flew HS -- the author's
        # intent (no collisions) silently inverted into the default physics.
        # Refuse by name, and name the field that actually controls it.
        # "hs1" (the legacy spelling of "hs") is normalized at INGEST, in
        # CollisionSpec.__post_init__; validate() checks and never
        # mutates.
        _KNOWN_MODELS = ("hs", "sds")
        _model = str(getattr(self.collisions, "model", "hs") or "").lower()
        if _model not in _KNOWN_MODELS:
            _hint = ("  Did you mean collisions.enabled: false? `model` "
                     "selects WHICH collision physics runs; `enabled` "
                     "decides WHETHER any runs."
                     if _model in ("none", "off", "", "null", "vacuum")
                     else "")
            errs.append(
                f"collisions.model={getattr(self.collisions, 'model', None)!r} "
                f"is not a known model; known models are "
                f"{list(_KNOWN_MODELS)}.{_hint}")
        if getattr(self.collisions, "enabled", False):
            for _f in ("P_torr", "T_k", "sigma_m2"):
                if getattr(self.collisions, _f, None) is None:
                    errs.append(
                        f"collisions enabled but {_f} is None -- set it "
                        f"(pressure is collisions.P_torr; P_pa is derived)")
            if all(getattr(self.collisions, _f, None) is not None
                   for _f in ("P_torr", "T_k", "sigma_m2")):
                errs.extend(self.collisions.validate())
            # per-ion cross-section model: unknown model, unknown fit
            # class, or a fit asked of a gas/charge it was not fitted to
            from ion_gym.physics.ccs import validate_sigma_model
            errs.extend(validate_sigma_model(self.collisions,
                                             self.source.charge))
        if self.transporter is not None:
            errs.extend(self.transporter.validate())
        # RADIAL CHANNELS need an axis: built in on r-z, declared
        # everywhere else. One rule, every route.
        if self.optic_axis is not None:
            errs.extend(self.optic_axis.validate())
            if (self.geometry.symmetry.coords == "rz"
                    and self.optic_axis.to_dict()
                    != RZ_OPTIC_AXIS.to_dict()):
                errs.append(
                    f"optic_axis {self.optic_axis.to_dict()} on an r-z "
                    f"deck: the r-z axis is built into the route "
                    f"({RZ_OPTIC_AXIS.to_dict()}); remove the declaration")
        _radial = [c for c in (self.integration.record_channels or [])
                   if c in RADIAL_CHANNELS]
        if _radial and self.radial_frame() is None:
            errs.append(
                f"record_channels {_radial} are measured about the "
                f"instrument axis, which this deck does not declare -- add "
                f'"optic_axis": {{"axis": "z", "through_mm": [x, y]}} (the '
                f"axis direction and the line's position in the other two "
                f"coordinates), or drop those channels")
        g = self.geometry
        _lim = electrode_limit_error(len(g.electrodes))
        if _lim:
            errs.append(_lim)               # programmatic specs
        errs.extend(g.symmetry.validate())
        # SIGNED FRAME (origin_mm) guards: (1) shape must be
        # [x_lo, y_lo]; (2) each origin must sit ON the pitch lattice, or
        # no node can land on an integer-mm plane; (3) a mirror axis in a
        # signed frame MUST declare plane_mm — the midline default is
        # 0.5*extent, which is a coordinate in the LEGACY frame; letting
        # it default would silently verify/fold about the wrong plane.
        if g.origin_mm is not None:
            o = g.origin_mm
            if (not isinstance(o, (list, tuple)) or len(o) != 2
                    or not all(isinstance(v, (int, float)) for v in o)):
                errs.append(f"geometry.origin_mm must be [x_lo, y_lo] mm, "
                            f"got {o!r}")
            elif g.symmetry.coords == "rz" and any(v != 0 for v in o):
                # The planar and 3-D builders place the lattice at
                # origin_mm; the r-z builder does not read it, so shapes
                # authored in the shifted frame landed in the unshifted
                # one (a rect -5..5 at origin -5 kept 11 of its 21 nodes,
                # and an ion inside it was reported leaving the box).
                errs.append(
                    f"geometry.origin_mm = {list(o)!r} on an r-z deck: the "
                    f"r-z builder does not apply origin_mm (the planar and "
                    f"3-D builders do), so the geometry would be placed in "
                    f"the unshifted frame. Author the r-z deck with the "
                    f"axis at r = 0 and x from 0 (omit origin_mm or give "
                    f"[0, 0]).")
            else:
                h = float(g.mm_per_gu)
                for a, v in zip("xy", o):
                    if abs(v / h - round(v / h)) > 1e-9:
                        errs.append(
                            f"geometry.origin_mm[{a}]={v:g} is not a "
                            f"multiple of mm_per_gu={h:g} — the node "
                            f"lattice could not hit integer-mm planes")
                sym = g.symmetry.normalized()
                for a in ("x", "y"):
                    if (sym.planes.get(a) == "mirror"
                            and a not in sym.plane_mm):
                        errs.append(
                            f"signed frame (origin_mm) with a declared "
                            f"{a}-mirror requires symmetry.plane_mm[{a!r}] "
                            f"(e.g. 0.0) — the midline default assumes "
                            f"the legacy [0, extent] frame and would "
                            f"place the plane in the wrong spot")
        if g.mm_per_gu <= 0:
            errs.append("mm_per_gu must be > 0")
        # Each of these used to surface far downstream with a message that
        # pointed elsewhere (review F7): no electrodes -> a bare
        # StopIteration; a zero width -> "Factor is exactly singular"; a
        # zero mass -> ZeroDivisionError.
        if not g.electrodes and self.builder != "scene3d":
            errs.append("geometry.electrodes is empty -- a field needs at "
                        "least one electrode")
        for key in ("width_mm", "height_mm"):
            v = getattr(g, key)
            if isinstance(v, (int, float)) and not v > 0:
                errs.append(f"geometry.{key} = {v!r}: the domain must have "
                            f"a size > 0 mm")
        if isinstance(g.depth_mm, (int, float)) and g.depth_mm < 0:
            errs.append(f"geometry.depth_mm = {g.depth_mm!r}: 0 for a 2-D "
                        f"deck, > 0 for 3-D")
        _bad_m = [m for m in (self.source.mz_list or [])
                  if isinstance(m, (int, float)) and not m > 0]
        if _bad_m:
            errs.append(f"source.mz_list holds {_bad_m}: every ion mass "
                        f"must be > 0 Da")
        else:
            # THE LATTICE IS GU-NATIVE: domain
            # extents and every symmetry-plane position are integer gu,
            # exactly, on every route and axis, mirrored or not. The
            # loader refuses here with the same message the builders'
            # counting function raises, so a non-conforming deck cannot
            # reach a solver whose per-route floor()/round() used to
            # absorb the remainder silently (a declared 20.0 mm
            # at 0.35 became a real 19.95 and displaced a 3-D deck's
            # symmetry plane by 0.86 cell in z). Metal edges stay in mm
            # and rasterize — shapes are deliberately NOT checked.
            from ion_gym.io.lattice import conformance_error
            _h = float(g.mm_per_gu)
            for _ax, _ext, _fld in (("x", g.width_mm, "width_mm"),
                                    ("y", g.height_mm, "height_mm"),
                                    ("z", g.depth_mm, "depth_mm")):
                if _fld == "depth_mm" and _ext == 0.0:
                    continue          # 0 = the 2-D route switch, not a span
                _e = conformance_error(_ext, _h, axis=_ax,
                                       what=f"{_fld} domain extent")
                if _e is not None:
                    errs.append(_e)
            _sym = g.symmetry.normalized()
            _o = {"x": 0.0, "y": 0.0, "z": 0.0}
            if g.origin_mm is not None and len(g.origin_mm) == 2:
                _o["x"], _o["y"] = (float(g.origin_mm[0]),
                                    float(g.origin_mm[1]))
            _extent = {"x": g.width_mm, "y": g.height_mm, "z": g.depth_mm}
            for _ax in ("x", "y", "z"):
                if _sym.planes.get(_ax) != "mirror":
                    continue
                # TWO plane conventions, both pre-existing (DESIGN_
                # CONTRACT_symmetry_planes vs the 3-D stored-half):
                # the 2-D routes (depth_mm == 0) default an undeclared
                # plane to the DOMAIN MIDLINE, which must land on a
                # node; the 3-D routes store the half-domain with the
                # plane at stored 0 — a node by construction — and
                # never consult the midline. Checking the midline on a
                # 3-D deck would refuse every conforming stored-half
                # spec (caught on real stored-half decks).
                if g.depth_mm == 0.0:
                    # 2-D ROUTES. raster2d.anchored_grid anchors the
                    # grid on the DECLARED plane_mm, defaulting to the
                    # domain midline. Both are stated in the spec frame,
                    # so the position is checked RELATIVE TO origin_mm —
                    # the same frame raster2d reads it in.
                    _p = (float(_sym.plane_mm[_ax]) if _ax in _sym.plane_mm
                          else 0.5 * float(_extent[_ax]))
                    _e = conformance_error(
                        _p - _o[_ax], _h, axis=_ax,
                        what="symmetry-plane position (relative to the "
                             "origin; planes lie ON nodes, slop belongs "
                             "at the outer walls)")
                    if _e is not None:
                        errs.append(_e)
                    continue
                # 3-D ROUTES store the HALF domain with the plane at
                # stored 0 — node 0 of the stored array, a node by
                # construction. There is no midline to check (checking
                # one refuses every conforming stored-half spec, caught
                # on real stored-half decks) and no origin to
                # subtract: origin_mm is the 2-D anchored-grid frame,
                # consumed by raster2d, and NO 3-D builder reads it.
                # Subtracting it here would mix frames.
                #
                # This branch used to be a bare `continue` — a silent
                # pass. What is actually checkable, and is
                # now checked: build_stl3d._declared_mirror_axes is
                # documented as THE one authority for declared mirror
                # planes on the 3-D routes, and it reads
                # scene.grid.mirror or symmetry.planes — it NEVER reads
                # plane_mm. So a 3-D deck declaring a plane anywhere but
                # 0 declares a location the builder will silently
                # ignore, and the field solved would not be the field
                # the spec describes (displayed equals
                # computed). CONFIGURATION-AGNOSTIC by construction:
                # every 3-D route — shapes3d, stl3d, scene3d — reaches
                # build_stl3d_run through that same one authority, so
                # this holds for all of them and special-cases none.
                if _ax in _sym.plane_mm and float(_sym.plane_mm[_ax]) != 0.0:
                    errs.append(
                        f"{_ax}-axis symmetry-plane position "
                        f"{float(_sym.plane_mm[_ax]):g} mm is declared on "
                        f"a 3-D deck (depth_mm={g.depth_mm:g} > 0), where "
                        f"the stored-half convention fixes the plane at "
                        f"stored 0. The 3-D builders' one authority for "
                        f"declared mirror planes "
                        f"(build_stl3d._declared_mirror_axes) reads "
                        f"symmetry.planes / scene.grid.mirror and never "
                        f"plane_mm, so this location would be silently "
                        f"ignored and the solved field would not be the "
                        f"one this spec describes. Either declare "
                        f"plane_mm[{_ax!r}] = 0.0 (or omit it) and author "
                        f"the half-domain about stored 0, or solve the "
                        f"full extent by setting planes[{_ax!r}] = "
                        f"'none'.")
                # The mirrored axis's STORED EXTENT is already checked by
                # the domain-extent loop above, which is the one
                # authority for that fact; re-checking it here would
                # create a second copy that can drift. The remaining
                # assertion — that the unfolded field really is the
                # mirror of its half, with the normal E identically zero
                # on the plane — is not decidable from declarations at
                # all: it is verified after every mirrored build by
                # physics.symmetry.assert_mirror_field_symmetry, which
                # refuses the build if it does not hold.
        if g.field_method not in ("electrode_aware", "plain_gradient"):
            errs.append(f"field_method must be 'electrode_aware' or "
                        f"'plain_gradient', got {g.field_method!r}")
        if g.channel_dtype not in ("float64", "float32"):
            errs.append(f"channel_dtype must be 'float64' or 'float32', "
                        f"got {g.channel_dtype!r}")
        # external-geometry builders carry masks from a file, not inline
        # shapes/STL — skip the per-electrode geometry requirement for them.
        skip_geo = self.builder in EXTERNAL_GEOMETRY_BUILDERS
        for e in g.electrodes:
            if e.shapes and e.stl:
                errs.append(f"electrode {e.name}: both shapes and stl set")
            if not skip_geo and not e.shapes and not e.stl:
                errs.append(f"electrode {e.name}: no geometry")
            for _i, _s in enumerate(e.shapes or []):
                errs.extend(_s.size_errors(f"electrode {e.name}: shape #{_i}"))
        # ---- DC ladders ------------------------------------------------
        names = {gr.name for gr in g.dc_groups}
        for gr in g.dc_groups:
            if gr.interp not in ("linear", "weights"):
                errs.append(f"dc_group {gr.name}: interp must be "
                            f"'linear'|'weights', got {gr.interp!r}")
            if getattr(gr, "mode", "set") not in ("set", "add"):
                errs.append(f"dc_group {gr.name}: mode must be "
                            f"'set' (group owns member dc) or 'add' "
                            f"(float rail added to authored dc), got "
                            f"{gr.mode!r}")
        seen = {}
        for e in g.electrodes:
            if e.dc_group is None:
                continue
            if e.dc_group not in names:
                errs.append(f"electrode {e.name}: dc_group "
                            f"{e.dc_group!r} is not defined")
                continue
            grp = next(x for x in g.dc_groups if x.name == e.dc_group)
            if grp.uniform:
                # ONE AUTHORITY: resolve_dc_groups() short-circuits a
                # uniform group before either ladder branch and writes
                # v_in to every member, consulting neither dc_index nor
                # dc_weight. validate() demanding them anyway made the
                # two disagree: a deck that resolves perfectly was
                # refused at load. That is a defect of THIS check for
                # every uniform group in every deck (the flag is a
                # first-class DCGroupSpec field, not a device option) —
                # it is not scoped to the geometry that exposed it, and
                # the member-count check twenty lines below already
                # treats uniform as its own kind.
                #
                # A member that DOES carry dc_index is NOT refused here,
                # and that is deliberate rather than an oversight. The
                # shipped fleet uses the index on uniform members as a
                # stable member ordinal — mirror pairs numbered L_* 0..2
                # against R_* 100..102 in the planar MRTOF and xevo
                # decks, plate ordinals in the OA decks — while the
                # group itself carries v_in == v_out. Refusing it here
                # rejects 19 decks in this tree that an earlier release
                # accepts and that resolve to exactly the same voltages
                # either way. An unused ordinal is not a contradiction;
                # the voltage the deck declares is the voltage the
                # solver receives, which is the invariant that matters.
                continue
            if grp.interp == "linear":
                if e.dc_index is None:
                    errs.append(f"electrode {e.name}: in dc_group "
                                f"{e.dc_group!r} but has no dc_index — the "
                                f"ladder order must be a NUMBER, never "
                                f"inferred from the name")
                    continue
                key = (e.dc_group, e.dc_index)
                if key in seen:
                    errs.append(f"dc_group {e.dc_group!r}: dc_index "
                                f"{e.dc_index} used by both {seen[key]} and "
                                f"{e.name} — the ladder order is ambiguous")
                seen[key] = e.name
            elif e.dc_weight is None:
                errs.append(f"electrode {e.name}: dc_group {e.dc_group!r} "
                            f"uses interp='weights' but it has no dc_weight")
        for gr in g.dc_groups:
            mem = [e for e in g.electrodes if e.dc_group == gr.name]
            # uniform groups put ONE voltage on every member -> 1 is fine;
            # only ladders need >=2 to interpolate v_in..v_out.
            if not gr.uniform and len(mem) < 2:
                errs.append(f"dc_group {gr.name}: ladder needs >=2 members to "
                            f"distribute v_in..v_out, has {len(mem)} "
                            f"(use a uniform group for a single-voltage set)")

        # An empty m/z list flies no ion: refused here, for every route (on
        # r-z it was only caught by a stray `mz_list[0]` expression in
        # make_rz_fly_fn, as a bare IndexError).
        if not list(self.source.mz_list or []):
            errs.append("source.mz_list is empty -- state at least one ion mass "
                        "(Da)")
        errs += self.source.value_errors()
        # The flight's step and length, for every route. A step <= 0 never
        # advances time (it was refused only at sizing, by record_volume,
        # without the field's name); a t_max_us <= 0 flew a zero-length
        # flight on r-z and 3-D, every ion a 'timeout', and failed planar
        # with "negative dimensions are not allowed".
        _it = self.integration

        def _positive(v):
            return (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v > 0)
        if _it.dt_ns is not None and not _positive(_it.dt_ns):
            errs.append(f"integration.dt_ns = {_it.dt_ns!r}: the step must "
                        f"be a number > 0 ns (or null, to derive it from "
                        f"the drives)")
        if not _positive(_it.t_max_us):
            errs.append(f"integration.t_max_us = {_it.t_max_us!r}: the "
                        f"flight must last a number > 0 us")
        if (isinstance(_it.rec_every, bool)
                or not isinstance(_it.rec_every, int) or _it.rec_every < 1):
            errs.append(f"integration.rec_every = {_it.rec_every!r}: record "
                        f"every N steps needs a whole number N >= 1")
        # (The P_torr/P_pa disagreement check that sat here is gone: P_pa
        # is derived from P_torr on every write, so they cannot disagree.
        # It also carried a retyped Torr constant.)
        if self.source.distribution not in SOURCE_DISTRIBUTIONS:
            errs.append(f"bad distribution {self.source.distribution!r}; "
                        f"known: {list(SOURCE_DISTRIBUTIONS)}")
        if self.source.distribution == "gaussian":
            try:
                self.source.gaussian_layout()
            except ValueError as e:
                errs.append(str(e))
        if self.source.distribution == "file" and not \
                self.source.births_file:
            errs.append("distribution=file but no births_file")
        if self.source.ke_hi < self.source.ke_lo:
            errs.append("ke_hi < ke_lo")
        for c in self.integration.record_channels:
            if c not in OPTIONAL_CHANNELS:
                errs.append(f"unknown record channel {c!r}")
        # (3-D geometry shown as a single 2-D slice is allowed; no check)
        return errs

    def advisories(self):
        """Ill-advised-but-legal configurations, each with its stated
        reason. BY DESIGN: the RF time-resolution and
        stroboscopic-sampling checks are ADVISORIES, not refusals —
        deliberately coarse scenarios can be warranted, so nothing
        breaks; the user is warned on the command line (build_run) and
        in the GUI status instead. validate() stays the refusal
        channel for configurations that cannot mean what was asked."""
        out = []
        g = self.geometry
        _it = self.integration
        # a beam energy with a zero direction is ignored (births at rest)
        _zd = self.source.zero_direction_note()
        if _zd:
            out.append(_zd)
        # gas physics the deck did not state, flown at the default
        _col = self.collisions
        _dflt = (sorted(getattr(_col, "_defaulted", ()) or ())
                 if _col.enabled else [])
        if _dflt:
            _unit = {"P_torr": " Torr", "T_k": " K", "gas": ""}
            out.append(
                "collisions: the deck does not state "
                + ", ".join(f"{k} (flying the default "
                            f"{getattr(_col, k)!r}{_unit.get(k, '')})"
                            for k in _dflt)
                + " -- state them in the deck if they matter")
        # TABLE DRIVES vs AUTO-dt: the
        # auto-dt heuristic derives the step from the slowest drive PERIOD,
        # and a table group's frequency_hz is a placeholder, not a
        # timescale -- the time-lag example's 1 Hz made auto-dt pick a ~ms
        # step and the Verlet exploded. Until the heuristic excludes table
        # channels, a spec that mixes them with an unpinned dt gets warned.
        if _it.dt_ns is None and any(gr.waveform == "table"
                                     for gr in g.rf_groups):
            out.append(
                "integration: dt_ns is unset with a table-waveform drive "
                "present; auto-dt keys on drive frequency, which is "
                "meaningless for a table, and can choose a step that "
                "destroys the integration -- pin dt_ns explicitly "
                "(the time-lag example uses 1.0).")
        _rf_f = [gr.frequency_hz for gr in g.rf_groups
                 if gr.frequency_hz and gr.amplitude_v]
        # the step the run WILL use; read, never written back -- an
        # advisory reports and must not change the spec (the
        # resolution itself lives in from_dict and sim_build.build_run)
        _dt = (_it.dt_ns if _it.dt_ns is not None
               else self.derive_dt_ns())
        _long = long_flight_note(_dt, _it.t_max_us,
                                 getattr(self.source, "n_ions", 1))
        if _long:
            out.append(_long)
        # GAS-TRANSPORT TIMING BIAS (validation spec
        # V10, scope extended by V11): gas-phase transport TIMING is
        # fast-biased O(dt). Measured, operating points inline: funnel
        # Stage-2 ladder +1.48% at dt=1 ns (500 kHz RF, 1 Torr N2,
        # m/z 556); Mason-Schamp drift tube ~-3%/ns DC-ONLY (1 Torr He,
        # C60) -- the bias is a GAS property, not an RF one. Fates,
        # transmission and positions are dt-robust (measured: 500/500
        # fates at 4 ns; plate |y| shift 1.1 um << SE). Warn-not-refuse
        # convention: optimizer loops/sweeps (relative rankings) and
        # fate/position work are legitimate at coarse dt.
        # VOXEL-SURFACE TERM (validation spec
        # V12): voxelized CURVED metal displaces near-surface
        # observables (impact positions, near-electrode trajectories) by
        # O(h). Measured on the Pilot D quad (4.6 mm rods, h=0.5 mm):
        # impact-z term ~1.5 mm, closing 66% per h-halving toward the
        # smooth-surface limit. Geometry-agnostic BY DESIGN
        # (a pitch number would be a geometry-specific rule): fires on
        # the presence of curved or mesh metal, whose staircase error
        # is intrinsic; axis-aligned rects land on voxel faces.
        _curved = any(
            (sh.to_dict().get("type") in ("ellipse", "polygon"))
            for el in g.electrodes for sh in el.shapes) or any(
            getattr(el, "stl", None) for el in g.electrodes)
        if _curved:
            out.append(
                f"geometry: curved/mesh metal voxelized at "
                f"{g.mm_per_gu:g} mm/gu carries an O(h) surface-"
                f"position term on NEAR-SURFACE observables (impact "
                f"positions, near-electrode paths) — measured ~1.5 mm "
                f"at h=0.5 on 4.6 mm rods, 66% smaller per h-halving. "
                f"If such observables are certified, demonstrate an "
                f"h-plateau or quote the term inline; interior/"
                f"paraxial observables are unaffected (measured).")
        # SUB-CELL SOURCE -- ROUTE-AGNOSTIC BY
        # CONSTRUCTION: it reads only the DECLARED source extent and the
        # pitch, both of which every route has, so planar, r-z, stl2d,
        # 3-D and tw2d all get it from the one place that already
        # warns about voxelized curved metal. A source narrower than a
        # cell means every derivative measured across the packet -- an
        # aberration coefficient, a lever sensitivity, an optimisation
        # gradient -- is dominated by the interpolation stencil.
        _s = self.source
        _h = float(g.mm_per_gu)
        _ext = {}
        if _s.distribution == "box":
            for _a, _v in zip("xyz", (list(_s.box_mm) + [0, 0, 0])[:3]):
                _ext[_a] = float(_v)
        elif _s.distribution in ("disc", "grid"):
            for _a in "xyz":
                _ext[_a] = (0.0 if _a == _s.axis else 2.0 * float(_s.r_mm))
        elif _s.distribution == "line":
            _ext[_s.axis] = float(_s.len_mm)
        elif _s.distribution == "point":
            _ext = {a: 0.0 for a in "xyz"}
        elif _s.distribution == "gaussian":
            # A Gaussian DECLARES its extent. FWHM is the width
            # the check compares to the pitch -- the same "declared
            # width" role box_mm plays -- and gaussian_layout() has
            # already refused malformed vectors by the time advisories
            # run on a validated spec; an unvalidated one falls back to
            # zeros here and validate() carries the refusal.
            _fw = list(_s.fwhm_mm) if (
                _s.fwhm_mm is not None
                and hasattr(_s.fwhm_mm, "__len__")
                and len(_s.fwhm_mm) == 3) else [0.0, 0.0, 0.0]
            for _a, _v in zip("xyz", _fw):
                _ext[_a] = float(_v)
        elif _s.distribution != "file":
            # NOT a known kind: say so. An empty extent here used to fall
            # through to the POINT-source note below;
            # validate() refuses the kind itself.
            _ext = None
        # 'file' births are not declared here: their extent is in the CSV,
        # so this reports a named skip rather than a silent pass.
        if _ext is None:
            out.append(
                f"source: distribution {_s.distribution!r} is not a known "
                f"kind, so the sub-cell resolution check could not be "
                f"applied (validate() refuses the distribution).")
        elif _s.distribution == "file":
            out.append(
                f"source: births come from a file, so the packet extent "
                f"is not declared in the spec and the sub-cell resolution "
                f"check ({SOURCE_CELLS_ADVISORY:g}-cell advisory) could "
                f"NOT be applied -- measure the flown spread in cells "
                f"before trusting any derivative across the packet.")
        else:
            _thin = {a: v / _h for a, v in _ext.items()
                     if 0.0 < v / _h < SOURCE_CELLS_ADVISORY}
            if _thin:
                out.append(
                    "source: declared extent is SUB-CELL on "
                    + ", ".join(f"{a} ({c:.2f} cells)"
                                for a, c in sorted(_thin.items()))
                    + f" at mm_per_gu={_h:g} mm. Field sampling between "
                    f"nodes is linear WITHIN a cell, so any derivative "
                    f"measured across the packet on those axes (aberration "
                    f"coefficients, lever sensitivities, optimisation "
                    f"gradients) is dominated by the interpolation "
                    f"stencil, not the solved field. Measured precedent: a "
                    f"0.19-cell packet reported an aberration 5x its "
                    f"converged value AND the wrong functional form. "
                    f"Demonstrate a pitch ladder before "
                    f"quoting any such number, and quote it with its "
                    f"pitch.")
            elif all(v == 0.0 for v in _ext.values()):
                out.append(
                    f"source: a POINT source has zero declared extent, so "
                    f"the packet's spread at any measurement plane is set "
                    f"by the optics and cannot be checked against the "
                    f"{_h:g} mm lattice from the spec alone. Measure the "
                    f"flown spread in cells before trusting a derivative "
                    f"across the packet.")
        if self.collisions.enabled and _dt >= 0.5:
            out.append(
                f"collisions: gas-phase transport at dt_ns="
                f"{_dt:g} carries an O(dt) FAST bias on TIMING "
                f"observables (~1.5-3%/ns measured; fates/positions "
                f"are dt-robust). For certified timing, demonstrate a "
                f"dt plateau (the adopted convention) or quote "
                f"the bias inline; dt_ns <= 0.25 reached the plateau "
                f"on both measured devices.")
        if _rf_f and _dt > 0:
            _per_ns = 1e9 / max(_rf_f)
            _steps = _per_ns / _dt
            if _steps < RF_MIN_STEPS_PER_PERIOD:
                out.append(
                    f"integration: dt_ns={_dt:g} resolves the "
                    f"{max(_rf_f):g} Hz RF drive with only {_steps:.0f} "
                    f"steps/period (< {RF_MIN_STEPS_PER_PERIOD}) — this "
                    f"produces numerical micro-heating (phantom eV-scale "
                    f"samples in the velocity ensemble). Use dt_ns <= "
                    f"{_per_ns/RF_RECOMMENDED_STEPS_PER_PERIOD:.3g} "
                    f"({RF_RECOMMENDED_STEPS_PER_PERIOD} steps/period, "
                    f"the stock-example convention).")
            _samp = _dt * max(1, int(_it.rec_every))
            if _samp >= 0.5 * _per_ns:
                _m = _samp % _per_ns
                _off = min(_m, _per_ns - _m) / _per_ns
                if _off < STROBE_WINDOW_FRAC:
                    out.append(
                        f"integration: recording every {_samp:g} ns is "
                        f"commensurate with the {_per_ns:g} ns RF period "
                        f"(within {STROBE_WINDOW_FRAC:.0%}) — "
                        f"stroboscopic sampling records ONE drive phase "
                        f"and biases velocity statistics. Change "
                        f"rec_every so the sampling interval is not an "
                        f"integer multiple of the period.")
        # SDS OUTSIDE ITS RF REGIME: warn, not refuse.
        # Carried as an advisory so it reaches every place advisories go:
        # build_run's console print AND the app's status line -- it used to
        # be printed by build_run only, so a UI user never saw it.
        from ion_gym.physics.sds import (rf_regime_note, sds_active,
                                         species_notes,
                                         SDS_NO_COLLISION_COUNT)
        _sds_note = rf_regime_note(self)
        if _sds_note:
            out.append(_sds_note)
        # the SDS mobility each species flies, and any converted between
        # gases (displayed equals solver input)
        out.extend(species_notes(self))
        # n_col UNDER SDS: the one rule every route applies (NaN column,
        # empty per-ion count); stated whenever the channel is recorded.
        if (sds_active(self.collisions)
                and "n_col" in (_it.record_channels or [])):
            out.append(SDS_NO_COLLISION_COUNT)
        # PER-ION CROSS SECTIONS under sigma_model='ccs_fit': the value
        # each declared m/z flies with, extrapolations flagged (displayed
        # equals solver input).
        from ion_gym.physics.ccs import (sigma_note, extrapolation_note,
                                         stated_sigma_note, fallback_note,
                                         charge_note)
        _sig_note = sigma_note(self)
        if _sig_note:
            out.append(_sig_note)
        # ...any extrapolated m/z as its OWN warning, so it is not buried
        # in that list, a stated sigma_m2 the fit leaves unused (on the
        # console and the app's advisory line), and
        # the static-sigma fallback where the fit does not apply
        for _note in (extrapolation_note(self), stated_sigma_note(self),
                      fallback_note(self), charge_note(self)):
            if _note:
                out.append(_note)
        # HARD-SPHERE STEP vs COLLISION TIME: a
        # loud warning, never a refusal.
        from ion_gym.physics.collision3d import hs_step_note
        _hs_note = hs_step_note(self, _dt)
        if _hs_note:
            out.append(_hs_note)
        return out


    # ------------------------------------------------------------ JSON IO
    def to_dict(self):
        d = {
            "schema_version": SCHEMA_VERSION, "name": self.name,
            "notes": self.notes, "geometry": self.geometry.to_dict(),
            "source": self.source.to_dict(),
            "collisions": self.collisions.to_dict(),
            "integration": self.integration.to_dict(),
            "bounds": self.bounds.to_dict(),
            "stations": [st.to_dict() for st in self.stations],
            "view": self.view.to_dict(), "builder": self.builder,
            "import_path": self.import_path,
            "scene": self.scene}
        # Written only when present, so a deck without one saves
        # byte-identically to before (it was never written).
        if self.transporter is not None:
            d["transporter"] = self.transporter.to_dict()
        if self.optic_axis is not None:            # only when declared
            d["optic_axis"] = self.optic_axis.to_dict()
        return d

    def drive_summary(self) -> str:
        """One-line operating point DERIVED from the live drives and DC
        values -- never from the authored name, which can go stale after
        manual retunes (a title drawn from the name
        misreported a retuned run; displayed numbers must equal solver
        input, so the display derives from the input)."""
        by_wave = {}
        for r in self.geometry.rf_groups:
            key = (round(float(r.frequency_hz), 3), float(r.amplitude_v),
                   r.waveform)
            by_wave.setdefault(key, []).append(r.name)
        parts = []
        for (f, a, wave), names in sorted(by_wave.items(),
                                          key=lambda kv: -kv[0][0]):
            n = len(names)
            fam = names[0] if n == 1 else "{0} x{1}".format(
                names[0].rstrip("0123456789") or names[0], n)
            parts.append("{0} {1:g} Vpp {2} @ {3:g} kHz".format(
                fam, 2.0 * a, wave, f / 1e3))
        dc = {e.name: e.dc for e in self.geometry.electrodes
              if not e.rf_groups and e.dc}
        if dc:
            parts.append("DC " + ", ".join(
                "{0} {1:g} V".format(k, v) for k, v in sorted(dc.items())))
        return " | ".join(parts) if parts else "no drives"

    def to_json(self, path=None, indent=2):
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            write_text_atomic(path, s)
        return s

    @classmethod
    def from_dict(cls, d):
        # Another kind of document is named as itself, with its loader,
        # before any single-deck check can blame a missing block.
        from ion_gym.io.spec_io import sniff
        _kind = sniff(d)
        if _kind in _OTHER_DOCUMENT_LOADERS:
            raise ValueError(f"SimSpec: this is {_OTHER_DOCUMENT_LOADERS[_kind]}")
        check_schema_version(d, "SimSpec")
        _known_top = {f.name for f in fields(cls)}
        # sanctioned metadata (documented allowlist): these
        # ride in spec files by design — _display_name is read by the
        # app's example menu; schema_version is checked above. They are
        # the ONLY tolerated non-field keys; everything else warns.
        # _ui_hidden: the deck ships and flies (notebooks reference it)
        # but is kept out of the example MENU. UI-side metadata, like
        # _display_name -- known, not a misspelling.
        _known_top |= {"schema_version", "_display_name", "_ui_hidden"}
        _extra = sorted(k for k in d if k not in _known_top)
        if _extra:
            import warnings
            warnings.warn(
                f"SimSpec: unknown top-level key(s) {_extra} retained in "
                f"the saved file but NOT interpreted — misspelled or "
                f"foreign parameter?", stacklevel=3)
        # The ion source and the integration
        # contract set the physics, and their defaults are one deck's
        # operating point -- a deck without them is refused, not flown
        # as the funnel. (No collisions block = vacuum.)
        for _blk in ("source", "integration"):
            if _blk not in d:
                raise MissingPhysicsKey(
                    f"SimSpec: the deck has no {_blk!r} block. It sets the "
                    f"physics, and its defaults are one deck's operating "
                    f"point -- state it in the deck.")
        if "name" not in d:
            import warnings
            warnings.warn("SimSpec: cosmetic key(s) not in the deck, "
                          "loaded at their defaults: name='simulation'",
                          stacklevel=3)
        spec = cls(
            geometry=GeometrySpec.from_dict(d["geometry"]),
            source=SourceSpec.from_dict(d["source"]),
            collisions=(CollisionSpec.from_dict(d["collisions"])
                        if "collisions" in d
                        else CollisionSpec(enabled=False)),   # absent: vacuum
            integration=IntegrationSpec.from_dict(d["integration"]),
            bounds=BoundsSpec.from_dict(d.get("bounds", {})),
            stations=[StationSpec.from_dict(x)
                      for x in d.get("stations", [])],
            transporter=(TransporterSpec.from_dict(d["transporter"])
                         if d.get("transporter") is not None else None),
            optic_axis=(OpticAxisSpec.from_dict(d["optic_axis"])
                        if d.get("optic_axis") is not None else None),
            view=ViewSpec.from_dict(d.get("view", {})),
            name=d.get("name", "simulation"), notes=d.get("notes", ""),
            builder=d.get("builder", ""),
            import_path=d.get("import_path", ""),
            scene=d.get("scene"))
        # PASSTHROUGH: top-level foreign keys survive too --
        # SimSpec.from_dict is a hand-built constructor, not a
        # _take_known site, so retention is explicit here. The sanctioned
        # menu metadata is retained the same way (it is known, so it does
        # not warn, but a save that dropped it changed the deck's menu
        # label and visibility).
        _keep = _extra + [k for k in ("_display_name", "_ui_hidden")
                          if k in d]
        if _keep:
            spec._extras = {k: d[k] for k in _keep}
        if spec.integration.dt_ns is None:
            spec.integration.set_derived_dt(spec.derive_dt_ns())
        return spec

    @classmethod
    def from_json(cls, path_or_str):
        # NOTE: when given a real file path, the loaded spec records it in
        # _loaded_from so relative resources (births_file, stl_dir) resolve
        # relative to the SPEC, not the CWD -- spec+resources is portable.
        s = str(path_or_str)
        loaded_from = None
        if s.lstrip().startswith("{"):
            raw = s
        else:
            p = Path(s)
            if p.exists():
                raw = p.read_text(encoding="utf-8")
                loaded_from = str(p)
            else:
                # No silent fallthrough:
                # this used to fall through to json.loads(<the path
                # string>) — every missing/stale/wrong-CWD path became a
                # baffling "Expecting value: char 0" instead of the
                # truth. A string that isn't JSON is a PATH, and a path
                # that doesn't exist is refused BY NAME.
                raise FileNotFoundError(
                    f"SimSpec.from_json: no such file {p!r} (cwd "
                    f"{Path.cwd()}). Anchor spec paths at the repo root "
                    f"— e.g. ion_gym.io.paths.repo_root() / "
                    f"'examples/...' — instead of relying on the "
                    f"notebook's working directory.")
        obj = cls.from_dict(json.loads(raw))
        obj._loaded_from = loaded_from
        return obj


# ------------------------------------------------------- example builders
@dataclass
class PitchChange(_StrictAttrs):
    """What set_pitch actually did, per axis. RETURNED, not just printed,
    so a headless caller can assert on it and a notebook can put it in a
    read-out — a print-only report is lost the moment nobody is looking."""
    axis: str
    old_extent_mm: float
    new_extent_mm: float
    old_origin_mm: Optional[float] = None
    new_origin_mm: Optional[float] = None
    cells: int = 0
    # WHY the origin moved, when it did. Two different things happen and
    # saying "grown symmetrically about the plane" for both would be a
    # report that claims something that did not happen.
    #   "symmetric" -- a 2-D mirrored axis with a declared plane, grown
    #                  equally on both sides so the plane stays on the
    #                  metal
    #   "covered"   -- a non-mirrored axis whose frame origin did not sit
    #                  on the new lattice; the low wall moved outward to
    #                  the node below it
    origin_reason: str = ""

    def moved(self):
        return (abs(self.new_extent_mm - self.old_extent_mm) > 1e-12
                or (self.old_origin_mm is not None
                    and abs((self.new_origin_mm or 0.0)
                            - self.old_origin_mm) > 1e-12))

    def describe(self):
        t = (f"{self.axis}: {self.old_extent_mm:g} -> "
             f"{self.new_extent_mm:g} mm ({self.cells} cells)")
        if (self.old_origin_mm is not None
                and abs((self.new_origin_mm or 0.0) - self.old_origin_mm) > 1e-12):
            why = {"symmetric": "grown symmetrically about the plane, "
                                "which did not move",
                   "covered": "low wall moved outward onto the new "
                              "lattice"}.get(self.origin_reason,
                                             self.origin_reason or "")
            t += (f", origin {self.old_origin_mm:g} -> "
                  f"{self.new_origin_mm:g} mm ({why})")
        return t


def min_rec_every(integ):
    """Smallest rec_every whose records cover the WHOLE declared flight
    within max_records -- the value check_recording_capacity's refusal
    names. THE one place that arithmetic lives: the refusal and any
    caller choosing a record interval (e.g. a decimation study) read it
    here instead of retyping a number that breaks when the deck changes.

    Returns 1 when the integration declares nothing to check against
    (dt_ns, t_max_us or max_records unset -- the same condition under
    which check_recording_capacity checks nothing). Refuses when
    max_records <= 2: the buffer holds only the two end rows, so no
    rec_every can cover the flight.
    """
    dt_ns = float(getattr(integ, "dt_ns", 0.0) or 0.0)
    t_max_us = float(getattr(integ, "t_max_us", 0.0) or 0.0)
    cap = int(getattr(integ, "max_records", 0) or 0)
    if dt_ns <= 0 or t_max_us <= 0 or cap <= 0:
        return 1
    if cap <= 2:
        raise ValueError(
            f"max_records={cap} holds only the two end rows; no rec_every "
            f"can cover a {t_max_us:g} us flight. Raise max_records.")
    steps = flight_steps(dt_ns, t_max_us)
    # the exact solution of _records_needed(steps, r) <= cap, then nudged
    # through THE SAME predicate so float rounding can never disagree
    # with the check. Rounding moves it by a step or two; the nudge is
    # bounded so a case it cannot settle fails loudly instead of looping.
    r = max(int(steps // (cap - 1)) + 1, 1)
    for _ in range(_REC_EVERY_NUDGES):
        if r > 1 and _records_needed(steps, r - 1) <= cap:
            r -= 1
        elif _records_needed(steps, r) > cap:
            r += 1
        else:
            return r
    raise RuntimeError(
        f"min_rec_every did not settle within {_REC_EVERY_NUDGES} steps of "
        f"its estimate (steps={steps:g}, max_records={cap}, last "
        f"rec_every={r}); this is a defect in the arithmetic, not the deck")


# A flight longer than this many integrator steps cannot be counted in a
# float exactly (2**53): steps / r stops changing with r, which is how the
# record arithmetic above once looped forever on a t_max_us of 1e27.
MAX_FLIGHT_STEPS = 2 ** 53
_REC_EVERY_NUDGES = 64


def flight_steps(dt_ns, t_max_us):
    """Integrator steps in a flight, refused by name when it is not a
    finite count below MAX_FLIGHT_STEPS."""
    steps = t_max_us * 1e3 / dt_ns
    if not math.isfinite(steps) or steps >= MAX_FLIGHT_STEPS:
        raise ValueError(
            f"integration: t_max_us={t_max_us:g} at dt_ns={dt_ns:g} is "
            f"{steps:g} integrator steps; a flight must be a finite number "
            f"of steps below 2**53 ({float(MAX_FLIGHT_STEPS):.3g}). Check "
            f"t_max_us and dt_ns.")
    return steps


def long_flight_note(dt_ns, t_max_us, n_ions):
    """The long-flight advisory, or None. Steps per ion x ions above
    LONG_FLIGHT_ION_STEPS is announced with a wall-time range from
    FLIGHT_NS_PER_STEP. Values the validator refuses (a non-positive step
    or flight) give None here; the refusal names them."""
    try:
        dt_ns, t_max_us = float(dt_ns), float(t_max_us)
        n = max(int(n_ions or 1), 1)
    except (TypeError, ValueError):
        return None
    if not (dt_ns > 0 and t_max_us > 0
            and math.isfinite(dt_ns) and math.isfinite(t_max_us)):
        return None
    per_ion = t_max_us * 1e3 / dt_ns
    total = per_ion * n
    if total <= LONG_FLIGHT_ION_STEPS:
        return None
    from ion_gym.physics.sizing import fmt_time
    lo, hi = (total * ns * 1e-9 for ns in FLIGHT_NS_PER_STEP)
    return (f"integration: a long flight -- up to {per_ion:.3g} steps per "
            f"ion (t_max_us={t_max_us:g} at dt_ns={dt_ns:g}) x {n} ions = "
            f"{total:.3g} integrator steps, roughly {fmt_time(lo)} to "
            f"{fmt_time(hi)} on one core (fewer if ions end early; divided "
            f"by the worker count). If that is not intended, check dt_ns "
            f"and t_max_us.")


def _records_needed(steps, every):
    """Records a flight of `steps` integrator steps needs at `every` --
    the count check_recording_capacity uses, kept in one
    place so min_rec_every solves the same inequality the check tests."""
    return int(steps / every) + 2


def check_recording_capacity(integ):
    """Refuse a recording that cannot cover the declared flight.

    max_records is a CAP ON THE BUFFER, and when the flight needs more
    records than the cap allows the integrator does not thin the trace --
    IT STOPS RECORDING. The trajectory returned then covers only the
    first part of the flight and carries nothing saying so.

    MEASURED: dt 2 ns, t_max 3000 us, rec_every 10,
    max_records 100000. 1,500,000 steps / 10 = 150,000 records wanted
    against a 100,000 cap, so every trace stopped around 2000 us of a
    3000 us flight and the last third was simply absent. Fates and
    positions still looked plausible; any timing observable read off the
    tail was wrong. That is a CORRECTNESS defect, which is why this
    refuses rather than warns -- a warning on this path already existed
    on the 3-D STL route and went unseen in the console for months.

    The message carries the arithmetic and BOTH levers with the value
    that would actually fit, so the fix is a copy-paste, not a
    calculation. Raising rec_every thins uniformly and keeps the whole
    flight; raising max_records keeps the sample rate and costs memory
    (records x ions x channels x 8 bytes).
    """
    dt_ns = float(getattr(integ, "dt_ns", 0.0) or 0.0)
    t_max_us = float(getattr(integ, "t_max_us", 0.0) or 0.0)
    every = max(int(getattr(integ, "rec_every", 1) or 1), 1)
    cap = int(getattr(integ, "max_records", 0) or 0)
    if dt_ns <= 0 or t_max_us <= 0 or cap <= 0:
        return None                      # nothing to check against
    steps = flight_steps(dt_ns, t_max_us)
    need = _records_needed(steps, every)
    if need <= cap:
        return None
    every_fit = min_rec_every(integ) if cap > 2 else None
    covered_us = cap * every * dt_ns * 1e-3
    return (
        f"recording cannot cover the flight: {steps:,.0f} steps at "
        f"dt_ns={dt_ns:g} over t_max_us={t_max_us:g}, recorded every "
        f"{every} -> {need:,} records wanted but max_records={cap:,}. "
        f"max_records does NOT thin the trace, it STOPS RECORDING, so "
        f"every trajectory would cover only the first {covered_us:,.0f} "
        f"us and the rest would be silently absent. Either set "
        f"rec_every={every_fit} (thins uniformly, keeps the whole "
        f"flight) or set max_records={need:,} (keeps the sample rate, "
        f"costs {need:,} records per ion).")


def set_pitch(spec, mm_per_gu):
    """Re-solve THIS geometry at a different resolution, in place.

    THE SPEC-LEVEL SIBLING of GeomScene.at_resolution, which has done this
    for scenes while every other caller had to remember
    by hand. Under the lattice rule `mm_per_gu` is NOT independently settable:
    the extents are lattice quantities counted in cells, so a pitch that
    does not divide them leaves the deck refusable. Measured on the
    certified oa_12plate deck, four of five ordinary pitches were refused
    by the loader after the UI set the pitch alone.

    WHAT MOVES, AND WHAT NEVER DOES:

      * ELECTRODE GEOMETRY NEVER MOVES. Not snapped, not rounded, not
        re-centred. Metal edges stay in mm and rasterize — the A7
        carve-out — so a 0.55 mm gap is still 0.55 mm at every pitch.
        Voltages, drives, source and gas are likewise untouched.
      * THE VACUUM BOX moves, by less than one cell, outward. Slop
        belongs at the outer walls; trimming would put a wall inside the
        metal.
      * WHICH NODES LAND INSIDE THE METAL changes. That IS the resolution
        change, and it is the point of the call.

    MIRRORED AXES, the part that needs care. Growing an extent while the
    origin stays put moves the MIDLINE, so on an axis that folds about
    its midline the plane drifts off the metal it is supposed to mirror —
    a field that is not the mirror of its half. Handled
    by convention, not by hope:

      * 3-D routes store the HALF domain with the plane at stored 0, so
        growing the far wall cannot move the plane. Covered up plainly.
      * 2-D with a DECLARED plane_mm: the domain is grown SYMMETRICALLY
        about that plane (origin shifts by half the growth), so the plane
        stays exactly where the metal is, still on a node, and the extent
        is an even cell count by construction.
      * 2-D with the MIDLINE DEFAULT and no declared plane: REFUSED when
        growth is needed. There is no way to keep an implicit midline on
        the metal while moving a wall, and covering anyway would fold
        about a plane the geometry does not have. The refusal names the
        two ways out.

    Returns a list of PitchChange. Raises ValueError, naming the axis and
    the reason, when the domain cannot be covered safely.
    """
    g = spec.geometry
    h = float(mm_per_gu)
    if h <= 0.0:
        raise ValueError(f"resolution must be > 0 mm/gu, got {mm_per_gu!r}")
    import math
    from ion_gym.io.lattice import (LATTICE_TOL_CELLS,
                                    cover_extent_mm, on_lattice)

    sym = g.symmetry.normalized()
    is_3d = float(g.depth_mm or 0.0) > 0.0
    origin = {"x": 0.0, "y": 0.0}
    if g.origin_mm is not None and len(g.origin_mm) == 2:
        origin["x"], origin["y"] = float(g.origin_mm[0]), float(g.origin_mm[1])
    has_origin = g.origin_mm is not None

    extents = {"x": g.width_mm, "y": g.height_mm, "z": g.depth_mm}
    changes, new_ext, new_org = [], {}, dict(origin)

    for ax in ("x", "y", "z"):
        ext = float(extents[ax])
        if ax == "z" and not is_3d:
            new_ext[ax] = 0.0
            continue
        mirrored = sym.planes.get(ax) == "mirror"
        o0 = origin.get(ax, 0.0)

        if not mirrored or is_3d:
            # Plain cover-up, OUTWARD ON BOTH WALLS. On a 3-D mirrored
            # axis the plane is stored 0 and only the far wall moves, so
            # this is safe there too.
            #
            # The origin is covered as well as the extent. The loader
            # requires origin_mm to sit on the pitch lattice (or no node
            # can land on an integer-mm plane), and an origin authored at
            # a different pitch generally will not — mirror_end_quarter's
            # origin of 40 mm is not a whole number of 0.3 mm cells. That
            # is not a reason to refuse a resolution change: the origin
            # of a NON-mirrored axis is a frame offset, not a physical
            # feature, so the low wall moves OUTWARD to the node below it
            # and the extent grows to keep the high wall where it was.
            # Slop at the outer walls, both of them.
            lo, hi = o0, o0 + ext
            if has_origin and ax in origin and not on_lattice(lo, h):
                lo_n = math.floor(lo / h + LATTICE_TOL_CELLS) * h
            else:
                lo_n = lo
            cov = cover_extent_mm(hi - lo_n, h)
            new_ext[ax] = cov
            if has_origin and ax in new_org:
                new_org[ax] = lo_n
            changes.append(PitchChange(
                axis=ax, old_extent_mm=ext, new_extent_mm=cov,
                old_origin_mm=(o0 if has_origin and ax in origin else None),
                new_origin_mm=(lo_n if has_origin and ax in origin else None),
                cells=int(round(cov / h)), origin_reason="covered"))
            continue

        # 2-D mirrored axis.
        if ax not in sym.plane_mm:
            # midline default: the plane is 0.5*extent and MOVES with it
            cov = cover_extent_mm(ext, h, mirrored=True)
            if abs(cov - ext) > 1e-12:
                raise ValueError(
                    f"cannot re-pitch to {h:g} mm/gu: the {ax}-axis folds "
                    f"about the DOMAIN MIDLINE (no symmetry.plane_mm["
                    f"{ax!r}] declared), and covering the extent "
                    f"{ext:g} -> {cov:g} mm would move that midline by "
                    f"{0.5 * (cov - ext):+g} mm, off the metal it is "
                    f"supposed to mirror. Folding about a plane the "
                    f"geometry does not have is exactly the defect the "
                    f"symmetry gate exists to catch, so this is refused "
                    f"rather than covered. Either declare "
                    f"symmetry.plane_mm[{ax!r}] so the plane is fixed and "
                    f"the domain can grow symmetrically about it, or "
                    f"choose a pitch that divides {ext:g} mm "
                    f"(nearest: {ext / round(ext / h):.6g} mm/gu).")
            new_ext[ax] = cov
            changes.append(PitchChange(axis=ax, old_extent_mm=ext,
                                       new_extent_mm=cov,
                                       cells=int(round(cov / h))))
            continue

        # 2-D mirrored axis with a DECLARED plane: grow symmetrically
        # about it. half = the larger side, rounded UP to whole cells, so
        # neither side is clipped and the plane lands on a node.
        pl = float(sym.plane_mm[ax])
        lo, hi = o0, o0 + ext
        half = cover_extent_mm(max(pl - lo, hi - pl), h)
        lo_n, ext_n = pl - half, 2.0 * half
        new_ext[ax] = ext_n
        new_org[ax] = lo_n
        changes.append(PitchChange(axis=ax, old_extent_mm=ext,
                                   new_extent_mm=ext_n, old_origin_mm=o0,
                                   new_origin_mm=lo_n,
                                   cells=int(round(ext_n / h)),
                                   origin_reason="symmetric"))

    g.width_mm, g.height_mm = new_ext["x"], new_ext["y"]
    if is_3d:
        g.depth_mm = new_ext["z"]
    if has_origin:
        g.origin_mm = [new_org["x"], new_org["y"]]
    g.mm_per_gu = h

    errs = spec.validate()
    if errs:
        raise ValueError(
            "set_pitch produced a spec its own loader refuses — this is a "
            "defect in set_pitch, not in the deck, and the geometry has "
            "been left at the new pitch for inspection:\n  "
            + "\n  ".join(errs))
    return changes


def reflectron_vacuum_spec():
    """A vacuum (collisions off), planar example skeleton — shows the
    schema handling a DC-only, no-gas instrument. It carries NO
    electrodes, so validate() refuses it until some are added; for a
    flyable deck start from examples/."""
    return SimSpec(
        geometry=GeometrySpec(
            width_mm=90.0, height_mm=40.0, mm_per_gu=0.1,
            symmetry=SymmetrySpec(coords="xyz")),
        source=SourceSpec(n_ions=50, distribution="point", x0_mm=45.0,
                          ke_lo=20000.0, ke_hi=20000.0, mz_list=[1000.0],
                          direction=[0.0, 1.0, 0.0]),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=0.5, t_max_us=100.0,
                                    record_channels=["speed", "ke_ev"]),
        view=ViewSpec(mode="2d", planes=["xy"]),
        name="reflectron (vacuum) skeleton")
