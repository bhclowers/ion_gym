"""
ion_gym.sim_spec
----------------
A single, self-contained JSON schema describing a WHOLE simulation, for
any geometry — not just the funnel. One SimSpec answers every setup
question:

  geometry   : electrodes as inline shapes (rect/ellipse/polygon/cutout,
               mm-space, matching ion_playground's vocabulary) OR a
               reference to STL files (one per electrode/basis). Symmetry
               is declared (planar | cylindrical | none) so the same spec
               drives a 2-D cross-section or a 3-D solve.
  voltages   : per electrode — dc, rf_amplitude, rf_frequency_hz,
               rf_phase_deg (ion_playground's exact field names). The
               field assembler turns these into A + sum_k s_k(t) B_k, so
               DC-only, single-RF (two-phase), and multi-phase drives are
               all expressible. Adjusting voltages never re-solves the
               bases — it re-weights them (the fast-adjust invariant), so
               the app can sweep voltages and redraw fields live.
  source     : ion births — a distribution spec (point | disc | grid |
               line | file), KE or velocity spread, m/z list, charge,
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

import json
import math
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import List, Optional, Dict, Any


SCHEMA_VERSION = 1

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
def _take_known(cls, d):
    """LOUD LOAD: unknown keys in a spec dict are WARNED about, never
    silently dropped — the silent filter is exactly how a legacy
    'dt_us' key ran an example at the 1.0 ns default while its author
    believed 2.0 was in force. Loading still succeeds (foreign metadata
    is tolerated), but loudly and by name."""
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


@dataclass
class ShapeSpec(_StrictAttrs):
    """A geometry primitive in mm-space. type in
    {rect, ellipse, polygon, cutout, group}. Fields mirror
    ion_playground.shapes so its scene JSONs translate directly.

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
                f"{legacy} (pre-2026-08-02 extrude_spec output). These "
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
    def _k(gr):
        tail = gr.name[len(prefix):]
        return int(tail) if tail.isdigit() else 0
    tw_sorted = sorted(tw, key=_k)
    len(tw_sorted)
    step = (phase_step_deg if phase_step_deg is not None
            else None)
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
    """
    name: str = "LADDER"
    v_in: float = 0.0
    v_out: float = 0.0
    interp: str = "linear"          # 'linear' | 'weights'
    uniform: bool = False           # plain equal-V set: all members at v_in
    # A 'uniform' group is a non-ladder DC group — every
    # member sits at the SAME voltage (v_in); v_out tracks it. It is exactly
    # a ladder with v_in==v_out, surfaced as its own kind so the UI shows one
    # voltage field instead of in/out/number. The ladder is unchanged.

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items()
                      if k in cls.__dataclass_fields__})


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
    `rf_group` into `rf_groups`, and drops obsolete per-electrode
    rf_amplitude/rf_frequency_hz/rf_phase_deg with those semantics moved
    to the group. New specs never write them."""
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
            parts.append(f"DC ladder '{self.dc_group}' tap {tap} "
                         "(dc derived, not authored)")
        if self.metal_depth_mm is not None:
            parts.append(f"metal depth {self.metal_depth_mm:g} mm "
                         "(overrides the stage declaration)")
        return " | ".join(parts)

    @property
    def is_rf(self):
        return bool(self.group_names())

    def to_dict(self):
        d = asdict(self)
        d["shapes"] = [s.to_dict() for s in self.shapes]
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["shapes"] = [ShapeSpec.from_dict(s) for s in d.get("shapes", [])]
        # LEGACY FOLD (clean drive model): singular rf_group ->
        # rf_groups; per-electrode rf_phase_deg 180 becomes a distinct
        # group only if the caller already split rails by group, so we
        # simply drop the obsolete per-electrode fields (their semantics
        # moved to RFGroupSpec). A legacy JSON that relied on per-electrode
        # phase to make a rail is refused loudly at compose time, not
        # silently mis-driven.
        groups = list(d.get("rf_groups", []) or [])
        if d.get("rf_group") and d["rf_group"] not in groups:
            groups = [d["rf_group"]] + groups
        d["rf_groups"] = groups
        # documented legacy folds — deliberately dropped,
        # so they must not trip the loud unknown-key warning
        d.pop("rf_group", None)
        d.pop("rf_phase_deg", None)
        return _take_known(cls, d)


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

    def _group_map(self):
        return {g.name: g for g in self.rf_groups}

    def electrode_rf(self, el):
        """Effective (amplitude_v, frequency_hz, phase_deg) for an
        electrode: from its named rf_group if set, else its legacy
        per-electrode RF fields, else zeros (DC-only)."""
        names = el.group_names()
        if names:
            g = self._group_map().get(names[0])
            if g is not None:
                return g.amplitude_v, g.frequency_hz, g.phase_deg
        return 0.0, 0.0, 0.0

    def electrode_drive(self, el):
        """Full drive-group resolution: the electrode's RFGroupSpec (from
        its named rf_group), or a synthesized sin group for legacy
        per-electrode RF fields, or None (DC-only). Unlike electrode_rf
        this carries the waveform/table/pe_mode — the general drive."""
        names = el.group_names()
        if names:
            g = self._group_map().get(names[0])
            if g is not None:
                return g
        return None

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
    def from_dict(cls, d):
        d = dict(d)
        d["electrodes"] = [ElectrodeSpec.from_dict(e)
                           for e in d.get("electrodes", [])]
        d["rf_groups"] = [RFGroupSpec.from_dict(g)
                          for g in d.get("rf_groups", [])]
        d["dc_groups"] = [DCGroupSpec.from_dict(g)
                          for g in d.get("dc_groups", [])]
        sym = d.get("symmetry")
        if isinstance(sym, str):
            # back-compat: old coarse enum
            coords = "rz" if sym == "cylindrical" else "xyz"
            d["symmetry"] = SymmetrySpec(coords=coords)
        else:
            d["symmetry"] = SymmetrySpec.from_dict(sym)
        return _take_known(cls, d)


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
                 gaussian_layout() for the refusal contract (L-248)
      file     : births_file (csv with x,y,z,vx,vy,vz[,tob]) — e.g. an external
    Kinematics: ke_lo..ke_hi (eV) directed along `direction`, OR
    thermal at temperature_k; m/z list flown as a mix; charge signed;
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


# ------------------------------------------------------------- collisions
TORR_PA = 133.32236842105263          # 1 Torr in Pa (exact-ish, 101325/760)


@dataclass
class CollisionSpec(_StrictAttrs):
    """HS buffer gas. enabled=false -> vacuum (reflectron/TOF).

    Pressure is authored in TORR (P_torr) — the working unit for this lab and
    for the SDS diffusion model. P_pa is derived from it for the kernel; if a legacy spec
    supplies only P_pa, it is honored and P_torr back-filled. Set exactly one."""
    enabled: bool = True
    gas: str = "N2"
    T_k: float = 273.0
    P_torr: float = 1.0
    P_pa: float = None
    sigma_m2: float = 2.27e-18
    model: str = "hs"                  # "hs" (hard-sphere) or "sds";
                                       # legacy "hs1" is accepted and
                                       # normalized at ingest (renamed
                                       # at ingest; schema too)
    gas_diam_nm: float = 0.366         # SDS gas diameter (air default)
    flow_mm_us: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        # single source of truth: Torr. P_pa is derived unless only P_pa given.
        if self.P_pa is None:
            self.P_pa = self.P_torr * TORR_PA
        else:
            # legacy path: P_pa authored directly -> back-fill P_torr
            self.P_torr = self.P_pa / TORR_PA

    def set_pressure_torr(self, torr):
        """Set BOTH pressure fields from Torr, atomically.

        Assigning P_torr and nulling P_pa does NOT re-derive (the derivation
        lives in __post_init__), so the two fields drift apart and the kernel
        -- which reads P_torr -- silently disagrees with anything reading
        P_pa. Two fields for one quantity is a standing hazard; this is the
        one door through it."""
        self.P_torr = float(torr)
        self.P_pa = float(torr) * TORR_PA
        return self

    def set_pressure_pa(self, pa):
        self.P_pa = float(pa)
        self.P_torr = float(pa) / TORR_PA
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


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

    def to_dict(self):
        return asdict(self)

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

    kind (ruled 2026-09-12 — record/detect was "a distinction
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
                  "aperture" 2026-09-11, Brian: an aperture is the
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
                        f"(2026-09-11 — an aperture is the opening; "
                        f"this kind is the plate). Update the deck.")
        elif self.kind == "record":
            errs.append(f"{prefix}station {self.name!r}: kind "
                        f"'record' was RETIRED (2026-09-12 — "
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

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return _take_known(cls, d)


# =================================================================== SimSpec
@dataclass
class SimSpec(_StrictAttrs):
    _RUNTIME_ATTRS = frozenset({"_loaded_from", "_extras"})
    geometry: GeometrySpec
    source: SourceSpec = field(default_factory=SourceSpec)
    collisions: CollisionSpec = field(default_factory=CollisionSpec)
    integration: IntegrationSpec = field(default_factory=IntegrationSpec)
    bounds: "BoundsSpec" = field(default_factory=BoundsSpec)
    # Declared planes (detectors, slits, diagnostic planes) the deck
    # carries as first-class objects; see StationSpec.
    stations: List[StationSpec] = field(default_factory=list)
    # optional device-agnostic periodic
    # transporter plane pair: dict with axis ('x'|'y'|'z'), accept_mm,
    # emit_mm (spec frame), direction (+1/-1), max_passes. None ->
    # disabled. Only the 3-D shapes route implements it; every other
    # route REFUSES a spec that carries it.
    transporter: dict | None = None
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
        L.append(f"    m/z {list(s.mz_list)}, charge {s.charge:+d}; birth "
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
            L.append(f"  periodic transporter: {self.transporter}")
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
            if getattr(grp, "uniform", False):
                # plain equal-V group: every member at v_in (v_out mirrors)
                grp.v_out = grp.v_in
                for e in mem:
                    e.dc = float(grp.v_in)
                continue
            if grp.interp == "weights":
                # weights mode: dc_weight IS the tap position;
                # dc_index is not consulted (it may be None). A
                # member without a weight is REFUSED by name — the
                # old silent default of 0.0 parked it at v_in, a
                # hidden branch (caught 2026-09-12 when per-plate
                # weighted ladders first exercised this path with
                # index-free members).
                _now = [e.name for e in mem if e.dc_weight is None]
                if _now:
                    raise ValueError(
                        f"dc_group {grp.name!r} (interp='weights'): "
                        f"member(s) {_now} have no dc_weight — a "
                        f"weighted ladder needs every tap placed")
                for e in mem:
                    f = float(e.dc_weight)
                    e.dc = float(grp.v_in) + (float(grp.v_out)
                                              - float(grp.v_in)) * f
                continue
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
            for e in mem:
                f = (e.dc_index - lo) / span
                e.dc = float(grp.v_in) + (float(grp.v_out)
                                          - float(grp.v_in)) * f
        return self

    def validate(self):
        errs = []
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
        # the UI's direction widgets, an IndexError-adjacent crash on
        # Brian's machine (2026-09-11). Refuse it here, by name, with
        # the fix in the message.
        _d = self.source.direction
        try:
            _dv = [float(v) for v in _d]
            _shape_ok = (len(_dv) == 3
                         and all(math.isfinite(v) for v in _dv))
        except TypeError:
            _dv, _shape_ok = None, False
        if not _shape_ok:
            errs.append(
                f"source.direction must be a 3-vector of finite "
                f"floats, got {_d!r} — e.g. [1.0, 0.0, 0.0] for +x "
                f"(a scalar sign is not the contract; `axis` is a "
                f"separate field)")
        elif not any(v != 0.0 for v in _dv):
            # ZERO NORM is legitimate ONLY as "no directed beam".
            # Combined with a declared kinetic energy it is a
            # CONTRADICTION that used to resolve silently: births
            # normalises with `d / (norm(d) or 1.0)`, so a zero vector
            # left the directed term at zero and threw ke_lo..ke_hi
            # away — the deck declared an energy the solver never
            # received. Refuse THAT, by name, and say which of the two
            # declarations to change; a zero direction with no energy
            # declared stays valid and means exactly what it says.
            if self.source.ke_lo > 0 or self.source.ke_hi > 0:
                errs.append(
                    f"source.direction is the zero vector while "
                    f"ke_lo..ke_hi = {self.source.ke_lo}.."
                    f"{self.source.ke_hi} eV declares a directed beam — "
                    f"the energy has nowhere to point and would be "
                    f"silently discarded (ions born exactly at rest). "
                    f"Either give the beam its direction (e.g. "
                    f"[1.0, 0.0, 0.0] for +x) or declare ke_lo = "
                    f"ke_hi = 0 to mean births at rest; for a random "
                    f"thermal spread use temperature_k.")
        for st in self.stations:
            errs += st.validate(prefix="stations: ")
        # A None pressure with collisions ON passed validation and then died
        # inside the numba kernel with a typing wall instead of a diagnostic
        # Refuse it here, by name.
        # The model enum was once UNCHECKED, so a deck saying
        # model="none" validated clean and then flew HS -- the author's
        # intent (no collisions) silently inverted into the default physics.
        # Refuse by name, and name the field that actually controls it.
        # "hs1" is the legacy spelling of "hs";
        # it is normalized here — the ONE ingest point — so every deck
        # written before the rename keeps loading, and everything
        # downstream sees only the canonical name.
        if str(getattr(self.collisions, "model", "") or "").lower() == "hs1":
            self.collisions.model = "hs"
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
            for _f in ("P_torr", "P_pa", "T_k", "sigma_m2"):
                if getattr(self.collisions, _f, None) is None:
                    errs.append(
                        f"collisions enabled but {_f} is None -- set it (use "
                        f"collisions.set_pressure_torr()/set_pressure_pa() "
                        f"for pressure, which sets both fields atomically)")
        if self.transporter is not None:
            tp = self.transporter
            if tp.get("axis") not in ("x", "y", "z"):
                errs.append(f"transporter.axis {tp.get('axis')!r} not x/y/z")
            for k in ("accept_mm", "emit_mm"):
                if not isinstance(tp.get(k), (int, float)):
                    errs.append(f"transporter.{k} missing or non-numeric")
            if isinstance(tp.get("accept_mm"), (int, float)) and \
                    tp.get("accept_mm") == tp.get("emit_mm"):
                errs.append("transporter accept_mm == emit_mm (zero-length)")
            if tp.get("direction", -1) not in (1, -1):
                errs.append("transporter.direction must be +1 or -1")
            if int(tp.get("max_passes", 1)) < 1:
                errs.append("transporter.max_passes must be >= 1")
        g = self.geometry
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
        # ---- DC ladders ------------------------------------------------
        names = {gr.name for gr in g.dc_groups}
        for gr in g.dc_groups:
            if gr.interp not in ("linear", "weights"):
                errs.append(f"dc_group {gr.name}: interp must be "
                            f"'linear'|'weights', got {gr.interp!r}")
        seen = {}
        for e in g.electrodes:
            if e.dc_group is None:
                continue
            if e.dc_group not in names:
                errs.append(f"electrode {e.name}: dc_group "
                            f"{e.dc_group!r} is not defined")
                continue
            grp = next(x for x in g.dc_groups if x.name == e.dc_group)
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
            if not getattr(gr, "uniform", False) and len(mem) < 2:
                errs.append(f"dc_group {gr.name}: ladder needs >=2 members to "
                            f"distribute v_in..v_out, has {len(mem)} "
                            f"(use a uniform group for a single-voltage set)")

        c = self.collisions
        if c.enabled and c.P_torr is not None and c.P_pa is not None:
            derived = float(c.P_torr) * 133.322368421
            if abs(derived - float(c.P_pa)) > 1e-3 * max(derived, 1e-12):
                errs.append(
                    f"collisions: P_torr ({c.P_torr:g} Torr = {derived:.4g} Pa) "
                    f"and P_pa ({c.P_pa:.4g} Pa) DISAGREE. The kernel reads "
                    f"P_torr and would silently use {derived:.4g} Pa, ignoring "
                    f"P_pa. Set one and let the other derive.")
        if self.source.distribution not in (
                "point", "disc", "line", "grid", "box", "gaussian",
                "file"):
            errs.append(f"bad distribution {self.source.distribution!r}")
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
        if _it.dt_ns is None:
            _it.dt_ns = self.derive_dt_ns()
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
        # 'file' births are not declared here: their extent is in the CSV,
        # so this reports a named skip rather than a silent pass.
        if _s.distribution == "file":
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
                    f"converged value AND the wrong functional form "
                    f"(ledger L-216). Demonstrate a pitch ladder before "
                    f"quoting any such number, and quote it with its "
                    f"pitch.")
            elif all(v == 0.0 for v in _ext.values()):
                out.append(
                    f"source: a POINT source has zero declared extent, so "
                    f"the packet's spread at any measurement plane is set "
                    f"by the optics and cannot be checked against the "
                    f"{_h:g} mm lattice from the spec alone. Measure the "
                    f"flown spread in cells before trusting a derivative "
                    f"across the packet (ledger L-216).")
        if self.collisions.enabled and _it.dt_ns >= 0.5:
            out.append(
                f"collisions: gas-phase transport at dt_ns="
                f"{_it.dt_ns:g} carries an O(dt) FAST bias on TIMING "
                f"observables (~1.5-3%/ns measured; fates/positions "
                f"are dt-robust). For certified timing, demonstrate a "
                f"dt plateau (adopted convention 2026-08-13) or quote "
                f"the bias inline; dt_ns <= 0.25 reached the plateau "
                f"on both measured devices.")
        if _rf_f and _it.dt_ns > 0:
            _per_ns = 1e9 / max(_rf_f)
            _steps = _per_ns / _it.dt_ns
            if _steps < RF_MIN_STEPS_PER_PERIOD:
                out.append(
                    f"integration: dt_ns={_it.dt_ns:g} resolves the "
                    f"{max(_rf_f):g} Hz RF drive with only {_steps:.0f} "
                    f"steps/period (< {RF_MIN_STEPS_PER_PERIOD}) — this "
                    f"produces numerical micro-heating (phantom eV-scale "
                    f"samples in the velocity ensemble). Use dt_ns <= "
                    f"{_per_ns/RF_RECOMMENDED_STEPS_PER_PERIOD:.3g} "
                    f"({RF_RECOMMENDED_STEPS_PER_PERIOD} steps/period, "
                    f"the stock-example convention).")
            _samp = _it.dt_ns * max(1, int(_it.rec_every))
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
        return out


    # ------------------------------------------------------------ JSON IO
    def to_dict(self):
        return {
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
            Path(path).write_text(s)
        return s

    @classmethod
    def from_dict(cls, d):
        _known_top = {f.name for f in fields(cls)}
        # sanctioned metadata (documented allowlist): these
        # ride in spec files by design — _display_name is read by the
        # app's example menu; schema_version is provenance. They are the
        # ONLY tolerated non-field keys; everything else warns.
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
        spec = cls(
            geometry=GeometrySpec.from_dict(d["geometry"]),
            source=SourceSpec.from_dict(d.get("source", {})),
            collisions=CollisionSpec.from_dict(d.get("collisions", {})),
            integration=IntegrationSpec.from_dict(
                d.get("integration", {})),
            bounds=BoundsSpec.from_dict(d.get("bounds", {})),
            stations=[StationSpec.from_dict(x)
                      for x in d.get("stations", [])],
            view=ViewSpec.from_dict(d.get("view", {})),
            name=d.get("name", "simulation"), notes=d.get("notes", ""),
            builder=d.get("builder", ""),
            import_path=d.get("import_path", ""),
            scene=d.get("scene"))
        if _extra:
            # PASSTHROUGH: top-level foreign keys survive too --
            # SimSpec.from_dict is a hand-built constructor, not a
            # _take_known site, so retention is explicit here.
            spec._extras = {k: d[k] for k in _extra}
        if spec.integration.dt_ns is None:
            spec.integration.dt_ns = spec.derive_dt_ns()
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
                raw = p.read_text()
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
    schema handling a DC-only, no-gas instrument."""
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
