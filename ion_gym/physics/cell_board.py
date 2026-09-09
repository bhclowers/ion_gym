"""
cell_board.py — parametric rectangular-channel SLIM cross-section
generator for geometry search (ALL FOUR WALLS are pattern-bearing
surfaces — side walls are not restricted to uniform guards).

The 2-D simulation plane is the channel CROSS-SECTION: rails run along
the transport axis (into the screen) and appear here as rectangles on
the walls of a rectangular box. Every wall carries its own repeating
rail pattern; a uniform DC "guard" wall is simply the k=1 pattern whose
single rail spans the wall (roles are OUTCOMES, not inputs). The
generator maps a declared parameter set to a SimSpec; every emitted
shape derives from the arguments and the emitted JSON is the single
source of truth the voxelizer/solver consume.

DESIGN CONSTRAINTS, enforced here with named refusals:
  * Total envelope (channel + walls) fits in 15 mm x 15 mm.
  * Minimum feature (rail width AND rail-to-rail gap) >= 0.125 mm.
  * Rectangular box, all angles 90 deg; up to 4 metal-bearing walls.
  * RF phases restricted to {0, 180} deg: each rail's RF is ONE SIGNED
    amplitude, sign = phase group, |a| in [50, 300] V when on.
    DECLARED MAPPING: |a| < rf_min_v (default 50) means RF OFF for that
    rail (a DC-only / TW-capable rail). The dead zone is part of the
    parametrization, stated here, not a hidden branch.
  * >= 1 RF-free rail per cell PER WALL PATTERN: the LAST rail of every
    wall's cell has its RF forced off (TW-capable by construction).
    Consequence: k=2 admits only single-phase arrays;
    two-phase alternation needs k>=3; the incumbent is k=4.

STARTING SEARCH SPACE (a declared dimensionality choice
— NOT an assumed field symmetry; the solver always solves the full
domain): mirror_tb=True copies the bottom pattern to the top wall and
mirror_lr=True copies the left pattern to the right wall. Setting the
flags False gives full per-wall independence in the SAME generator;
lifting the restriction is a deliberate search-space decision, not a
default.

Template-based emission: geometry/RF/DC are REPLACED on a deep copy of
a caller-supplied template SimSpec; integration, collisions, source m/z
set and bounds are INHERITED from the template. The source is
re-centered to the channel interior.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

from ion_gym.io.lattice import cover_extent_mm

# Design bounds (module-level named constants, not magic numbers in code)
ENVELOPE_MM = 15.0        # max outer extent, each axis
MIN_FEATURE_MM = 0.125    # rail-to-rail ISOLATION GAP (125 um).
                          # This is the min AND the
                          # max — any wider gap is exposed dielectric
                          # that charges up under ion impact. Boards are
                          # fully metallized except isolation cuts.
RAIL_MIN_MM = 0.4         # minimum ELECTRODE width (fabrication
                          # floor): rails narrower than ~400 um
                          # are not buildable/usable here.
RF_MIN_V = 50.0           # |a| below this = RF off (declared dead zone)
RF_MAX_V = 300.0          # amplitude ceiling

WALLS = ("bottom", "top", "left", "right")


class CellBoardInfeasible(ValueError):
    """A candidate that violates a declared constraint. Screens must record
    these as infeasible-by-construction — never silently skip them."""


@dataclass
class RailParam:
    """One rail of a wall's repeating cell (cross-section rectangle).
    width/gap are measured ALONG the wall (x for bottom/top, y for
    left/right); rail depth into the channel is the wall thickness."""
    width_mm: float           # >= MIN_FEATURE_MM
    gap_after_mm: float       # gap to the next rail, >= MIN_FEATURE_MM
    rf_signed_v: float        # sign = phase group (+ -> 0 deg, - -> 180)
    dc_v: float = 0.0


@dataclass
class WallParam:
    """One wall's pattern: a cell of k rails tiled n times, centered on
    the wall's channel-facing span. active=False leaves the wall bare
    (a declared absence, e.g. an open/dielectric side)."""
    cell: List[RailParam] = field(default_factory=list)
    n_tiles: int = 1
    active: bool = True
    force_last_rf_off: bool = True
    allow_wide_gaps: bool = False
    absorb_edges: bool = True
    # The "gap == 125 um isolation cut" rule is for
    # the board/wall-less case (exposed dielectric). Box WALLS may use
    # larger gaps, so they set allow_wide_gaps=True to reopen the gap as
    # a searchable position. The lower bound (>= isolation) still holds.
    # absorb_edges=False keeps the OUTERMOST rails at their exact size
    # (no margin absorption), leaving a real inset gap at the wall ends —
    # so box-SLIM wall RF stays 2 mm and does not touch the corners.
    # The last-rail-RF-off rule is a TRAVELLING-WAVE
    # convention (>=1 DC-free tap per cell). It is WRONG for a symmetric
    # confinement cell, where the last rail is a mirror partner and
    # grounding it breaks the left-right RF symmetry. Default True keeps
    # existing (TW) behavior; MirrorPairSpace sets False.


@dataclass
class CellBoardParam:
    """Full declared parameter set for one candidate cross-section."""
    channel_w_mm: float           # interior width  (x)
    channel_h_mm: float           # interior height (y)
    bottom: WallParam = field(default_factory=WallParam)
    top: Optional[WallParam] = None      # ignored when mirror_tb
    left: WallParam = field(default_factory=lambda: WallParam(active=False))
    right: Optional[WallParam] = None    # ignored when mirror_lr
    mirror_tb: bool = True        # declared search-space restriction
    mirror_lr: bool = True
    frequency_hz: float = 1.0e6   # shared RF frequency, both phase groups
    rail_thick_mm: float = 0.2    # rail (copper) thickness, all walls
    rf_min_v: float = RF_MIN_V
    envelope_mm: float = ENVELOPE_MM
    min_feature_mm: float = MIN_FEATURE_MM


def uniform_dc_wall(span_mm: float, dc_v: float) -> WallParam:
    """The guard-as-outcome helper: a k=1 pattern whose single RF-free
    rail spans the wall. Exists for readability; it is NOT a special
    code path in the emitter."""
    return WallParam(cell=[RailParam(span_mm, MIN_FEATURE_MM, 0.0, dc_v)],
                     n_tiles=1)


def _wall_tiled(wall: WallParam, span_mm: float, pattern_mm: float):
    """The tiled rail list [(start_along, width, rf_signed, dc, forced)]
    in wall-local coordinates, pattern centered on the span."""
    out = []
    margin = 0.5 * (span_mm - pattern_mm)
    along = margin
    for tile in range(wall.n_tiles):
        for j, r in enumerate(wall.cell):
            forced_off = (wall.force_last_rf_off
                          and j == len(wall.cell) - 1)
            out.append([along, r.width_mm, r.rf_signed_v, r.dc_v,
                        forced_off])
            along += r.width_mm + r.gap_after_mm
    # EDGE ABSORPTION (no-exposed-dielectric rule):
    # the tiling remainder at the wall ends would be bare board, so the
    # OUTERMOST rails absorb it — first rail extends to the wall start,
    # last to the wall end. Declared, deterministic, symmetric (margins
    # equal), so mirror proofs and the census are unaffected.
    if out and margin > 1e-9 and wall.absorb_edges:
        out[0][0] = 0.0
        out[0][1] += margin
        out[-1][1] += margin
    return [tuple(r) for r in out]


def _wall_mirror_symmetric(wall: WallParam, span_mm: float,
                           pattern_mm: float, rf_min_v: float,
                           tol: float = 1e-9) -> bool:
    """True iff the wall's tiled pattern maps to itself under mirror
    about the span midline WITH matching drive (same effective RF
    amplitude+phase and DC on mirror partners). Mirror does not flip RF
    phase, so signed amplitudes must MATCH, not negate. Used to decide
    whether a y-mirror may be DECLARED — proving per-problem at the
    declaration site, not assuming."""
    if not wall.active:
        return True
    rails = _wall_tiled(wall, span_mm, pattern_mm)

    def eff(rf_signed, forced):
        if forced or abs(rf_signed) < rf_min_v:
            return 0.0
        return rf_signed

    fwd = [(s, w, eff(a, f), dc) for s, w, a, dc, f in rails]
    rev = [(span_mm - (s + w), w, eff(a, f), dc)
           for s, w, a, dc, f in reversed(rails)]
    for (s1, w1, a1, d1), (s2, w2, a2, d2) in zip(fwd, rev):
        if (abs(s1 - s2) > tol or abs(w1 - w2) > tol
                or abs(a1 - a2) > tol or abs(d1 - d2) > tol):
            return False
    return True


def rail_is_rf(wall: WallParam, idx_in_cell: int, rf_min_v: float) -> bool:
    """Declared RF-on rule, PER WALL PATTERN: last rail of the cell is
    FORCED off (>=1 RF-free, TW-capable rail per cell); others are on
    iff |a| >= rf_min_v."""
    if wall.force_last_rf_off and idx_in_cell == len(wall.cell) - 1:
        return False
    return abs(wall.cell[idx_in_cell].rf_signed_v) >= rf_min_v


def _resolved_walls(p: CellBoardParam) -> dict:
    """Apply the mirror flags; refuse contradictory input rather than
    silently prefer one wall."""
    if p.mirror_tb and p.top is not None:
        raise CellBoardInfeasible(
            "mirror_tb=True but an independent top pattern was given — "
            "drop one (silent preference would hide a wrong geometry)")
    if p.mirror_lr and p.right is not None:
        raise CellBoardInfeasible(
            "mirror_lr=True but an independent right pattern was given")
    top = p.bottom if p.mirror_tb else (p.top or WallParam(active=False))
    right = p.left if p.mirror_lr else (p.right or WallParam(active=False))
    return {"bottom": p.bottom, "top": top, "left": p.left, "right": right}


def _validate_wall(name: str, wall: WallParam, span_mm: float,
                   p: CellBoardParam) -> float:
    """Check one wall's declared constraints; return its pattern length."""
    if not wall.active:
        return 0.0
    if not wall.cell:
        raise CellBoardInfeasible(name + " wall active but cell empty")
    if wall.n_tiles < 1:
        raise CellBoardInfeasible(
            name + " wall n_tiles must be >= 1, got " + repr(wall.n_tiles))
    for i, r in enumerate(wall.cell):
        if r.width_mm < RAIL_MIN_MM:
            raise CellBoardInfeasible(
                "{0} wall rail {1} width {2:.4f} mm < electrode minimum "
                "{3} mm".format(name, i, r.width_mm, RAIL_MIN_MM))
        if r.gap_after_mm < p.min_feature_mm - 1e-9:
            raise CellBoardInfeasible(
                "{0} wall rail {1} gap {2:.4f} mm < isolation gap {3} mm"
                .format(name, i, r.gap_after_mm, p.min_feature_mm))
        if (not wall.allow_wide_gaps
                and r.gap_after_mm > p.min_feature_mm + 1e-9):
            raise CellBoardInfeasible(
                "{0} wall rail {1} gap {2:.4f} mm > isolation gap {3} mm"
                " — exposed dielectric charges up (PI 2026-07-28); "
                "gaps are exactly the isolation cut".format(
                    name, i, r.gap_after_mm, p.min_feature_mm))
        if abs(r.rf_signed_v) > RF_MAX_V:
            raise CellBoardInfeasible(
                "{0} wall rail {1} |RF| {2:.1f} V > {3} V ceiling"
                .format(name, i, abs(r.rf_signed_v), RF_MAX_V))
    cell_len = sum(r.width_mm + r.gap_after_mm for r in wall.cell)
    pattern = wall.n_tiles * cell_len - wall.cell[-1].gap_after_mm
    if pattern > span_mm + 1e-9:
        raise CellBoardInfeasible(
            "{0} wall tiled pattern {1:.3f} mm exceeds its span {2:.3f} mm"
            .format(name, pattern, span_mm))
    return pattern


def build_spec(p: CellBoardParam, template):
    """Emit a SimSpec for the candidate on a deep copy of `template`.

    Frame: channel interior spans x in [t, t+W], y in [t, t+H] with
    t = rail_thick_mm. Each wall's rails live in its own 0..t frame and
    stay within the channel-facing span (corners are bare — declared).
    Electrode naming: <wall>_R<i> aggregates rail i of that wall's cell
    across its tiles AND across a mirrored partner wall (one electrode,
    both walls — the mirrored copy is the same declared conductor).
    Each RF-on rail role gets its own RF group <wall>_R<i>_RF (phase 0
    or 180, own amplitude, shared frequency).
    """
    from ion_gym.io.sim_spec import ShapeSpec

    walls = _resolved_walls(p)
    t = p.rail_thick_mm
    W, H = p.channel_w_mm, p.channel_h_mm
    outer_w, outer_h = W + 2.0 * t, H + 2.0 * t
    if outer_w > p.envelope_mm or outer_h > p.envelope_mm:
        raise CellBoardInfeasible(
            "outer envelope {0:.2f} x {1:.2f} mm exceeds {2} mm box"
            .format(outer_w, outer_h, p.envelope_mm))
    spans = {"bottom": W, "top": W, "left": H, "right": H}
    patterns = {n: _validate_wall(n, w, spans[n], p)
                for n, w in walls.items()}

    spec = copy.deepcopy(template)
    g = spec.geometry
    g.width_mm, g.height_mm = outer_w, outer_h

    def rect(x, y, w, h):
        return ShapeSpec.from_dict({"type": "rect", "x_mm": x, "y_mm": y,
                                    "width_mm": w, "height_mm": h})

    def rail_shape(wall_name, along0, w_mm):
        """One rail rectangle in absolute frame coordinates (width
        passed explicitly so edge absorption can extend it)."""
        if wall_name == "bottom":
            return rect(along0, 0.0, w_mm, t)
        if wall_name == "top":
            return rect(along0, t + H, w_mm, t)
        if wall_name == "left":
            return rect(0.0, along0, t, w_mm)
        return rect(t + W, along0, t, w_mm)           # right

    # group mirrored partners under the source wall's electrodes
    emit_as = {"bottom": ["bottom"], "left": ["left"],
               "top": [] if p.mirror_tb else ["top"],
               "right": [] if p.mirror_lr else ["right"]}
    if p.mirror_tb:
        emit_as["bottom"].append("top")
    if p.mirror_lr:
        emit_as["left"].append("right")

    electrodes, rf_groups = [], []
    el_proto, rf_proto = g.electrodes[0], g.rf_groups[0]
    for src, targets in emit_as.items():
        wall = walls[src]
        if not wall.active or not targets:
            continue
        margin = 0.5 * (spans[src] - patterns[src])
        offset0 = t + margin
        # EDGE ABSORPTION (no-exposed-dielectric):
        # the tiling remainder would be bare board, so the outermost
        # rails extend to the wall ends. Declared, symmetric — mirror
        # proofs and the census are unaffected.
        for i, r in enumerate(wall.cell):
            shapes = []
            for tgt in targets:
                along = offset0
                for tile in range(wall.n_tiles):
                    for j, rj in enumerate(wall.cell):
                        if j == i:
                            a0, wj = along, rj.width_mm
                            if margin > 1e-9 and wall.absorb_edges:
                                if tile == 0 and j == 0:
                                    a0 -= margin
                                    wj += margin
                                if (tile == wall.n_tiles - 1
                                        and j == len(wall.cell) - 1):
                                    wj += margin
                            shapes.append(rail_shape(tgt, a0, wj))
                        along += rj.width_mm + rj.gap_after_mm
            el = copy.deepcopy(el_proto)
            el.name = "{0}_R{1}".format(src, i)
            el.shapes = shapes
            el.is_grid = False
            el.dc = float(r.dc_v)
            if rail_is_rf(wall, i, p.rf_min_v):
                gname = el.name + "_RF"
                el.rf_groups = [gname]
                grp = copy.deepcopy(rf_proto)
                grp.name = gname
                grp.frequency_hz = float(p.frequency_hz)
                grp.amplitude_v = float(abs(r.rf_signed_v))
                grp.phase_deg = 0.0 if r.rf_signed_v >= 0.0 else 180.0
                grp.duty = 0.5
                grp.offset_v = 0.0
                grp.waveform = "sin"
                rf_groups.append(grp)
            else:
                el.rf_groups = []
            electrodes.append(el)

    if not electrodes:
        raise CellBoardInfeasible("no active wall emitted any electrode")
    g.electrodes = electrodes
    g.rf_groups = rf_groups
    g.dc_groups = []
    # BUILD CONVENTION: ions are born at the channel
    # CENTER — the (0,0) of the centered authoring convention. The spec/
    # solver frame is corner-based (planar lattice authority), so the
    # center sits at (t+W/2, t+H/2); the declared plane below pins it to
    # a lattice node. A literal -H/+H (negative-coordinate) spec frame
    # is a core change to the lattice authority, deliberately not done here.
    spec.source.x0_mm = t + 0.5 * W
    spec.source.y0_mm = t + 0.5 * H
    # Declare the y mirror plane ONLY when construction guarantees it:
    # mirror_tb gives top==bottom (same conductor); every active side
    # wall must additionally be palindromic about mid-height with
    # matching drive. The fold authority re-verifies discretely and
    # halves the solve when it holds.
    sides_ok = all(_wall_mirror_symmetric(walls[n], spans[n], patterns[n],
                                          p.rf_min_v)
                   for n in ("left", "right"))
    if p.mirror_tb and sides_ok:
        g.symmetry.planes["y"] = "mirror"
        g.symmetry.plane_mm["y"] = t + 0.5 * H
    ks = {n: len(w.cell) for n, w in walls.items() if w.active}
    spec.name = ("cell_board {0} W={1:g} H={2:g} f={3:g}kHz"
                 .format(" ".join("{0}:k{1}".format(n, k)
                                  for n, k in sorted(ks.items())),
                         W, H, p.frequency_hz / 1e3))
    spec.notes = ("Generated by physics.cell_board.build_spec; parameters "
                  "are the declaration of record. Last rail of every wall "
                  "cell RF-forced-off (TW-capable). mirror_tb={0} "
                  "mirror_lr={1} (declared Stage-B1 space)."
                  .format(p.mirror_tb, p.mirror_lr))
    spec.builder = "cell_board"
    return spec


def incumbent_analog() -> CellBoardParam:
    """The tetramer-like anchor: RF+ / TW / RF- / TW alternation (k=4,
    rail 1 off via the dead zone, rail 3 forced off), mirrored sandwich,
    side walls as uniform 4 V DC (the k=1 guard-as-outcome). An ANALOG
    (12 rails vs the tetramer's 11) used by smokes/gates as a known
    anchor, never as a bound."""
    return CellBoardParam(
        channel_w_mm=12.0, channel_h_mm=2.75,
        bottom=WallParam(cell=[RailParam(0.406, 0.127, +100.0, 0.0),
                               RailParam(0.406, 0.127, 0.0, 0.0),
                               RailParam(0.406, 0.127, -100.0, 0.0),
                               RailParam(0.406, 0.127, 0.0, 0.0)],
                         n_tiles=3),
        left=uniform_dc_wall(2.75, 4.0),
        frequency_hz=0.8e6)


def five_wire_analog(*, channel_w_mm: float = 12.0,
                     channel_h_mm: float = 12.0,
                     w_dc_mm: float = 2.0, w_rf_mm: float = 1.0,
                     w_gnd_mm: float = 0.4, gap_mm: float = MIN_FEATURE_MM,
                     rf_v: float = 150.0, dc_v: float = 2.0,
                     frequency_hz: float = 1.0e6,
                     top_plate_dc: Optional[float] = None
                     ) -> CellBoardParam:
    """Canonical surface-trap FIVE-WIRE arrangement (DC | RF | GND |
    RF | DC) captured as a named anchor of our space — the pattern of
    Hong 2016 / Abbasov 2023 / Gerasin 2024, scaled from their um-scale
    UHV chips to our mm-scale, 125-um-fab, collisional SLIM context
    (regime differs; the ARRANGEMENT is what is captured). Both RF
    rails share ONE phase (same sign): the null forms against the
    grounded center/outer metal, exactly the same-phase-pair-vs-ground
    topology the pairing construct admits. Single-wall: top ABSENT
    (mirror_tb=False), tall channel above the board for the ion-height
    dynamic. The asymmetric-RF-width variant (Gerasin's principal-axis
    tilt) is this preset with unequal w_rf on the two RF rails —
    construct by editing the returned param's cell widths."""
    return CellBoardParam(
        channel_w_mm=channel_w_mm, channel_h_mm=channel_h_mm,
        bottom=WallParam(cell=[
            RailParam(w_dc_mm, gap_mm, 0.0, dc_v),
            RailParam(w_rf_mm, gap_mm, rf_v, 0.0),
            RailParam(w_gnd_mm, gap_mm, 0.0, 0.0),   # grounded center
            RailParam(w_rf_mm, gap_mm, rf_v, 0.0),   # SAME phase
            RailParam(w_dc_mm, gap_mm, 0.0, dc_v),   # forced-off last
        ], n_tiles=1),
        top=(uniform_dc_wall(channel_w_mm, top_plate_dc)
             if top_plate_dc is not None else WallParam(active=False)),
        mirror_tb=False,
        left=WallParam(active=False), mirror_lr=True,
        frequency_hz=frequency_hz)


def mirrored_five_wire_analog(*, channel_w_mm: float = 12.0,
                              gap_h_mm: float = 4.0,
                              rf_v: float = 150.0, dc_v: float = 2.0,
                              frequency_hz: float = 1.0e6
                              ) -> CellBoardParam:
    """MIRRORED surface-trap anchor: the five-wire
    cell on BOTH boards, facing each other across gap_h_mm — i.e.
    sandwich mode with the surface-trap cell. Screens of this class are
    simply wall_mode='sandwich', K_BOTTOM=5 (tag sandwich_k5s1_rf2);
    this preset is the named anchor of that space."""
    fw = five_wire_analog(channel_w_mm=channel_w_mm,
                          channel_h_mm=gap_h_mm, rf_v=rf_v, dc_v=dc_v,
                          frequency_hz=frequency_hz)
    return CellBoardParam(
        channel_w_mm=channel_w_mm, channel_h_mm=gap_h_mm,
        bottom=fw.bottom, mirror_tb=True,
        left=WallParam(active=False), mirror_lr=True,
        frequency_hz=frequency_hz)


def asym_five_wire_analog(*, channel_w_mm: float = 12.0,
                          gap_h_mm: float = 4.0,
                          rf_v: float = 150.0,
                          frequency_hz: float = 1.0e6
                          ) -> CellBoardParam:
    """ASYMMETRIC surface-trap anchor: five-wire bottom, and a top cell
    with UNEQUAL RF rail widths (the Gerasin-style asymmetry that tilts
    the principal axes) — an interior point of wall_mode='asym'."""
    fw = five_wire_analog(channel_w_mm=channel_w_mm,
                          channel_h_mm=gap_h_mm, rf_v=rf_v,
                          frequency_hz=frequency_hz)
    top = WallParam(cell=[
        RailParam(2.0, MIN_FEATURE_MM, 0.0, 2.0),
        RailParam(0.5, MIN_FEATURE_MM, rf_v, 0.0),      # thin RF
        RailParam(0.4, MIN_FEATURE_MM, 0.0, 0.0),
        RailParam(1.5, MIN_FEATURE_MM, rf_v, 0.0),      # wide RF (asymmetry)
        RailParam(2.0, MIN_FEATURE_MM, 0.0, 2.0),
    ], n_tiles=1)
    return CellBoardParam(
        channel_w_mm=channel_w_mm, channel_h_mm=gap_h_mm,
        bottom=fw.bottom, top=top, mirror_tb=False,
        left=WallParam(active=False), mirror_lr=True,
        frequency_hz=frequency_hz)


class FiveWireWidthSpace:
    """Focused study space: the five-wire TOPOLOGY is
    FIXED — DC | RF(ph0) | GND | RF(ph180) | DC, symmetric RF, wall-less
    (guards do the transverse confinement) — and only the physically
    interesting knobs vary: the 5 rail WIDTHS, the guard DC voltage, the
    shared RF amplitude, and the RF frequency. ~9 dims vs the open k=5
    search's ~19, so the map is dense and the width dependence is
    directly readable (this is the 'try different electrode widths for a
    5-wire system' question). channel width is derived from the widths +
    fixed 125um gaps; channel height is a knob.

    This is a NAMED, declared space — not a reparametrization of
    CellSpace — because its axes ARE the design story (widths), and it
    plugs into the same screen/evaluate/bank machinery via to_param.
    """
    names_ = ["w_dc_guard_mm", "w_rf_mm", "w_gnd_mm", "channel_h_mm",
              "rf_amp_v", "frequency_hz", "guard_dc_v", "gnd_dc_v"]

    def __init__(self, *, gap_mm=MIN_FEATURE_MM, rf_ph=(0.0, 180.0),
                 min_channel_w_mm=8.0, max_channel_w_mm=12.0):
        self.gap_mm = gap_mm
        self.rf_ph = rf_ph
        # x-axis width BAND: 4 mm winners were too
        # compressed, so the channel width is constrained to a window,
        # both bounds declared refusals (not silent clamps).
        self.min_channel_w_mm = min_channel_w_mm
        self.max_channel_w_mm = max_channel_w_mm

    def names(self):
        return list(self.names_)

    def bounds(self):
        import numpy as np
        lo = [1.0, 1.0, 0.5, 2.0, 50.0, 0.5e6, -30.0, -10.0]
        hi = [3.5, 2.8, 2.0, 12.0, 300.0, 2.5e6, 30.0, 10.0]
        return np.array(lo, float), np.array(hi, float)

    def to_param(self, x):
        (w_g, w_rf, w_gnd, ch_h, amp, freq, gdc, gnddc) = [float(v)
                                                           for v in x]
        cell = [
            RailParam(w_g, self.gap_mm, 0.0, gdc),                 # DC guard
            RailParam(w_rf, self.gap_mm, amp, 0.0),                # RF ph0
            RailParam(w_gnd, self.gap_mm, 0.0, gnddc),             # center
            RailParam(w_rf, self.gap_mm, -amp, 0.0),               # RF ph180
            RailParam(w_g, self.gap_mm, 0.0, gdc),                 # DC guard
        ]
        ch_w = sum(r.width_mm for r in cell) + 4 * self.gap_mm
        if ch_w > self.max_channel_w_mm + 1e-9:
            raise CellBoardInfeasible(
                "channel width {0:.2f} mm > {1:.2f} mm x ceiling"
                .format(ch_w, self.max_channel_w_mm))
        if ch_w < self.min_channel_w_mm - 1e-9:
            raise CellBoardInfeasible(
                "channel width {0:.2f} mm < {1:.2f} mm x floor "
                "(too compressed)".format(ch_w, self.min_channel_w_mm))
        return CellBoardParam(
            channel_w_mm=ch_w, channel_h_mm=ch_h,
            bottom=WallParam(cell=cell, n_tiles=1), mirror_tb=True,
            left=WallParam(active=False), mirror_lr=True,
            frequency_hz=freq)

    def tag(self):
        return "fivewire_width_wl"


class MirrorPairSpace:
    """Symmetric pair-space: a centre DC rail plus N
    mirror-PAIRS radiating outward. Geometric mirror is a HARD CONSTRAINT
    — only the right half is searched; the left half is its reflection —
    so every candidate is fabrication-symmetric by construction (a 1 mm
    rail on the left has a 1 mm rail on the right).

    Each pair carries a PARITY-TAGGED drive, which is the 'pseudo-
    symmetry': the two mirror rails are geometrically identical but their
    drive is either
      * DC  (both rails the same DC voltage),
      * RF even  (both rails same-phase RF), or
      * RF odd   (rails at OPPOSITE phase 0/180 — the trap mechanism;
                  phi(-x) = -phi(x), the ONLY way an RF pair confines).
    So the search picks, per pair: width, gap-from-previous, and a drive
    = {DC V | RF(amp, even|odd)}. The five-wire is one point in this
    space (centre DC + 1 odd-RF pair + 1 DC pair). 'RF rails to the
    guards' is the optimiser choosing RF on the OUTER pair.

    Centre rail is DC-only for now (an on-axis RF rail has no mirror
    partner to be anti-symmetric with — it would sit at the RF null).

    Encoding per pair p (right half, inner->outer): width_p, gap_p,
    drive_kind_p in {0:DC, 1:RF-even, 2:RF-odd}, level_p (DC volts if
    DC, RF amplitude if RF). Plus global: channel_h, frequency,
    centre width, centre DC. n_pairs is FIXED at construction (run 1, 2,
    3 as separate campaigns) so the vector length is stable for CMA.
    """
    def __init__(self, n_pairs=2, *, gap_mm=MIN_FEATURE_MM,
                 min_channel_w_mm=8.0, max_channel_w_mm=12.0,
                 rf_ph=(0.0, 180.0)):
        self.n_pairs = int(n_pairs)
        self.gap_mm = gap_mm
        self.min_channel_w_mm = min_channel_w_mm
        self.max_channel_w_mm = max_channel_w_mm
        self.rf_ph = rf_ph

    def names(self):
        n = ["ch_h_mm", "frequency_hz", "w_center_mm", "center_dc_v"]
        for p in range(self.n_pairs):
            n += ["w_p{0}_mm".format(p), "kind_p{0}".format(p),
                  "level_p{0}".format(p)]
        return n

    def bounds(self):
        import numpy as np
        # gaps are NOT searchable — they are the fixed 125 um isolation cut
        # (any wider gap is exposed dielectric that charges up).
        # Per pair: width, kind (0 DC / 1 RF-even / 2 RF-odd), level.
        lo = [2.0, 0.5e6, 0.4, -10.0]
        hi = [12.0, 2.5e6, 2.5, 10.0]
        for _ in range(self.n_pairs):
            #     width  kind(floor->0,1,2)  level
            lo += [0.4, 0.0,   -30.0]
            hi += [3.5, 2.999, 300.0]
        return np.array(lo, float), np.array(hi, float)

    def to_param(self, x):
        x = [float(v) for v in x]
        ch_h, freq, w_c, c_dc = x[:4]
        rest = x[4:]
        g = self.gap_mm                     # fixed 125 um isolation cut
        pairs = []
        n_rf_pairs = 0
        rf_ordinal = 0                      # counts RF pairs, inner->outer
        for p in range(self.n_pairs):
            w, kind_f, level = rest[3*p:3*p+3]
            kind = int(min(2, max(0, int(kind_f))))   # 0 DC,1 RF even,2 RF odd
            if kind in (1, 2):
                n_rf_pairs += 1
            if kind == 0:
                rail_r = RailParam(w, g, 0.0, level)            # DC
                rail_l = RailParam(w, g, 0.0, level)            # mirror same V
            else:
                amp = max(RF_MIN_V, level)
                if kind == 1:                                   # RF even
                    rail_r = RailParam(w, g, amp, 0.0)
                    rail_l = RailParam(w, g, amp, 0.0)          # same phase
                else:                                           # RF odd
                    # ALTERNATING MULTIPOLE: flip the pair's phase sense on
                    # every other RF pair going outward, so the right half
                    # reads ph0, ph180, ph0, ... Each pair still opposite
                    # (RF+ left <-> RF- right); only the pair's orientation
                    # alternates. rf_ordinal 0 -> right=+ (ph0), 1 -> right=-
                    s = 1.0 if (rf_ordinal % 2 == 0) else -1.0
                    rail_r = RailParam(w, g,  s * amp, 0.0)
                    rail_l = RailParam(w, g, -s * amp, 0.0)
                rf_ordinal += 1
            pairs.append((rail_l, rail_r))
        # assemble symmetric cell: [outer..inner LEFT] + centre + [inner..outer RIGHT]
        left_half = [pl for (pl, pr) in reversed(pairs)]
        right_half = [pr for (pl, pr) in pairs]
        centre = RailParam(w_c, g, 0.0, c_dc)
        cell = left_half + [centre] + right_half
        ch_w = sum(r.width_mm for r in cell) + (len(cell) - 1) * self.gap_mm
        if ch_w > self.max_channel_w_mm + 1e-9:
            raise CellBoardInfeasible(
                "channel width {0:.2f} > {1:.2f} mm ceiling"
                .format(ch_w, self.max_channel_w_mm))
        if ch_w < self.min_channel_w_mm - 1e-9:
            raise CellBoardInfeasible(
                "channel width {0:.2f} < {1:.2f} mm floor"
                .format(ch_w, self.min_channel_w_mm))
        if n_rf_pairs < 1:
            raise CellBoardInfeasible(
                "no RF pair — an all-DC board has no dynamic trap "
                "(PI 2026-07-29: require >=1 RF pair)")
        # geometric symmetry is enforced HERE by construction (cell is
        # built mirror-symmetric); mirror_lr stays True for the tb fold.
        return CellBoardParam(
            channel_w_mm=ch_w, channel_h_mm=ch_h,
            bottom=WallParam(cell=cell, n_tiles=1,
                             force_last_rf_off=False),   # keep RF symmetric
            mirror_tb=True,
            left=WallParam(active=False), mirror_lr=True,
            frequency_hz=freq)

    def tag(self):
        return "mirrorpair_n{0}".format(self.n_pairs)


class MirrorBoxSpace(MirrorPairSpace):
    """MirrorPairSpace + searchable LEFT/RIGHT walls (the 'box'
    option). The horizontal top/bottom rails are inherited
    (symmetric pairs, alternating multipole, >=1 RF pair). Added: N_wall
    vertical wall-rail PAIRS on the left wall, MIRRORED to the right by
    mirror_lr — so the walls are symmetric by construction (a rail at
    height h on the left has one at h on the right).

    UNRESTRAINED by design: on the walls the rail HEIGHT (extent along y)
    and the GAP between wall rails are BOTH search dimensions (the fixed
    125 um isolation cut is a wall-LESS/board rule; the box walls may have
    larger gaps). Wall rails can be DC or RF; the wall force_last_rf_off
    is off so wall RF stays symmetric too.

    Extra dims (appended after the horizontal-rail block): per wall rail
    p: h_wall_p (height mm), gap_wall_p (gap to next, mm, >=125um but
    UNBOUNDED above), kind_wall_p, level_wall_p.
    """
    def __init__(self, n_pairs=2, n_wall=2, *, gap_mm=MIN_FEATURE_MM,
                 min_channel_w_mm=8.0, max_channel_w_mm=12.0,
                 max_wall_gap_mm=4.0, rf_ph=(0.0, 180.0)):
        super().__init__(n_pairs=n_pairs, gap_mm=gap_mm,
                         min_channel_w_mm=min_channel_w_mm,
                         max_channel_w_mm=max_channel_w_mm, rf_ph=rf_ph)
        self.n_wall = int(n_wall)
        self.max_wall_gap_mm = max_wall_gap_mm

    def names(self):
        n = super().names()
        for p in range(self.n_wall):
            n += ["h_wall{0}_mm".format(p), "gap_wall{0}_mm".format(p),
                  "kind_wall{0}".format(p), "level_wall{0}".format(p)]
        return n

    def bounds(self):
        import numpy as np
        lo0, hi0 = super().bounds()
        lo, hi = list(lo0), list(hi0)
        for _ in range(self.n_wall):
            #     height gap(>=125um, UNbounded-ish)  kind    level
            lo += [0.4, self.gap_mm,                   0.0,   -30.0]
            hi += [3.0, self.max_wall_gap_mm,          2.999, 300.0]
        return np.array(lo, float), np.array(hi, float)

    def to_param(self, x):
        x = [float(v) for v in x]
        n_horiz = 4 + 3 * self.n_pairs          # base MirrorPairSpace vec
        base = super().to_param(x[:n_horiz])
        wall_rest = x[n_horiz:]
        # build the LEFT wall cell (mirror_lr copies to right). Gaps here
        # are searchable and may exceed 125 um.
        g_iso = self.gap_mm
        wall_cell = []
        n_wall_rf = 0
        for p in range(self.n_wall):
            h, gap, kind_f, level = wall_rest[4*p:4*p+4]
            gap = max(g_iso, gap)               # floor at the isolation cut
            kind = int(min(2, max(0, int(kind_f))))
            if kind == 0:
                rail = RailParam(h, gap, 0.0, level)             # DC
            else:
                amp = max(RF_MIN_V, level)
                n_wall_rf += 1
                # wall RF: single wall, so phase sign alternates by ordinal
                s = 1.0 if (n_wall_rf % 2 == 1) else -1.0
                rail = RailParam(h, gap, s * amp, 0.0)
            wall_cell.append(rail)
        base.left = WallParam(cell=wall_cell, n_tiles=1,
                              force_last_rf_off=False, active=True,
                              allow_wide_gaps=True)   # box walls: gap searchable
        base.mirror_lr = True                   # right = mirror of left
        return base

    def tag(self):
        return "mirrorbox_n{0}w{1}".format(self.n_pairs, self.n_wall)


class MirrorRingSpace:
    """Closed-loop alternating multipole: the RF phase
    alternates A(+)/B(-) CONTINUOUSLY around the whole perimeter — top,
    right wall, bottom, left wall — with a DC rail between each RF rail,
    instead of the four sides being phased independently. Geometry is
    symmetric (top=bottom, left=right in WIDTH/HEIGHT/position), but the
    PHASES are anti-symmetric across both mirrors (the alternation makes
    top/bottom and left/right phase-swapped), so all four walls are built
    EXPLICITLY (mirror_tb=mirror_lr=False) with phases assigned by
    clockwise perimeter position. Wall rail GAPS stay searchable.

    Topology, clockwise:
      top/bottom (each L->R): DC | RF | DCcentre | RF | DC   (5 rails)
      each wall (bottom->top): RF | DC | RF                  (3 rails)
    8 RF rails total, alternating +,-,+,-,... around the loop (closes:
    even count). Because the field is ODD under both reflections, NO
    mirror fold is declared (full solve) — correct, not an oversight.

    Searchable: channel w/h, frequency, RF amp, centre & guard & wall DC,
    top/bottom rail widths, wall rail heights, and the wall GAPS.
    """
    def __init__(self, *, gap_mm=MIN_FEATURE_MM, max_wall_gap_mm=4.0,
                 min_channel_w_mm=8.0, max_channel_w_mm=12.0):
        self.gap_mm = gap_mm
        self.max_wall_gap_mm = max_wall_gap_mm
        self.min_channel_w_mm = min_channel_w_mm
        self.max_channel_w_mm = max_channel_w_mm

    names_ = ["ch_h_mm", "ch_w_mm", "frequency_hz", "rf_amp_v",
              "center_dc_v", "guard_dc_v", "wall_dc_v",
              "w_guard_mm", "w_rf_mm", "w_center_mm",
              "h_wall_rf_mm", "h_wall_dc_mm",
              "gap_wall_lo_mm", "gap_wall_hi_mm"]

    def names(self):
        return list(self.names_)

    def bounds(self):
        import numpy as np
        #     ch_h ch_w  freq   amp  cdc  gdc  wdc  wg   wrf  wc   hrf  hdc  gl    gh
        lo = [6.0, 8.0, 0.5e6,  30., -10.,-30.,-30., 0.4, 0.4, 0.4, 0.4, 0.4, gap_lo(self), gap_lo(self)]
        hi = [12., 12., 2.5e6, 300.,  10., 30., 30., 3.0, 3.0, 2.0, 3.0, 3.0, self.max_wall_gap_mm, self.max_wall_gap_mm]
        return np.array(lo, float), np.array(hi, float)

    def to_param(self, x):
        x = [float(v) for v in x]
        (ch_h, ch_w, freq, amp, cdc, gdc, wdc,
         wg, wrf, wc, hrf, hdc, g_lo, g_hi) = x
        g = self.gap_mm
        A, B = amp, -amp        # phase 0 / phase 180
        # top L->R: DC(guard) | RF(B) | DC(centre) | RF(A) | DC(guard)
        top = [RailParam(wg, g, 0.0, gdc), RailParam(wrf, g, B, 0.0),
               RailParam(wc, g, 0.0, cdc), RailParam(wrf, g, A, 0.0),
               RailParam(wg, g, 0.0, gdc)]
        # bottom L->R: phase-SWAPPED (A<->B) to continue the alternation
        bottom = [RailParam(wg, g, 0.0, gdc), RailParam(wrf, g, A, 0.0),
                  RailParam(wc, g, 0.0, cdc), RailParam(wrf, g, B, 0.0),
                  RailParam(wg, g, 0.0, gdc)]
        # walls bottom->top, gaps SEARCHABLE (g_lo below RF, g_hi above DC)
        gl = max(g, g_lo); gh = max(g, g_hi)
        left = [RailParam(hrf, gl, B, 0.0), RailParam(hdc, gh, 0.0, wdc),
                RailParam(hrf, g, A, 0.0)]
        right = [RailParam(hrf, gl, A, 0.0), RailParam(hdc, gh, 0.0, wdc),
                 RailParam(hrf, g, B, 0.0)]
        for w in (top, bottom):
            cw = sum(r.width_mm for r in w) + (len(w)-1)*g
            if not (self.min_channel_w_mm-1e-9 <= cw <= self.max_channel_w_mm+1e-9):
                raise CellBoardInfeasible(
                    "channel width {0:.2f} out of [{1},{2}] mm".format(
                        cw, self.min_channel_w_mm, self.max_channel_w_mm))
        return CellBoardParam(
            channel_w_mm=ch_w, channel_h_mm=ch_h,
            bottom=WallParam(cell=bottom, n_tiles=1, force_last_rf_off=False),
            top=WallParam(cell=top, n_tiles=1, force_last_rf_off=False),
            left=WallParam(cell=left, n_tiles=1, force_last_rf_off=False,
                           allow_wide_gaps=True, active=True),
            right=WallParam(cell=right, n_tiles=1, force_last_rf_off=False,
                            allow_wide_gaps=True, active=True),
            mirror_tb=False, mirror_lr=False,      # phases differ; build all 4
            frequency_hz=freq)

    def tag(self):
        return "mirrorring"


def gap_lo(space):
    return space.gap_mm


def box_slim_params(*, side_mm=None, w_corner_dc=1.5, w_rf=2.0,
                    w_center_dc=0.5, h_wall_rf=None, h_wall_dc=0.5,
                    wall_edge_margin=0.5, tb_gap=MIN_FEATURE_MM,
                    rf_amp_v=261.0, frequency_hz=1.2e6,
                    corner_dc_v=3.47, center_dc_v=-7.07, wall_dc_v=-3.29):
    """Generic BOX-SLIM generator. A square (or w!=h)
    closed box whose perimeter is an alternating multipole:

      top/bottom (L->R): DCcorner | RF | DCcentre | RF | DCcorner
      each wall (edge->edge): RF | DCcentre | RF, with the RF rails
        placed `wall_edge_margin` from the top/bottom board edges.

    All electrode DIMENSIONS are named args so this is adjustable:
      w_corner_dc / w_rf / w_center_dc : top/bottom rail WIDTHS (x)
      h_wall_rf / h_wall_dc            : wall rail HEIGHTS (y)
      wall_edge_margin                 : RF inset from board edges
    Geometry is symmetric (up/down, left/right); the centre DC rails
    (the "purple" DC-) sit symmetrically on all four sides.

    SQUARE by default: if side_mm is None it is set to the top/bottom
    pattern width (so the horizontal rails fill the width exactly), and
    the wall gaps are then DERIVED so the wall RF lands `wall_edge_margin`
    from each edge. Returns a CellBoardParam (feed to build_spec).

    Returns the CellBoardParam AND the derived wall gap, so the caller
    can see/adjust it.
    """
    tb_gap = float(tb_gap)
    if h_wall_rf is None:
        h_wall_rf = w_rf        # wall RF HEIGHT = top/bottom RF WIDTH
    # top/bottom pattern width = 2 corners + 2 RF + centre + 4 gaps
    tb_width = 2*w_corner_dc + 2*w_rf + w_center_dc + 4*tb_gap
    side = float(side_mm) if side_mm is not None else tb_width
    if side < tb_width - 1e-9:
        raise CellBoardInfeasible(
            "side {0:.2f} mm < top/bottom pattern {1:.2f} mm".format(
                side, tb_width))
    # wall gap derived so RF sits wall_edge_margin from each edge:
    # side = 2*margin + 2*h_wall_rf + h_wall_dc + 2*gap
    wall_gap = (side - 2*wall_edge_margin - 2*h_wall_rf - h_wall_dc) / 2.0
    if wall_gap < MIN_FEATURE_MM - 1e-9:
        raise CellBoardInfeasible(
            "derived wall gap {0:.3f} mm < isolation {1} mm — raise "
            "side_mm or shrink wall rails/margin".format(
                wall_gap, MIN_FEATURE_MM))
    A, B = rf_amp_v, -rf_amp_v
    g = tb_gap
    # top L->R: DCcorner | RF(B) | DCcentre | RF(A) | DCcorner
    top = [RailParam(w_corner_dc, g, 0.0, corner_dc_v),
           RailParam(w_rf, g, B, 0.0),
           RailParam(w_center_dc, g, 0.0, center_dc_v),
           RailParam(w_rf, g, A, 0.0),
           RailParam(w_corner_dc, g, 0.0, corner_dc_v)]
    # bottom: phase-swapped to continue the perimeter alternation
    bottom = [RailParam(w_corner_dc, g, 0.0, corner_dc_v),
              RailParam(w_rf, g, A, 0.0),
              RailParam(w_center_dc, g, 0.0, center_dc_v),
              RailParam(w_rf, g, B, 0.0),
              RailParam(w_corner_dc, g, 0.0, corner_dc_v)]
    # walls bottom->top: RF | DCcentre | RF, gaps = derived wall_gap
    left = [RailParam(h_wall_rf, wall_gap, B, 0.0),
            RailParam(h_wall_dc, wall_gap, 0.0, wall_dc_v),
            RailParam(h_wall_rf, g, A, 0.0)]
    right = [RailParam(h_wall_rf, wall_gap, A, 0.0),
             RailParam(h_wall_dc, wall_gap, 0.0, wall_dc_v),
             RailParam(h_wall_rf, g, B, 0.0)]
    p = CellBoardParam(
        channel_w_mm=side, channel_h_mm=side,
        bottom=WallParam(cell=bottom, n_tiles=1, force_last_rf_off=False),
        top=WallParam(cell=top, n_tiles=1, force_last_rf_off=False),
        left=WallParam(cell=left, n_tiles=1, force_last_rf_off=False,
                       allow_wide_gaps=True, absorb_edges=False, active=True),
        right=WallParam(cell=right, n_tiles=1, force_last_rf_off=False,
                        allow_wide_gaps=True, absorb_edges=False, active=True),
        mirror_tb=False, mirror_lr=False, frequency_hz=frequency_hz)
    return p, wall_gap


class CarpetDensitySpace:
    """Temperature-first carpet exploration: vary the
    NUMBER and PITCH of alternating RF rails on the top/bottom boards,
    inside up to a 12x12 mm box, to find geometries that let ions sit
    COLD mid-channel. Control surfaces deliberately DEFERRED — walls are
    single grounded electrodes (placeholders for later control), so this
    space searches confinement quality only.

    Board layout (each of top/bottom): guard | N alternating RF | guard.
    - N rails (kind: even count keeps the perimeter alternation clean;
      floor of a continuous dim), rail width >= 0.5 mm (declared minimum),
      gaps fixed at the 125 um isolation cut (board rule).
    - Guards at both ends, searchable width + one DC level.
    - Channel width DERIVED from the board pattern (guards + rails +
      gaps); infeasible if outside [min_channel_w_mm, max_channel_w_mm].
    - Channel height searchable up to 12 mm.
    - Top board phase-swapped from bottom (perimeter alternation).

    Dims: [ch_h_mm, n_rf (floored, even), w_rf_mm, w_guard_mm,
           guard_dc_v, rf_amp_v, frequency_hz]
    """
    def __init__(self, *, gap_mm=MIN_FEATURE_MM, min_channel_w_mm=6.0,
                 max_channel_w_mm=12.0, max_channel_h_mm=12.0,
                 n_rf_lo=4, n_rf_hi=16, min_rail_w_mm=0.5):
        self.gap_mm = gap_mm
        self.min_channel_w_mm = min_channel_w_mm
        self.max_channel_w_mm = max_channel_w_mm
        self.max_channel_h_mm = max_channel_h_mm
        self.n_rf_lo, self.n_rf_hi = int(n_rf_lo), int(n_rf_hi)
        self.min_rail_w_mm = min_rail_w_mm

    def names(self):
        return ["ch_h_mm", "n_rf", "w_rf_mm", "w_guard_mm",
                "guard_dc_v", "rf_amp_v", "frequency_hz"]

    def bounds(self):
        import numpy as np
        lo = [4.0, float(self.n_rf_lo), self.min_rail_w_mm, 0.5,
              -30.0, 50.0, 0.5e6]
        hi = [self.max_channel_h_mm, self.n_rf_hi + 0.999, 2.5, 2.5,
              30.0, 300.0, 2.5e6]
        return np.array(lo, float), np.array(hi, float)

    def to_param(self, x):
        x = [float(v) for v in x]
        ch_h, n_f, w_rf, w_g, g_dc, amp, freq = x
        n = int(n_f)
        n -= n % 2                       # even count for clean alternation
        n = max(self.n_rf_lo - (self.n_rf_lo % 2), n)
        if w_rf < self.min_rail_w_mm - 1e-9:
            raise CellBoardInfeasible(
                "rail width {0:.3f} < {1} mm minimum".format(
                    w_rf, self.min_rail_w_mm))
        g = self.gap_mm
        A = max(RF_MIN_V, amp)
        def board(s0):
            cell = [RailParam(w_g, g, 0.0, g_dc)]
            for k in range(n):
                cell.append(RailParam(
                    w_rf, g, (s0 if k % 2 == 0 else -s0) * A, 0.0))
            cell.append(RailParam(w_g, g, 0.0, g_dc))
            return cell
        bot, top = board(-1.0), board(1.0)      # phase-swapped pair
        ch_w = (sum(r.width_mm for r in bot)
                + (len(bot) - 1) * g)
        if not (self.min_channel_w_mm - 1e-9 <= ch_w
                <= self.max_channel_w_mm + 1e-9):
            raise CellBoardInfeasible(
                "derived channel width {0:.2f} outside [{1}, {2}] mm "
                "(n={3}, w_rf={4:.2f}, w_guard={5:.2f})".format(
                    ch_w, self.min_channel_w_mm, self.max_channel_w_mm,
                    n, w_rf, w_g))
        # OPEN planar carpets: NO vertical walls (two facing boards with
        # end guards only). Inactive walls carry no metal, so the sides of
        # the domain are open field boundaries, not grounded conductors.
        return CellBoardParam(
            channel_w_mm=ch_w, channel_h_mm=ch_h,
            bottom=WallParam(cell=bot, n_tiles=1, force_last_rf_off=False),
            top=WallParam(cell=top, n_tiles=1, force_last_rf_off=False),
            left=WallParam(active=False), right=WallParam(active=False),
            mirror_tb=False, mirror_lr=False, frequency_hz=freq)

    def pitch_mm(self, x):
        """Rail pitch (width + gap) for a vector — the RF decay scale."""
        return float(x[2]) + self.gap_mm

    def tag(self):
        return "carpetdensity"


class PlanarCarpetSpace:
    """Open planar-carpet evaluation. Two facing carpet
    boards, END GUARDS, NO vertical walls. Searched variables, all
    independent:

      board WIDTH, surface GAP, RF rail WIDTH (>= 500 um MINIMUM), and
      GUARD WIDTH. The rail COUNT N is derived to fill the board width
      given the rail width and guards (density emerges from width, rail
      width, and guard). Rail width and guard width are separate dims
      (they need not match). 500 um is the rail-width FLOOR, not a
      cap.

    gap_surface_mm is the CLEAR face-to-face distance (ions cross it in y);
    channel height = gap + two 0.2 mm board thicknesses. Guard DC stays in
    the few-volt band the narrow-gap sweep showed holds. Sides OPEN.

    Dims: [board_w_mm, gap_surface_mm, w_rf_mm, w_guard_mm, guard_dc_v,
           rf_amp_v, frequency_hz]  (N derived to fill the width)
    """
    def __init__(self, *, gap_mm=MIN_FEATURE_MM, board_w_lo=6.0,
                 board_w_hi=13.0, surf_gap_lo=2.0, surf_gap_hi=10.0,
                 rail_w_lo=0.5, rail_w_hi=2.0, guard_w_lo=0.5, guard_w_hi=3.0,
                 guard_dc_abs_max=12.0, n_rf_min=4, n_rf_max=20,
                 board_thick_mm=0.2):
        self.gap_mm = gap_mm
        self.board_w_lo, self.board_w_hi = board_w_lo, board_w_hi
        self.surf_gap_lo, self.surf_gap_hi = surf_gap_lo, surf_gap_hi
        self.rail_w_lo, self.rail_w_hi = rail_w_lo, rail_w_hi   # 500um floor
        self.guard_w_lo, self.guard_w_hi = guard_w_lo, guard_w_hi
        self.guard_dc_abs_max = guard_dc_abs_max
        self.n_rf_min, self.n_rf_max = int(n_rf_min), int(n_rf_max)
        self.board_thick_mm = board_thick_mm

    def names(self):
        return ["board_w_mm", "gap_surface_mm", "w_rf_mm", "w_guard_mm",
                "guard_dc_v", "rf_amp_v", "frequency_hz"]

    def bounds(self):
        import numpy as np
        lo = [self.board_w_lo, self.surf_gap_lo, self.rail_w_lo,
              self.guard_w_lo, -self.guard_dc_abs_max, 50.0, 0.5e6]
        hi = [self.board_w_hi, self.surf_gap_hi, self.rail_w_hi,
              self.guard_w_hi, self.guard_dc_abs_max, 300.0, 2.5e6]
        return np.array(lo, float), np.array(hi, float)

    def _n_rails(self, board_w, w_rf, w_g):
        # rails span between the two guards (with a gap either side of the
        # rail block): span = board_w - 2*guard - 2*gap. Fit N rails of
        # width w_rf with 125um internal gaps: N*w_rf + (N-1)*g <= span.
        g = self.gap_mm
        span = board_w - 2.0 * w_g - 2.0 * g
        if span < w_rf - 1e-9:
            return 0
        n = int((span + g) // (w_rf + g))
        n -= n % 2                       # even for clean perimeter alternation
        return n

    def to_param(self, x):
        x = [float(v) for v in x]
        board_w, surf_gap, w_rf, w_g, g_dc, amp, freq = x
        if w_rf < self.rail_w_lo - 1e-9:
            raise CellBoardInfeasible(
                "rail width {0:.3f} < {1} mm floor".format(w_rf,
                                                           self.rail_w_lo))
        n = self._n_rails(board_w, w_rf, w_g)
        if n < self.n_rf_min:
            raise CellBoardInfeasible(
                "board {0:.2f}mm with {1:.2f}mm guards fits only {2} rails "
                "of {3:.2f}mm (< {4} min)".format(
                    board_w, w_g, n, w_rf, self.n_rf_min))
        n = min(n, self.n_rf_max)
        g = self.gap_mm; A = max(RF_MIN_V, amp)
        def board(s0):
            cell = [RailParam(w_g, g, 0.0, g_dc)]
            for k in range(n):
                cell.append(RailParam(w_rf, g, (s0 if k % 2 == 0 else -s0)*A, 0.0))
            cell.append(RailParam(w_g, g, 0.0, g_dc))
            return cell
        bot, top = board(-1.0), board(1.0)
        ch_w = sum(r.width_mm for r in bot) + (len(bot)-1)*g
        ch_h = surf_gap + 2.0 * self.board_thick_mm
        return CellBoardParam(
            channel_w_mm=ch_w, channel_h_mm=ch_h,
            bottom=WallParam(cell=bot, n_tiles=1, force_last_rf_off=False),
            top=WallParam(cell=top, n_tiles=1, force_last_rf_off=False),
            left=WallParam(active=False), right=WallParam(active=False),
            mirror_tb=False, mirror_lr=False, frequency_hz=freq)

    def n_rails_of(self, x):
        return self._n_rails(float(x[0]), float(x[2]), float(x[3]))

    def pitch_mm(self, x):
        return float(x[2]) + self.gap_mm

    def surface_gap_mm(self, x):
        return float(x[1])

    def tag(self):
        return "planarcarpet"


# ---------------------------------------------------------------------
# PARAMETRIC 2-D SLIM CROSS-SECTION (see the SLIM notebook):
# the tetramer confinement-slice family as an explicit parameter set —
# n RF rails alternating RF_A/RF_B on both boards, guards at the ends,
# declared board gap and thickness, optional x-mirror declaration for
# folded solves. Emits a STANDALONE native SimSpec (inline rects; no
# template, no reference geometry). Voltages are ASSIGNED AFTERWARD by
# the caller (groups are created here at 0 V so the geometry can be
# inspected before any drive exists).
def slim2d_cross_section(*, n_rf: int = 6,
                         rail_w_mm: float = 0.406,
                         rail_h_mm: float = 0.2,
                         pitch_mm: float = 1.067,
                         guard_w_mm: float = 3.15,
                         guard2_w_mm: float = None,
                         guard_gap_mm: float = 0.254,
                         center_dc: bool = False,
                         center_w_mm: float = 0.406,
                         dc_every_pairs: int = None,
                         dc_w_mm: float = 0.406,
                         gap_mm: float = 2.75,
                         board_t_mm: float = 0.2,
                         edge_gap_mm: float = 0.254,
                         mirror_x: bool = False,
                         mirror_y: bool = False,
                         h_mm: float = 0.05,
                         name: str = "parametric 2-D SLIM slice"):
    """Build the cross-section SimSpec.

    Layout along x: [guard][edge_gap][n_rf rails at pitch][edge_gap]
    [guard]; identical top and bottom boards separated by gap_mm (board
    inner faces), each board_t_mm thick — total height = gap + 2*t.
    Rails alternate dc-less RF groups RF_A / RF_B (0 V, 0 Hz until the
    caller assigns); guards are one DC electrode "Guard" (0 V).
    mirror_x declares an x-mirror at the domain midline (folded solve;
    only valid when the layout is x-symmetric, which this constructor
    guarantees). Returns the SimSpec (validate()d — refuses on error).
    """
    from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                     ShapeSpec, SourceSpec, CollisionSpec,
                                     IntegrationSpec, ViewSpec, RFGroupSpec,
                                     SymmetrySpec)
    if n_rf < 1:
        raise ValueError(f"n_rf must be >= 1, got {n_rf}")
    if rail_w_mm > pitch_mm:
        raise ValueError(f"rail_w_mm {rail_w_mm} exceeds pitch_mm "
                         f"{pitch_mm} — rails would overlap")
    if center_dc and n_rf % 2:
        raise ValueError(
            "center_dc needs an EVEN n_rf (half the rails each side of "
            f"the centre electrode); got n_rf={n_rf}")
    if dc_every_pairs:
        # interstitial DC rails at RF-pair boundaries (reference layout:
        # 8 rails -> 3 DC rails, one at every boundary;
        # the centre electrode is just the middle member of this family)
        if n_rf % 2:
            raise ValueError(
                f"dc_every_pairs needs an EVEN n_rf (complete A/B "
                f"pairs); got n_rf={n_rf}")
        if center_dc:
            raise ValueError(
                "center_dc and dc_every_pairs overlap (the middle "
                "interstitial IS the centre electrode) — use one.")
        if mirror_x:
            raise ValueError(
                "mirror_x with dc_every_pairs is NOT a symmetry: the "
                "fold reverses each A/B pair's phase order. Drop "
                "mirror_x (mirror_y remains the natural SLIM fold).")
        if int(dc_every_pairs) != 1:
            raise ValueError(
                f"dc_every_pairs={dc_every_pairs}: only 1 (a DC rail "
                f"between EVERY RF pair) is defined so far — say the "
                f"word and coarser periods get built.")
    if mirror_x:
        # an x-mirror maps rail k onto rail n-1-k; with A/B alternation
        # that is only the SAME phase group when n_rf is odd (or when a
        # centre electrode makes the run even-but-symmetric). Declaring
        # a false symmetry would fold different drives onto each other —
        # refuse rather than solve a wrong field.
        ok = (n_rf % 2 == 1) or center_dc
        if not ok:
            raise ValueError(
                "mirror_x with even n_rf and no centre electrode is NOT "
                "a symmetry: the fold maps RF_A rails onto RF_B rails. "
                "Use odd n_rf, add center_dc=True, or drop mirror_x.")

    # lateral layout: [G1][gap?G2][edge_gap][rails (+ centre)][mirrored]
    g2 = float(guard2_w_mm) if guard2_w_mm else 0.0
    end_w = guard_w_mm + ((guard_gap_mm + g2) if g2 else 0.0)
    n_dc_rails = (n_rf // 2 - 1) if dc_every_pairs else 0
    span = (n_rf - 1) * pitch_mm + rail_w_mm
    if center_dc:
        span += pitch_mm                      # one extra slot mid-run
    span += n_dc_rails * pitch_mm             # interstitial DC slots
    # LATTICE RULE: W and H are LATTICE quantities and must be an exact
    # integer number of cells at h_mm. They are derived from rail
    # pitches, guard widths and board thicknesses -- all arbitrary
    # physical floats -- so the raw sums almost never conform, and this
    # generator REFUSED ITS OWN OUTPUT at the validate() call below
    # (measured: W = 8.116 mm = 54.107 cells
    # at h = 0.15). The classic defect of a
    # framework generator sizing a domain from geometry without counting
    # in cells.
    #
    # COVER-UP, never trim: trimming would move a wall inside the guard
    # electrode flush against it. The extent is DERIVED from the count
    # through the lattice owner, not rounded here.
    #
    # SYMMETRY IS PRESERVED DELIBERATELY, which is why the width slop is
    # split rather than appended:
    #   * H: the boards sit at y = 0 and y = H - board_t, i.e. flush to
    #     BOTH walls, so growing H keeps them equidistant from the new
    #     midline and a declared y-mirror still folds. No interior shift
    #     is needed.
    #   * W: the guards are likewise flush (0 and W - guard_w, and G2 at
    #     x2 and W - x2 - g2), but the RAIL PATTERN starts at a fixed
    #     offset from the LEFT. Growing W alone would decentre it, the
    #     discrete masks would stop being mirror-equal, and a declared
    #     x-mirror would silently stop folding (reported by
    #     planar_fold_axes, but a silent loss of the fold all the same).
    #     Half the slop is therefore added to the rail start below.
    _W_raw = 2 * end_w + 2 * edge_gap_mm + span
    _H_raw = gap_mm + 2 * board_t_mm

    W = cover_extent_mm(_W_raw, h_mm, mirrored=bool(mirror_x))
    H = cover_extent_mm(_H_raw, h_mm, mirrored=bool(mirror_y))
    _x_shift = 0.5 * (W - _W_raw)      # keeps the rail pattern centred
    y_rows = (0.0, H - board_t_mm)          # bottom board, top board

    def rects(x0, w):
        return [ShapeSpec.from_dict(dict(type="rect", x_mm=x0, y_mm=y,
                                         width_mm=w, height_mm=board_t_mm))
                for y in y_rows]

    els = []
    # + _x_shift: half the width slop from the lattice snap above, so the rail
    # pattern stays centred between the flush guards and a declared
    # x-mirror still folds.
    x = end_w + edge_gap_mm + _x_shift
    slots = n_rf + (1 if center_dc else 0) + n_dc_rails
    mid = slots // 2
    k_rf = 0
    dc_shapes = []
    for slot in range(slots):
        if center_dc and slot == mid:
            # centre electrode occupies its slot, width centred in it
            cx = x + (rail_w_mm - center_w_mm) / 2.0
            els.append(ElectrodeSpec(name="DC_center", dc=0.0,
                                     shapes=rects(cx, center_w_mm)))
        elif dc_every_pairs and slot % 3 == 2:
            # every third slot is a DC rail: RF RF | DC | RF RF | DC ...
            cx = x + (rail_w_mm - dc_w_mm) / 2.0
            dc_shapes.extend(rects(cx, dc_w_mm))
        else:
            grp = "RF_A" if k_rf % 2 == 0 else "RF_B"
            els.append(ElectrodeSpec(name=f"rf{k_rf+1}", dc=0.0,
                                     rf_groups=[grp],
                                     shapes=rects(x, rail_w_mm)))
            k_rf += 1
        x += pitch_mm
    if dc_shapes:
        # ONE electrode, one bias knob for the whole interstitial family
        # (split into separately biased rails later if tuning needs it)
        els.append(ElectrodeSpec(name="DC_rails", dc=0.0,
                                 shapes=dc_shapes))
    g1 = ElectrodeSpec(name="G1", dc=0.0,
                       shapes=(rects(0.0, guard_w_mm)
                               + rects(W - guard_w_mm, guard_w_mm)))
    els.append(g1)
    if g2:
        # second guard set INSIDE the first, both ends, own electrode
        x2 = guard_w_mm + guard_gap_mm
        g2_el = ElectrodeSpec(name="G2", dc=0.0,
                              shapes=(rects(x2, g2)
                                      + rects(W - x2 - g2, g2)))
        els.append(g2_el)

    sym = SymmetrySpec(coords="xyz",
                       planes={"x": ("mirror" if mirror_x else "none"),
                               "y": ("mirror" if mirror_y else "none"),
                               "z": "none"})
    geo = GeometrySpec(width_mm=W, height_mm=H, mm_per_gu=h_mm,
                       electrodes=els, symmetry=sym,
                       rf_groups=[RFGroupSpec("RF_A", frequency_hz=0.0,
                                              amplitude_v=0.0, phase_deg=0.0),
                                  RFGroupSpec("RF_B", frequency_hz=0.0,
                                              amplitude_v=0.0,
                                              phase_deg=180.0)])
    spec = SimSpec(name=name, geometry=geo,
                   source=SourceSpec(n_ions=20, x0_mm=W / 2, y0_mm=H / 2,
                                     distribution="disc", r_mm=0.3,
                                     ke_lo=0.02, ke_hi=0.05,
                                     mz_list=[322.0]),
                   collisions=CollisionSpec(),
                   integration=IntegrationSpec(),
                   view=ViewSpec())
    errs = spec.validate()
    if errs:
        raise ValueError("slim2d_cross_section produced an invalid spec "
                         f"— parameter conflict: {errs}")
    return spec
