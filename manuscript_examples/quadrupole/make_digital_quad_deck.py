"""Author the digital (two-level square drive) variant of the shipped 3-D
STL quadrupole, operated in a higher stability zone per Hu and
Gundlach-Graham, Rapid Commun. Mass Spectrom. 38 (2024) e9753.

The geometry, STL rods, lattice and source are inherited UNCHANGED from
examples/quadrupole_stl_rods_full_3-d.json. Only the drive changes: the
two sinusoids become two-level square waveforms of opposite sense, the
pair with duty >= 0.5 on the x rods and its complement on the y rods.
With no DC between the pairs (a = 0), the duty cycle alone selects the
stability zone, and the drive frequency selects which m/z sits at the
zone's q:  q = 4 z e V / (m r0^2 Omega^2).

r0 provenance: the rod assembly's closest-approach radius measured from
the STLs is 3.856 mm (tests/test_stl_assembly.py), the design statement
is 3.84 mm, and the shipped deck's own sinusoidal amplitude (241.335 V
at 2 MHz for q = 0.4 at m/z 100) inverts to r0 = 3.840 mm exactly, so
3.840 mm is the number the shipped deck already embodies.

Zone table (Hu 2024, Table 1): duty, operating q at a = 0, and the
baseline resolution R_BL = q/dq of the zone. The zone bands were
reproduced here with ion_gym's exact Meissner monodromy
(ion_gym.physics.mathieu.meissner_half_trace) before this deck was cut:
centers within 0.03 %, R_BL within 5 % of the published values.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

# Hu 2024 Table 1. duty is the x-pair high fraction.
ZONES = {
    "3,1": dict(duty=0.8350, q=4.59, r_bl=556),
    "3,2": dict(duty=0.6250, q=8.00, r_bl=1519),
    "3,3": dict(duty=0.5000, q=12.50, r_bl=2532),
    "4,1": dict(duty=0.8750, q=8.00, r_bl=12500),
}

E_CHG = 1.602176634e-19
KG_AMU = 1.66053906660e-27

# Fraction of the zone's duty tolerance the integration step may consume.
# The flown waveform's switch edges are resolved to dt/T of a period, an
# effective duty error; the deck's step keeps that error at half the
# tolerance that would shift the zone center by half its own width.
DT_DUTY_TOLERANCE_FRACTION = 0.5


def _zone_duty_tolerance(duty, q_hint):
    """quad_lib's measured tolerance, imported beside this module."""
    import sys
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from quad_lib import zone_duty_tolerance
    return zone_duty_tolerance(duty, q_hint)


def digital_frequency_hz(q, v0p, mz, r0_mm, charge=1.0):
    """Drive frequency that puts an ion of mz at Mathieu q, from the
    ideal-quadrupole relation q = 4 e V / (m r0^2 Omega^2) inverted for
    Omega. V is the zero-to-peak pole voltage v0p; r0 the field radius."""
    m_kg = mz * KG_AMU
    r0 = r0_mm * 1e-3
    omega = np.sqrt(4.0 * charge * E_CHG * v0p / (m_kg * r0 ** 2 * q))
    return omega / (2.0 * np.pi)


def solved_field_curvature(deck, deck_dir, v0p):
    """Quadrupole curvature A (V/mm^2) of the deck's SOLVED field at the
    axis, with the x pair at +v0p and the y pair at -v0p DC (the
    two-level drive's instantaneous pattern), such that the principal-
    axis potential is A (u^2 - v^2) / 2.

    Why this exists: the frequency that places an ion at the zone's q
    must come from the field the ions actually fly in. The ideal-quad
    relation with the design r0 missed the flown band entirely at this
    deck's 0.5 mm voxelization -- the voxelized curvature differs from
    the ideal by several percent, and the zones are tenths of a percent
    wide. (The planar control measured the same effect: recalibrating
    V-per-q at 0.5 mm pitch moved it 9% and restored the flown band.)

    Rotation-agnostic ON PURPOSE: this deck's rod pairs sit on the
    diagonals, so a 1-D fit along lab x reads zero curvature by
    construction. The probe fits the full 2-D quadratic on a vacuum
    window around the axis and takes A from the Hessian eigenvalues
    (lambda = +/-A for a Laplace field, whatever the rod orientation);
    the fitted saddle position and the eigenvalue asymmetry are both
    checked, so a mis-centered window or a non-quadrupole field refuses
    rather than calibrating to a wrong number. Uses the deck's own
    electrode bases (served by the stl3d solve cache), composed as a
    DC-only probe.
    """
    import numpy as np
    from ion_gym.io.sim_spec import SimSpec
    from ion_gym.physics.sim_build import build_run

    probe = copy.deepcopy(deck)
    g = probe["geometry"]
    raw = g.get("stl_dir")
    if raw and not Path(raw).is_absolute():
        # the probe is built from a dict (no file location), so the
        # deck-relative stl_dir must become absolute here
        g["stl_dir"] = str((Path(deck_dir) / raw).resolve())
    xnames = {el["name"] for el in g["electrodes"]
              if "DIGX" in el.get("rf_groups", [])}
    ynames = {el["name"] for el in g["electrodes"]
              if "DIGY" in el.get("rf_groups", [])}
    if not xnames or not ynames:
        raise ValueError(f"curvature probe: deck has no DIGX/DIGY "
                         f"electrodes (x: {sorted(xnames)}, "
                         f"y: {sorted(ynames)})")
    for el in g["electrodes"]:
        el["dc"] = float(v0p) if el["name"] in xnames else -float(v0p)
        el["rf_groups"] = []
    g["rf_groups"] = []
    spec = SimSpec.from_dict(probe)
    model, _fly, _cols, _births = build_run(spec)
    nz = model.A.shape[2]
    xs, ys, pe, ele = model.pe_surface(mz=None, plane="xy", index=nz // 2)
    x_src, y_src = probe["source"]["x0_mm"], probe["source"]["y0_mm"]
    half = 2.0
    mx = np.abs(xs - x_src) <= half
    my = np.abs(ys - y_src) <= half
    win = np.ix_(mx, my)
    if int(np.count_nonzero(ele[win])):
        raise ValueError("curvature probe: the fit window touches metal; "
                         "the source position or window is off-axis")
    X, Y = np.meshgrid(xs[mx], ys[my], indexing="ij")
    x, y, f = X.ravel(), Y.ravel(), np.asarray(pe)[win].ravel()
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    c, *_ = np.linalg.lstsq(basis, f, rcond=None)
    hess = np.array([[2 * c[3], c[4]], [c[4], 2 * c[5]]])
    lam = np.linalg.eigvalsh(hess)
    a_curv = (lam[1] - lam[0]) / 2.0
    if a_curv <= 0:
        raise ValueError(f"curvature probe: Hessian eigenvalues {lam} "
                         f"carry no quadrupole term")
    if abs(lam[1] + lam[0]) > 0.2 * a_curv:
        raise ValueError(f"curvature probe: Hessian trace {lam[0] + lam[1]:.3f} "
                         f"V/mm^2 is not small against the quadrupole "
                         f"strength {a_curv:.3f} -- the window is not "
                         f"reading a 2-D quadrupole field")
    saddle = np.linalg.solve(hess, -np.array([c[1], c[2]]))
    if np.hypot(saddle[0] - x_src, saddle[1] - y_src) > 0.5:
        raise ValueError(f"curvature probe: fitted saddle at "
                         f"({saddle[0]:.2f}, {saddle[1]:.2f}) mm, more "
                         f"than 0.5 mm from the deck source "
                         f"({x_src:g}, {y_src:g}) -- the probe is not "
                         f"reading the quad center")
    return float(a_curv)


def digital_frequency_from_field(q, v0p, mz, curvature_v_mm2, charge=1.0):
    """Drive frequency that puts an ion of mz at Mathieu q, from the
    MEASURED axis curvature A = |d2 phi/dx2| of the solved field at
    pole voltage v0p: q = 2 e A / (m Omega^2) (the ideal-quad special
    case A = 2 v0p / r0^2 recovers q = 4 e V / (m r0^2 Omega^2))."""
    m_kg = mz * KG_AMU
    a_si = curvature_v_mm2 * 1e6          # V/mm^2 -> V/m^2
    omega = np.sqrt(2.0 * charge * E_CHG * a_si / (m_kg * q))
    return omega / (2.0 * np.pi)


def make_digital_quad_deck(source_deck, out_path, zone="3,1", v0p=200.0,
                           mz=100.0, r0_mm=3.840, dt_ns=None):
    """Write the digital-drive deck and return it as a dict.

    source_deck : the shipped 3-D STL quadrupole deck (geometry source).
    zone        : Hu 2024 zone label, one of ZONES.
    v0p         : two-level amplitude, V zero-to-peak (paper: 200).
    mz          : the m/z placed at the zone's q by the frequency.
    r0_mm       : field radius (provenance in the module docstring).
    dt_ns       : integration step, or None (the default) to derive it
                  from the zone itself: the switch edges are resolved to
                  dt/T of a period, an effective duty error, and the
                  zone tolerates only the duty error that shifts its
                  band center by half the band width (zone 3,1: 2.6e-4;
                  measured via zone_duty_tolerance from the exact
                  Meissner monodromy). The derived step spends
                  DT_DUTY_TOLERANCE_FRACTION of that tolerance. A flat
                  1 ns default shipped first and was wrong: at 537 kHz
                  it put 5.4e-4 of duty error on a zone that tolerates
                  2.6e-4, and the flown band vanished.
    """
    if zone not in ZONES:
        raise ValueError(f"zone {zone!r} is not in the table; known zones: "
                         f"{sorted(ZONES)}")
    z = ZONES[zone]
    deck = copy.deepcopy(json.loads(Path(source_deck).read_text()))
    out_dir = Path(out_path).resolve().parent
    # stl_dir resolves against the DECK's own location (stl_resolve), so a
    # deck cut into another folder must carry the path from THERE back to
    # the source deck's assets. Computed from the two locations, whatever
    # they are; an stl_dir copied verbatim points at nothing and the build
    # refuses with FileNotFoundError on the first rod.
    import os
    raw = deck.get("geometry", {}).get("stl_dir")
    if raw and not Path(raw).is_absolute():
        src_assets = (Path(source_deck).resolve().parent / raw)
        deck["geometry"]["stl_dir"] = os.path.relpath(src_assets, out_dir)
    duty = z["duty"]
    # x pair: high fraction `duty`, phase 0. y pair: the inverted
    # waveform, which for a two-level square is duty (1 - d) with the
    # high window shifted to start where x goes low: phase 360(1 - d).
    groups = deck["geometry"]["rf_groups"]
    if len(groups) != 2:
        raise ValueError(f"expected the quad's two RF groups, found "
                         f"{[g['name'] for g in groups]}")
    for g, (nm, d, ph) in zip(groups, [("DIGX", duty, 0.0),
                                       ("DIGY", 1.0 - duty,
                                        360.0 * (1.0 - duty))]):
        old = g["name"]
        g.update(name=nm, waveform="square", duty=d, phase_deg=ph,
                 amplitude_v=float(v0p))
        for el in deck["geometry"]["electrodes"]:
            el["rf_groups"] = [nm if r == old else r
                               for r in el["rf_groups"]]
    # frequency from the SOLVED field's curvature (see
    # solved_field_curvature); the design-r0 figure stays in the notes as
    # a cross-check only
    curv = solved_field_curvature(deck, out_dir, v0p)
    charge = deck["source"].get("charge", 1)
    f_hz = digital_frequency_from_field(z["q"], v0p, mz, curv, charge)
    f_r0 = digital_frequency_hz(z["q"], v0p, mz, r0_mm, charge)
    for g in deck["geometry"]["rf_groups"]:
        g["frequency_hz"] = f_hz
    if dt_ns is None:
        # TWO requirements, both measured in flight (2-D control, zone
        # 3,1): (1) the step must be COMMENSURATE with the drive -- dt =
        # T/N with N a multiple of the duty's denominator, so every
        # switch edge lands exactly on a step. An incommensurate step
        # puts a per-period jitter of order dt/T on the duty, and that
        # jitter killed the flown band even at dt/T = 7e-5, while
        # edge-aligned steps held it. (2) Within edge-aligned steps,
        # dt small enough that truncation over the exponential-growth
        # segments is converged: the static duty-tolerance budget below
        # caps dt, and the flown ladder confirmed T/14400 reproduces the
        # dt->0 transmission.
        from fractions import Fraction
        tol = _zone_duty_tolerance(duty, z["q"])
        per_ns = 1e9 / f_hz
        cap_ns = DT_DUTY_TOLERANCE_FRACTION * tol * per_ns
        den = Fraction(duty).limit_denominator(10000).denominator
        n_steps_per = math.ceil(per_ns / cap_ns / den) * den
        dt_ns = per_ns / n_steps_per
    deck["integration"]["dt_ns"] = float(dt_ns)
    # the inherited rec_every was chosen for the source deck's step; at
    # this deck's (much smaller) derived step it would overflow the
    # inherited max_records and build_run refuses the deck. Thin the
    # recording so it covers the whole flight at the same record budget.
    integ = deck["integration"]
    n_steps = math.ceil(integ["t_max_us"] * 1e3 / float(dt_ns))
    integ["rec_every"] = max(1, math.ceil(n_steps / integ["max_records"]))

    deck["name"] = f"quadrupole digital drive, zone {zone} (3-D STL rods)"
    deck["notes"] = (
        f"Digital (two-level square) drive variant of "
        f"quadrupole_stl_rods_full_3-d.json: geometry, STL rods, lattice "
        f"and source inherited unchanged. Drive per Hu and "
        f"Gundlach-Graham, RCM 38 (2024) e9753: duty "
        f"{100 * duty:.2f}/{100 * (1 - duty):.2f} on the x pair (DIGX, "
        f"phase 0) with the inverted waveform on the y pair (DIGY, duty "
        f"{1 - duty:g}, phase {360 * (1 - duty):g} deg), amplitude "
        f"{v0p:g} V zero-to-peak, no DC between pairs (a = 0). Zone "
        f"{zone}: q = {z['q']:g}, published R_BL = {z['r_bl']}. "
        f"Frequency {f_hz / 1e3:.3f} kHz places m/z {mz:g} at that q "
        f"via the SOLVED field's measured axis curvature "
        f"{curv:.4f} V/mm^2 at {v0p:g} V (q = 2eA/(m Omega^2)); the "
        f"ideal-quad figure at the design r0 = {r0_mm:g} mm would be "
        f"{f_r0 / 1e3:.3f} kHz, off by "
        f"{abs(f_hz / f_r0 - 1):.1%} -- far more than the zone's "
        f"~0.2% band, which is why the calibration reads the field, "
        f"not the drawing. "
        f"dt {dt_ns:.6g} ns = T/{round(1e9 / f_hz / dt_ns)} is "
        f"COMMENSURATE with the drive (every switch edge lands on an "
        f"integration step; an incommensurate step jitters the duty per "
        f"period, and the flown band died from that jitter even at "
        f"dt/T = 7e-5) and sits inside the zone's measured static duty "
        f"tolerance (the error that shifts the band center by half its "
        f"width) at dt/T = {dt_ns * 1e-9 * f_hz:.1e}. Stability-zone numbers "
        f"validated against the exact Meissner monodromy "
        f"(physics.mathieu.meissner_half_trace) before cutting.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(deck, indent=1) + "\n")
    return deck


def main():
    from ion_gym.io.paths import examples_dir
    here = Path(__file__).resolve()
    deck = make_digital_quad_deck(
        examples_dir() / "quadrupole_stl_rods_full_3-d.json",
        here.parent / "decks" / "quadrupole_digital_3d.json")
    print(deck["name"])
    print(deck["notes"])


if __name__ == "__main__":
    main()
