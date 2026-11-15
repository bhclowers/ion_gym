"""Shared machinery of the digital_quad study: the 2-D analytic rod
quadrupole both stability notebooks fly, its q calibration, and the
exact Meissner band helpers the deck generator and the digital notebook
share.

The flown quadrupole is the V02 validation quad (four round rods at
rod/field-radius ratio 1.148), borrowed with its estimators from
notebooks/V02_mathieu_stability.ipynb, extended only by the drive
options the digital study needs: a two-level square waveform of duty d
on the x pair with the inverted waveform (duty 1 - d, phase 360(1 - d)
degrees) on the y pair, and a common phase shift that sets the entry
phase of an ion born at t = 0.
"""
from __future__ import annotations

import numpy as np

from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec, GeometrySpec,
                                 IntegrationSpec, RFGroupSpec, ShapeSpec,
                                 SimSpec, SourceSpec)
from ion_gym.physics.mathieu import hill_beta_njit, meissner_half_trace
from ion_gym.physics.sim_build import build_run
from ion_gym.physics.symmetry import SymmetrySpec

AMU = 1.66053906660e-27
E_CHG = 1.602176634e-19

# The V02 quad: field radius, rod ratio (the dodecapole-minimizing value
# for round rods), raster pitch.
R0_MM = 3.0
ROD_FACTOR = 1.148
PITCH_MM = 0.1


def quad_spec(v0, *, waveform="sin", duty=0.5, phase_frac=0.0,
              f_rf_mhz=1.0, mz=100.0, x0=0.5, y0=None, ke=0.0,
              t_max=160.0, dt_ns=1.0, rec_every=25, pitch_mm=PITCH_MM,
              rod_factor=ROD_FACTOR, dc_v=0.0,
              name="digital_quad study quad", rf_names=None):
    """One-ion spec of the analytic rod quadrupole at one drive.

    waveform 'sin': the V02 drive, x pair at phase 0 and y at 180.
    waveform 'square': the digital drive, x pair high for `duty` of the
    period and the y pair carrying the exact inverse. phase_frac shifts
    both waveforms together, so an ion born at t = 0 enters at that
    fraction of the waveform period.

    dc_v: DC volts on the x pair, with -dc_v on the y pair -- the
    Mathieu a axis. Convert a to volts with dc_for_a, which carries the
    measured sign convention; the magnitude relation |a|/q = 2|U|/V is
    geometry-free because a and q scale with the same field curvature.

    name: the deck's name field. The default is the study label; figures
    for publication pass a plain device name.
    rf_names: (x pair, y pair) drive-group names. None keeps the study
    names, ('RFX', 'RFY') for 'sin' and ('DIGX', 'DIGY') for 'square'.
    """
    ph0 = 360.0 * phase_frac
    if rf_names is not None and len(rf_names) != 2:
        raise ValueError(f"rf_names takes two names (x pair, y pair), got "
                         f"{rf_names!r}")
    if waveform == "square":
        nx, ny = rf_names or ("DIGX", "DIGY")
        rf = [RFGroupSpec(name=nx, waveform="square", duty=duty,
                          frequency_hz=f_rf_mhz * 1e6, amplitude_v=v0,
                          phase_deg=ph0 % 360.0),
              RFGroupSpec(name=ny, waveform="square", duty=1.0 - duty,
                          frequency_hz=f_rf_mhz * 1e6, amplitude_v=v0,
                          phase_deg=(ph0 + 360.0 * (1.0 - duty)) % 360.0)]
    elif waveform == "sin":
        nx, ny = rf_names or ("RFX", "RFY")
        rf = [RFGroupSpec(name=nx, waveform="sin",
                          frequency_hz=f_rf_mhz * 1e6, amplitude_v=v0,
                          phase_deg=ph0 % 360.0),
              RFGroupSpec(name=ny, waveform="sin",
                          frequency_hz=f_rf_mhz * 1e6, amplitude_v=v0,
                          phase_deg=(ph0 + 180.0) % 360.0)]
    else:
        raise ValueError(f"waveform {waveform!r}: this quad takes 'sin' "
                         f"or 'square'")
    gx, gy = rf[0].name, rf[1].name
    rrod = rod_factor * R0_MM
    cdist = R0_MM + rrod
    w = 2 * (R0_MM + 2 * rrod) + 2.0
    w = float(np.ceil(w / pitch_mm - 1e-9) * pitch_mm)
    ctr = w / 2

    def rod(name, dx, dy, grp, dc):
        return ElectrodeSpec(name=name, dc=dc, rf_groups=[grp], shapes=[
            ShapeSpec(type="ellipse", params={
                "cx_mm": ctr + dx, "cy_mm": ctr + dy,
                "rx_mm": rrod, "ry_mm": rrod})])

    geom = GeometrySpec(
        width_mm=w, height_mm=w, mm_per_gu=pitch_mm,
        symmetry=SymmetrySpec(coords="xyz"), rf_groups=rf,
        electrodes=[rod("xp", cdist, 0, gx, dc_v),
                    rod("xm", -cdist, 0, gx, dc_v),
                    rod("yp", 0, cdist, gy, -dc_v),
                    rod("ym", 0, -cdist, gy, -dc_v)])
    if y0 is None:
        y0 = 0.7 * x0
    return SimSpec(
        name=name, geometry=geom,
        source=SourceSpec(n_ions=1, distribution="point", x0_mm=ctr + x0,
                          y0_mm=ctr + y0, ke_lo=ke, ke_hi=ke,
                          direction=[1.0, 0.0, 0.0], temperature_k=0.0,
                          mz_list=[mz], tob_span_us=0.0, seed=1),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(t_max_us=t_max, dt_ns=dt_ns,
                                    rec_every=rec_every))


# One palette for every chart in the two stability notebooks, so the x
# axis, the y axis, and an operating island carry the same color in
# every figure a reader meets.
CHART_COLORS = {
    "x_band": "#2E6FB7",     # x-axis stable stripe
    "y_band": "#E8842C",     # y-axis stable stripe
    "island": "#1A1A1A",     # two-axis stable (operating) set
    "zone_mark": "#C8102E",  # published Table-1 operating points
    "flown_ok": "#2E8B57",   # flown: transmitted
    "flown_lost": "#C8102E", # flown: lost
}


def apply_chart_style():
    """rcParams shared by both stability notebooks: sans-serif, open
    spines, 300 dpi on save. Applied once per notebook in the imports
    cell; figures stay self-describing (operating point in the title)
    per the rendering rules."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.dpi": 115,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.formatter.useoffset": False,
    })


def halo_text(ax, x, y, text, **kw):
    """Axis text with a white halo, so labels stay readable on top of a
    colored map."""
    import matplotlib.patheffects as pe
    t = ax.annotate(text, (x, y), **kw)
    t.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])
    return t


def dc_for_a(a, v_per_q):
    """Volts for quad_spec's dc_v that realize Mathieu parameter a.

    Magnitude: a and q scale with the SAME measured field curvature
    (one geometric factor times the applied volts), so their ratio is
    geometry-free and exact, |a|/q = 2|U|/V, i.e. |U| = |a| V_PER_Q / 2
    at any frequency -- no second calibration exists to disagree with
    the first. Sign: MEASURED, not assumed. meissner_half_trace's
    x-labeled axis is defocused during its high segment (p = a - 2q),
    while the flown x pair sits at +amplitude during its duty fraction,
    which focuses a positive ion -- so the flown x pair realizes the
    theory's y-labeled axis and U = -a V_PER_Q / 2. Flight check:
    volts from a = +0.05 transmit on the
    theory's a = -0.05 band center and not on the +0.05 one.
    """
    return -a * v_per_q / 2.0


def fly_one(sp, *, record_budget_gb=None):
    """(model, trajectory, column index, status) of the spec's one ion.
    record_budget_gb goes to build_run (None derives it from RAM)."""
    errs = sp.validate()
    if errs:
        raise ValueError("spec refused:\n  " + "\n  ".join(errs))
    model, fly, cols, births = build_run(sp, record_budget_gb=record_budget_gb)
    tr, status = fly(0)
    return model, np.asarray(tr), {c: j for j, c in enumerate(cols)}, status


def f_zero_cross(tr, ci, ref, f_rf_mhz=1.0, col="x"):
    """Secular frequency by zero-crossing count of the RF-period-smoothed
    coordinate (the V02 estimator, bin-free unlike an FFT peak)."""
    t = tr[:, ci["t"]]
    x = tr[:, ci[col]] - ref
    dt = t[1] - t[0]
    n = max(1, int(round(1.0 / f_rf_mhz / dt)))
    xs = np.convolve(x - x.mean(), np.ones(n) / n, mode="valid")
    ts = t[n - 1:]
    s = np.signbit(xs)
    z = np.nonzero(s[1:] != s[:-1])[0]
    if len(z) < 4:
        return np.nan
    tz = []
    for i in (z[0], z[-1]):
        a, b = xs[i], xs[i + 1]
        tz.append(ts[i] + dt * a / (a - b))
    return (len(z) - 1) / (2 * (tz[1] - tz[0]))


def calibrate_v_per_q(f_rf_mhz=1.0, v_cal=30.0, mz=100.0):
    """(V per unit q at f_rf_mhz, q at v_cal) from the SOLVED field.

    The V02 calibration: the RF pseudopotential curvature of the solved
    quad at v_cal gives omega_pe, q = 2 sqrt(2) omega_pe / Omega inverts
    it, and q is exactly linear in V. q scales as 1/Omega^2, so the
    calibration at one frequency prices every other through the
    (f / f_cal)^2 factor applied here. One number per geometry, read
    from the field the ions fly in, never a textbook constant.
    """
    f_cal = 1.0
    sp = quad_spec(v_cal, f_rf_mhz=f_cal, mz=mz)
    ctr = sp.geometry.width_mm / 2
    model, _tr, _ci, _st = fly_one(sp)
    xs, ys, pe, _ele = model.pe_surface(mz=mz)
    cy = int(np.argmin(np.abs(ys - ctr)))
    sel = np.abs(xs - ctr) <= 0.6
    a2, _b2, _c2 = np.polyfit(xs[sel], pe[:, cy][sel], 2)
    d2 = 2.0 * a2
    m_kg = mz * AMU
    om = 2 * np.pi * f_cal
    f_pe = np.sqrt(max(d2, 0.0) * E_CHG * 1e6 / m_kg) * 1e-6 / (2 * np.pi)
    q_cal = (f_pe * 2 * np.pi) * 2 * np.sqrt(2) / om
    if q_cal <= 0:
        raise ValueError("calibration found no RF curvature: the solved "
                         "pseudopotential vertex fit returned zero")
    v_per_q = (v_cal / q_cal) * (f_rf_mhz / f_cal) ** 2
    return v_per_q, q_cal


def a0_stable_bands(stable, q_lo, q_hi, n=40001):
    """[(q_lo, q_hi)] of the stable runs of `stable(q)` on [q_lo, q_hi],
    scanned at n points. The resolution of an edge is the scan step,
    (q_hi - q_lo)/(n - 1), which the caller states with any width."""
    qs = np.linspace(q_lo, q_hi, n)
    ok = np.array([bool(stable(q)) for q in qs])
    runs = []
    i = 0
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            runs.append((float(qs[i]), float(qs[j])))
            i = j + 1
        else:
            i += 1
    return runs


def sin_stable(a, q):
    """Both transverse axes stable under the sinusoidal drive. The
    validated RK4 monodromy clips the half-trace, so instability lands
    exactly on beta = 0 or 1 and the stable interior is the open
    interval; the x axis takes (a, q), the y axis (-a, q)."""
    bx = hill_beta_njit(a, q)
    by = hill_beta_njit(-a, q)
    return 0.0 < bx < 1.0 and 0.0 < by < 1.0


def digital_stable(a, q, duty):
    """Both transverse axes stable under the digital drive: the x axis
    takes (a, q, duty), the y axis the inverted waveform (-a, q, 1-duty)."""
    return (abs(meissner_half_trace(a, q, duty)) < 1.0
            and abs(meissner_half_trace(-a, q, 1.0 - duty)) < 1.0)


# Band-edge bisection: an edge is located to EDGE_TOL_REL times the band's
# q (absolute tolerance = EDGE_TOL_REL * q_hint). 1e-12 is far below any
# published zone width (the narrowest, zone 4,1, is ~8.5e-5 of its q) and
# above the double-precision spacing of q near 16 (~3.6e-15). The cap on
# halvings is the exit of the bisection loop: 200 halvings shrink any
# scan step below the tolerance many times over.
EDGE_TOL_REL = 1e-12
EDGE_MAX_HALVINGS = 200


def refine_band_edge(stable, q_out, q_in, tol, max_halvings=EDGE_MAX_HALVINGS):
    """Bisect the stability edge between q_out (unstable) and q_in (stable)
    until the bracket is narrower than tol; return the stable side, so the
    returned q is inside the band by at most tol. Refuses a bracket whose
    ends do not straddle the edge, and a bracket that has not closed after
    max_halvings."""
    if bool(stable(q_out)) or not bool(stable(q_in)):
        raise ValueError(f"edge bracket ({q_out!r}, {q_in!r}) does not "
                         f"straddle a stability edge: stable(q_out) = "
                         f"{bool(stable(q_out))}, stable(q_in) = "
                         f"{bool(stable(q_in))}")
    for _ in range(max_halvings):
        if abs(q_in - q_out) <= tol:
            return q_in
        mid = 0.5 * (q_out + q_in)
        if bool(stable(mid)):
            q_in = mid
        else:
            q_out = mid
    raise ValueError(f"edge bisection did not reach tol = {tol:g} in "
                     f"{max_halvings} halvings (bracket {q_out!r} to "
                     f"{q_in!r})")


def zone_band_a0(duty, q_hint, rel_span=0.02, n=40001, a=0.0,
                 tol_rel=EDGE_TOL_REL):
    """(q_lo, q_hi) of the digital x-AND-y stable band nearest q_hint,
    on the a = 0 axis by default; pass `a` to cut the band at a DC
    offset (the band tilts in (a, q), so its center moves with a).

    Two stages. A scan of n points over q_hint * (1 +/- rel_span) finds
    the stable runs, and the run containing q_hint (else the first) is
    taken. Each edge of that run is then bisected with refine_band_edge
    between the last unstable and the first stable scan point, to
    tol_rel * q_hint. The edge accuracy therefore does not depend on n.
    The scan only has to detect the band: a band narrower than the scan
    step, 2 * rel_span * q_hint / (n - 1), can fall between points and
    be missed, and is then reported by the refusal below.

    The fixed-grid version returned the outermost STABLE scan points as
    the edges, which places each edge up to one step inside the band. For
    zone 4,1 (84 steps across at the default n) that made R_BL 11 905;
    the bisected width gives 11 707.
    """
    def stable(q):
        return digital_stable(a, q, duty)

    q0, q1 = q_hint * (1 - rel_span), q_hint * (1 + rel_span)
    runs = a0_stable_bands(stable, q0, q1, n)
    if not runs:
        raise ValueError(f"no stable band within {rel_span:.0%} of "
                         f"q = {q_hint} at duty {duty}, a = {a} at scan "
                         f"step {(q1 - q0) / (n - 1):.3g}; a band narrower "
                         f"than the step can be missed, so raise n")
    hit = [r for r in runs if r[0] <= q_hint <= r[1]]
    lo_s, hi_s = (hit or runs)[0]
    step = (q1 - q0) / (n - 1)
    if lo_s <= q0 or hi_s >= q1:
        raise ValueError(f"the band at duty {duty}, a = {a} reaches the "
                         f"edge of the scanned span [{q0:.6g}, {q1:.6g}], "
                         f"so its edge is not bracketed; widen rel_span")
    tol = tol_rel * q_hint
    lo = refine_band_edge(stable, lo_s - step, lo_s, tol)
    hi = refine_band_edge(stable, hi_s + step, hi_s, tol)
    return lo, hi


def zone_duty_tolerance(duty, q_hint):
    """The duty error that moves the zone's a = 0 band center by half the
    band's own width, from the center's measured sensitivity to duty."""
    lo, hi = zone_band_a0(duty, q_hint)
    center, width = 0.5 * (lo + hi), hi - lo
    eps = 1e-4
    lo2, hi2 = zone_band_a0(duty + eps, q_hint)
    sensitivity = abs(0.5 * (lo2 + hi2) - center) / eps
    if sensitivity == 0.0:
        raise ValueError(f"zone center at duty {duty} shows no duty "
                         f"sensitivity; the tolerance is undefined")
    return (width / 2.0) / sensitivity


# Samples per drive segment when the peak excursion of an ideal-field
# orbit is read off the exact segment propagators. Inside one segment the
# orbit is a cos/sin or cosh/sinh pair with at most one interior extremum,
# so 400 samples place that extremum to well under 0.1% of its value.
PEAK_SAMPLES_PER_SEGMENT = 400


def meissner_peak_excursion(a, q, duty, phase_frac, u0, n_periods, *,
                            axis="x", samples=PEAK_SAMPLES_PER_SEGMENT):
    """Largest |u| reached in n_periods drive periods by an ion at rest at
    u0 in the IDEAL quadrupole field under the digital drive, entering at
    phase_frac of the x-pair waveform (the quad_spec phase convention).

    The motion is u'' + p(tau) u = 0 with tau = Omega t / 2. The x pair is
    high (m = +1) while the phase fraction is below duty, as in
    build_planar.drive_voltage. The x axis sees p = a - 2 q m and the y
    axis, whose rods carry the inverse waveform, p = -a + 2 q m (the
    meissner_half_trace convention). Each segment is propagated exactly,
    so the only approximation is the sampling of |u| inside a segment.
    Returns the peak in the units of u0.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis {axis!r}: 'x' or 'y'")
    if not (0.0 < duty < 1.0):
        raise ValueError(f"duty {duty} must lie strictly between 0 and 1")
    if not (0.0 <= phase_frac < 1.0):
        raise ValueError(f"phase_frac {phase_frac} must lie in [0, 1)")
    if n_periods < 1:
        raise ValueError(f"n_periods {n_periods} must be at least 1")
    sgn = 1.0 if axis == "x" else -1.0
    u, v, frac, peak = float(u0), 0.0, float(phase_frac), abs(float(u0))
    tau_left = n_periods * np.pi
    while tau_left > 0.0:          # exits: each pass removes L > 0 of tau
        m, nxt = (1.0, duty) if frac < duty else (-1.0, 1.0)
        L = min((nxt - frac) * np.pi, tau_left)
        p = sgn * (a - 2.0 * q * m)
        s = np.linspace(0.0, L, samples)
        if p > 0.0:
            w = np.sqrt(p)
            us = u * np.cos(w * s) + v / w * np.sin(w * s)
            vs = -u * w * np.sin(w * s) + v * np.cos(w * s)
        elif p < 0.0:
            w = np.sqrt(-p)
            us = u * np.cosh(w * s) + v / w * np.sinh(w * s)
            vs = u * w * np.sinh(w * s) + v * np.cosh(w * s)
        else:
            us, vs = u + v * s, np.full_like(s, v)
        peak = max(peak, float(np.abs(us).max()))
        u, v = float(us[-1]), float(vs[-1])
        tau_left -= L
        frac = 0.0 if nxt == 1.0 else nxt
    return peak
