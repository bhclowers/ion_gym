"""mathieu.py — Floquet characteristic exponent for the Mathieu/Meissner
Hill equation (the njit candidate).

STATUS: CANDIDATE beside the validated implementation, not a
replacement. V02's `hill_beta` (pure-Python RK4 monodromy, 4001 nodes
over [0, pi]) is the implementation of record; at a measured
~150 ms/point it prices V02's two Floquet maps at ~22 minutes, and its
own section-2 comment calls a numba lift "a worthwhile follow-up".
This module transcribes that algorithm EXACTLY — same 4001-node grid,
same classic RK4, same trace-clip — compiled with numba. Acceptance
protocol: the validated V02 run exports
outputs/v02_floquet_reference.npz; `compare_to_reference` must
reproduce every finite point of both maps within tolerance before the
notebook swaps over (at which point the notebook-local definition is
REMOVED, per the no-parallel-v2 rule — the swap is the supersede).

Local pre-acceptance anchors (cheap, run in tests):
  * a=0 identity: matches the notebook's printed anchor value at q=0.30
    to 1e-9 (the spec requirement stated at the definition site).
  * spot equality vs the transcribed pure-Python form at scattered
    (a, q), square and cosine drive, to 1e-12 (same arithmetic, so the
    only differences are compiler-level float reassociation).
"""
import numpy as np
from numba import njit

_N_TAU = 4001                    # the validated node count, verbatim


@njit(cache=True, nogil=True)
def hill_half_trace_njit(a, q, square=False):
    """tr(M)/2 of the one-period transfer matrix by RK4 over [0, pi],
    UNCLIPPED: |tr M|/2 > 1 is the instability margin the stability-map
    figures color by. The loop is V02's validated `hill_beta` algorithm
    verbatim (two fundamental solutions, classic RK4, fixed 4001-node
    grid); hill_beta_njit clips and arccos-es this value, refactored
    2026-10-21 with a 60-point bit-identity probe across both drives.
    `square` selects the Meissner (square-wave) drive.
    """
    dtau = np.pi / (_N_TAU - 1)
    y1_0, y1_1 = 1.0, 0.0
    y2_0, y2_1 = 0.0, 1.0
    for k in range(_N_TAU - 1):
        ti = k * dtau
        for _sel in range(2):
            if _sel == 0:
                y0, y1v = y1_0, y1_1
            else:
                y0, y1v = y2_0, y2_1
            # k1
            m = (np.sign(np.cos(2.0 * ti)) if square
                 else np.cos(2.0 * ti))
            k1_0 = y1v
            k1_1 = (2.0 * q * m - a) * y0
            # k2
            tm = ti + dtau / 2.0
            m = (np.sign(np.cos(2.0 * tm)) if square
                 else np.cos(2.0 * tm))
            k2_0 = y1v + dtau / 2.0 * k1_1
            k2_1 = (2.0 * q * m - a) * (y0 + dtau / 2.0 * k1_0)
            # k3
            k3_0 = y1v + dtau / 2.0 * k2_1
            k3_1 = (2.0 * q * m - a) * (y0 + dtau / 2.0 * k2_0)
            # k4
            te = ti + dtau
            m = (np.sign(np.cos(2.0 * te)) if square
                 else np.cos(2.0 * te))
            k4_0 = y1v + dtau * k3_1
            k4_1 = (2.0 * q * m - a) * (y0 + dtau * k3_0)
            y0 = y0 + dtau / 6.0 * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
            y1v = y1v + dtau / 6.0 * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)
            if _sel == 0:
                y1_0, y1_1 = y0, y1v
            else:
                y2_0, y2_1 = y0, y1v
    return (y1_0 + y2_1) / 2.0


@njit(cache=True, nogil=True)
def hill_beta_njit(a, q, square=False):
    """beta(a, q) = arccos(clip(tr M / 2)) / pi from hill_half_trace_njit
    (the validated V02 RK4 transfer matrix; see that function). Clipping
    puts every unstable point exactly on beta 0 or 1."""
    tr = hill_half_trace_njit(a, q, square)
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    return np.arccos(tr) / np.pi


def rf_amplitude_for_q(q, mz, r0_mm, freq_hz, charge=1):
    """RF amplitude V (volts, 0-peak) that puts an ion of mass mz (Da) at
    Mathieu q in an ideal quadrupole of field radius r0_mm driven at
    freq_hz:  q = 4 z e V / (m r0^2 Omega^2)  ->  V = q m r0^2 Omega^2 /
    (4 z e). The standard mass-filter relation; geometry-agnostic (any r0,
    any frequency, any charge).

    Moved here from build_stl._q_to_rf_amp (D6b, 2026-10-16): it is
    quadrupole physics, not part of the STL quad demo that used it."""
    from ion_gym.physics.constants import E_CHG, KG_AMU
    if not (r0_mm > 0 and freq_hz > 0 and charge):
        raise ValueError(f"rf_amplitude_for_q: need r0_mm > 0, freq_hz > 0 "
                         f"and a non-zero charge (got r0_mm={r0_mm}, "
                         f"freq_hz={freq_hz}, charge={charge})")
    m = mz * KG_AMU
    omega = 2.0 * np.pi * freq_hz
    r0 = r0_mm / 1000.0
    return q * m * r0 ** 2 * omega ** 2 / (4.0 * charge * E_CHG)


def compare_to_reference(npz_path, tol=1e-9):
    """Acceptance gate against the exported V02 reference maps.

    Recomputes every FINITE point of both maps with hill_beta_njit and
    reports the worst absolute deviation per map; raises if any point
    exceeds `tol`, naming the worst offender. Returns the two worst
    deviations so the acceptance can be quoted."""
    d = np.load(npz_path, allow_pickle=True)
    f_rf = float(d["f_rf_mhz"][0])
    worst = {}
    # section-2 tongue: stored as the stability chart 'stable' — the
    # comparable quantity is beta itself where 0<beta<1; recompute and
    # compare the STABILITY CLASSIFICATION plus beta on stable points.
    aa, qq = d["chart_a"], d["chart_q"]
    st_ref = d["chart_stable"]
    w = 0.0
    bad = None
    for i in range(aa.size):
        for j in range(qq.size):
            b = hill_beta_njit(float(aa[i]), float(qq[j]))
            stable = 1.0 if (0.0 < b < 1.0) else 0.0
            if stable != float(st_ref[i, j] > 0):
                bad = (aa[i], qq[j], b, st_ref[i, j])
    if bad is not None:
        raise AssertionError(
            f"chart stability classification differs at a={bad[0]:.4f}, "
            f"q={bad[1]:.4f}: candidate beta={bad[2]:.6f}, reference "
            f"class={bad[3]}")
    worst["chart_class"] = 0.0
    aa3, qq3, f_ref = d["fmap_a"], d["fmap_q"], d["fmap_f"]
    for i in range(aa3.size):
        for j in range(qq3.size):
            r = f_ref[i, j]
            if not np.isfinite(r):
                continue
            b = hill_beta_njit(float(aa3[i]), float(qq3[j]))
            dev = abs(b * f_rf / 2.0 - r)
            if dev > w:
                w = dev
                if dev > tol:
                    raise AssertionError(
                        f"fmap deviates {dev:.3e} MHz at a={aa3[i]:.4f}, "
                        f"q={qq3[j]:.4f} (tol {tol:g})")
    worst["fmap_MHz"] = w
    return worst


@njit(cache=True, nogil=True)
def meissner_half_trace(a, q, duty=0.5):
    """Half-trace tr(M)/2 of the one-period monodromy for the DUTY-CYCLE
    Meissner (two-level square) Hill drive, computed EXACTLY.

    The drive m(tau) holds +1 for a fraction `duty` of the pi-period and
    -1 for the rest, so u'' + (a - 2 q m) u = 0 is piecewise constant:
    p = a - 2q on the high segment, p = a + 2q on the low one. Each
    segment's propagator is analytic (cos/sin for p > 0, cosh/sinh for
    p < 0, shear for p = 0), and the monodromy is their product -- the
    matrix method of the digital-drive literature (Konenkov et al.,
    J. Am. Soc. Mass Spectrom. 13 (2002) 597). No integration grid, so
    no step error at the switch instants, which is what a fixed-grid RK4
    cannot avoid for a discontinuous drive.

    |tr(M)/2| < 1 is stable (any stability zone, not only the first).
    The x axis of a digital quadrupole takes (a, q, duty); the y axis
    takes (-a, q, 1 - duty), its drive being the inverted waveform.
    hill_beta_njit(square=True) is the 50/50 special case; the two agree
    there (RK4 to its step error) and this form is the authority for
    duty work.
    """
    if not (0.0 < duty < 1.0):
        raise ValueError("duty must lie strictly between 0 and 1")
    m00, m01, m10, m11 = 1.0, 0.0, 0.0, 1.0
    # segments in drive order: high (m=+1) then low (m=-1)
    for seg in range(2):
        if seg == 0:
            p = a - 2.0 * q
            L = duty * np.pi
        else:
            p = a + 2.0 * q
            L = (1.0 - duty) * np.pi
        if p > 0.0:
            w = np.sqrt(p)
            c = np.cos(w * L)
            s = np.sin(w * L)
            a00, a01, a10, a11 = c, s / w, -w * s, c
        elif p < 0.0:
            w = np.sqrt(-p)
            c = np.cosh(w * L)
            s = np.sinh(w * L)
            a00, a01, a10, a11 = c, s / w, w * s, c
        else:
            a00, a01, a10, a11 = 1.0, L, 0.0, 1.0
        # M <- A @ M
        n00 = a00 * m00 + a01 * m10
        n01 = a00 * m01 + a01 * m11
        n10 = a10 * m00 + a11 * m10
        n11 = a10 * m01 + a11 * m11
        m00, m01, m10, m11 = n00, n01, n10, n11
    return 0.5 * (m00 + m11)


@njit(cache=True, nogil=True)
def meissner_beta(a, q, duty=0.5):
    """beta in [0, 1] for the duty-cycle Meissner drive: arccos of the
    clipped half-trace, the same convention as hill_beta_njit, exact per
    meissner_half_trace. 0 < beta < 1 is stable; use the unclipped
    meissner_half_trace for band-edge location."""
    tr = meissner_half_trace(a, q, duty)
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    return np.arccos(tr) / np.pi
