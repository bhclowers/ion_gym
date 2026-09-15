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
def hill_beta_njit(a, q, square=False):
    """beta(a, q) by RK4 monodromy over one half-period [0, pi].

    Verbatim algorithm of V02's validated `hill_beta`: two fundamental
    solutions advanced by classic RK4 on the fixed 4001-node grid;
    beta = arccos(clip(trace/2)) / pi. `square` selects the Meissner
    (square-wave) drive exactly as the original does.
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
    tr = (y1_0 + y2_1) / 2.0
    if tr > 1.0:
        tr = 1.0
    elif tr < -1.0:
        tr = -1.0
    return np.arccos(tr) / np.pi


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
