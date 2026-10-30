"""Digital-drive stability zones against the published reference
(Hu and Gundlach-Graham, Rapid Commun. Mass Spectrom. 38 (2024) e9753,
Table 1). Requested by Brian 2026-10-21.

Pins physics.mathieu.meissner_half_trace — the exact two-segment
one-period transfer matrix — to the paper's four operating zones: band
center q and band-limited resolving power R_BL = q_c / dq at a = 0. A
future edit to mathieu.py that shifts a zone fails here, not in a
notebook.

Fast: no field solve and no flight. A coarse scan brackets each band,
then bisection on the stability predicate sharpens the edges, so the
R_BL figures carry no scan-resolution error.
"""
import numpy as np
import pytest

from ion_gym.physics.mathieu import hill_beta_njit, meissner_half_trace

# Hu 2024 Table 1: (duty = x-pair high fraction, q at band center,
# published band-limited resolving power, R_BL gate). Zone 4,1's gate is
# wider: its band is the narrowest (relative width 8e-5) and the most
# sensitive to the duty's fourth decimal, and the exact transfer matrix
# finds R_BL = 11905 against the published 12500 (4.8% low).
TABLE_1 = {
    "3,1": dict(duty=0.8350, q=4.59, r_bl=556, r_tol=0.05),
    "3,2": dict(duty=0.6250, q=8.00, r_bl=1519, r_tol=0.05),
    "3,3": dict(duty=0.5000, q=12.50, r_bl=2532, r_tol=0.05),
    "4,1": dict(duty=0.8750, q=8.00, r_bl=12500, r_tol=0.07),
}
# The paper quotes q to three figures (rounding alone allows 1.1e-3 at
# 4.59); the centers land within 3.1e-4 of the quoted values.
Q_CENTER_RTOL = 2e-3
# Coarse bracket scan: the widest zone band is 1.8e-3 of q, so 20001
# points over +/-2% (dq = 2e-6 of q) oversample every band's interior.
N_SCAN = 20001
SCAN_REL_SPAN = 0.02
# Edge bisection target, relative to q.
EDGE_RTOL = 1e-8


def both_axes_stable(a, q, duty):
    """x axis sees (a, q, duty); y carries the inverted waveform,
    (-a, q, 1 - duty). Stable strictly inside |half trace| < 1."""
    return (abs(meissner_half_trace(a, q, duty)) < 1.0
            and abs(meissner_half_trace(-a, q, 1.0 - duty)) < 1.0)


def _bisect_edge(q_in, q_out, duty):
    """The stability boundary between a stable q and an unstable q."""
    if not both_axes_stable(0.0, q_in, duty):
        raise AssertionError(f"bisection started outside the band: "
                             f"q_in = {q_in} is not stable at duty {duty}")
    while abs(q_out - q_in) > EDGE_RTOL * q_in:
        mid = 0.5 * (q_in + q_out)
        if both_axes_stable(0.0, mid, duty):
            q_in = mid
        else:
            q_out = mid
    return 0.5 * (q_in + q_out)


def band_at(duty, q_hint):
    """(q_lo, q_hi) of the a = 0 band containing q_hint's neighborhood:
    coarse bracket, then bisected edges."""
    qs = np.linspace(q_hint * (1 - SCAN_REL_SPAN),
                     q_hint * (1 + SCAN_REL_SPAN), N_SCAN)
    ok = np.array([both_axes_stable(0.0, q, duty) for q in qs])
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        raise AssertionError(f"no stable band within "
                             f"{SCAN_REL_SPAN:.0%} of q = {q_hint} at "
                             f"duty {duty}")
    # the run nearest the hint; a gap in idx would mean two bands in the
    # window, which the zone table does not describe -- refuse loudly
    if np.any(np.diff(idx) > 1):
        raise AssertionError(f"multiple stable runs within "
                             f"{SCAN_REL_SPAN:.0%} of q = {q_hint} at "
                             f"duty {duty}; the window is not one band")
    i0, i1 = idx[0], idx[-1]
    if i0 == 0 or i1 == N_SCAN - 1:
        raise AssertionError(f"band at duty {duty} touches the scan "
                             f"window edge; widen SCAN_REL_SPAN")
    lo = _bisect_edge(qs[i0], qs[i0 - 1], duty)
    hi = _bisect_edge(qs[i1], qs[i1 + 1], duty)
    return lo, hi


@pytest.mark.parametrize("zone", sorted(TABLE_1))
def test_table1_zone(zone):
    z = TABLE_1[zone]
    lo, hi = band_at(z["duty"], z["q"])
    q_c = 0.5 * (lo + hi)
    width = hi - lo
    assert width > 0.0
    assert q_c == pytest.approx(z["q"], rel=Q_CENTER_RTOL), (
        f"zone {zone}: band center {q_c:.5f} vs published {z['q']}")
    r_bl = q_c / width
    assert r_bl == pytest.approx(z["r_bl"], rel=z["r_tol"]), (
        f"zone {zone}: R_BL {r_bl:.0f} vs published {z['r_bl']} "
        f"(gate {z['r_tol']:.0%})")


def test_meissner_matches_square_rk4():
    """At duty 0.5 the exact segment product and the validated RK4
    square-drive integration build the same transfer matrix, within the
    RK4's own limits: its fixed 4001-node grid steps across the switch
    instants, an O(dtau) local error that the exponential segment
    growth amplifies with q (measured 3.5e-4 in beta at q = 12.5,
    under 1e-6 at q <= 0.6). That amplification is the reason
    meissner_half_trace exists, so the high-q point gets the looser
    gate on purpose."""
    for a, q, beta_abs_tol in ((0.0, 0.3, 1e-5), (0.05, 0.6, 1e-5),
                               (0.0, 12.5, 1e-3)):
        ht = meissner_half_trace(a, q, 0.5)
        if abs(ht) < 1.0:
            beta_exact = np.arccos(ht) / np.pi
            beta_rk4 = hill_beta_njit(a, q, square=True)
            assert beta_rk4 == pytest.approx(beta_exact,
                                             abs=beta_abs_tol), (
                f"(a, q) = ({a}, {q})")
        else:
            # RK4 clips its trace, so instability lands on beta 0 or 1
            b = hill_beta_njit(a, q, square=True)
            assert b in (0.0, 1.0), (f"(a, q) = ({a}, {q}): exact says "
                                     f"unstable, RK4 beta = {b}")
