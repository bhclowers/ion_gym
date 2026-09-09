"""
w1_wiley.py  --  W1: the stack against its own analytics
========================================================
A 1-D, on-axis, piecewise-uniform-field model of ANY plate stack: the
Wiley-McLaren idealisation, generalised exactly from two fields to N.
This is the ANALYTIC REFERENCE the flown 2-D solve is judged against in
test_W1.py.  Nothing here knows an instrument constant: the model is built
from a plate table (positions + node voltages), whoever supplies it.

THE IDEALISATION, stated: on the axis, the potential is the node voltage
across each plate's thickness and LINEAR across each gap (uniform E).  What
this throws away is exactly what the 2-D solve adds back -- slot penetration
and slit lensing -- so the (flown - analytic) difference is not an error
bar, it is a MEASUREMENT of the slot physics.

Closed forms used (all exact for piecewise-uniform E):
  * region with field:  v1 = sqrt(v0^2 + 2 a L),  t = (v1 - v0)/a
  * field-free:         t = L / v0
  * turnaround:         an ion born with -v_s returns to its birth plane
                        with +v_s after  dt = 2 v_s / a_1  (a_1 = field at
                        the birth point).  This is THE irreducible TOF
                        spread -- no downstream optic can undo it, because
                        the two ions are identical after the turnaround
                        except for the time offset.
Units: mm, us, V, Da.  a = ACC * E / m with ACC = 96.485 (q = +1).
"""
# PROVENANCE
#   origin   : Wiley & McLaren, Rev. Sci. Instrum. 26 (1955) 1150 -- "Time-of-
#              Flight Mass Spectrometer with Improved Resolution".  The
#              two-field space-focus condition.
#   derived  : generalised here from TWO fields to N, exactly (not fitted).
#   verified : gate W1 -- the stack against its own analytics.
from __future__ import annotations

import numpy as np

ACC = 96.485          # mm/us^2 per (V/mm / Da)


# ------------------------------------------------------------- the model
class AxisModel:
    """Piecewise-linear on-axis potential from a plate table.

    plates : [(s_front, s_back, volts)] mm/V, ascending, non-overlapping.
    Across a plate: phi = volts.  Across a gap: linear between neighbours.
    Beyond the last plate: field-free at the last plate's potential.
    """

    def __init__(self, plates):
        self.plates = sorted((float(a), float(b), float(v))
                             for a, b, v in plates)
        # breakpoints: (s, phi) at every plate face
        pts = []
        for a, b, v in self.plates:
            pts += [(a, v), (b, v)]
        self.s = np.array([p[0] for p in pts])
        self.phi = np.array([p[1] for p in pts])

    def pot(self, s):
        return np.interp(s, self.s, self.phi)

    def E(self, s):
        """-dphi/ds, piecewise constant (V/mm)."""
        s = np.atleast_1d(np.asarray(s, float))
        out = np.zeros_like(s)
        for i in range(len(self.s) - 1):
            a, b = self.s[i], self.s[i + 1]
            if b <= a:
                continue
            m = (s >= a) & (s < b)
            out[m] = -(self.phi[i + 1] - self.phi[i]) / (b - a)
        return out if out.size > 1 else float(out[0])

    # ------------------------------------------------------------- TOF
    def tof(self, s0, s_end, mz, K0=0.0, v0_sign=+1.0):
        """Exact TOF from s0 to s_end (s_end > s0) for charge +1.

        K0 (eV) is the initial kinetic energy ALONG s; v0_sign=-1 starts the
        ion moving backwards (turnaround handled exactly: the ion climbs,
        stops, and returns to s0 with the sign flipped -- valid while the
        field at s0 region is uniform over the excursion, which it is for
        thermal energies in any real extraction gap).
        """
        m = float(mz)
        v = v0_sign * np.sqrt(2.0 * ACC * max(K0, 0.0) / m)
        t = 0.0
        if v < 0.0:
            a1 = ACC * self.E(s0) / m
            if a1 <= 0:
                raise ValueError("ion launched backwards into a non-"
                                 "accelerating field never returns")
            t += 2.0 * (-v) / a1          # the turnaround, exactly
            v = -v
        # march the breakpoints
        edges = [s0] + [float(x) for x in self.s if s0 < x < s_end] + [s_end]
        for a, b in zip(edges[:-1], edges[1:]):
            L = b - a
            if L <= 0:
                continue
            acc = ACC * self.E(0.5 * (a + b)) / m
            if abs(acc) < 1e-15:
                t += L / v
            else:
                v1sq = v * v + 2.0 * acc * L
                if v1sq <= 0:
                    raise ValueError(f"ion reflected inside [{a},{b}] -- "
                                     "not a transmission stack")
                v1 = np.sqrt(v1sq)
                t += (v1 - v) / acc
                v = v1
        return t, v

    # ----------------------------------------------------- space focus
    def space_focus(self, s_beam, s_exit, mz, ds=0.25, n=7):
        """The plane L* (mm past s_exit) where dT/ds0 = 0, and the residual
        curvature d2T/ds0^2 there.  Computed from the exact TOFs of a fan of
        birth positions -- no small-signal formula, so it is valid for any
        number of stages."""
        s0s = s_beam + ds * (np.arange(n) - (n - 1) / 2)
        te = np.empty(n)
        ve = np.empty(n)
        for i, s0 in enumerate(s0s):
            te[i], ve[i] = self.tof(s0, s_exit, mz)
        # T_i(L) = te_i + L/ve_i is linear in L: same closed form as the
        # flown estimator (validate_gbstack_gateC.space_focus), so the two
        # sides of the W1 comparison use ONE convention.
        a = 1.0 / ve
        b = te
        va = a - a.mean()
        vb = b - b.mean()
        L = -float(np.dot(va, vb) / np.dot(va, va))
        T = b + a * L
        c2 = float(np.polyfit(s0s - s_beam, T, 2)[0])   # us/mm^2
        return L, c2, (s0s, te, ve)

    def turnaround_ns(self, s_beam, mz, K_th_eV):
        """The WM turnaround: dt between the +v and -v thermal twins."""
        a1 = ACC * self.E(s_beam) / mz
        v = np.sqrt(2.0 * ACC * K_th_eV / mz)
        return 2.0 * v / a1 * 1e3          # ns


# ------------------------------------------- the classic two-field check
def wm_two_field_sf(s0, d, E1, E2):
    """Wiley & McLaren 1955, eq. for the first-order space focus of a
    two-field source: ion born a distance s0 before the first grid, field
    E1; acceleration region length d, field E2; drift field-free.  Returns
    the drift length D to the space focus.  (k0 is the total-energy ratio.)
    """
    k0 = (s0 * E1 + d * E2) / (s0 * E1)
    return 2.0 * s0 * k0**1.5 * (1.0 - (d / s0) / (k0 + np.sqrt(k0)))


def two_field_model(s0, d, E1, E2, back=None):
    """The same two-field source as an AxisModel, for the identity check:
    the generalised integrator must reproduce the 1955 closed form exactly.

    Frame: the ion is born at s = 0, a distance s0 BEFORE grid 1; the source
    region (field E1) extends back to a repeller at s = -back, so a fan of
    birth positions around 0 stays inside a UNIFORM E1 -- which is the WM
    premise.  Zero-thickness grids get a token 1e-6 mm of metal so every
    interval is non-degenerate."""
    back = 2.0 * s0 if back is None else back
    t = 1e-6
    phi_rep = (s0 + back) * E1 + d * E2
    return AxisModel([(-back - t, -back, phi_rep),
                      (s0, s0 + t, d * E2),
                      (s0 + d, s0 + d + t, 0.0)])
