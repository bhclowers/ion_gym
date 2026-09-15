"""
validate_collisions_c1.py — C-1 gates for the HS collision kernel,
run at an IMS benchmark's conditions (Gillig et al.,
C60: m = 720 u in He at 1 Torr / 298 K), where the example's own
sigma = 1.25e-18 m^2 was DERIVED from the measured K0 = 4.31 cm^2/V/s
via Mason-Schamp — so hard-sphere theory closes exactly on published
experiment and every gate has an absolute target.

  C1-a THERMALIZATION: hot ions (3 mm/us ~ 33 eV), zero field. After
       equilibration the ensemble must satisfy <KE> = (3/2) kT and
       per-axis velocity variance kT/m. GATE: both within 5% / 8%
       (predicted standard errors printed; ~3,500 collisions/ion).
       This is the moment the flux-biased partner sampling exists for —
       unbiased Maxwell partners equilibrate measurably cold (the publishee
       issue I362).
  C1-b MOBILITY: uniform E = 0.2 V/mm (7 Td — genuinely low-field,
       unlike the example's own 50 V/cm ~ 150 Td drift cell, which is
       a focusing demo, not a mobility benchmark). Ensemble drift speed
       vs v_d = K E with K = K0 (P0/P)(T/T0) = 3.574e3 cm^2/V/s.
       GATE: within 5% (statistical SE + first-order Chapman-Enskog is
       good to ~2% for hard spheres).
  C1-c DIFFUSION (the requested gate), two forms:
       (i)  absolute: field-free ensemble variance growth, pooled over
            3 axes: slope = 2D vs Einstein D = kT K / q. GATE: 10%.
       (ii) INTERNAL Einstein closure: D_meas q / (kT K_meas) = 1 using
            C1-b's measured K — fluctuation-dissipation consistency of
            the kernel itself, independent of Chapman-Enskog accuracy.
            GATE: 10%.

Runtime ~2-4 min (serial deterministic ensembles, numba).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.physics.collision3d import ensemble, KB, E_CHG

M_ION, Q = 720.0, 1.0
M_GAS = 4.0
T, P, SIGMA = 298.0, 133.28, 1.25e-18
K0 = 4.31                                    # cm^2/V/s (Gillig)
K = K0 * (760.0 / 1.0) * (T / 273.15)        # cm^2/V/s at 1 Torr, 298 K
K_mm = K * 1e2 / 1e6                         # mm^2/(V*us)
D_TH = KB * T / E_CHG * K_mm                 # mm^2/us (Einstein)
KT_EV = KB * T / E_CHG


def c1a():
    recs, cols = ensemble(240, M_ION, Q, (0, 0, 0), (0, 0, 3.0),
                          (0, 0, 0), T, P, SIGMA, dt_us=4e-4, t_us=60.0,
                          record_every=2500, seed0=11)
    # pool velocities from the equilibrated second half
    V = np.concatenate([r[r[:, 0] > 30.0, 4:7] for r in recs])
    m_kg = M_ION * 1.6605402e-27
    ke = 0.5 * m_kg * (V ** 2).sum(axis=1) * 1e6 / E_CHG   # eV
    ke_ratio = ke.mean() / (1.5 * KT_EV)
    var_th = KB * T / m_kg * 1e-6                          # (mm/us)^2
    vr = (V.var(axis=0) / var_th)
    n_eff = len(recs) * 3                # ~decorrelated snapshots/ion
    se = np.sqrt(2.0 / 3.0 / n_eff)
    ok = abs(ke_ratio - 1) < 0.05 and np.all(np.abs(vr - 1) < 0.08)
    print(f"C1-a thermalization ({int(cols.mean())} collisions/ion): "
          f"<KE>/(3kT/2) = {ke_ratio:.4f} (SE~{se:.3f}); per-axis "
          f"var/(kT/m) = {vr[0]:.3f},{vr[1]:.3f},{vr[2]:.3f}  ->  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def c1b():
    E = 0.2                                                # V/mm
    recs, _ = ensemble(240, M_ION, Q, (0, 0, 0), (0, 0, 0),
                       (0, 0, E), T, P, SIGMA, dt_us=4e-4, t_us=150.0,
                       record_every=2500, seed0=71)
    t = recs[0][:, 0]
    Z = np.stack([r[:, 3] for r in recs])
    zm = Z.mean(axis=0)
    sel = t > 30.0
    vd, _ = np.polyfit(t[sel], zm[sel], 1)                 # mm/us
    vd_th = K_mm * E
    # SE of the drift estimate from ensemble spread of endpoint slopes
    vds = (Z[:, -1] - Z[:, sel.argmax()]) / (t[-1] - t[sel.argmax()])
    se = vds.std() / np.sqrt(len(recs)) / vd_th
    ok = abs(vd / vd_th - 1) < 0.05
    print(f"C1-b mobility: v_d = {vd:.5f} mm/us vs K*E = {vd_th:.5f} "
          f"(K = {K:.0f} cm2/V/s from Gillig K0 = {K0}); ratio "
          f"{vd/vd_th:.4f} (SE~{se:.3f})  ->  {'PASS' if ok else 'FAIL'}")
    return ok, vd / E                                       # K_meas mm units


def c1c(K_meas_mm):
    # 600 ions / 220 us with bootstrap SE: a 240-ion first draft drew a
    # 1.2-sigma-low D (ratio 0.923) — worth chasing because Einstein's
    # relation is EXACT at zero field by fluctuation-dissipation, so a
    # real deficit would be a kernel bug; the high-statistics run gave
    # 1.021 +- 0.039, i.e. noise. Gate kept at 10% with SE ~4%.
    recs, _ = ensemble(600, M_ION, Q, (0, 0, 0), (0, 0, 0),
                       (0, 0, 0), T, P, SIGMA, dt_us=4e-4, t_us=220.0,
                       record_every=2500, seed0=977)
    t = recs[0][:, 0]
    R = np.stack([r[:, 1:4] for r in recs])                # (N, nt, 3)
    sel = t > 30.0                                          # equilibrated
    var = R[:, sel, :].var(axis=0)                          # (nt_sel, 3)
    ts = t[sel]
    slopes = [np.polyfit(ts, var[:, a], 1)[0] for a in range(3)]
    D = np.mean(slopes) / 2.0                               # mm^2/us
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(120):
        idx = rng.choice(len(recs), len(recs))
        v = R[idx][:, sel, :].var(axis=0)
        boots.append(np.mean([np.polyfit(ts, v[:, a], 1)[0]
                              for a in range(3)]) / 2.0)
    se = np.std(boots) / D_TH
    r1 = D / D_TH
    r2 = D * E_CHG / (KB * T) / K_meas_mm * 1e0             # Einstein closure
    # units: D[mm^2/us] * q/(kT) [1/V] / K[mm^2/(V us)] -> dimensionless
    g1 = abs(r1 - 1) < 0.10
    g2 = abs(r2 - 1) < 0.10
    print(f"C1-c diffusion: D = {D:.5f} mm2/us vs Einstein(K0) "
          f"{D_TH:.5f}; ratio {r1:.4f} (SE~{se:.3f})  ->  "
          f"{'PASS' if g1 else 'FAIL'}")
    print(f"     Einstein closure D q/(kT K_meas) = {r2:.4f}  ->  "
          f"{'PASS' if g2 else 'FAIL'}   [fluctuation-dissipation "
          f"consistency, Chapman-Enskog-independent]")
    return g1 and g2


def main():
    a = c1a()
    b, K_meas = c1b()
    c = c1c(K_meas)
    print("\nC-1 COLLISION KERNEL GATES:",
          "ALL PASS" if (a and b and c) else "FAIL")


if __name__ == "__main__":
    main()
