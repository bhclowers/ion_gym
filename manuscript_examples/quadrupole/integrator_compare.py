"""Velocity Verlet (production planar route) against RK4 (tracer_tw2d) on
the manuscript quadrupole: same solved bases, same drive channels, same
birth state. The findings are written up in INTEGRATOR_COMPARISON.md
beside this script, and the banked output is integrator_compare.out.txt.

Sections:
  A. zone 3,1 band scan (17 voltages x 16 phases) + starting-offset
     ladder, dt 0.1 ns (the Figure 4B operating point)
  B. 13 x 13 (a, q) grid of zone 1, sinusoidal, dt 1 ns (Figure 4A)
  C. secular-frequency ladder q = 0.10..0.80 against the transfer matrix
  D. dt-halving ladders at two production points: self-convergence per
     integrator and the cross-integrator gap at the production step
  E. the disagreeing ladder cells located against the peak-excursion
     threshold, each re-flown at half the production step

Run:  python manuscript_examples/quadrupole/integrator_compare.py
A measurement probe, not a gate: it registers nowhere in tiers.py.
"""
import sys
import time
from pathlib import Path

import numpy as np

# sibling modules (quad_lib, the figure producers) live beside this probe
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import quad_lib as QL                                  # noqa: E402
import make_fig4_stability as F                        # noqa: E402
from make_digital_quad_deck import ZONES               # noqa: E402
from ion_gym.physics.sim_build import build_run        # noqa: E402
from ion_gym.physics.build_planar import V_BASIS       # noqa: E402
from ion_gym.physics.tracer_tw2d import (build_tw2d_fields,  # noqa: E402
                                         fly_tw2d_ion)
from ion_gym.physics.mathieu import hill_beta_njit     # noqa: E402

T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:6.1f} s] {msg}", flush=True)


def planar_fly(sp):
    """(traj, cols-index, summary, births0, model) through the production
    planar route (velocity Verlet)."""
    errs = sp.validate()
    if errs:
        raise ValueError("spec refused:\n  " + "\n  ".join(errs))
    model, fly, cols, births = build_run(sp)
    tr, st = fly(0)
    ci = {c: j for j, c in enumerate(cols)}
    return np.asarray(tr), ci, st, births[0], model


def tw2d_fields_for(sp, model):
    assign = {}
    dc = {}
    for k, el in enumerate(sp.geometry.electrodes, start=1):
        if k not in model.bases:
            raise AssertionError(f"electrode index {k} missing from bases "
                                 f"{sorted(model.bases)}")
        if len(el.rf_groups) != 1:
            raise AssertionError(f"electrode {el.name!r} carries "
                                 f"{len(el.rf_groups)} groups; this probe "
                                 f"maps exactly one")
        assign[k] = el.rf_groups[0]
        dc[k] = float(el.dc)
    unit = {k: b / V_BASIS for k, b in model.bases.items()}
    return build_tw2d_fields(unit, sp.geometry.rf_groups, assign, dc,
                             h_mm=model.mm_per_gu,
                             ele=np.asarray(model.ele, bool))


def both_fly(sp):
    """Fly one spec on both integrators from the identical birth state.
    Returns (planar traj, ci, planar summary, tw2d out dict)."""
    tr, ci, st, b0, model = planar_fly(sp)
    bx, by, bz, bvx, bvy, bvz, btob = (float(v) for v in b0)
    f2 = tw2d_fields_for(sp, model)
    o2 = fly_tw2d_ion(sp, f2, 0, r0_mm=(bx, by),
                      v0_mm_us=(bvx, bvy, bvz), tob_us=btob,
                      dt_ns=sp.integration.dt_ns,
                      t_max_us=sp.integration.t_max_us,
                      record_every=sp.integration.rec_every)
    return tr, ci, st, o2


def survived_pair(sp):
    tr, ci, st, o2 = both_fly(sp)
    return st["kind"] == 2, int(o2["kind"]) == 2


say("calibrating v_per_q from the solved field")
v_per_q, q_cal = QL.calibrate_v_per_q(f_rf_mhz=F.F_RF_MHZ, v_cal=F.V_CAL,
                                      mz=F.MZ)
say(f"v_per_q = {v_per_q:.6f} V (q at {F.V_CAL} V: {q_cal:.6f})")

duty = ZONES[F.ZONE]["duty"]
lo, hi = QL.zone_band_a0(duty, ZONES[F.ZONE]["q"])
qc = 0.5 * (lo + hi)
say(f"zone {F.ZONE} band [{lo:.6f}, {hi:.6f}], centre {qc:.6f}")

phases = np.linspace(0.0, 1.0, F.N_PHASE, endpoint=False)

# ---- A. zone 3,1 scan + starting-offset ladder -----------------------------
sv_v = np.zeros((len(F.REL_SCAN), F.N_PHASE), bool)
sv_r = np.zeros_like(sv_v)
for i, rel in enumerate(F.REL_SCAN):
    for j, pf in enumerate(phases):
        sv_v[i, j], sv_r[i, j] = survived_pair(
            F._scan_spec(rel, pf, qc, duty, v_per_q))
say(f"A. scan {sv_v.size} points flown on both")
agree = (sv_v == sv_r)
print("   rel      in-band  verlet/16  rk4/16  disagree-cells")
inband = (F.REL_SCAN * qc >= lo) & (F.REL_SCAN * qc <= hi)
for i, rel in enumerate(F.REL_SCAN):
    print(f"   {rel:.4f}  {str(inband[i]):5}    "
          f"{int(sv_v[i].sum()):2d}        {int(sv_r[i].sum()):2d}      "
          f"{int((~agree[i]).sum())}")
print(f"A. TOTAL disagreeing cells: {int((~agree).sum())} of {agree.size}")

lad_v = []
lad_r = []
for x0 in F.X0_LADDER_MM:
    vv = rr = 0
    for pf in phases:
        a, b = survived_pair(F._scan_spec(1.0, pf, qc, duty, v_per_q, x0))
        vv += a
        rr += b
    lad_v.append(vv)
    lad_r.append(rr)
say(f"A. offset ladder {list(F.X0_LADDER_MM)} mm: "
    f"verlet {lad_v} rk4 {lad_r} (manuscript, verlet: [2, 4, 7, 10])")

# ---- B. 13 x 13 sinusoidal grid -------------------------------------------
n_grid = 13
a_fl = np.linspace(*F.A_GRID_SPAN, n_grid)
q_fl = np.linspace(*F.Q_GRID_SPAN, n_grid)
gv = np.zeros((n_grid, n_grid), bool)
gr = np.zeros_like(gv)
gp = np.zeros_like(gv)
for i, a in enumerate(a_fl):
    for j, q in enumerate(q_fl):
        gv[i, j], gr[i, j] = survived_pair(F._grid_spec(a, q, v_per_q))
        gp[i, j] = QL.sin_stable(a, q)
say(f"B. grid {gv.size} points flown on both")
print(f"B. verlet vs chart disagreements: {int((gv != gp).sum())}")
print(f"B. rk4   vs chart disagreements: {int((gr != gp).sum())}")
print(f"B. verlet vs rk4  disagreements: {int((gv != gr).sum())}")

# ---- C. secular-frequency ladder ------------------------------------------
print("C.   q     f_tm (MHz)   verlet      rel err     rk4         rel err")
for q in F.Q_LADDER:
    sp = QL.quad_spec(q * v_per_q, waveform="sin", f_rf_mhz=F.F_RF_MHZ,
                      mz=F.MZ, t_max=F.T_LADDER_US, dt_ns=F.DT_A_NS)
    ctr = sp.geometry.width_mm / 2
    tr, ci, st, o2 = both_fly(sp)
    if st["kind"] != 2 or int(o2["kind"]) != 2:
        raise RuntimeError(f"ladder ion lost at q = {q}: planar "
                           f"{st['kind']}, tw2d {o2['kind']}")
    fv = QL.f_zero_cross(tr, ci, ctr, F.F_RF_MHZ)
    tr2 = np.column_stack([np.asarray(o2["t_us"]), np.asarray(o2["x"])])
    fr = QL.f_zero_cross(tr2, {"t": 0, "x": 1}, ctr, F.F_RF_MHZ)
    ftm = hill_beta_njit(0.0, q) / 2 * F.F_RF_MHZ
    print(f"C. {q:5.2f}  {ftm:.6f}   {fv:.6f}  {abs(fv/ftm-1):.2e}  "
          f"{fr:.6f}  {abs(fr/ftm-1):.2e}")
say("C. ladder done")


# ---- D. dt-halving ladders -------------------------------------------------
def end_state(sp_maker, kernel, dt_ns, t_max, rec_us):
    """Final recorded (x, y) at t = t_max under one integrator.
    rec_us: record cadence in us, held fixed across dt so the final
    record lands at the same lab time."""
    sp = sp_maker()
    sp.integration.dt_ns = dt_ns
    sp.integration.rec_every = max(1, int(round(rec_us * 1e3 / dt_ns)))
    sp.integration.t_max_us = t_max
    if kernel == "verlet":
        tr, ci, st, b0, model = planar_fly(sp)
        if st["kind"] != 2:
            return None
        return float(tr[-1, ci["x"]]), float(tr[-1, ci["y"]])
    tr, ci, st, b0, model = planar_fly(sp)        # births + fields
    bx, by, bz, bvx, bvy, bvz, btob = (float(v) for v in b0)
    f2 = tw2d_fields_for(sp, model)
    o2 = fly_tw2d_ion(sp, f2, 0, r0_mm=(bx, by),
                      v0_mm_us=(bvx, bvy, bvz), tob_us=btob,
                      dt_ns=dt_ns, t_max_us=t_max,
                      record_every=sp.integration.rec_every,
                      max_records=2000)
    if int(o2["kind"]) != 2:
        return None
    return float(np.asarray(o2["x"])[-1]), float(np.asarray(o2["y"])[-1])


def dt_ladder(label, sp_maker, dts, t_max, rec_us, production_dt):
    print(f"D. {label}: t_max {t_max} us, records every {rec_us} us")
    ends = {}
    for k in ("verlet", "rk4"):
        ends[k] = {dt: end_state(sp_maker, k, dt, t_max, rec_us)
                   for dt in dts}
        row = []
        for d1, d2 in zip(dts[:-1], dts[1:]):
            e1, e2 = ends[k][d1], ends[k][d2]
            if e1 is None or e2 is None:
                row.append("lost")
                continue
            row.append(f"{np.hypot(e1[0]-e2[0], e1[1]-e2[1]):.3e}")
        print(f"D.   {k:6}: |end(dt) - end(dt/2)| mm over {dts}: {row}")
    for dt in dts:
        ev, er = ends["verlet"][dt], ends["rk4"][dt]
        if ev is None or er is None:
            print(f"D.   dt {dt} ns: lost on one integrator "
                  f"(verlet {ev is not None}, rk4 {er is not None})")
            continue
        gap = np.hypot(ev[0] - er[0], ev[1] - er[1])
        tag = "  <-- production" if dt == production_dt else ""
        print(f"D.   dt {dt} ns: verlet-vs-rk4 end gap {gap:.3e} mm{tag}")


# D1: sinusoidal zone 1 interior point (a = 0, q = 0.5), production dt 1 ns
dt_ladder("sin zone 1 (a=0, q=0.5)",
          lambda: F._grid_spec(0.0, 0.5, v_per_q),
          [4.0, 2.0, 1.0, 0.5, 0.25], F.T_GRID_US, 4.0, 1.0)

# D2: zone 3,1 band centre at a transmitted phase, production dt 0.1 ns
surv_idx = [j for j in range(F.N_PHASE) if sv_v[len(F.REL_SCAN) // 2, j]]
pf_ok = phases[surv_idx[0]]
say(f"D2 uses transmitted phase {pf_ok:.4f} at the band centre")
dt_ladder(f"digital zone 3,1 centre, phase {pf_ok:.4f}",
          lambda: F._scan_spec(1.0, pf_ok, qc, duty, v_per_q),
          [0.4, 0.2, 0.1, 0.05], F.T_SCAN_US, 2.0, 0.1)

# ---- E. disagreeing ladder cells against the excursion threshold -----------
print("E. ladder cells that disagree, or sit in the 1.36-1.50 mm band")
print("E. x0_um  phase   peak_mm   verlet  rk4   note")
for x0 in F.X0_LADDER_MM:
    for pf in phases:
        pk = F._peak_mm(qc, duty, pf, x0)
        sv, sr = survived_pair(F._scan_spec(1.0, pf, qc, duty, v_per_q, x0))
        if sv == sr and not (1.36 <= pk <= 1.50):
            continue
        note = "DISAGREE" if sv != sr else "in the undecided band"
        print(f"E. {x0*1e3:5.1f}  {pf:.4f}  {pk:8.4f}  {str(sv):5}  "
              f"{str(sr):5}  {note}")
        if sv != sr:
            sp = F._scan_spec(1.0, pf, qc, duty, v_per_q, x0)
            sp.integration.dt_ns = 0.05
            sp.integration.rec_every *= 2
            tr, ci, st, b0, model = planar_fly(sp)
            f2 = tw2d_fields_for(sp, model)
            bx, by, bz, bvx, bvy, bvz, btob = (float(v) for v in b0)
            o2 = fly_tw2d_ion(sp, f2, 0, r0_mm=(bx, by),
                              v0_mm_us=(bvx, bvy, bvz), tob_us=btob,
                              dt_ns=0.05,
                              t_max_us=sp.integration.t_max_us,
                              record_every=sp.integration.rec_every)
            print(f"E.        at dt 0.05 ns: verlet {st['kind'] == 2}, "
                  f"rk4 {int(o2['kind']) == 2}")

say("all sections done")
