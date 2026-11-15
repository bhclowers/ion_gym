"""Figure 4 of the ion_gym framework manuscript: the stability charts of
the rod quadrupole checked in flight.

Panel A repeats the 49-point flown (a, q) grid of sin_stability.ipynb over
the first stability zone of the sinusoidal drive, computed from the
one-period transfer matrix. Panel B repeats the zone 3,1 transmission scan
of digital_stability.ipynb, flown across the band computed from the exact
Meissner transfer matrix. The operating points below are the notebooks'
parameter cells, and the flights call the same quad_lib functions, so the
figure and the notebooks fly the same ions.

The flights are banked once (io.results_bank) and the figure is rendered
from the bank, so a restyle never refies. A bank is never overwritten.
Delete it by hand to refly.

The run also prints the two numbers the Figure 4 paragraph quotes, the
largest relative error of the flown secular frequency on the zone 1 ladder
and the number of grid points where flight and chart disagree.

Run:  python manuscript_examples/quadrupole/make_fig4_stability.py
Run:  ... make_fig4_stability.py --grid 13   (denser flown grid)
Writes outputs/manuscript/fig4_stability_g<N>.{npz,png,pdf} under the repo
root.
"""
from __future__ import annotations

import os
import time

import numpy as np

import quad_lib as QL
from make_digital_quad_deck import ZONES

from ion_gym.io.paths import outputs_dir
from ion_gym.io.results_bank import bank_run, load_bank
from ion_gym.physics.mathieu import hill_beta_njit, hill_half_trace_njit
from ion_gym.viz.stability_figure import (BandScan, StabilityChart,
                                          stability_flight_figure)
from ion_gym.viz.viz_core import apply_house_style, save_figure

# ---- shared (both notebooks) ---------------------------------------------
MZ = 100.0              # Th
F_RF_MHZ = 1.0          # MHz
V_CAL = 30.0            # V, curvature calibration amplitude

# ---- panel A: sin_stability.ipynb ----------------------------------------
NA, NQ = 161, 161                     # computed chart grid
A_SPAN, Q_SPAN = (-0.30, 0.30), (0.0, 1.0)
N_GRID = 7                            # flown points per axis (notebook: 7)
A_GRID_SPAN = (-0.25, 0.25)           # flown a range
Q_GRID_SPAN = (0.15, 0.95)            # flown q range
X0_A_MM, Y0_A_MM = 0.10, 0.07         # launch offset from the axis
T_GRID_US = 80.0                      # survival time of a grid flight
DT_A_NS = 1.0
Q_LADDER = (0.10, 0.25, 0.40, 0.60, 0.80)   # secular-frequency ladder
T_LADDER_US = 160.0

# ---- panel B: digital_stability.ipynb ------------------------------------
ZONE = "3,1"
N_PHASE = 16
REL_SCAN = np.linspace(0.9960, 1.0040, 17)  # q / q_center
X0_B_MM, Y0_B_MM = 0.010, 0.007
T_SCAN_US = 50.0
DT_B_NS = 0.1
# start offsets flown at the zone center to show that transmission is set
# by the size of the orbit (y offset keeps the Y0_B_MM / X0_B_MM ratio)
X0_LADDER_MM = (0.020, 0.010, 0.005, 0.0025)

SHOW_NOTE = False       # operating notes under the panels (the manuscript
                        # caption carries them, so the print omits them)

OUT_STEM = "fig4_stability"           # files are <stem>_g<N_GRID>.*


def _survives(sp):
    return QL.fly_one(sp)[3]["kind"] == 2


def _grid_spec(a, q, v_per_q):
    return QL.quad_spec(q * v_per_q, waveform="sin", f_rf_mhz=F_RF_MHZ,
                        mz=MZ, dc_v=QL.dc_for_a(a, v_per_q), x0=X0_A_MM,
                        y0=Y0_A_MM, t_max=T_GRID_US, dt_ns=DT_A_NS)


def _scan_spec(rel, pf, qc, duty, v_per_q, x0=X0_B_MM):
    return QL.quad_spec(rel * qc * v_per_q, waveform="square", duty=duty,
                        phase_frac=pf, f_rf_mhz=F_RF_MHZ, mz=MZ,
                        x0=x0, y0=x0 * Y0_B_MM / X0_B_MM, t_max=T_SCAN_US,
                        dt_ns=DT_B_NS)


def _peak_mm(q, duty, pf, x0):
    """Largest ideal-field excursion (mm) on either axis over the flight
    of one scan ion, from the exact digital-drive propagators."""
    n_per = int(round(T_SCAN_US * F_RF_MHZ))
    y0 = x0 * Y0_B_MM / X0_B_MM
    return max(QL.meissner_peak_excursion(0.0, q, duty, pf, x0, n_per,
                                          axis="x"),
               QL.meissner_peak_excursion(0.0, q, duty, pf, y0, n_per,
                                          axis="y"))


def fly_and_bank(path, n_grid=N_GRID):
    """Compute both charts, fly both checks, and bank every array."""
    t0 = time.time()
    v_per_q, q_cal = QL.calibrate_v_per_q(f_rf_mhz=F_RF_MHZ, v_cal=V_CAL,
                                          mz=MZ)
    aa = np.linspace(*A_SPAN, NA)
    qq = np.linspace(*Q_SPAN, NQ)
    margin = np.array([[np.log10(max(abs(hill_half_trace_njit(a, q)),
                                     abs(hill_half_trace_njit(-a, q)),
                                     1e-12)) for q in qq] for a in aa])
    a_fl = np.linspace(*A_GRID_SPAN, n_grid)
    q_fl = np.linspace(*Q_GRID_SPAN, n_grid)
    ag, qg = np.meshgrid(a_fl, q_fl, indexing="ij")
    survived = np.array([[_survives(_grid_spec(a, q, v_per_q))
                          for q in q_fl] for a in a_fl])
    predicted = np.array([[QL.sin_stable(a, q) for q in q_fl]
                          for a in a_fl])

    f_flown, f_tm = [], []
    for q in Q_LADDER:
        sp = QL.quad_spec(q * v_per_q, waveform="sin", f_rf_mhz=F_RF_MHZ,
                          mz=MZ, t_max=T_LADDER_US, dt_ns=DT_A_NS)
        _m, tr, ci, st = QL.fly_one(sp)
        if st["kind"] != 2:
            raise RuntimeError(f"ladder ion at q = {q} was lost at "
                               f"{st['tof']:.1f} us inside zone 1")
        f_flown.append(QL.f_zero_cross(tr, ci, sp.geometry.width_mm / 2,
                                       F_RF_MHZ))
        f_tm.append(hill_beta_njit(0.0, q) / 2 * F_RF_MHZ)

    duty = ZONES[ZONE]["duty"]
    lo, hi = QL.zone_band_a0(duty, ZONES[ZONE]["q"])
    qc = 0.5 * (lo + hi)
    phases = np.linspace(0.0, 1.0, N_PHASE, endpoint=False)
    surv_phase = np.array([[_survives(_scan_spec(r, pf, qc, duty, v_per_q))
                            for pf in phases] for r in REL_SCAN])
    trans = surv_phase.mean(axis=1)
    peak_phase = np.array([[_peak_mm(r * qc, duty, pf, X0_B_MM)
                            for pf in phases] for r in REL_SCAN])
    surv_ladder = np.array([[_survives(_scan_spec(1.0, pf, qc, duty,
                                                  v_per_q, x0))
                             for pf in phases] for x0 in X0_LADDER_MM])
    peak_ladder = np.array([[_peak_mm(qc, duty, pf, x0) for pf in phases]
                            for x0 in X0_LADDER_MM])
    print(f"flown and computed in {time.time() - t0:.0f} s")

    meta = {"operating_point": (f"rod quadrupole r0 {QL.R0_MM} mm, m/z "
                                f"{MZ:g}, {F_RF_MHZ:g} MHz; A sin drive "
                                f"grid; B digital duty {duty} zone {ZONE}"),
            "v_per_q": v_per_q, "q_cal": q_cal, "duty": duty,
            "band": [lo, hi], "q_center": qc, "zone": ZONE,
            "n_grid": n_grid}
    arrays = {"a": aa, "q": qq, "margin": margin, "a_flown": ag,
              "q_flown": qg, "survived": survived, "predicted": predicted,
              "q_ladder": np.array(Q_LADDER), "f_flown": np.array(f_flown),
              "f_tm": np.array(f_tm), "rel_scan": REL_SCAN,
              "transmission": trans, "phases": phases,
              "survived_phase": surv_phase, "peak_phase_mm": peak_phase,
              "x0_ladder_mm": np.array(X0_LADDER_MM),
              "survived_ladder": surv_ladder, "peak_ladder_mm": peak_ladder}
    return bank_run(path, meta=meta, arrays=arrays)


def build_figure(path, show_note=SHOW_NOTE):
    """Figure 4 from the bank at path (returns the Figure)."""
    meta, z = load_bank(path)
    v_per_q, qc, duty = meta["v_per_q"], meta["q_center"], meta["duty"]
    lo, hi = meta["band"]
    # the notes are read from specs built exactly as the flights built
    # theirs, so the printed operating point is what the solver received
    sp_a = _grid_spec(0.0, 0.5, v_per_q)
    sp_b = _scan_spec(1.0, 0.0, qc, duty, v_per_q)
    ctr = sp_a.geometry.width_mm / 2
    src_a, src_b = sp_a.source, sp_b.source
    rf_b = sp_b.geometry.rf_groups[0]
    n_grid = z["survived"].size
    note_a = (f"Sinusoidal drive, m/z {src_a.mz_list[0]:g}, "
              f"{sp_a.geometry.rf_groups[0].frequency_hz / 1e6:g} MHz, "
              f"{v_per_q:.1f} V per unit q from the solved field. "
              f"{n_grid} ions, one per point, at rest "
              f"({src_a.x0_mm - ctr:.2f}, {src_a.y0_mm - ctr:.2f}) mm off "
              f"axis; survival is {sp_a.integration.t_max_us:g} µs without "
              f"a rod strike, dt = {sp_a.integration.dt_ns:g} ns. Region "
              f"from the one-period transfer matrix, {NA} × {NQ} grid.")
    note_b = (f"Digital drive, duty {rf_b.duty:g}, m/z "
              f"{src_b.mz_list[0]:g}, {rf_b.frequency_hz / 1e6:g} MHz, "
              f"zone {meta['zone']} at a = 0, q_c = {qc:.4f} "
              f"({rf_b.amplitude_v:.1f} V). {N_PHASE} entry phases per "
              f"point from ({src_b.x0_mm - ctr:.3f}, "
              f"{src_b.y0_mm - ctr:.3f}) mm; survival is "
              f"{sp_b.integration.t_max_us:g} µs, dt = "
              f"{sp_b.integration.dt_ns:g} ns. Band from the exact "
              f"Meissner transfer matrix.")
    chart = StabilityChart(q=z["q"], a=z["a"], margin=z["margin"],
                           q_flown=z["q_flown"], a_flown=z["a_flown"],
                           survived=z["survived"].astype(bool),
                           note=note_a)
    scan = BandScan(x=z["rel_scan"], transmission=z["transmission"],
                    band=(lo / qc, hi / qc), note=note_b,
                    x_label="$q/q_c$")
    apply_house_style()
    return stability_flight_figure(chart, scan, operating_note=show_note)


def report(path):
    """Print the numbers the Figure 4 paragraph quotes, from the bank."""
    meta, z = load_bank(path)
    rel = np.abs(z["f_flown"] / z["f_tm"] - 1.0)
    mism = z["survived"].astype(bool) != z["predicted"].astype(bool)
    print(f"V per q {meta['v_per_q']:.2f} V at {F_RF_MHZ:g} MHz, m/z {MZ:g}")
    for q, f, ft, r in zip(z["q_ladder"], z["f_flown"], z["f_tm"], rel):
        print(f"  ladder q {q:.2f}: flown {f:.5f} MHz, transfer matrix "
              f"{ft:.5f} MHz, rel err {r:.1e}")
    print(f"secular frequency: max relative error {rel.max():.1e}")
    print(f"grid: {int(mism.sum())} of {mism.size} points disagree with "
          f"the chart, {int(z['survived'].sum())} survived")
    for (i, j) in zip(*np.nonzero(mism)):
        print(f"  disagree at (a, q) = ({z['a_flown'][i, j]:+.3f}, "
              f"{z['q_flown'][i, j]:.3f}): flight "
              f"{'survived' if z['survived'][i, j] else 'lost'}")
    lo, hi = meta["band"]
    qc = meta["q_center"]
    print(f"zone {meta['zone']}: band q {lo:.5f} to {hi:.5f}, q_c {qc:.5f}, "
          f"R_BL {qc / (hi - lo):.0f}")
    for r, t in zip(z["rel_scan"], z["transmission"]):
        inb = lo / qc <= r <= hi / qc
        print(f"  q/q_c {r:.4f}: transmission {t:.3f}"
              + ("  (in band)" if inb else ""))
    # why the in-band transmission is partial: the size of the orbit
    inb = (z["rel_scan"] >= lo / qc) & (z["rel_scan"] <= hi / qc)
    sv, pk = z["survived_phase"][inb], z["peak_phase_mm"][inb]
    x0 = float(z["x0_ladder_mm"][list(z["x0_ladder_mm"]).index(X0_B_MM)])
    print(f"in band, start {x0} mm: ideal-field peak excursion "
          f"{pk.min():.3f} to {pk.max():.2f} mm "
          f"(gain {pk.min() / x0:.0f} to {pk.max() / x0:.0f}); largest "
          f"peak that survived {pk[sv].max():.2f} mm, smallest peak lost "
          f"{pk[~sv].min():.2f} mm")
    for x0, svl, pkl in zip(z["x0_ladder_mm"], z["survived_ladder"],
                            z["peak_ladder_mm"]):
        print(f"  center, start {x0:g} mm: {int(svl.sum())}/{svl.size} "
              f"phases transmitted, largest surviving peak "
              f"{pkl[svl].max():.2f} mm, smallest lost peak "
              f"{pkl[~svl].min():.2f} mm")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", type=int, default=N_GRID,
                    help=f"flown points per axis in panel A "
                         f"(default {N_GRID})")
    ap.add_argument("--note", action="store_true",
                    help="draw the operating notes under the panels")
    args = ap.parse_args()
    n = args.grid
    if n < 2:
        raise SystemExit(f"--grid {n}: need at least 2 points per axis")
    stem = f"{OUT_STEM}_g{n}"
    bank = str(outputs_dir("manuscript_examples", "quadrupole", f"{stem}.npz"))
    if os.path.exists(bank):
        print(f"rendering from the existing bank {bank}")
    else:
        os.makedirs(os.path.dirname(bank), exist_ok=True)
        print(f"wrote {fly_and_bank(bank, n)}")
    report(bank)
    fig = build_figure(bank, show_note=args.note or SHOW_NOTE)
    for ext in ("png", "pdf"):
        p = save_figure(fig, outputs_dir("manuscript_examples", "quadrupole", f"{stem}.{ext}"))
        print(f"wrote {p}")
