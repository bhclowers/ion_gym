"""test_ionbench_native_equiv.py — native kernel-equivalence gate.

ionbench_numba.fly_ionbench_numba reproduces the Python orchestrator
(ionbench.fly_ionbench) to machine precision, and is fast — the SAME
contract the retired numba-equiv test enforced (see
tests/retired/RETIRED.md), reconstructed natively: the
the assembly here is two FAs solved by ion_gym's own solver (solve3d, nz=1,
native 'sw' stencil) and placed with nontrivial rigid transforms.

The equivalence claim needs ANY multi-FA. What it
DOES need is every kernel code path exercised, so the fixture deck is
chosen (and ASSERTED, so the gate cannot silently degenerate) to cover:
  * a metal impact (the 40-iteration bisection),
  * an assembly-bounds exit,
  * a multi-instance traversal (seam trigger fired both ways),
  * a pure free-flight ion that never enters an instance,
  * two masses,
and the two instances differ in scale, rotation, and translation.

Bars (identical to the retired gate; a port, not a reimplementation):
  EQUIVALENCE  per ion: dTOF < 1e-6 ns, dimpact < 1e-3 nm, dKE < 1e-6 eV,
               impact kind name identical.
  SPEEDUP      numba >= 20x the Python orchestrator per ion.

One-time cross-check: this gate's measurement, run
verbatim on a legacy tof fixture before it was dropped, reproduced
the retired gate's PASS — recorded, fixture deleted.

Run: python tests/test_ionbench_native_equiv.py   (~1-3 min incl JIT)
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import time

import numpy as np

from ion_gym.physics.ionbench import PlacedFA, Ionbench2D, fly_ionbench
from ion_gym.physics.ionbench_numba import flatten_ionbench, fly_ionbench_numba, impact_name
from ion_gym.physics.solver3d import solve3d

FAILED = []


def gate(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
    if not ok:
        FAILED.append(n)


# ---------------------------------------------------------------- fixture
def _solve_pa(fixed2d, val2d):
    """Native nz=1 solve -> (phi, ele) 2-D arrays, PlacedFA-ready."""
    fixed = fixed2d[:, :, None]
    val = val2d[:, :, None]
    phi, sweeps, delta = solve3d(fixed, val, stencil="sw", tol=1e-9)
    return phi[:, :, 0], fixed2d


def build_native_assembly():
    """Two native PAs.  Local frames are half-planes (row v=0 is the axis;
    ionbench mirror-extends about it, same as the tof convention).

    accel   81x41 @ 0.5 mm/gu: 1000 V plate (holed) then grounded plate ->
            ~1 keV/charge axial kick, then drift.
    reflect 61x41 @ 0.4 mm/gu: grounded holed entrance column, solid back
            wall at +1300 V -> reflects the ~1 keV ions, impacts a 2 keV one.
    Placed with differing scale (0.5 vs 0.4), rotation (0 vs 170 deg) and
    translation — the transform machinery under test.
    """
    HOLE = 8  # gu, axis hole radius on every apertured plate

    f = np.zeros((81, 41), bool)
    v = np.zeros((81, 41))
    f[5:8, HOLE:] = True          # plate A (holed), 1000 V
    v[5:8, HOLE:] = 1000.0
    f[30:33, HOLE:] = True        # plate B (holed), grounded
    accel_phi, accel_ele = _solve_pa(f, v)

    f = np.zeros((61, 41), bool)
    v = np.zeros((61, 41))
    f[2:5, 12:] = True            # grounded entrance column (4.8 mm hole:
                                  # the tilted round trip displaces the
                                  # reflected ion laterally BOTH ways)
    f[54:57, :] = True            # solid back wall, +1600 V
    v[54:57, :] = 1600.0
    refl_phi, refl_ele = _solve_pa(f, v)

    pas = [
        PlacedFA("accel", accel_phi, accel_ele, 0.5, 0.0, (0.0, 0.0),
                 "cylindrical"),
        # entrance FACES the beam (an earlier 170 deg placement presented
        # the back corner and every axial ion impacted; at 8 deg the
        # REFLECTED ion returned displaced and impacted on the entrance
        # column — COV.multi_seam caught both). 3 deg keeps the rotation
        # matrix nontrivially exercised and the round trip threads the
        # 4.8 mm entrance hole both ways.
        PlacedFA("reflect", refl_phi, refl_ele, 0.4, 3.0, (50.0, -0.2),
                 "cylindrical"),
    ]
    wb_box = (-30.0, -30.0, 0.0, 130.0, 60.0, 0.0)   # iob-style 6-tuple
    return wb_box, Ionbench2D(pas)


# deck: (name, mz_Da, x0, y0, vx0_mm_us, vy0_mm_us)  [wb mm, mm/us]
# born-KE arithmetic (accel adds ~1000 eV; wall 1600 V):
#   300 eV born -> 1300 eV at the wall: REFLECTS, re-climbs accel in
#   reverse, leaves with ~300 eV through the left bound (>=4 seams);
#   1200 eV born -> 2200 eV: PENETRATES, metal-impacts on the wall.
DECK = [
    ("axial_reflected", 100.0, 4.0, 0.0, 24.05, 0.0),   # ~300 eV born
    ("axial_wall_impact", 100.0, 4.0, 0.0, 48.1, 0.0),   # ~1200 eV born
    ("heavy_reflected", 500.0, 4.0, 0.0, 10.76, 0.0),   # ~300 eV, m/z 500
    ("offaxis_plate_impact", 100.0, 1.0, 1.0, 5.0, 20.0),
    ("free_flight_bounds", 100.0, 20.0, 40.0, -5.0, 5.0),
]


def main():
    print("M-1 NATIVE EQUIVALENCE — ionbench_numba vs ionbench.fly_ionbench")
    t0 = time.time()
    wb_box, wb = build_native_assembly()
    flat = flatten_ionbench(wb)
    print(f"  native two-FA assembly solved + flattened in "
          f"{time.time()-t0:.0f} s (solve3d nz=1, 'sw' stencil)")

    # warm the JIT off the clock
    s = DECK[0]
    fly_ionbench_numba(flat, s[1], s[2], s[3], s[4], s[5],
                        dt_ns=1.0, t_max_us=60.0, record_every=10,
                        bounds=wb_box)

    print("\nEQUIVALENCE (bars: dTOF<1e-6 ns, dimpact<1e-3 nm, dKE<1e-6 eV, "
          "impact name identical)")
    wt = wp = wk = 0.0
    kinds = {}
    seams = {}
    for name, mz, x0, y0, vx, vy in DECK:
        rp = fly_ionbench(wb, mz, x0, y0, vx, vy, dt_ns=1.0, t_max_us=60.0,
                           record_every=10, bounds=wb_box)
        rn = fly_ionbench_numba(flat, mz, x0, y0, vx, vy, dt_ns=1.0,
                                 t_max_us=60.0, record_every=10,
                                 bounds=wb_box)
        dt = abs(rp["tof_us"] - rn["tof_us"]) * 1e3            # us -> ns
        dp = np.hypot(rp["x"][-1] - rn["x"][-1],
                      rp["y"][-1] - rn["y"][-1]) * 1e6          # mm -> nm
        dk = abs(rp["KE_eV"] - rn["KE_eV"])
        nm = impact_name(wb, rn["impact_kind"])
        sp_ok = nm == rp["impact"]
        kinds[name] = rp["impact"]
        # instance visits from the python trajectory (fixture-coverage audit)
        seams[name] = sum(
            1 for i in range(1, len(rp["x"]))
            for pa in wb.instances
            if pa.contains(rp["x"][i], rp["y"][i])
            != pa.contains(rp["x"][i - 1], rp["y"][i - 1]))
        wt, wp, wk = max(wt, dt), max(wp, dp), max(wk, dk)
        gate(f"EQ.{name}", dt < 1e-6 and dp < 1e-3 and dk < 1e-6 and sp_ok,
             f"dTOF {dt:.2e} ns, dimpact {dp:.2e} nm, dKE {dk:.2e} eV, "
             f"impact '{rp['impact']}'{'' if sp_ok else ' vs numba ' + nm}")
    print(f"  worst: dTOF {wt:.2e} ns, dimpact {wp:.2e} nm, dKE {wk:.2e} eV")

    # the fixture must actually exercise the paths the port could break on
    print("\nFIXTURE COVERAGE (asserted, so the gate cannot degenerate)")
    inst_names = {pa.name for pa in wb.instances}   # DECLARED, not sniffed
    has_metal = any(k in inst_names for k in kinds.values())
    has_bounds = any(k not in inst_names for k in kinds.values())
    gate("COV.metal_impact", has_metal, f"impacts: {kinds}")
    gate("COV.bounds_exit", has_bounds, "at least one assembly-bounds exit")
    gate("COV.multi_seam", max(seams.values()) >= 3,
         f"seam crossings per ion: {seams} (>=3 somewhere: instance "
         f"entered AND left AND re-entered or a second instance reached)")
    gate("COV.free_flight", seams["free_flight_bounds"] == 0,
         "one ion never enters any instance (pure drift + bounds)")

    print("\nSPEEDUP (bar: >= 20x)")
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        for name, mz, x0, y0, vx, vy in DECK:
            fly_ionbench(wb, mz, x0, y0, vx, vy, dt_ns=1.0, t_max_us=60.0,
                          record_every=10, bounds=wb_box)
    tp = (time.perf_counter() - t0) / (reps * len(DECK))
    t0 = time.perf_counter()
    for _ in range(reps):
        for name, mz, x0, y0, vx, vy in DECK:
            fly_ionbench_numba(flat, mz, x0, y0, vx, vy, dt_ns=1.0,
                                t_max_us=60.0, record_every=10,
                                bounds=wb_box)
    tn = (time.perf_counter() - t0) / (reps * len(DECK))
    gate("SPEEDUP", tp / tn >= 20.0,
         f"python {tp*1e3:.1f} ms/ion, numba {tn*1e3:.2f} ms/ion -> "
         f"{tp/tn:.0f}x")

    print()
    if FAILED:
        print("M-1 NATIVE EQUIVALENCE: FAIL ->", ", ".join(FAILED))
        return 1
    print("M-1 NATIVE EQUIVALENCE: ALL PASS — ionbench_numba certified "
          "against the Python orchestrator on a fully native assembly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
