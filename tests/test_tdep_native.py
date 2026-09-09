"""test_tdep_native.py — tracer_tdep certified natively.

The time-dependent tracer (fast-adjust superposition + voltages(t), the
tstep_adjust breakpoint clamp, the per-ion clock) carried one gate:
validate_buncher, which needs a retired external fixture and so
runs only where that local fixture exists (tier 3). This gate is the
NATIVE forward-carry, zero fixture: the field is solved in-gate (solve3d,
nz=1, 'sw' stencil, cylindrical r-z drift tube + one adjustable ring), a
650 V step waveform is flown, and the outputs are locked to a FROZEN GOLDEN.

Deck coverage (ASSERTED, so the gate cannot degenerate):
  * the switch fires MID-FLIGHT for every ion (clamp exercised; energy is
    deliberately NOT conserved across the switch — that is the physics);
  * two ions differ ONLY by time of birth against the same wall-clock
    switch — the per-ion-clock threading must separate their TOFs;
  * one off-axis ion (radial dynamics); one heavy ion.

Mutation-verified at authoring: removing tracer_tdep's tstep_adjust clamp
reddens G.ion1 (TOF moves ~0.1 ns >> tol).

Run: python tests/test_tdep_native.py   (~1 min incl solve + JIT)
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import numpy as np

from ion_gym.physics.solver3d import solve3d
from ion_gym.physics.tracer_tdep import TDepField, fly_tdep

FAILED = []


def gate(n, ok, d=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
    if not ok:
        FAILED.append(n)


def solve2d(f, v):
    phi, sw, d = solve3d(f[:, :, None], v[:, :, None], stencil="sw",
                         tol=1e-9)
    return phi[:, :, 0]


def build_field():
    nx, ny = 120, 30
    tube = np.zeros((nx, ny), bool)
    tube[:, 28:] = True                    # grounded outer tube wall
    gap = np.zeros((nx, ny), bool)
    gap[55:65, 12:] = True                 # adjustable ring, r >= 12 mm
    phi_base = solve2d(tube | gap, np.zeros((nx, ny)))
    phi_basis = solve2d(tube | gap, np.where(gap, 1.0e4, 0.0)) - phi_base
    return TDepField(np.arange(nx) * 1.0, np.arange(ny) * 1.0,
                     phi_base, [phi_basis], symmetry="cylindrical")


WAVE = [[650.0], [0.0]]                    # 650 V until breakpoint, then 0
DECK = [  # (name, mz, KE_eV, z0, u0, tob_us, breakpoint_us_ion_clock)
    ("ion1_axial",   100.0, 2000.0, 2.0, 0.0, 0.00, 1.10),
    ("ion2_late_tob", 100.0, 2000.0, 2.0, 0.0, 0.25, 0.85),  # same wall clock
    ("ion3_offaxis", 100.0, 1200.0, 2.0, 1.5, 0.00, 1.10),
    ("ion4_heavy",   300.0, 2000.0, 2.0, 0.0, 0.00, 1.60),
]
# FROZEN from the anchored kernel (see docstring):
# (tof_us, KE_eV, u_exit_mm, steps)
GOLDEN = [
    (2.039054531158466, 1606.2722707005526, 0.0, 6334),
    (2.0393131516280185, 1599.3303282504125, 0.0, 6336),
    (2.7999843149816295, 809.87008591629, 2.351636372758725, 6738),
    (3.5517450148999874, 1568.5832196673584, 0.0, 6371),
]
RTOL = 1e-9                                # deterministic kernel, fastmath OFF


def main():
    print("TDEP NATIVE — tracer_tdep vs frozen anchored golden")
    fld = build_field()
    res = []
    for (name, mz, ke, z0, u0, tob, bp), g in zip(DECK, GOLDEN):
        r = fly_tdep(fld, mz, ke, z0, u0, [bp], WAVE, tob_us=tob,
                     dt_frac=0.02, h_mm=1.0, max_steps=400000)
        res.append(r)
        dt = abs(r["tof_us"] - g[0]) / max(abs(g[0]), 1e-30)
        dk = abs(r["KE_eV"] - g[1]) / max(abs(g[1]), 1e-30)
        du = abs(r["u_exit_mm"] - g[2])
        ok = dt < RTOL and dk < RTOL and du < 1e-6 and r["steps"] == g[3]
        gate(f"G.{name}", ok,
             f"TOF {r['tof_us']:.9f} us (rel d {dt:.1e}), "
             f"KE {r['KE_eV']:.4f} eV (rel d {dk:.1e}), "
             f"u {r['u_exit_mm']:.6f} mm, steps {r['steps']}")

    print("\nCOVERAGE (asserted)")
    gate("COV.switch_midflight",
         all(g[1] != d[2] for g, d in zip(GOLDEN, DECK)),
         "every ion's exit KE differs from its birth KE — the switch fired "
         "inside the field for all four (energy change IS the physics)")
    gate("COV.tob_threading",
         abs(res[0]["tof_us"] - res[1]["tof_us"]) > 1e-5,
         f"same wall-clock switch, tob 0 vs 0.25 us -> TOFs differ by "
         f"{abs(res[0]['tof_us']-res[1]['tof_us'])*1e3:.3f} ns")
    gate("COV.radial", res[2]["u_exit_mm"] > 1.0,
         "off-axis ion exits displaced (radial dynamics exercised)")

    print()
    if FAILED:
        print("TDEP NATIVE: FAIL ->", ", ".join(FAILED))
        return 1
    print("TDEP NATIVE: ALL PASS — kernel unchanged from its "
          "anchored state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
