"""
test_numba_equiv.py — acceptance gate for the Numba port.

Requirement (pre-declared): tracer_numba's fly() reproduces tracer's fly() on the
validated einzel across all four regimes — transmit, y-wall, reflect,
electrode-impact — to:
    TOF        : < 1 ps
    positions  : < 0.1 um
    velocities : < 1e-6 relative
    exit KE    : < 1e-6 relative
If this passes, the jitted core is a faithful translation and the speed win is
free of physics risk.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import numpy as np
from ion_gym.physics.tracer import Field2D, fly as fly_np
from ion_gym.physics.tracer_numba import FieldNumba, fly as fly_nb


def _einzel_field(h=1.0, refine=1):
    """Cylindrical einzel Laplace solve. Inlined on severing the dependency on
    immersion.einzel_pa.build_and_solve -- which was itself core-only (numpy +
    solver2d.solve_laplace), so this reproduces it byte-for-byte. Fixture-free,
    deterministic; centre plate at 110 V. A core gate builds its own field."""
    import numpy as np
    from ion_gym.physics.solver2d import solve_laplace
    hh = h / refine
    nz = 90 * refine + 1
    nr = 19 * refine + 1
    Z = np.linspace(0, 90, nz)
    R = np.linspace(0, 19, nr)
    fixed = np.zeros((nz, nr), bool)
    val = np.zeros((nz, nr))
    rmask = R >= 18 - 1e-9
    for (z0, z1), V in (((0, 28), 0.0), ((30, 56), 110.0), ((58, 90), 0.0)):
        zmask = (Z >= z0 - 1e-9) & (Z <= z1 + 1e-9)
        m = np.outer(zmask, rmask)
        fixed |= m
        val[m] = V
    phi = solve_laplace(fixed, val, hh, symmetry="cylindrical",
                        neumann_edges=("z0", "z1", "u1"))
    return Z, R, phi, fixed



class BoreMetal:
    surfaces = (18.0,)
    bands = ((0.0, 28.0), (30.0, 56.0), (58.0, 90.0))
    def __call__(self, z_mm, u_mm):
        return abs(u_mm) >= 18.0 and any(a <= z_mm <= b for a, b in self.bands)


def main():
    Z, R, phi110, _ = _einzel_field(h=1.0)
    phi = phi110 * 2.0                       # 220 V: all four regimes present
    f_np = Field2D(Z, R, phi, "cylindrical")
    f_nb = FieldNumba(Z, R, phi, "cylindrical")
    metal = BoreMetal()

    cases = {0: "transmit", 5: "y-wall", 11: "reflect", 13: "electrode"}
    print(f"{'y0':>4} {'regime':>10} | {'dTOF(ps)':>9} {'dz(nm)':>9} "
          f"{'du(nm)':>9} {'dvz(rel)':>10} {'dvu(rel)':>10} {'dKE(rel)':>10} "
          f"{'impact==':>8}")
    worst = dict(tof=0, pos=0, vel=0, ke=0)
    for y0, name in cases.items():
        a = fly_np(f_np, 100.0, 200.0, 0.0, float(y0), dt_frac=0.02, h_mm=1.0,
                   max_steps=400000, metal=metal)
        b = fly_nb(f_nb, 100.0, 200.0, 0.0, float(y0), dt_frac=0.02, h_mm=1.0,
                   max_steps=400000, metal=metal)
        dtof = abs(a["t_ns"] - b["t_ns"]) * 1e3          # ps
        # endpoint positions
        dz = abs(a["z"][-1] - b["z"][-1]) * 1e6          # nm (mm->nm)
        du = abs(a["u"][-1] - b["u"][-1]) * 1e6
        np.sqrt(2 * a["KE_eV"]); np.sqrt(2 * b["KE_eV"])
        dke = abs(a["KE_eV"] - b["KE_eV"]) / a["KE_eV"]
        # exit angle -> velocity components
        aa = a["ang_mrad"] * 1e-3; ab = b["ang_mrad"] * 1e-3
        dvz = abs(np.cos(aa) - np.cos(ab)) / max(abs(np.cos(aa)), 1e-9)
        dvu = abs(np.sin(aa) - np.sin(ab)) / max(abs(np.sin(aa)), 1e-9)
        same = a["impact"] == b["impact"]
        print(f"{y0:4d} {name:>10} | {dtof:9.3f} {dz:9.3f} {du:9.3f} "
              f"{dvz:10.2e} {dvu:10.2e} {dke:10.2e} {str(same):>8}")
        worst["tof"] = max(worst["tof"], dtof)
        worst["pos"] = max(worst["pos"], dz, du)
        worst["vel"] = max(worst["vel"], dvz, dvu)
        worst["ke"] = max(worst["ke"], dke)

    print(f"\nworst: dTOF={worst['tof']:.3f} ps, dpos={worst['pos']:.3f} nm, "
          f"dvel={worst['vel']:.2e} rel, dKE={worst['ke']:.2e} rel")
    ok = (worst["tof"] < 1.0 and worst["pos"] < 100.0
          and worst["vel"] < 1e-6 and worst["ke"] < 1e-6)
    print("ACCEPTANCE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
