"""test_solver_convergence.py -- gate SC: a solver that cannot converge SAYS SO.

THE DEFECT (found while chasing "were the solver fixes mathematically
sound?").  solver3d.solve3d ended with:

    for s in range(max_sweeps):
        ...
        if d < tol:
            return phi, s + 1, d
    return phi, max_sweeps, d          # <-- fell out of the loop

It ran out of sweeps and returned the UNCONVERGED FIELD, with no exception and
no warning, in a tuple indistinguishable from a converged one.  solve_bases
never compared the returned delta to tol.  On the einzel geometry (520 x 80 --
long and thin, precisely where SOR's convergence collapses) it burned all 60,000
sweeps and handed back a field with a 3.2% error, reporting delta 1.96e-02
against a tol of 1e-04 nobody checked.  `tol` was decorative.

It surfaced only because a `except Exception` fallback was silently routing
between multigrid and SOR, and the two answers were 3.2% apart.  MULTIGRID IS
THE CORRECT ONE: it converges (2e-9), SOR stalls.  Both are exact on parallel
plates at every nz, so this is not a discretisation difference -- it is one
solver quietly failing.

WHY THIS MATTERS BEYOND THE BUG.  Once every solver either reaches tol or
REFUSES, the solver stops being an input to the answer: a converged field is a
property of the DISCRETISATION, not of who solved it.  That is the only thing
that makes it legitimate for the basis cache key to omit the solver -- which it
does.  The old code violated that silently, and a cache written by one solver
was served to the other.
"""
import _bootstrap  # noqa: F401
import sys

import numpy as np
from ion_gym.physics.solver3d import solve3d, solve_bases, SolveNotConverged
from ion_gym.physics.multigrid3d import solve_bases_mg

V = 1e4
FAILED = []


def check(name, ok, detail=""):
    print(f"  {name:52s} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        FAILED.append(name)


def main():
    # SC-1: a stalling geometry must RAISE, not return a plausible field.
    nx, ny = 520, 80
    metal = np.zeros((nx, ny, 1), bool)
    metal[100:110, 30:50, :] = True
    metal[250:260, 30:50, :] = True
    metal[400:410, 30:50, :] = True
    val = np.zeros(metal.shape)
    val[400:410, 30:50, :] = V
    try:
        solve3d(metal, val, theta=None, mirror=(False, False, False),
                tol=1e-4, stencil="ghost_linear", omega=None, max_sweeps=200)
        check("SC-1 stalled SOR refuses", False, "returned an unconverged field")
    except SolveNotConverged as e:
        check("SC-1 stalled SOR refuses", "too loose" in str(e))

    # SC-2: the tolerance must actually CONTROL something.  If tightening tol
    # cannot change the answer, tol is decorative -- that was the symptom.
    m = {1: np.zeros((48, 24, 1), bool)}
    m[1][0, :, :] = True
    m2 = np.zeros((48, 24, 1), bool); m2[-1, :, :] = True
    masks = {1: m[1], 2: m2}
    a = solve_bases(masks, mirror=(False, False, False), v_basis=V,
                    tol=1e-4, stencil="ghost_linear", omega=None)
    b = solve_bases(masks, mirror=(False, False, False), v_basis=V,
                    tol=1e-10, stencil="ghost_linear", omega=None)
    moved = np.abs(a[1] - b[1]).max()
    check("SC-2 tol controls the answer", moved > 0.0,
          f"tightening tol moved the field by {moved:.2e} V")

    # SC-3: SOR and MG AGREE where SOR actually converges.  Exact ground truth
    # (parallel plates -> linear ramp), so neither gets to grade its own work.
    i = np.arange(48)[:, None, None]
    exact = V * (1.0 - i / 47.0) * np.ones((48, 24, 1))
    g = solve_bases_mg(masks, mirror=(False, False, False), v_basis=V,
                       tol=1e-8, stencil="ghost_linear")
    es = np.abs(b[1] - exact).max() / V
    eg = np.abs(g[1] - exact).max() / V
    check("SC-3 SOR exact on plates", es < 1e-6, f"err {es:.2e}")
    check("SC-3 MG  exact on plates", eg < 1e-6, f"err {eg:.2e}")
    check("SC-3 SOR and MG agree",
          np.abs(b[1] - g[1]).max() / V < 1e-6,
          f"max|SOR-MG| {np.abs(b[1]-g[1]).max()/V:.2e} of V")

    print()
    if FAILED:
        print(f"SOLVER CONVERGENCE GATES: {len(FAILED)} FAILED -> "
              f"{', '.join(FAILED)}")
        return 1
    print("SOLVER CONVERGENCE GATES: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
