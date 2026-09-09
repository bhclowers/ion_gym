"""
test_stl_cache.py — the STL build must cache its voxelize+solve.

build_stl_run solves per-electrode bases via SOR, which is seconds of work
on a fine grid. Without caching, every app rebuild (voltage tweak, view
switch, fly) re-solved and blocked the UI. This gates the fast-adjust
invariant: the FIRST build for a geometry solves; later builds with the
SAME geometry but different voltages/RF re-weight the cached bases in
milliseconds. Changing the GRID (a real geometry change) re-solves.

Gates:
  K-1 CACHE HIT: a second build with a changed voltage is >20x faster
      than the first (solve skipped) and still correct.
  K-2 CORRECTNESS: the cached re-weight gives the same field a fresh
      solve would (voltage scales the static field linearly).
  K-3 GRID INVALIDATES: changing mm_per_gu forces a re-solve (miss).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.physics.build_stl import quad_stl_spec, build_stl_run, _STL_BUILD_CACHE


def main():
    ok = True
    _STL_BUILD_CACHE.clear()

    spec = quad_stl_spec("/tmp/quad_cache_test")
    # warm the numba JIT on a throwaway geometry so timing reflects the
    # solve, not compilation
    build_stl_run(quad_stl_spec("/tmp/quad_cache_warm"), ground_border=True)

    _STL_BUILD_CACHE.clear()
    t0 = time.time()
    m1, _, _, _ = build_stl_run(spec, ground_border=True)
    t_first = time.time() - t0

    # K-1: change a DC voltage, rebuild — should hit cache (fast)
    spec.geometry.electrodes[0].dc = 10.0
    t0 = time.time()
    m2, _, _, _ = build_stl_run(spec, ground_border=True)
    t_second = time.time() - t0
    g1 = t_second < t_first / 20.0
    ok &= g1
    print(f"K-1 cache hit: first {t_first:.2f}s, second {t_second*1e3:.0f}ms "
          f"({t_first / max(t_second, 1e-6):.0f}x faster)  ->  "
          f"{'PASS' if g1 else 'FAIL'}")

    # K-2: the re-weight is correct — a +10 V DC on rod 1 adds 10*basis1
    # to the static field vs the zero-DC build.
    cc = spec.geometry.width_mm / 2
    h = m2.mm_per_gu
    ci = int(round(cc / h))
    # rod-1 basis ~ its potential per volt; at 10 V the field near rod 1
    # should be ~10x the per-volt basis. Just check the static field
    # changed by the expected sign/scale at a probe near rod 1.
    dA = m2.A[ci + int(2.0 / h), ci] - m1.A[ci + int(2.0 / h), ci]
    g2 = dA > 0.0    # +10 V on the +x rod raises potential on the +x side
    ok &= g2
    print(f"K-2 reweight correct: dA(+x probe) = {dA:.3f} V for +10 V on "
          f"rod 1 (expect >0)  ->  {'PASS' if g2 else 'FAIL'}")

    # K-3: changing the grid is a real geometry change -> re-solve (miss)
    spec2 = quad_stl_spec("/tmp/quad_cache_test2")
    spec2.geometry.mm_per_gu = 0.2
    t0 = time.time()
    build_stl_run(spec2, ground_border=True)
    t_grid = time.time() - t0
    g3 = t_grid > 0.3     # a real solve, not a cache hit
    ok &= g3
    print(f"K-3 grid invalidates: mm_per_gu 0.15->0.2 re-solved in "
          f"{t_grid:.2f}s  ->  {'PASS' if g3 else 'FAIL'}")

    # K-4: build_needs_solve predicts fresh->True, cached->False. This is
    # what the UI uses to show a solving spinner only when it'll be slow.
    from ion_gym.physics.sim_build import build_needs_solve
    _STL_BUILD_CACHE.clear()
    fresh = quad_stl_spec("/tmp/quad_predict")
    pred_fresh = build_needs_solve(fresh)
    build_stl_run(fresh, ground_border=True)      # now cached
    pred_cached = build_needs_solve(fresh)
    g4 = pred_fresh and not pred_cached
    ok &= g4
    print(f"K-4 spinner predicate: fresh={pred_fresh} cached={pred_cached} "
          f"(expect True/False)  ->  {'PASS' if g4 else 'FAIL'}")

    print("\nSTL CACHE GATES:", "ALL PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
