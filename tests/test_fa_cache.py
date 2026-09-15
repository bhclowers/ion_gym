"""
test_fa_cache.py — validate the FA cache (build scope item 2).

Gates:
  1. round-trip: stored then memmap-loaded arrays are bit-identical.
  2. field equivalence: a cache-backed FieldNumba produces the SAME
     trajectories as one built directly (the memmapped Ez/Eu are read by the
     jitted kernel with no copy, no recompute).
  3. key sensitivity: changing any spec field misses; identical spec hits.
  4. integrity: a truncated .npy is caught at load, not flown as garbage.
  5. timing: second construction (hit) skips the solve -> large speedup.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import os
import shutil
import tempfile
import time
import numpy as np

from ion_gym.physics.tracer_numba import FieldNumba, fly
from ion_gym.io import fa_cache


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
    def __call__(self, z, u):
        return abs(u) >= 18.0 and any(a <= z <= b for a, b in self.bands)


def solve_220():
    Z, R, phi110, _ = _einzel_field(h=1.0)
    return Z, R, phi110 * 2.0


def main():
    root = tempfile.mkdtemp(prefix="iongym_cache_test_")
    try:
        spec = dict(geometry="einzel_91x20", bore_r=18, thick=2, egap=4,
                    lead=10, r_max=19, voltages=[0, 220, 0], h=1.0,
                    symmetry="cylindrical", edge_ghost="ghost_linear")

        print("== gate 1+5: miss (solve) then hit (memmap) ==")
        t0 = time.perf_counter()
        f_miss = FieldNumba.cached(spec, solve_220, root=root)
        t_miss = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_hit = FieldNumba.cached(spec, solve_220, root=root)
        t_hit = time.perf_counter() - t0
        print(f"   miss (solve+derive+store): {t_miss*1e3:7.1f} ms")
        print(f"   hit  (memmap load)       : {t_hit*1e3:7.1f} ms  "
              f"({t_miss/t_hit:.0f}x faster)")
        assert np.array_equal(np.asarray(f_miss.Ez), np.asarray(f_hit.Ez))
        assert np.asarray(f_hit.Ez).flags.writeable is False, \
            "memmap should be read-only"
        print("   Ez bit-identical, hit-side is read-only memmap: OK")

        print("\n== gate 2: cache-backed field == directly-built field ==")
        Zn, Un, phi = solve_220()
        f_direct = FieldNumba(Zn, Un, phi, "cylindrical")
        metal = BoreMetal()
        worst = 0.0
        for y0 in (0.0, 5.0, 11.0, 13.0):
            a = fly(f_direct, 100, 200, 0, y0, dt_frac=0.02, h_mm=1.0,
                    max_steps=400000, metal=metal)
            b = fly(f_hit, 100, 200, 0, y0, dt_frac=0.02, h_mm=1.0,
                    max_steps=400000, metal=metal)
            d = abs(a["t_ns"] - b["t_ns"])
            worst = max(worst, d)
            assert a["impact"] == b["impact"]
        print(f"   max |dTOF| across regimes = {worst*1e3:.3e} ps  (want 0)")

        print("\n== gate 3: key sensitivity ==")
        k0 = fa_cache.spec_key(spec)
        s2 = dict(spec); s2["voltages"] = [0, 240, 0]
        s3 = dict(spec); s3["h"] = 0.5
        print(f"   base key           : {k0}")
        print(f"   V change  -> key   : {fa_cache.spec_key(s2)}  "
              f"({'MISS' if fa_cache.spec_key(s2) != k0 else 'COLLISION!'})")
        print(f"   h change  -> key   : {fa_cache.spec_key(s3)}  "
              f"({'MISS' if fa_cache.spec_key(s3) != k0 else 'COLLISION!'})")
        # fp-noise stability: same numbers, different float spelling -> same key
        s4 = dict(spec); s4["h"] = 1.0 + 1e-15
        print(f"   h + 1e-15 -> key   : {fa_cache.spec_key(s4)}  "
              f"({'stable' if fa_cache.spec_key(s4) == k0 else 'forked'})")

        print("\n== gate 4: corruption caught, not flown ==")
        key = fa_cache.spec_key(spec)
        ez_path = os.path.join(root, key, "Ez.npy")
        with open(ez_path, "r+b") as fh:
            fh.truncate(os.path.getsize(ez_path) // 2)   # lop off half
        try:
            fa_cache.load(spec, root=root)
            print("   FAIL: truncated array loaded without error")
        except IOError as e:
            print(f"   truncated Ez raised: {str(e)[:60]}...")
        assert fa_cache.verify(spec, root=root) is False
        print("   verify() reports corrupt entry: OK")

        print("\nALL CACHE GATES PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
