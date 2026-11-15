"""
test_fa_cache.py — validate the FA cache (build scope item 2).

Gates:
  1. round-trip: stored then memmap-loaded arrays are bit-identical, and
     the hit side is a read-only memmap.
  2. field equivalence: the cached Z/Ue/Ez/Eu equal a direct build_field of
     the same solve, bit for bit (what a kernel reads from a hit is what it
     would have computed on a miss).
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

from ion_gym.physics.tracer_numba import build_field
from ion_gym.io import fa_cache

FIELD = ("Z", "Ue", "Ez", "Eu")


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



def solve_220():
    Z, R, phi110, _ = _einzel_field(h=1.0)
    return Z, R, phi110 * 2.0


def cached_field(spec, solve_fn, root):
    """The cache-backed field build: load on a hit, else solve, derive the
    mirror-extended V/m arrays, store them and return them."""
    arrs, _meta = fa_cache.load(spec, mmap=True, root=root)
    if arrs is not None:
        return {k: arrs[k] for k in FIELD}
    Zn, Un, phi = solve_fn()
    field = dict(zip(FIELD, build_field(Zn, Un, phi, "cylindrical")))
    fa_cache.store(dict(spec), dict(field, phi=np.ascontiguousarray(phi)),
                   root=root)
    return field


def main():
    root = tempfile.mkdtemp(prefix="iongym_cache_test_")
    try:
        spec = dict(geometry="einzel_91x20", bore_r=18, thick=2, egap=4,
                    lead=10, r_max=19, voltages=[0, 220, 0], h=1.0,
                    symmetry="cylindrical", edge_ghost="ghost_linear")

        print("== gate 1+5: miss (solve) then hit (memmap) ==")
        t0 = time.perf_counter()
        f_miss = cached_field(spec, solve_220, root)
        t_miss = time.perf_counter() - t0
        t0 = time.perf_counter()
        f_hit = cached_field(spec, solve_220, root)
        t_hit = time.perf_counter() - t0
        print(f"   miss (solve+derive+store): {t_miss*1e3:7.1f} ms")
        print(f"   hit  (memmap load)       : {t_hit*1e3:7.1f} ms  "
              f"({t_miss/t_hit:.0f}x faster)")
        for k in FIELD:
            assert np.array_equal(np.asarray(f_miss[k]),
                                  np.asarray(f_hit[k])), k
        assert np.asarray(f_hit["Ez"]).flags.writeable is False, \
            "memmap should be read-only"
        print("   Z/Ue/Ez/Eu bit-identical, hit-side is read-only memmap: OK")

        print("\n== gate 2: cache-backed field == directly-built field ==")
        Zn, Un, phi = solve_220()
        direct = dict(zip(FIELD, build_field(Zn, Un, phi, "cylindrical")))
        for k in FIELD:
            assert np.array_equal(direct[k], np.asarray(f_hit[k])), \
                f"cached {k} differs from a direct build"
        print("   cached arrays == direct build_field, bit for bit: OK")

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
        assert fa_cache.spec_key(s2) != k0, "voltage change collides"
        assert fa_cache.spec_key(s3) != k0, "pitch change collides"
        assert fa_cache.spec_key(s4) == k0, "fp noise forks the key"

        print("\n== gate 4: corruption caught, not flown ==")
        key = fa_cache.spec_key(spec)
        ez_path = os.path.join(root, key, "Ez.npy")
        with open(ez_path, "r+b") as fh:
            fh.truncate(os.path.getsize(ez_path) // 2)   # lop off half
        try:
            fa_cache.load(spec, root=root)
        except IOError as e:
            print(f"   truncated Ez raised: {str(e)[:60]}...")
        else:
            raise AssertionError("truncated Ez loaded without error")
        assert fa_cache.verify(spec, root=root) is False
        print("   verify() reports corrupt entry: OK")

        print("\nALL CACHE GATES PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
