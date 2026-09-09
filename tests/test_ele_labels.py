"""
test_ele_labels.py -- `ele` must carry INT16 LABELS on every model that feeds
the display layer. Cold cache AND warm cache.

WHY THIS EXISTS
    `ele` answers two different questions and was being asked to be one array:
        kernel:  "is this node metal?"        -> a boolean is fine
        display: "WHICH electrode is this?"   -> needs a LABEL
    As a bool, `ele == 1` is true for EVERY conductor (True == 1) and `ele == 3`
    is true for none. So `viz_core`'s `ele == i` colour lookup, the electrode
    legend, and the PE view drape every electrode at electrode #1's voltage.
    The solve is unaffected -- bases are built from the per-electrode masks --
    so the numbers are right and the PICTURE IS WRONG, which is exactly the
    failure mode "display must equal solver input" exists to forbid.

    It was fixed in build_stl3d, then found again in build_rz, then found again
    in build_planar. Three times, because the fixes were symptomatic.

THE ACTUAL ROOT CAUSE was neither builder: `basis_cache.load()` did
    ele = np.ascontiguousarray(arrs["ele"]).astype(bool)
so a builder could store a perfectly good int16 label array and the cache would
hand it back as a boolean mask. COLD runs looked correct and the bug appeared
the moment the cache warmed. That is why the warm-cache case below is not
decoration: it is the case that actually broke.

    basis_cache.CACHE_FORMAT now versions the stored array layout, so a change
    of this kind invalidates the entries it invalidates.

WHAT IS NOT A BUG
    ionbench's `ele` is a per-FA metal mask (one FA IS one electrode) fed to
    field-aware interpolation. There are no labels to collapse and it is not in
    the display path. Boolean is correct there. Do not "fix" it.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, ShapeSpec,
                      SourceSpec, IntegrationSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILED.append((name, e))
        print(f"  [FAIL] {name}\n         {type(e).__name__}: {e}")


def _rect(x, y, w, h):
    return ShapeSpec(type="rect",
                     params=dict(x_mm=x, y_mm=y, width_mm=w, height_mm=h))


def planar_spec(n=3):
    els = [ElectrodeSpec(name=f"p{i}", shapes=[_rect(2 + 4 * i, 2, 2, 6)],
                         dc=float(10 * i)) for i in range(n)]
    return SimSpec(
        geometry=GeometrySpec(
            width_mm=16, height_mm=10, depth_mm=0.0, mm_per_gu=0.2,
            symmetry=SymmetrySpec(coords="xyz",
                                  planes={"x": "none", "y": "none",
                                          "z": "none"}),
            electrodes=els),
        source=SourceSpec(seed=0, n_ions=1, distribution="point", x0_mm=1.0,
                          y0_mm=5.0, mz_list=[100.0]),
        integration=IntegrationSpec(dt_ns=2.0, t_max_us=1.0))


def rz_spec(n=3):
    """r-z: x is the axis, y is radius. Three stacked rings."""
    els = [ElectrodeSpec(name=f"r{i}", shapes=[_rect(2 + 5 * i, 6, 3, 2)],
                         dc=float(10 * i)) for i in range(n)]
    return SimSpec(
        geometry=GeometrySpec(
            width_mm=18, height_mm=9, depth_mm=0.0, mm_per_gu=0.2,
            symmetry=SymmetrySpec(coords="rz",
                                  planes={"x": "none", "y": "none",
                                          "z": "none"}),
            electrodes=els),
        source=SourceSpec(seed=0, n_ions=1, distribution="point", x0_mm=1.0,
                          y0_mm=1.0, mz_list=[100.0]),
        integration=IntegrationSpec(dt_ns=2.0, t_max_us=1.0))


def assert_labels(ele, n_expect, who):
    ele = np.asarray(ele)
    assert ele.dtype != bool, (
        f"{who}: `ele` is a BOOLEAN MASK. Every `ele == i` lookup in the "
        f"display layer collapses to electrode #1.")
    assert np.issubdtype(ele.dtype, np.integer), \
        f"{who}: `ele` dtype is {ele.dtype}, expected an integer label array"
    labels = sorted(int(v) for v in np.unique(ele) if v > 0)
    assert labels == list(range(1, n_expect + 1)), (
        f"{who}: labels {labels}, expected 1..{n_expect} — an electrode is "
        f"missing or two have been merged")
    # bool consumers must be unharmed: the kernels test nonzero-ness
    assert (ele > 0.5).sum() == (ele != 0).sum(), \
        f"{who}: `ele > 0.5` no longer selects the metal"


def _no_disk_cache():
    """Force the BUILDER to construct `ele` itself.

    Clearing the in-memory dict is not enough: the builder also consults the
    on-disk basis cache, which will happily serve a good array built by an
    earlier run — so a reintroduced bug in the builder's own construction is
    invisible. My first version of this test made exactly that mistake, and the
    mutation (restoring the bool OR-fold in build_planar) passed 6/6. The test
    had the same blind spot as the bug.
    """
    from ion_gym.io import basis_cache
    return basis_cache, basis_cache.load


def run():
    from ion_gym.physics import build_planar
    from ion_gym.physics import build_rz

    print("\n=== build_planar ===")
    sp = planar_spec(3)

    def planar_cold():
        bc, real_load = _no_disk_cache()
        build_planar._PLANAR_BASIS_CACHE.clear()
        bc.load = lambda *a, **k: (None, None)      # genuine miss
        try:
            m = build_planar.build_planar_model(sp, verbose=False)
            assert_labels(m.ele, 3, "build_planar (cold)")
        finally:
            bc.load = real_load
    check("build_planar: int16 labels, cold cache", planar_cold)

    def planar_warm():
        # second build hits the in-memory AND the on-disk cache -- the path
        # where basis_cache.load()'s .astype(bool) silently undid the fix
        build_planar._PLANAR_BASIS_CACHE.clear()
        m = build_planar.build_planar_model(sp, verbose=False)
        assert_labels(m.ele, 3, "build_planar (warm)")
    check("build_planar: labels SURVIVE the basis cache", planar_warm)

    print("\n=== build_rz ===")
    sz = rz_spec(3)

    def rz_cold():
        bc, real_load = _no_disk_cache()
        build_rz._RZ_BASIS_CACHE.clear()
        bc.load = lambda *a, **k: (None, None)
        try:
            m = build_rz.build_rz_model(sz, verbose=False)
            assert_labels(m.ele, 3, "build_rz (cold)")
        finally:
            bc.load = real_load
    check("build_rz: int16 labels, cold cache", rz_cold)

    def rz_warm():
        build_rz._RZ_BASIS_CACHE.clear()
        m = build_rz.build_rz_model(sz, verbose=False)
        assert_labels(m.ele, 3, "build_rz (warm)")
    check("build_rz: labels SURVIVE the basis cache", rz_warm)

    print("\n=== basis_cache ===")

    def cache_format_versioned():
        from ion_gym.io import basis_cache
        assert getattr(basis_cache, "CACHE_FORMAT", 1) >= 2, \
            "CACHE_FORMAT missing: a change to the stored array layout cannot " \
            "invalidate the entries it invalidates"
        k = basis_cache.geometry_key_dict(planar_spec(3))
        assert "_fmt" in k, "the cache key does not carry the array format"
    check("basis_cache: array FORMAT is in the key", cache_format_versioned)

    def cache_roundtrip_preserves_labels():
        """BEHAVIOUR, not a source grep. Store an int16 label array, load it
        back, and demand the labels are still there. (My first version of this
        test grepped load() for 'astype(bool)' — and matched the COMMENT that
        explains why the cast was removed. A test that reads the source instead
        of running it will fail on prose.)"""
        from ion_gym.io import basis_cache
        import tempfile
        sp = planar_spec(4)
        nx, ny = 40, 25
        ele = np.zeros((nx, ny), np.int16)
        for i in range(1, 5):                       # four distinct labels
            ele[i * 6:i * 6 + 4, 5:12] = i
        bases = {i: np.zeros((nx, ny)) for i in range(1, 5)}
        with tempfile.TemporaryDirectory() as root:
            basis_cache.store(sp, bases, ele, root=root)
            _b, back = basis_cache.load(sp, root=root)
            assert back is not None, "stored, then missed on load"
            assert_labels(back, 4, "basis_cache round-trip")
    check("basis_cache: labels survive store -> load",
          cache_roundtrip_preserves_labels)

    print("\n" + "=" * 66)
    print(f"PASSED {len(PASSED)}   FAILED {len(FAILED)}")
    for nm, e in FAILED:
        print(f"  FAIL {nm}: {e}")
    print("=" * 66)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(run())
