"""test_ke_axis_channels.py — ke_x/ke_y/ke_z are first-class channels and
correct in BOTH the planar (2-D) and stl3d/scene3d (3-D) recorders.

Per-axis KE was added as recordable optional channels. Because
the two routes fill columns in structurally different ways (the planar
recorder is a numba-unrolled if-chain; the 3-D recorder is a name-keyed
dict), the same channel is written in two places — exactly the split that
once shipped field components under speed's name. These gates are the
desync guard: they check not just that the columns EXIST but that each
carries the physically-correct value, and that the per-axis parts sum to
the recorded total.

Gates (per route):
  G-1 NAMES: build_run's col_names == spec.column_names(), and the
      recorded array width equals the column count.
  G-2 VALUES: ke_x == 1/2 m (vx*1e3)^2 / e (and y, z), to float tol.
  G-3 CLOSURE: ke_x + ke_y + ke_z == ke_ev at every recorded step.
  G-4 TOTAL: ke_ev == 1/2 m (speed*1e3)^2 / e (speed self-consistent).
"""
import _bootstrap  # noqa: F401

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.sim_build import build_run, build_route

KG_AMU = 1.66053906660e-27
E_CHG = 1.602176634e-19
MS = 1.0e3   # mm/us -> m/s

KE_CHANNELS = ["speed", "ke_ev", "ke_x", "ke_y", "ke_z"]


def _check_route(name, spec, mz_da):
    ok = True
    cols = None
    # DECLARED seed: seed 0 restores
    # the exact draws this gate's life was measured under;
    # unseeded, a random birth can land in/near metal and refuse a build
    # whose assertions are otherwise draw-insensitive.
    spec.source.seed = 0
    model, fly, cols, births = build_run(spec)

    # G-1 NAMES
    g1 = (cols == spec.column_names())
    tr, _summ = fly(0)
    g1 = g1 and (tr.shape[1] == len(cols))
    print(f"  [{name}] G-1 names match & width: {cols == spec.column_names()}"
          f" / width {tr.shape[1]}=={len(cols)}  -> {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    ic = {c: cols.index(c) for c in cols}
    vx, vy, vz = tr[:, ic["vx"]], tr[:, ic["vy"]], tr[:, ic["vz"]]
    m_kg = mz_da * KG_AMU

    # G-2 VALUES: each per-axis KE equals 1/2 m v_i^2 / e
    def ke_of(v):
        return 0.5 * m_kg * (v * MS) ** 2 / E_CHG
    g2 = (np.allclose(tr[:, ic["ke_x"]], ke_of(vx), rtol=1e-6, atol=1e-12)
          and np.allclose(tr[:, ic["ke_y"]], ke_of(vy), rtol=1e-6, atol=1e-12)
          and np.allclose(tr[:, ic["ke_z"]], ke_of(vz), rtol=1e-6, atol=1e-12))
    print(f"  [{name}] G-2 ke_x/y/z == 1/2 m v_i^2/e  -> "
          f"{'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # G-3 CLOSURE: per-axis parts sum to the recorded total
    parts = tr[:, ic["ke_x"]] + tr[:, ic["ke_y"]] + tr[:, ic["ke_z"]]
    close = float(np.max(np.abs(parts - tr[:, ic["ke_ev"]])))
    g3 = close < 1e-9
    print(f"  [{name}] G-3 ke_x+ke_y+ke_z == ke_ev (max dev {close:.2e} eV) "
          f"-> {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    # G-4 TOTAL: ke_ev consistent with speed
    speed = tr[:, ic["speed"]]
    g4 = np.allclose(tr[:, ic["ke_ev"]], ke_of(speed), rtol=1e-6, atol=1e-12)
    # and speed == |v|
    g4 = g4 and np.allclose(speed, np.sqrt(vx**2 + vy**2 + vz**2),
                            rtol=1e-6, atol=1e-9)
    print(f"  [{name}] G-4 ke_ev==1/2 m speed^2/e & speed==|v|  -> "
          f"{'PASS' if g4 else 'FAIL'}")
    ok &= g4
    return ok


def _planar_spec():
    s = SimSpec.from_json(
        "examples/slim_tetramer_confinement_2-d_acrossxgap.json")
    s.geometry.field_method = "plain_gradient"
    s.source.n_ions = 3
    s.integration.record_channels = KE_CHANNELS[:]      # all five
    s.integration.t_max_us = min(getattr(s.integration, "t_max_us", 20.0), 20.0)
    return s, s.source.mz_list[0]


def _three_d_spec():
    s = SimSpec.from_json("examples/quadrupole_stl_rods_full_3-d.json")
    # COARSEN FOR SPEED -- but mm_per_gu is NOT independently settable on
    # a conforming deck (A7). This deck is 20.5 x 20.5 x 120.5 mm, an
    # exact 41/41/241 cells at its own 0.5 mm pitch; setting the pitch to
    # 1.0 alone left every axis on a HALF cell and the loader refused the
    # spec. Changing a pitch means re-deriving the extents at the new
    # one, which is what cover_extent_mm does -- outward, so no rod is
    # clipped. This deck declares no mirror plane, so no axis needs an
    # even count.
    from ion_gym.io.lattice import cover_extent_mm
    _h = 1.0                              # coarse -> fast solve
    g = s.geometry
    g.width_mm = cover_extent_mm(g.width_mm, _h)
    g.height_mm = cover_extent_mm(g.height_mm, _h)
    g.depth_mm = cover_extent_mm(g.depth_mm, _h)
    g.mm_per_gu = _h
    s.source.n_ions = 2
    s.integration.record_channels = KE_CHANNELS[:]
    s.integration.t_max_us = min(getattr(s.integration, "t_max_us", 5.0), 5.0)
    return s, s.source.mz_list[0]


def main():
    ok = True
    print("PLANAR (2-D) route:")
    sp, mz = _planar_spec()
    assert build_route(sp).builder == "planar", build_route(sp).builder
    ok &= _check_route("planar", sp, mz)

    print("3-D route:")
    s3, mz3 = _three_d_spec()
    assert build_route(s3).builder in ("stl3d", "scene3d"), \
        build_route(s3).builder
    ok &= _check_route("3-D", s3, mz3)

    print(f"\nKE-AXIS CHANNEL GATES: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
