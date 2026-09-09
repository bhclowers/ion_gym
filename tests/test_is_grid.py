"""
test_is_grid.py  --  the Dirichlet mask is not the hit mask
===========================================================
`ElectrodeSpec.is_grid` has been in the schema since the beginning and was
never honoured by the planar builder: `ele` (the mask the kernel hit-tests
against) unioned EVERY electrode, so a mesh, a gridless mirror's entrance
plane, a detector face at potential, or a drift-copper column that pins a
gauge was a WALL.  Ions died on a boundary condition.

  G1  a grid electrode is a Dirichlet boundary in the SOLVE: the field is
      bit-identical to the same geometry with the plane declared solid.
  G2  (a grid electrode is TRANSPARENT to the kernel -- absent from the hit
      mask, so an ion flies THROUGH it) is the BENCH kernel's own contract and
      now lives in test_bench_is_grid.py, the gate that exercises that kernel.
  G3  no existing spec changes: with no is_grid electrode anywhere, the hit
      mask is exactly what it was (the union of every electrode).
  G4  the disk FA cache DISCRIMINATES is_grid: flipping it changes the
      geometry key, so a stale hit mask can never be served.
"""

import numpy as np

from ion_gym.io import basis_cache
from ion_gym.physics.build_planar import build_planar_model
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, ShapeSpec,
                      SourceSpec, IntegrationSpec, CollisionSpec)


def _spec(grid: bool, live: bool = True):
    """A drift box with ONE transverse plane across the middle of the beam
    path, held at 0 V.  Nothing instrument-specific: the point is the plane."""
    # live=False -> every electrode at 0 V: a FIELD-FREE box.  G2 uses it so
    # that transparency is isolated from dynamics entirely: in a dead field
    # the ion flies ballistically, and the ONLY thing that can stop it is the
    # hit mask -- which is exactly the thing is_grid changes.
    vb, vp = (20.0, 10.0) if live else (0.0, 0.0)
    els = [
        ElectrodeSpec(name="back", dc=vb, shapes=[ShapeSpec(
            type="rect", params=dict(x_mm=0.0, y_mm=0.0,
                                     width_mm=0.4, height_mm=8.0))]),
        ElectrodeSpec(name="front", dc=0.0, shapes=[ShapeSpec(
            type="rect", params=dict(x_mm=19.6, y_mm=0.0,
                                     width_mm=0.4, height_mm=8.0))]),
        # the plane under test: a REAL Dirichlet (it shapes the field), and
        # solid or transparent depending only on is_grid.
        ElectrodeSpec(name="plane", dc=vp, is_grid=grid, shapes=[ShapeSpec(
            type="rect", params=dict(x_mm=9.8, y_mm=0.0,
                                     width_mm=0.4, height_mm=8.0))]),
    ]
    return SimSpec(
        name=f"grid_{int(grid)}_{int(live)}",
        geometry=GeometrySpec(width_mm=20.0, height_mm=8.0, mm_per_gu=0.1,
                              electrodes=els),
        source=SourceSpec(n_ions=1, distribution="grid", x0_mm=2.0,
                          y0_mm=4.0, ke_lo=50.0, ke_hi=50.0, axis="x",
                          mz_list=[100.0], seed=7),
        # collisions OFF, explicitly.  The default SourceSpec/CollisionSpec
        # ships with hs N2 @ 1 torr enabled, which damps a light ion to a
        # crawl -- that is what was quietly eating this fixture's velocity
        # before the bench REFUSED the spec and said so.  Vacuum flight is
        # what this gate is about.
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=0.5, t_max_us=5.0, rec_every=5,
                                    record_channels=[]))


def test_G1_grid_is_dirichlet_in_the_solve():
    from ion_gym.physics.build_planar import build_planar_model
    m_grid = build_planar_model(_spec(True))
    m_solid = build_planar_model(_spec(False))
    assert np.array_equal(m_grid.A, m_solid.A), \
        "declaring a plane is_grid changed the FIELD -- it must not: a grid " \
        "is still a Dirichlet boundary, it is only transparent to ions"
    assert np.array_equal(m_grid.ExA, m_solid.ExA)
    assert np.array_equal(m_grid.EyA, m_solid.EyA)


def test_G3_no_is_grid_anywhere_means_nothing_changed():
    """Behaviour-preserving for every existing spec: with no grid declared,
    the hit mask is the union of every electrode, exactly as before.

    AMENDED (node-count family): the earlier
    amendment moved the reference to node-at-i*h coordinates but kept the
    pre-v2 node COUNT (round(W/h), one node short per axis) and a BOOL
    reference against the builder's int16 LABEL array — shape (nx,ny) vs
    (nx+1,ny+1) and dtype both wrong, so array_equal was False on
    structure before any physics was compared (the stl3d family).
    The reference now comes from THE grid authority (anchored_grid), so
    a future raster change moves gate and builder together; occupancy is
    compared label-aware ((ele != 0) == union), and shape/dtype are
    asserted explicitly as tripwires."""
    spec = _spec(False)
    m = build_planar_model(spec)
    from ion_gym.physics.raster2d import (electrode_mask, anchored_grid)
    xs, ys, anchor = anchored_grid(spec)
    assert anchor == (0.0, 0.0)          # no declared plane: no anchoring
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    assert m.ele.shape == (len(xs), len(ys)), (
        f"ele shape {m.ele.shape} != authority grid "
        f"({len(xs)}, {len(ys)})")
    assert np.issubdtype(m.ele.dtype, np.integer), (
        f"ele must be the int LABEL array (labels doctrine), got "
        f"{m.ele.dtype}")
    ref = np.zeros((len(xs), len(ys)), bool)
    for el in spec.geometry.electrodes:
        ref |= np.asarray(electrode_mask(el, X, Y))
    assert np.array_equal(m.ele != 0, ref), (
        "hit-mask occupancy differs from the union of electrode masks")
    # label identity where coverage is unambiguous: each electrode's own
    # nodes carry its 1-based index (overlaps excluded from the check)
    singles = np.zeros((len(xs), len(ys)), np.int16)
    counts = np.zeros((len(xs), len(ys)), np.int16)
    for i, el in enumerate(spec.geometry.electrodes, start=1):
        mm = np.asarray(electrode_mask(el, X, Y))
        singles[mm] = i
        counts += mm
    solo = counts == 1
    assert np.array_equal(m.ele[solo], singles[solo]), (
        "label identity broken on unambiguously-owned nodes")


def test_G4_disk_cache_discriminates_is_grid():
    k_grid = basis_cache.key(_spec(True))
    k_solid = basis_cache.key(_spec(False))
    assert k_grid != k_solid, \
        "the FA cache key ignores is_grid -- a stale HIT MASK could be served"
