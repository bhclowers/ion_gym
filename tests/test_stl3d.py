"""
test_stl3d.py — gate the FULL 3-D STL solve/flight (build_stl3d).

Known answer: the user's assembly rods are z-invariant (straight quad),
so the 3-D solve must reproduce the validated 2-D slice solve at mid-z.
Then the composed drive must actually work: an RF-driven ion (groups
A/B at 0/180 deg, one frequency) stays confined and drifts axially,
while an ion in the DC-only saddle is expelled. Bounds fate wiring and
the phase/frequency refusals are checked too.

Coarse pitch (0.4 mm) keeps the gate fast; the mid-z comparison uses the
SAME pitch for both paths so discretization cancels.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from ion_gym.io.stl_upload import load_mesh_bytes, propose_sizing, spec_from_upload

PITCH = 0.4


def _specs():
    files = sorted(glob.glob("examples/quad_assembly/*.STL"))
    meshes = [load_mesh_bytes(f.split("/")[-1], open(f, "rb").read())
              for f in files]
    n = [m.name for m in meshes]
    DC = {n[0]: 100.0, n[2]: 100.0, n[1]: -100.0, n[3]: -100.0}
    RF = {n[0]: "A", n[2]: "A", n[1]: "B", n[3]: "B"}
    s = propose_sizing(meshes, pitch=PITCH, margin_mm=2.0)
    sp2 = spec_from_upload(meshes, DC, s, "/tmp/stl3d_2d", rf_groups=RF,
                           solve_3d=False)
    sp3 = spec_from_upload(meshes, DC, s, "/tmp/stl3d_3d", rf_groups=RF,
                           solve_3d=True)
    # DECLARED seed: tob_span over an RF cycle is a random
    # draw that moves r_max/dz in a driven quad; the banked expectations
    # (vacuum dz 55.6 mm, r_max 0.39 mm) were measured at the historical
    # implicit seed 0.
    sp2.source.seed = 0
    sp3.source.seed = 0
    return sp2, sp3


def test_midz_matches_2d():
    from ion_gym.physics.build_stl import build_stl_run
    from ion_gym.physics.build_stl3d import stl_masks_3d, build_stl3d_run
    sp2, sp3 = _specs()

    masks3 = stl_masks_3d(sp3)
    assert set(masks3) == {1, 2, 3, 4}
    nz = next(iter(masks3.values())).shape[2]
    assert nz > 5, f"3-D grid must have real z extent (nz={nz})"

    model3, fly3, cols3, births3 = build_stl3d_run(sp3)
    x3, y3, img3, em3 = model3.potential_image()

    model2, fn2, cols2, births2 = build_stl_run(sp2, ground_border=None)
    x2, y2, img2, em2 = model2.potential_image()

    # compare on the overlapping grid (identical pitch); interior only
    n = min(img2.shape[0], img3.shape[0]), min(img2.shape[1], img3.shape[1])
    a = img2[:n[0], :n[1]]
    b = img3[:n[0], :n[1]]
    # AMENDED (root-cause of a long-standing ~576% red).  The
    # em arrays became INT16 LABELS when the labels doctrine landed
    # (build_stl's 2-D path got labels in the Phase B fix) -- and this
    # gate's mask algebra, written against booleans, silently rotted:
    # `~(em2 | em3)` on int16 is BITWISE COMPLEMENT (~0 == -1), `w` was
    # zeros_like(int), and `a[sel]` became integer FANCY INDEXING over a
    # (ny, nx, nx) tensor.  The reported 576% was the rms of that
    # nonsense; with boolean masks the 3-D mid-z slice reproduces the 2-D
    # solve at rms 0.06% / max 0.42%.  Same silent-coercion family
    # as is_grid G3.  The physics never regressed; the comparison had.
    free = ~((em2[:n[0], :n[1]] > 0) | (em3[:n[0], :n[1]] > 0))
    # away from z-boundary effects: central 60% window
    w = np.zeros(n, bool)
    w[int(.2 * n[0]):int(.8 * n[0]), int(.2 * n[1]):int(.8 * n[1])] = True
    sel = free & w
    assert sel.dtype == np.bool_ and a[sel].ndim == 1   # mask, not indices
    scale = np.abs(a[sel]).max()
    rms = np.sqrt(np.mean((a[sel] - b[sel]) ** 2)) / scale
    mx = np.max(np.abs(a[sel] - b[sel])) / scale
    assert rms < 0.02 and mx < 0.08, f"mid-z vs 2-D: rms {rms:.3f} max {mx:.3f}"
    print(f"  3-D mid-z reproduces 2-D slice (z-invariant rods): "
          f"rms {rms*100:.2f}%, max {mx*100:.2f}% of peak  OK")


def test_rf_confinement_and_fates():
    from ion_gym.physics.sim_build import build_run
    sp2, sp3 = _specs()
    # OPERATING POINT DECLARED (root-cause of the second
    # long-standing stl3d red).  These are VACUUM-dynamics assertions:
    # ballistic axial drift, KE-conserving RF confinement.  The gate was
    # written while the 3-D path had a bug that silently flew EVERY ion in
    # vacuum ("the spec's gas/pressure were parsed, shown, and thrown
    # away" -- build_stl3d's own comment).  When that bug was fixed and
    # spec.collisions became honoured, the spec DEFAULT (hs N2 at 1 Torr)
    # correctly thermalised the 5 eV ion in ~6 us (ke_end 0.088 eV,
    # dz 0.05 mm) and the gate's IMPLICIT operating point broke it.  A
    # certified expectation carries its operating point inline; this one
    # is vacuum, so it says so.  (Measured in vacuum: dz 55.6 mm, r_max
    # 0.39 mm, KE conserved at 5.08 eV.)
    sp3.collisions.enabled = False

    # drive: A/B 200 V @ 1 MHz, 0/180; stable m/z at these settings
    for g in sp3.geometry.rf_groups:
        g.amplitude_v = 200.0
        g.frequency_hz = 1.0e6
        g.phase_deg = 0.0 if g.name == "A" else 180.0
    for e in sp3.geometry.electrodes:
        e.dc = 0.0
    src = sp3.source
    src.n_ions = 3
    src.distribution = "point"
    # start on the quad axis (measured, in build frame = original - domain_min)
    src.x0_mm = 10.248 + 2.0   # domain_min ~ -2 margin -> +2
    src.y0_mm = 10.145 + 2.0
    src.z0_mm = 6.0
    src.direction = [0.0, 0.0, 1.0]
    src.ke_lo, src.ke_hi = 5.0, 5.0
    src.mz_list = [500.0]
    sp3.integration.dt_ns = 5.0
    sp3.integration.t_max_us = 40.0
    sp3.integration.rec_every = 20

    model, fly_fn, cols, births = build_run(sp3)
    tr, summ = fly_fn(0)
    # confined: never hits a rod (kind 0) within the run; advances in z
    assert summ["kind"] != 0, f"RF ion splat on electrode: {summ}"
    dz = tr[-1, 3] - tr[0, 3]
    r_max = np.hypot(tr[:, 1] - src.x0_mm, tr[:, 2] - src.y0_mm).max()
    assert dz > 5.0, f"axial drift too small ({dz:.1f} mm)"
    assert r_max < 3.0, f"not confined (r_max {r_max:.2f} mm)"
    print(f"  RF quad (3-D): confined r_max {r_max:.2f} mm over "
          f"dz {dz:.1f} mm axial  OK")

    # DC saddle expels: same start, DC-only +-100 V
    sp2b, sp3b = _specs()
    sp3b.collisions.enabled = False      # same vacuum operating point
    for e, v in zip(sp3b.geometry.electrodes, (100.0, -100.0, 100.0, -100.0)):
        e.dc = v
    for g in sp3b.geometry.rf_groups:
        g.amplitude_v = 0.0
    sp3b.source = src
    model_b, fly_b, _, _ = build_run(sp3b)
    trb, sb = fly_b(0)
    rb = np.hypot(trb[:, 1] - src.x0_mm, trb[:, 2] - src.y0_mm).max()
    assert sb["kind"] in (0, 1) or rb > r_max, \
        "DC saddle should expel (splat 0 / exit 1) or exceed RF excursion"
    print(f"  DC-only saddle expels (kind {sb['kind']}, r_max {rb:.2f})  OK")

    # bounds: z_max plane -> fate 3
    sp3c = sp3
    sp3c.bounds.z_max_on = True
    sp3c.bounds.z_max = src.z0_mm + 8.0
    model_c, fly_c, _, _ = build_run(sp3c)
    trc, sc = fly_c(0)
    assert sc["kind"] == 3 and abs(trc[-1, 3] - sp3c.bounds.z_max) < 2.0
    print(f"  bounding plane fate 3 at z={sp3c.bounds.z_max}  OK")


def test_cache_and_velocities():
    """Second build must be a cache HIT (no re-solve, bit-identical
    potential); trajectory velocity columns must be populated and
    energy-consistent (record-0 speed == birth KE)."""
    import time as _t
    from ion_gym.physics.sim_build import build_run
    from ion_gym.io import fa_cache

    sp2, sp3 = _specs()
    sp3.collisions.enabled = False   # vacuum operating point (record-0
    #                                  KE == birth KE is a vacuum claim;
    #                                  see test_rf_confinement_and_fates)
    for g in sp3.geometry.rf_groups:
        g.amplitude_v = 200.0
        g.frequency_hz = 1.0e6
        g.phase_deg = 0.0 if g.name == "A" else 180.0
    src = sp3.source
    src.n_ions = 1
    src.distribution = "point"
    src.x0_mm, src.y0_mm, src.z0_mm = 12.25, 12.15, 6.0
    src.direction = [0.0, 0.0, 1.0]
    src.ke_lo = src.ke_hi = 5.0
    src.mz_list = [500.0]
    sp3.integration.dt_ns = 5.0
    sp3.integration.t_max_us = 10.0
    sp3.integration.rec_every = 20

    fa_cache.clear_all()
    t0 = _t.time()
    model1, fly1, _, _ = build_run(sp3)
    t_first = _t.time() - t0
    t0 = _t.time()
    model2, fn2, _, _ = build_run(sp3)
    t_hit = _t.time() - t0
    assert t_hit < t_first / 3, f"cache hit not fast: {t_hit:.1f}s vs {t_first:.1f}s"
    a1 = model1.potential_image()[2]
    a2 = model2.potential_image()[2]
    assert np.array_equal(a1, a2), "cached potential must be bit-identical"
    print(f"  3-D bases cache: first {t_first:.1f}s -> hit {t_hit:.1f}s, "
          f"bit-identical  OK")

    tr, summ = fn2(0)
    v = tr[:, 4:7]
    assert np.abs(v).max() > 0, "velocity columns empty"
    from ion_gym.physics.collision3d import KG_AMU, E_CHG
    m_kg = 500.0 * KG_AMU
    sp0 = np.linalg.norm(v[0]) * 1e3            # mm/us -> m/s
    ke0 = 0.5 * m_kg * sp0 ** 2 / E_CHG
    assert abs(ke0 - 5.0) < 0.05, f"record-0 KE {ke0:.3f} eV != 5"
    print(f"  velocity channels: record-0 KE {ke0:.3f} eV (birth 5.000)  OK")


def test_refusals():
    """AMENDED: _compose_AB was SUPERSEDED by
    compose_drive_channels, which handles arbitrary phases and multiple
    frequencies BY DESIGN (its docstring says so) — the old refusals
    (90-deg phase, second frequency) are now capabilities, so this gate
    flips: it certifies the capabilities and the refusals that remain
    (undefined group membership; defined-but-memberless driven group).
    Deleting the superseded function without updating this gate is what
    kept the suite red — the gate lagged the supersession."""
    import math
    from ion_gym.physics.build_stl3d import (compose_drive_channels,
                                             stl_masks_3d)
    from ion_gym.physics.solver3d import solve_bases
    sp2, sp3 = _specs()
    for g in sp3.geometry.rf_groups:
        g.amplitude_v = 100.0
        g.frequency_hz = 1e6
    sp3.geometry.rf_groups[1].phase_deg = 90.0        # now a CAPABILITY
    masks = stl_masks_3d(sp3)
    bases = solve_bases(masks, mirror=(False, False, False),
                        v_basis=1e4, tol=1e-3, stencil="ghost_linear",
                        omega=1.9)
    bases = {i: b / 1e4 for i, b in bases.items()}
    ch = compose_drive_channels(sp3, bases)
    assert len(ch["ch_om"]) == 2, f"want 2 channels, got {len(ch['ch_om'])}"
    assert any(abs(p - math.pi / 2) < 1e-9 for p in ch["ch_ph"]), \
        f"90-deg phase not carried: {ch['ch_ph']}"
    sp3.geometry.rf_groups[1].frequency_hz = 2e6      # now a CAPABILITY
    ch = compose_drive_channels(sp3, bases)
    oms = sorted(ch["ch_om"])
    # ch_om is rad/us (the kernels' microsecond time base): 1 MHz -> 2pi
    assert abs(oms[0] - 2 * math.pi * 1.0) < 1e-9 and \
        abs(oms[1] - 2 * math.pi * 2.0) < 1e-9, f"two frequencies: {oms}"
    # what STILL refuses: membership in an undefined group ...
    el = sp3.geometry.electrodes[0]
    keep = list(el.rf_groups)
    el.rf_groups = keep + ["ghost_group"]
    try:
        compose_drive_channels(sp3, bases)
        assert False, "undefined group membership must be refused"
    except ValueError as e:
        assert "ghost_group" in str(e)
    el.rf_groups = keep
    # ... and a defined, non-zero group nobody belongs to
    from ion_gym.io.sim_spec import RFGroupSpec
    sp3.geometry.rf_groups.append(RFGroupSpec(
        name="lonely", amplitude_v=50.0, frequency_hz=1e6))
    try:
        compose_drive_channels(sp3, bases)
        assert False, "memberless driven group must be refused"
    except ValueError as e:
        assert "lonely" in str(e)
    sp3.geometry.rf_groups.pop()
    print("  compose_drive_channels: 90-deg + 2-frequency compose; "
          "ghost membership + memberless group refuse  OK")


if __name__ == "__main__":
    test_midz_matches_2d()
    test_rf_confinement_and_fates()
    test_cache_and_velocities()
    test_refusals()
    print("\nSTL 3-D GATE: ALL PASS")
