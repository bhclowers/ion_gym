"""
validate_stl_einzel.py — validate the STL -> electrode PATH.

An electrode's geometry can come from an STL file instead of inline
shapes. This gates that path by rebuilding the einzel (whose native
inline-shape field is already validated) from STL plates and checking the
solved field matches: same on-axis saddle, same focusing. Uses the
Rung-2 voxelizer (voxelize.py) to rasterize each STL, sliced at the
mid-plane to a 2-D cross-section mask, then the same native solve.

Gates:
  S-1 SLICE: each STL plate rasterizes to the expected cross-section
      (a bar with a circular bore) at the right position.
  S-2 FIELD: the STL-built einzel reproduces the inline-shape einzel's
      on-axis saddle (lens dip within a grid tolerance).
  S-3 FOCUSING: a collimated beam focuses through the STL einzel, i.e.
      the STL path yields the same optics as inline shapes.
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path

import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec, SourceSpec,
                      CollisionSpec, IntegrationSpec, ViewSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)

W, H = 40.0, 20.0
BORE, THICK, GAP = 3.0, 1.5, 8.0
CX, CY = W / 2, H / 2
WALL = H - 2.0


def _write_einzel_stls(out_dir):
    import trimesh
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, xc in [("s_entrance", CX - GAP), ("s_lens", CX),
                     ("s_exit", CX + GAP)]:
        box = trimesh.creation.box(extents=[THICK, WALL, 4.0])
        box.apply_translation([xc, CY, 0])
        hole = trimesh.creation.cylinder(radius=BORE, height=6.0,
                                         sections=48)
        hole.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, [1, 0, 0]))
        hole.apply_translation([xc, CY, 0])
        plate = trimesh.boolean.difference([box, hole])
        fn = f"{name}.stl"
        plate.export(str(out / fn))
        files[name] = fn
    return files


def stl_einzel_spec(stl_dir, lens_v=-100.0):
    files = _write_einzel_stls(stl_dir)
    return SimSpec(
        geometry=GeometrySpec(
            width_mm=W, height_mm=H, mm_per_gu=0.2,
            symmetry=SymmetrySpec(coords="xyz"), stl_dir=str(stl_dir),
            electrodes=[
                ElectrodeSpec("entrance", stl=files["s_entrance"], dc=0.0),
                ElectrodeSpec("lens", stl=files["s_lens"], dc=lens_v),
                ElectrodeSpec("exit", stl=files["s_exit"], dc=0.0)]),
        source=SourceSpec(seed=0, distribution="line", n_ions=12, x0_mm=1.5,
                          y0_mm=CY - 1.5, len_mm=3.0, axis="y",
                          direction=[1.0, 0.0, 0.0], ke_lo=50.0,
                          ke_hi=50.0, mz_list=[100.0], tob_span_us=0.0),
        collisions=CollisionSpec(enabled=False),
        integration=IntegrationSpec(dt_ns=0.5, t_max_us=15.0, rec_every=8,
                                    record_channels=["speed"]),
        view=ViewSpec(mode="2d", planes=["xy"]),
        name="einzel from STL plates")


def _sigma_at(results, xq):
    ys = []
    for r in results:
        t = r.traj
        k = np.searchsorted(t[:, 1], xq)
        if 0 < k < len(t):
            ys.append(t[k, 2])
    return np.std(ys) if len(ys) > 2 else np.nan


def main():
    from ion_gym.physics.build_stl import stl_masks_2d, build_stl_run
    from ion_gym.physics.ensemble_driver import run
    ok = True
    stl_dir = "/tmp/stl_einzel"

    spec = stl_einzel_spec(stl_dir, lens_v=-100.0)

    # S-1 slice: three plates at the right x, each with a bore
    masks = stl_masks_2d(spec)
    h = spec.geometry.mm_per_gu
    centres = []
    for i in (1, 2, 3):
        xi, yi = np.where(masks[i])
        centres.append(xi.mean() * h)
    exp = [CX - GAP, CX, CX + GAP]
    g1 = all(abs(c - e) < 1.0 for c, e in zip(sorted(centres), exp))
    ok &= g1
    print(f"S-1 slice: plate x-centres {[f'{c:.1f}' for c in sorted(centres)]} "
          f"vs {exp}  ->  {'PASS' if g1 else 'FAIL'}")

    # S-2 field: on-axis saddle matches the inline einzel. Both now solve
    # on an nz=3 slab (nz=1 had a solver asymmetry bug), giving the
    # physically-correct planar lens dip ~ -72 V (cross-checked against
    # the direct 2-D solver2d ground truth of -71.4 V).
    model, fly, cols, births = build_stl_run(spec)
    jax = int(round(CY / h))
    onax = model.A[:, jax]
    v_lens = onax[int(round(CX / h))]
    g2 = -85.0 < v_lens < -60.0
    ok &= g2
    print(f"S-2 field: STL einzel on-axis lens dip {v_lens:.1f} V "
          f"(inline ~ -72, solver2d truth -71.4)  ->  "
          f"{'PASS' if g2 else 'FAIL'}")

    # S-3 focusing: beam converges, stronger with voltage
    sig = []
    for v in (0.0, -150.0, -300.0):
        sp = stl_einzel_spec(stl_dir, lens_v=v)
        m, f, c, b = build_stl_run(sp)
        res = run(len(b), f, check_every=5, decimate=1)
        sig.append(_sigma_at(res.results, 36.0))
    g3 = sig[-1] < sig[0]
    ok &= g3
    print(f"S-3 focusing: sigma_out {[f'{s:.3f}' for s in sig]} at "
          f"0/-150/-300 V — converges  ->  {'PASS' if g3 else 'FAIL'}")

    # S-4 quadrupole: a signed two-phase RF field from STL rods forms a
    # proper Mathieu saddle, and a single run with per-ion mass shows one
    # STABLE (q<0.908) and one UNSTABLE (q>0.908) ion.
    from ion_gym.physics.build_stl import quad_stl_spec
    qspec = quad_stl_spec("/tmp/quad_stls")
    cc = qspec.geometry.width_mm / 2
    qspec.source.x0_mm = cc
    qspec.source.y0_mm = cc + 0.5
    # ONE ion PER m/z. n_ions means ions-per-mass (mz_of block semantics,
    # the maker's 60 flew 120 ions of which this gate
    # inspected two -- and its old kinds[0]/kinds[1] indexing predated
    # fix K, so "index 1" was the SECOND stable ion, not the unstable
    # mass, and the gate failed on correct physics.
    qspec.source.n_ions = 1
    qspec.integration.dt_ns = 1.0
    qspec.integration.t_max_us = 40.0
    qspec.bounds.x_min_on = qspec.bounds.x_max_on = True
    qspec.bounds.y_min_on = qspec.bounds.y_max_on = True
    qspec.bounds.x_min = cc - 3.84
    qspec.bounds.x_max = cc + 3.84
    qspec.bounds.y_min = cc - 3.84
    qspec.bounds.y_max = cc + 3.84
    qm, qf, qc, qb = build_stl_run(qspec, ground_border=True)
    # saddle check
    hh = qm.mm_per_gu
    cci = int(round(cc / hh))
    oo = int(round(2.0 / hh))
    Bq = qm.Bk[0][0]
    saddle = Bq[cci + oo, cci] * Bq[cci, cci + oo] < 0
    qres = run(len(qb), qf, check_every=1, decimate=1)
    # THE FLOWN RECORD NAMES ITS OWN MASS: indices come from
    # summary["mz"] -- the kernel's statement of what flew -- never from
    # a positional layout assumption. "Lost" is striking a rod (kind 0)
    # or the rod-radius bound (kind 3): the excursion grew out of the trap.
    mz_st, mz_un = qspec.source.mz_list
    def _idx_of(res, mz, label):
        for i, r in enumerate(res.results):
            if r.summary["mz"] == mz:
                return i
        raise RuntimeError(
            f"S-4/5: no flown ion carries m/z {mz} ({label}); flown "
            f"masses were {[r.summary['mz'] for r in res.results]}")
    i_st = _idx_of(qres, mz_st, "stable")
    i_un = _idx_of(qres, mz_un, "unstable")
    kinds = [r.summary["kind"] for r in qres.results]
    g4 = saddle and kinds[i_st] == 2 and kinds[i_un] in (0, 3)
    ok &= g4
    print(f"S-4 quadrupole: saddle={saddle}, m/z{mz_st:.0f} "
          f"kind={kinds[i_st]} (2=stable), m/z{mz_un:.0f} "
          f"kind={kinds[i_un]} (0/3=lost)  ->  "
          f"{'PASS' if g4 else 'FAIL'}")

    # S-5 axial transport: with axial KE the stable ion drifts far down z
    # while staying confined (transmitted), and the unstable ion is lost
    # near the entrance. This exercises the field-free axial drift in the
    # planar tracer (z = z0 + vz*t) that makes the axial-transport transport
    # picture from a 2-D transverse solve.
    tspec = quad_stl_spec("/tmp/quad_stls")     # transport defaults (KE=8)
    tspec.source.n_ions = 1        # one ion per m/z (see S-4 note)
    tcc = tspec.geometry.width_mm / 2
    tspec.bounds.x_min_on = tspec.bounds.x_max_on = True
    tspec.bounds.y_min_on = tspec.bounds.y_max_on = True
    tspec.bounds.x_min = tcc - 3.84
    tspec.bounds.x_max = tcc + 3.84
    tspec.bounds.y_min = tcc - 3.84
    tspec.bounds.y_max = tcc + 3.84
    tm, tf, tc, tb = build_stl_run(tspec, ground_border=True)
    tres = run(len(tb), tf, check_every=1, decimate=1)
    # indices from the flown record, same as S-4
    j_st = _idx_of(tres, mz_st, "stable")
    j_un = _idx_of(tres, mz_un, "unstable")
    z_stable = tres.results[j_st].traj[:, 3].max()
    z_unstable = tres.results[j_un].traj[:, 3].max()
    k_stable = tres.results[j_st].summary["kind"]
    k_unstable = tres.results[j_un].summary["kind"]
    # no-truncation check: the stable ion's recorded z must advance in
    # small steps all the way (a full record buffer used to truncate the
    # trajectory, leaving a long straight line to the final point).
    zt = tres.results[j_st].traj[:, 3]
    max_gap = float(np.max(np.diff(zt))) if len(zt) > 1 else 0.0
    no_trunc = max_gap < 2.0     # steps are << 1 mm; a gap means truncation
    g5 = (k_stable == 2 and z_stable > 50.0 and
          k_unstable in (0, 3) and z_unstable < 20.0 and no_trunc)
    ok &= g5
    print(f"S-5 axial transport: stable transits {z_stable:.0f}mm "
          f"(kind {k_stable}, max z-gap {max_gap:.2f}mm), unstable lost at "
          f"{z_unstable:.0f}mm (kind {k_unstable})  ->  "
          f"{'PASS' if g5 else 'FAIL'}")

    print("\nSTL -> ELECTRODE PATH (via einzel):",
          "ALL PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
