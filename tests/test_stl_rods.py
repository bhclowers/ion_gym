"""
test_stl_rods.py — gate STL import on REAL user CAD (examples/rods,
four rod electrodes, RF pairing 1+3 / 2+4).

What it proves, in order:
  1. Part-frame trap CAUGHT: the rods as-uploaded were exported per-part
     (assembly placement lost) and interpenetrate; the voxelizer refuses
     and check_mask_overlaps surfaces the refusal. Fast-looking-wrong is
     not an option here by construction.
  2. Assembled at r0 = 3.84 mm (circle-fit each rod's arc, translate its
     centre to the diagonal at r0 + R_fit, flats outward): four disjoint,
     single-component masks through the VALIDATED stl_masks_2d path.
  3. Solved on the validated build path, the field is a quadrupole:
     phi(axis) ~ 0, 90-degree rotation antisymmetry < 5% on an r = 2 mm
     ring, cos(2θ) >= 95% of low harmonics, ring amplitude within 15% of
     the ideal hyperbolic 100*(r/r0)^2.
Pairing wiring (dc: 1,3 = +100; 2,4 = -100; rf_group A/B) mirrors the
user's stated convention and lands in the spec.
"""
import io
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from ion_gym.io.stl_upload import (load_mesh_bytes, propose_sizing, spec_from_upload,
                        check_mask_overlaps)

R0 = 3.84
DIAG = {1: (-1, 1), 2: (1, 1), 3: (1, -1), 4: (-1, -1)}   # flats outward
DC = {"Rod_1": 100.0, "Rod_3": 100.0, "Rod_2": -100.0, "Rod_4": -100.0}
RF = {"Rod_1": "A", "Rod_3": "A", "Rod_2": "B", "Rod_4": "B"}


def _raw():
    return {i: open(f"examples/rods/Rod_{i}.STL", "rb").read()
            for i in (1, 2, 3, 4)}


def _fit_circle(pts):
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(axis=1)
    cx, cy, c0 = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(cx), float(cy), float(np.sqrt(c0 + cx * cx + cy * cy))


def assemble(raw, r0=R0):
    out = []
    for i in (1, 2, 3, 4):
        m = trimesh.load(io.BytesIO(raw[i]), file_type="stl",
                         force="mesh", process=True)
        sec = m.section(plane_origin=[0, 0, m.bounds[:, 2].mean()],
                        plane_normal=[0, 0, 1])
        pts = np.vstack([p[:, :2] for p in sec.discrete])
        cx, cy, R = _fit_circle(pts)
        ux, uy = DIAG[i]
        t = np.array([ux, uy]) / np.hypot(ux, uy) * (r0 + R)
        m.apply_translation([t[0] - cx, t[1] - cy, 0.0])
        out.append(load_mesh_bytes(
            f"Rod_{i}.STL", trimesh.exchange.export.export_stl(m)))
    return out


def test_part_frame_caught():
    raw = _raw()
    meshes = [load_mesh_bytes(f"Rod_{i}.STL", raw[i]) for i in (1, 2, 3, 4)]
    s = propose_sizing(meshes, pitch=0.25, margin_mm=1.5)
    spec = spec_from_upload(meshes, DC, s, "/tmp/rods_gate_asis",
                            rf_groups=RF)
    ov = check_mask_overlaps(spec)
    assert ov and "overlaps" in ov[0][0], \
        "part-frame interpenetration must be refused and surfaced"
    print("  part-frame export caught (voxelizer refusal surfaced)  OK")


def test_assembled_quad_field():
    assembled = assemble(_raw())
    s = propose_sizing(assembled, pitch=0.2, margin_mm=2.0)
    spec = spec_from_upload(assembled, DC, s, "/tmp/rods_gate_asm",
                            rf_groups=RF)
    # DECLARED seed: this spec is built and flown; seed 0
    # = the historical draws its expectations were measured under.
    spec.source.seed = 0
    # pairing landed in the spec
    e = {el.name: el for el in spec.geometry.electrodes}
    assert e["Rod_1"].dc == e["Rod_3"].dc == 100.0
    assert e["Rod_2"].dc == e["Rod_4"].dc == -100.0
    # AMENDED: rf_group (scalar) was folded into rf_groups
    # (membership list) by the drive-schema migration; the
    # gate lagged the fold.
    assert e["Rod_1"].rf_groups == e["Rod_3"].rf_groups == ["A"]
    assert e["Rod_2"].rf_groups == e["Rod_4"].rf_groups == ["B"]

    assert not check_mask_overlaps(spec), "assembled rods must be disjoint"
    from ion_gym.physics.build_stl import stl_masks_2d
    from scipy import ndimage
    masks = stl_masks_2d(spec)
    assert set(masks) == {1, 2, 3, 4}
    for k in masks:
        assert ndimage.label(masks[k])[1] == 1, f"electrode {k} fragmented"

    from ion_gym.physics.sim_build import build_run
    model, fly, cols, births = build_run(spec)
    z, r_full, phi, em = model.potential_image()
    ax0 = -s.domain_min[:2]
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((np.asarray(z), np.asarray(r_full)), phi)
    phi_axis = float(f([ax0])[0])
    assert abs(phi_axis) < 2.0, f"axis potential {phi_axis} not ~0"

    th = np.linspace(0, 2 * np.pi, 73)[:-1]
    rr = 2.0
    xs, ys = ax0[0] + rr * np.cos(th), ax0[1] + rr * np.sin(th)
    xr, yr = ax0[0] - rr * np.sin(th), ax0[1] + rr * np.cos(th)
    v, vr = f(np.c_[xs, ys]), f(np.c_[xr, yr])
    res_anti = np.max(np.abs(v + vr)) / np.max(np.abs(v))
    assert res_anti < 0.05, f"antisymmetry residual {res_anti:.3f}"
    H = np.abs(np.fft.rfft(v))
    purity = H[2] / H[1:8].sum()
    assert purity > 0.95, f"cos(2θ) purity {purity:.3f}"
    A2 = 2 * H[2] / len(v)
    ideal = 100.0 * (rr / R0) ** 2
    assert abs(A2 - ideal) / ideal < 0.15, f"A2 {A2:.2f} vs ideal {ideal:.2f}"
    print(f"  assembled quad: axis {phi_axis:+.2f} V, antisym "
          f"{res_anti*100:.1f}%, cos2θ {purity*100:.1f}%, "
          f"A2 {A2:.1f}/{ideal:.1f} V  OK")


if __name__ == "__main__":
    test_part_frame_caught()
    test_assembled_quad_field()
    print("\nSTL RODS GATE: ALL PASS")
