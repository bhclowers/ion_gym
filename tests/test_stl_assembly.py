"""
test_stl_assembly.py — the ANSWER to "how should users export complex
geometry": one STL PER ELECTRODE, exported FROM THE ASSEMBLY (placement
preserved). Companion to test_stl_rods.py, which proves the per-part
export of the SAME rods is refused (interpenetration).

Fixtures: examples/quad_assembly (user's Quad_Assembly Rod A-1..D-4,
pairing A+C / B+D = the 1+3 / 2+4 convention). Asserts, untouched:
  * zero overlaps as-exported (assembly placement intact),
  * four disjoint single-component masks (validated stl_masks_2d),
  * quadrupole field: phi(axis) ~ 0; on r = 1.5/2.0/2.5 mm rings the
    90-degree antisymmetry < 5%, cos(2θ) purity > 95%, and A2/ideal is
    CONSTANT across radii (round-rod field constant; spread < 2%) at
    0.95-1.00 of hyperbolic with r0 = 3.856 mm (closest-approach r0
    measured from the assembly itself — which also settled the r0
    question at ~the user's stated 3.84).
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import glob
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from ion_gym.io.stl_upload import (load_mesh_bytes, propose_sizing, spec_from_upload,
                        check_mask_overlaps)

R0_TRUE = 3.856          # measured closest-approach from the assembly
AXIS = np.array([10.248, 10.145])   # measured quad axis (original frame)


def main():
    files = sorted(glob.glob("examples/quad_assembly/*.STL"))
    assert len(files) == 4, "assembly fixtures missing"
    meshes = [load_mesh_bytes(f.split("/")[-1], open(f, "rb").read())
              for f in files]
    n = [m.name for m in meshes]           # A-1, B-2, C-3, D-4 (sorted)
    DC = {n[0]: 100.0, n[2]: 100.0, n[1]: -100.0, n[3]: -100.0}
    RF = {n[0]: "A", n[2]: "A", n[1]: "B", n[3]: "B"}

    s = propose_sizing(meshes, pitch=0.15, margin_mm=2.0)
    spec = spec_from_upload(meshes, DC, s, "/tmp/quad_asm_gate",
                            rf_groups=RF)
    # DECLARED seed: spec_from_upload defaults seed=None =
    # fresh entropy; seed 0 = the historical draws.
    spec.source.seed = 0
    assert not check_mask_overlaps(spec), \
        "assembly export must import with ZERO overlaps"

    from ion_gym.physics.build_stl import stl_masks_2d
    from scipy import ndimage
    masks = stl_masks_2d(spec)
    assert set(masks) == {1, 2, 3, 4}
    for k in masks:
        assert ndimage.label(masks[k])[1] == 1

    from ion_gym.physics.sim_build import build_run
    model, fly, cols, births = build_run(spec)
    z, r_full, phi, em = model.potential_image()
    ax0 = AXIS - s.domain_min[:2]
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((np.asarray(z), np.asarray(r_full)), phi)
    assert abs(float(f([ax0])[0])) < 2.0

    th = np.linspace(0, 2 * np.pi, 145)[:-1]
    ratios = []
    for rr in (1.5, 2.0, 2.5):
        xs, ys = ax0[0] + rr * np.cos(th), ax0[1] + rr * np.sin(th)
        xr, yr = ax0[0] - rr * np.sin(th), ax0[1] + rr * np.cos(th)
        v, vr = f(np.c_[xs, ys]), f(np.c_[xr, yr])
        anti = np.max(np.abs(v + vr)) / np.max(np.abs(v))
        H = np.abs(np.fft.rfft(v))
        purity = H[2] / H[1:8].sum()
        A2 = 2 * H[2] / len(v)
        ratio = A2 / (100.0 * (rr / R0_TRUE) ** 2)
        assert anti < 0.05 and purity > 0.95, f"r={rr}: {anti}, {purity}"
        ratios.append(ratio)
    ratios = np.array(ratios)
    assert 0.95 <= ratios.mean() <= 1.00, f"field constant {ratios.mean()}"
    assert np.ptp(ratios) < 0.02, \
        f"A2/ideal must be radius-constant (round-rod signature): {ratios}"
    print(f"  assembly import: 0 overlaps, 4x1 components, field constant "
          f"{ratios.mean()*100:.1f}% (spread {np.ptp(ratios)*100:.2f}%)  OK")


if __name__ == "__main__":
    main()
    print("\nSTL ASSEMBLY GATE: ALL PASS")
