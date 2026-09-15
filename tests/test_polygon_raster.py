"""
test_polygon_raster.py -- gates for the planar polygon rasteriser.

The bug this pins: matplotlib's Path.contains_points has a WINDING-DEPENDENT
boundary rule, so a polygon and its mirror image classify on-edge sample
points differently.  Geometry whose faces land on cell centres then rasters
asymmetrically and verify_symmetry refuses to fold a plane that is, in
truth, exact.

The fix is a closed-solid rule (strict interior OR on-boundary, boundary by
DISTANCE, which is reflection-invariant).  These gates are deliberately
NOT about any one instrument: two unalike fixtures, plus adversarial
on-grid placement, so the fix cannot be configuration-specific.

  R0  area convergence           (both fixtures, vs analytic area)
  R1  reflection exactness       (x and y mirrors, edges ON cell centres)
  R2  winding independence       (CW vertex order == CCW vertex order)
  R3  closed solid               (a vertex/edge exactly on a sample point
                                  is METAL, matching the flyer's tangent rule)
  R4  rect/polygon agreement     (a polygon square == the rect primitive)
"""
import _bootstrap  # noqa: F401  -- repo root on sys.path
import sys
from pathlib import Path as _P

import numpy as np

sys.path.insert(0, str(_P(__file__).resolve().parent))
from ion_gym.io.sim_spec import ShapeSpec
from ion_gym.physics.raster2d import (_shape_mask)

FAILED = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILED.append(name)


def grid(w, h_mm):
    n = int(round(w / h_mm))
    xs = (np.arange(n) + 0.5) * h_mm
    return np.meshgrid(xs, xs, indexing="ij")


def poly(pts):
    return ShapeSpec("polygon", {"points_mm": [list(p) for p in pts]})


def shoelace(p):
    p = np.asarray(p, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# ---- FIXTURE A: an L-bracket (concave, all edges axis-aligned, placed so
# every face lands EXACTLY on a cell centre at h = 0.1 -- the adversarial
# case that broke the old rasteriser)
W = 10.0
A = [(2.05, 2.05), (7.05, 2.05), (7.05, 4.05), (4.05, 4.05),
     (4.05, 7.05), (2.05, 7.05)]
# mirrored about the domain centre x = 5
A_MX = [(W - x, y) for (x, y) in A]

# ---- FIXTURE B: a rotated hexagon (no axis-aligned edges, generic angles)
th = np.deg2rad(np.arange(6) * 60.0 + 17.0)
B = [(5.0 + 3.0 * np.cos(t), 5.0 + 3.0 * np.sin(t)) for t in th]
B_MY = [(x, W - y) for (x, y) in B]

print("\nR0  area convergence")
for nm, P in (("L-bracket", A), ("hexagon", B)):
    a_true = shoelace(P)
    prev = None
    for h in (0.2, 0.1, 0.05):
        X, Y = grid(W, h)
        a = _shape_mask(poly(P), X, Y).sum() * h * h
        err = (a - a_true) / a_true
        print(f"      {nm:10s} h={h:.2f}  area {a:8.4f} vs {a_true:8.4f}"
              f"  ({100 * err:+.3f}%)")
        prev = abs(err)
    gate(f"R0.{nm}", prev < 0.01, f"finest-grid error {100 * prev:.3f}%")

print("\nR1  reflection exactness (edges ON cell centres)")
for h in (0.1, 0.05, 0.02):
    X, Y = grid(W, h)
    ma = _shape_mask(poly(A), X, Y)
    ma_m = _shape_mask(poly(A_MX), X, Y)
    # the mirrored POLYGON must equal the mirrored MASK, cell for cell
    d = (ma[::-1, :] ^ ma_m).sum()
    gate(f"R1.L-bracket h={h}", d == 0, f"mismatch {d} cells")
    mb = _shape_mask(poly(B), X, Y)
    mb_m = _shape_mask(poly(B_MY), X, Y)
    d = (mb[:, ::-1] ^ mb_m).sum()
    gate(f"R1.hexagon  h={h}", d == 0, f"mismatch {d} cells")

print("\nR2  winding independence")
X, Y = grid(W, 0.05)
for nm, P in (("L-bracket", A), ("hexagon", B)):
    d = (_shape_mask(poly(P), X, Y) ^
         _shape_mask(poly(P[::-1]), X, Y)).sum()
    gate(f"R2.{nm}", d == 0, f"CW vs CCW mismatch {d} cells")

print("\nR3  closed solid (sample point exactly on a face/vertex is METAL)")
# a square whose faces land exactly on cell centres at h = 0.1
S = [(2.05, 2.05), (6.05, 2.05), (6.05, 6.05), (2.05, 6.05)]
X, Y = grid(W, 0.1)
m = _shape_mask(poly(S), X, Y)
xs = (np.arange(int(W / 0.1)) + 0.5) * 0.1
i_face = int(np.argmin(abs(xs - 6.05)))       # cell centre ON the +x face
j_mid = int(np.argmin(abs(xs - 4.05)))
gate("R3.face_is_metal", bool(m[i_face, j_mid]),
     f"cell at x={xs[i_face]:.2f} (on the face) is "
     f"{'metal' if m[i_face, j_mid] else 'VACUUM'}")
i_v = int(np.argmin(abs(xs - 2.05)))
gate("R3.vertex_is_metal", bool(m[i_v, i_v]),
     f"cell at the (2.05, 2.05) vertex is "
     f"{'metal' if m[i_v, i_v] else 'VACUUM'}")
# and the closure is symmetric: both x faces behave the same
i_face2 = int(np.argmin(abs(xs - 2.05)))
gate("R3.both_faces", bool(m[i_face, j_mid]) == bool(m[i_face2, j_mid]),
     "the -x and +x faces classify identically")

print("\nR4  polygon square == rect primitive")
X, Y = grid(W, 0.05)
r = _shape_mask(ShapeSpec("rect", {"x_mm": 2.05, "y_mm": 2.05,
                                   "width_mm": 4.0, "height_mm": 4.0}), X, Y)
p = _shape_mask(poly(S), X, Y)
gate("R4.agree", (r ^ p).sum() == 0,
     f"rect vs polygon mismatch {(r ^ p).sum()} cells")

print("\n" + "=" * 60)
print("FAILED: " + (", ".join(FAILED) if FAILED else "none"))
print("=" * 60)
sys.exit(1 if FAILED else 0)
