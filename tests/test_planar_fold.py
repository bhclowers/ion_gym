"""test_planar_fold.py — gate the 2a anchored fold (SCOPE_symmetry_slice2,
anchored-grid folding).

The fold is a pure optimisation: crop the masks at the anchored plane
node, solve the half (the solver's universal even-reflection edge at
index 0 IS the mirror BC),
reflect the bases back. Certification:

  F1  KNOWN ANSWER: for a declared, anchored, discretely-symmetric
      geometry, the folded solve equals the FULL-domain solve of the
      identical masks to solver tolerance, at two pitches.
  F2  MUTATION (refuse direction): a geometry whose shapes are NOT
      symmetric about the declared plane is REFUSED at Tier N (raise).
  F3  MIDLINE-DEFAULT FOLD (2b operating point): an undeclared midline
      mirror with exactly symmetric masks now FOLDS (the 2a declared-
      only boundary was lifted with the closed-edge recert), and the
      folded solve equals the full solve to solver tolerance — the same
      known-answer as F1, on the midline-default path.
  F4  BY-CONSTRUCTION invariant: the folded field is exactly even about
      the plane node (Ey(plane) == 0.0 identically).

OPERATING POINT: fixture is a symmetric slotted plate pair declared
about y = 5.5 mm (22 cells at 0.25, 55 cells at 0.1 — conforming at both
pitches); vacuum, DC only; solver tol 1e-6, agreement limit 50*tol
(iteration-path difference between the half and full solves).

SUB-CELL ANCHORING IS RETIRED.
This fixture previously declared y = 5.6975 mm, chosen to be OFF-GRID at
both pitches so the node lattice had to SHIFT to put the plane on a
node. The lattice rule ended that: a plane must sit an integer number of cells
from the origin, and `raster2d.anchored_grid` now REFUSES an off-lattice
plane before any shift can happen. The conflict was not merely with the
loader — an off-grid plane here also forces an off-grid EXTENT, because
this fixture sets H = 2*PLANE. You can have the plane on a node or the
walls on nodes, never both, unless the plane is an integer count from
the origin.

What anchoring still does, and what this file no longer covers: the
whole-frame `origin_mm` translation remains live and is exercised by 8
of the 45 two-dimensional decks in the tree (compactx_leg0b, four
of the 45 two-dimensional decks in the tree, all planar MRT and
all on conforming lattices. The SUB-CELL shift is now unreachable, and
this file was its only exercise. It is therefore uncovered because it is
GONE, not because it was forgotten. Restoring an off-grid plane here
would not restore the capability; it would only re-break the file.
"""
import _bootstrap  # noqa: F401
import numpy as np

from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec, SourceSpec)
from ion_gym.io.sim_spec import (SymmetrySpec)
from ion_gym.physics.build_planar import (verify_symmetry_shapes)
from ion_gym.physics.raster2d import (anchored_grid, planar_fold_axes, _fold_crop, _fold_reflect, electrode_mask)

# A7-conforming at BOTH pitches this file solves at: 5.5/0.25 = 22 cells
# exactly, 5.5/0.1 = 55 cells exactly. H = 2*PLANE keeps the domain
# symmetric about the plane and is likewise exact (44 and 110 cells), and
# W is 56 and 140. See the module docstring for why this is no longer the
# off-grid value it used to be.
PLANE = 5.5
W, H = 14.0, 11.0         # H = 2*PLANE: domain symmetric about the plane


def _pair_spec(h, declare=True, skew_mm=0.0):
    """Two identical slotted plates mirror-placed about PLANE; skew_mm
    shifts the TOP plate to break the symmetry for F2."""
    b_lo, b_hi = 1.7, 3.1                       # bottom plate y-span
    t_lo, t_hi = 2 * PLANE - b_hi, 2 * PLANE - b_lo
    # SELF-mirrored bases (the SLIM/mirror_export structure): the plate
    # PAIR is one conductor carrying both halves; a second self-mirrored
    # electrode exercises the multi-basis path. Distinct lo/hi electrodes
    # would be a mirror PAIR — detected and refused by the authority
    # (pair-folding is a flagged extension).
    els = [ElectrodeSpec(name="pair", dc=-250.0, shapes=[
               ShapeSpec("rect", dict(x_mm=2.0, y_mm=b_lo, width_mm=10.0,
                                      height_mm=b_hi - b_lo)),
               ShapeSpec("rect", dict(x_mm=2.0, y_mm=t_lo + skew_mm,
                                      width_mm=10.0,
                                      height_mm=t_hi - t_lo))]),
           ElectrodeSpec(name="wall", dc=40.0, shapes=[
               ShapeSpec("rect", dict(x_mm=0.3, y_mm=0.6, width_mm=0.7,
                                      height_mm=2 * PLANE - 1.2))])]
    sym = (SymmetrySpec(coords="xyz",
                        planes={"x": "none", "y": "mirror", "z": "none"},
                        plane_mm={"y": PLANE})
           if declare else SymmetrySpec())
    g = GeometrySpec(width_mm=W, height_mm=H, depth_mm=0.0, mm_per_gu=h,
                     symmetry=sym, electrodes=els)
    return SimSpec(name=f"fold_fixture_h{h}", geometry=g,
                   source=SourceSpec(seed=0, n_ions=1))


def _solve_masks(masks, tol=1e-6):
    from ion_gym.physics.multigrid3d import solve_bases_mg
    return solve_bases_mg(masks, mirror=(False, False, False), tol=tol,
                          stencil="ghost_linear", verbose=False)


def _masks_on(spec):
    xs, ys, anchor = anchored_grid(spec)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    m = {i: electrode_mask(e, X, Y)[:, :, None]
         for i, e in enumerate(spec.geometry.electrodes, start=1)}
    return xs, ys, anchor, m


def test_F1_folded_equals_full():
    tol = 1e-6
    for h in (0.25, 0.1):
        sp = _pair_spec(h)
        xs, ys, anchor, masks = _masks_on(sp)
        # WAS: assert anchor[1] != 0.0 — "fixture must force a nonzero
        # anchor". That assertion WAS the sub-cell anchoring coverage,
        # retired with the capability. The invariant
        # that replaces it is the one the lattice rule guarantees and the fold relies
        # on: the declared plane lands EXACTLY on a node index, so the
        # crop at that node is the mirror plane and not a neighbour of
        # it. Asserted as an exact integer, because "close to a node" is
        # the defect this whole gate exists to catch.
        _p_idx = (PLANE - ys[0]) / h
        assert abs(_p_idx - round(_p_idx)) < 1e-12, (
            f"declared plane {PLANE} mm must sit exactly on a node at "
            f"h={h}: got node index {_p_idx!r}")
        assert anchor[1] == 0.0, (
            f"a conforming plane needs no lattice shift; anchor={anchor}. "
            f"A non-zero y-anchor here would mean anchored_grid moved the "
            f"lattice for a plane that was already on a node.")
        sym = sp.geometry.symmetry.normalized()
        m2 = {i: m[:, :, 0] for i, m in masks.items()}
        fold = planar_fold_axes(sym, m2, xs, ys, {"x": W, "y": H})
        assert fold, "fixture must be foldable (anchored, symmetric)"
        half = _solve_masks({i: _fold_crop(m, fold)
                             for i, m in masks.items()}, tol)
        folded = {i: _fold_reflect(p, fold) for i, p in half.items()}
        full = _solve_masks(masks, tol)
        for i in sorted(full):
            d = float(np.abs(folded[i] - full[i]).max())
            print(f"  h={h}: basis {i} |folded-full|max = {d:.2e} "
                  f"(limit {50*tol:.0e})")
            assert d <= 50 * tol, (h, i, d)
    print("  F1 folded == full to solver tol at both pitches  OK")


def test_F2_asymmetric_shapes_refuse():
    sp = _pair_spec(0.25, skew_mm=0.3)
    g = sp.geometry
    ok, rep = verify_symmetry_shapes(g.symmetry, g.electrodes,
                                     {"x": W, "y": H})
    assert not ok, "0.3 mm skewed plate must fail the exact shape proof"
    from ion_gym.physics.build_planar import build_planar_model
    try:
        build_planar_model(sp)
    except ValueError as e:
        assert "refusing to fold a false plane" in str(e)
        print("  F2 skewed geometry refused at Tier N  OK")
    else:
        raise AssertionError("builder accepted a false declaration")


def test_F3_midline_default_folds_and_matches():
    # Self-contained midline-default fixture (no project import — test-
    # isolation rule): y-mirror declared WITHOUT plane_mm, binary-exact
    # pitch, node-aligned symmetric plates. 2b operating point: the
    # closed-edge raster makes the masks exactly mirror-equal, the 2a
    # declared-only boundary is lifted, so this FOLDS — and the folded
    # solve must equal the full solve to solver tolerance (the field
    # movement the 2b recert certifies).
    h, tol = 0.25, 1e-6
    els = [ElectrodeSpec(name="plates", dc=-100.0, shapes=[
               ShapeSpec("rect", dict(x_mm=2.0, y_mm=1.0, width_mm=8.0,
                                      height_mm=2.0)),
               ShapeSpec("rect", dict(x_mm=2.0, y_mm=9.0, width_mm=8.0,
                                      height_mm=2.0))])]
    sym = SymmetrySpec(coords="xyz",
                       planes={"x": "none", "y": "mirror", "z": "none"})
    g = GeometrySpec(width_mm=12.0, height_mm=12.0, depth_mm=0.0,
                     mm_per_gu=h, symmetry=sym, electrodes=els)
    sp = SimSpec(name="midline_default", geometry=g,
                 source=SourceSpec(seed=0, n_ions=1))
    xs, ys, anchor, masks = _masks_on(sp)
    assert anchor == (0.0, 0.0)          # no declared plane: no anchoring
    sn = sp.geometry.symmetry.normalized()
    m2 = {i: m[:, :, 0] for i, m in masks.items()}
    fold = planar_fold_axes(sn, m2, xs, ys,
                            {"x": g.width_mm, "y": g.height_mm})
    assert fold == {1: (len(ys) - 1) // 2}, (
        f"midline-default y-mirror must fold at the centre node "
        f"post-2b; got {fold}")
    half = _solve_masks({i: _fold_crop(m, fold)
                         for i, m in masks.items()}, tol)
    folded = {i: _fold_reflect(ph, fold) for i, ph in half.items()}
    full = _solve_masks(masks, tol)
    for i in sorted(full):
        d = float(np.abs(folded[i] - full[i]).max())
        print(f"  midline: basis {i} |folded-full|max = {d:.2e} "
              f"(limit {50*tol:.0e})")
        assert d <= 50 * tol, (i, d)
    print("  F3 midline-default fold active and == full solve  OK")


def test_F4_folded_field_exactly_even():
    sp = _pair_spec(0.25)
    xs, ys, anchor, masks = _masks_on(sp)
    sym = sp.geometry.symmetry.normalized()
    m2 = {i: m[:, :, 0] for i, m in masks.items()}
    fold = planar_fold_axes(sym, m2, xs, ys, {"x": W, "y": H})
    (ax_i, p), = fold.items()
    half = _solve_masks({i: _fold_crop(m, fold) for i, m in masks.items()})
    folded = _fold_reflect(next(iter(half.values())), fold)
    assert np.array_equal(folded, np.flip(folded, axis=ax_i))
    print("  F4 folded basis exactly even about the plane node  OK")


if __name__ == "__main__":
    test_F1_folded_equals_full()
    test_F2_asymmetric_shapes_refuse()
    test_F3_midline_default_folds_and_matches()
    test_F4_folded_field_exactly_even()
    print("PLANAR FOLD: ALL PASS")
