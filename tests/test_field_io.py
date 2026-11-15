"""
Gate: field_io — field portability.

A saved field is the deck's solve-cache entry exported as one file (with
the deck beside it), the same on every route; the dashboard and notebooks
call these same functions.

Covers:
  F1  a solved field saves, survives a RENAME, and loads into an empty
      cache: bases and labels come back bit-identical, and the geometry
      is a cache hit
  F2  a field loaded for a different geometry HARD-REFUSES, naming both
      cache entries
  F3  fingerprint anchors an identical solve; a perturbed basis DIVERGES
      with the offending basis/node named
  F4  trajectory save/load: matching spec -> no warning; different
      geometry -> warning string (overlay allowed, never a refusal)
  F5  an older array-level field file is refused by name, and the
      saved-fields list shows why rather than dropping it
  F6  a label is kept in the registry beside a custom save folder; the
      list matches the deck and shows the label
  F7  r-z: save -> open from the file alone (the deck it was solved for,
      then the field) round-trip; a file that does not record its deck
      refuses
  F8  provenance: many-to-one recorded, never keyed

No solver run: a solve is stood in for by storing synthetic bases the way
the solver stores them (basis_cache.store). All files under a tempdir;
the repo fields/ directory is never touched.

Run: python tests/test_field_io.py     (PASS/FAIL lines, exit code)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ion_gym.io import field_io, fa_cache, basis_cache
from ion_gym.io.sim_spec import (SimSpec, GeometrySpec, ElectrodeSpec,
                                 ShapeSpec)

FAILED = []


def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except AssertionError as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


def _rect(x, y, w, h):
    return ShapeSpec(type="rect",
                     params=dict(x_mm=x, y_mm=y, w_mm=w, h_mm=h))


def _spec(width=20.0, name="gate_field"):
    g = GeometrySpec(
        width_mm=width, height_mm=12.0, mm_per_gu=0.5,
        electrodes=[
            ElectrodeSpec(name="top", dc=100.0,
                          shapes=[_rect(2.0, 10.0, 16.0, 1.0)]),
            ElectrodeSpec(name="bot", dc=-50.0,
                          shapes=[_rect(2.0, 1.0, 16.0, 1.0)]),
        ])
    return SimSpec(geometry=g, name=name)


def _solved(spec, seed=0):
    """Synthetic 'solve' consistent with the spec: bases per electrode +
    an int16 label array, one fixed grid."""
    rng = np.random.default_rng(seed)
    shape = (40, 24)
    bases = {i + 1: rng.standard_normal(shape) * 1e4
             for i in range(len(spec.geometry.electrodes))}
    ele = np.zeros(shape, np.int16)
    ele[-2:, :] = 1
    ele[:2, :] = 2
    return bases, ele


def main():
    spec = _spec()
    bases, ele = _solved(spec)
    tmp = tempfile.mkdtemp(prefix="field_io_gate.")
    cache_root = os.path.join(tmp, "cache")

    # ---------------------------------------------------------------- F1
    def f1():
        basis_cache.store(spec, bases, ele, root=cache_root)
        p, deck = field_io.save_field(spec, directory=tmp, root=cache_root)
        assert deck and os.path.exists(deck), "deck JSON not beside it"
        renamed = os.path.join(tmp, "totally arbitrary name (sent).npz")
        os.replace(p, renamed)
        fresh = os.path.join(tmp, "fresh_cache")
        assert field_io.load_field(renamed, spec, root=fresh) == \
            field_io.field_key(spec)
        b2, e2 = basis_cache.load(spec, root=fresh)
        assert b2 is not None, "no cache hit after load_field"
        assert sorted(b2) == sorted(int(k) for k in bases), "basis keys"
        for k in bases:
            assert np.array_equal(b2[int(k)], bases[k]), f"basis {k} bits"
        assert np.array_equal(e2, ele), "ele bits"
        # the file read directly, without the cache: per-volt bases (the
        # solver's V_BASIS divided out)
        from ion_gym.physics.build_planar import V_BASIS
        fb, fl, fm = field_io.read_field_arrays(renamed)
        assert sorted(fb) == sorted(int(k) for k in bases)
        for k in bases:
            assert np.array_equal(fb[int(k)], bases[k] / V_BASIS), \
                f"file basis {k}"
        assert np.array_equal(fl, ele) and fm["spec_name"] == spec.name
    check("F1 save -> rename -> load into an empty cache, bit-identical", f1)

    # ---------------------------------------------------------------- F2
    def f2():
        basis_cache.store(spec, bases, ele, root=cache_root)
        p, _ = field_io.save_field(spec, directory=os.path.join(tmp, "f2"),
                                   root=cache_root)
        other = _spec(width=25.0)          # ONE geometry field differs
        try:
            field_io.load_field(p, other, root=cache_root)
        except field_io.FieldMismatch as e:
            msg = str(e)
            assert "different geometry" in msg, msg
            assert field_io.field_key(other)[:8] in msg, msg
            return
        raise AssertionError("wrong-geometry load did NOT refuse")
    check("F2 wrong-geometry load hard-refuses, names both entries", f2)

    # ---------------------------------------------------------------- F3
    def f3():
        fp = field_io.fingerprint(spec, bases, ele)
        dv = field_io.verify_fingerprint(fp, spec, bases, ele)
        assert dv == 0.0, f"self-anchor deviation {dv}"
        bad = {k: v.copy() for k, v in bases.items()}
        node = tuple(fp["probe_nodes"][0])
        bad[1][node] += 0.5                # > atol_v, < anything physical? no: named
        try:
            field_io.verify_fingerprint(fp, spec, bad, ele)
        except field_io.FieldMismatch as e:
            assert "b1" in str(e), f"offending basis not named: {e}"
            return
        raise AssertionError("perturbed solve passed the fingerprint")
    check("F3 fingerprint anchors; perturbation diverges, names b1", f3)

    # ---------------------------------------------------------------- F4
    def f4():
        class R:                            # minimal IonResult shape
            def __init__(self, i, t, s):
                self.index, self.traj, self.summary = i, t, s
        res = [R(0, np.arange(12.0).reshape(4, 3), {"kind": 0}),
               R(1, None, {"kind": 2}),     # unrecorded -> skipped+counted
               R(2, np.ones((2, 3)), {"kind": 1})]
        p = field_io.save_trajectories(res, spec,
                                       path=os.path.join(tmp, "t.traj.npz"),
                                       columns=["t", "x", "y"])
        trajs, meta, warn = field_io.load_trajectories(p, spec)
        assert warn is None, f"matching spec warned: {warn}"
        assert sorted(trajs) == [0, 2] and meta["n_skipped"] == 1
        _, _, warn2 = field_io.load_trajectories(p, _spec(width=25.0))
        assert warn2 and "different geometry" in warn2, warn2
    check("F4 trajectory round-trip; overlay warns, never refuses", f4)

    # ---------------------------------------------------------------- F5
    def f5():
        old_dir = os.path.join(tmp, "older")
        os.makedirs(old_dir)
        old = os.path.join(old_dir, "older_format.npz")
        field_io._atomic_savez(
            old, b1=bases[1], ele=ele,
            **{field_io._META: field_io._meta_to_member(
                {"kind": "ion_gym_field", "field_format": 1})})
        try:
            field_io.load_field(old, spec, root=cache_root)
        except field_io.FieldMismatch as e:
            assert "older field format" in str(e), str(e)
        else:
            raise AssertionError("an older-format field loaded")
        rows = field_io.scan_fields(directory=old_dir, spec=spec)
        assert len(rows) == 1 and "older field format" in rows[0]["problem"]
    check("F5 an older-format field is refused by name and listed", f5)

    # ---------------------------------------------------------------- F6
    def f6():
        sub = os.path.join(tmp, "custom_dir")
        basis_cache.store(spec, bases, ele, root=cache_root)
        field_io.save_field(spec, directory=sub, label="here",
                            root=cache_root)
        assert os.path.exists(os.path.join(sub, "fields_registry.json")), \
            "registry not written beside the saved file"
        rows = field_io.scan_fields(directory=sub, spec=spec)
        assert len(rows) == 1 and rows[0]["matches"] is True, rows
        assert rows[0]["label"] == "here", rows[0]["label"]
    check("F6 label kept beside a custom save dir; scan matches", f6)

    # ---------------------------------------------------------------- F7
    def f7():
        # r-z keys through the same canonical machinery (symmetry.coords
        # distinguishes it -- no r-z branch) and reopens from the file
        # alone: the deck it was solved for, then the field.
        from ion_gym.io.sim_spec import SymmetrySpec
        g = GeometrySpec(
            width_mm=20.0, height_mm=8.0, mm_per_gu=0.5,
            symmetry=SymmetrySpec(coords="rz"),
            electrodes=[
                ElectrodeSpec(name="ring", dc=75.0,
                              shapes=[_rect(4.0, 3.0, 4.0, 2.0)]),
                ElectrodeSpec(name="cap", dc=0.0, is_grid=True,
                              shapes=[_rect(14.0, 0.0, 1.0, 6.0)]),
            ])
        rspec = SimSpec(geometry=g, name="gate_rz")
        rbases, rele = _solved(rspec, seed=7)
        assert basis_cache.key(rspec) != basis_cache.key(spec), \
            "rz spec must key differently from the planar fixture"
        basis_cache.store(rspec, rbases, rele, root=cache_root)
        p, _ = field_io.save_field(rspec, directory=os.path.join(tmp, "rz"),
                                   root=cache_root)
        s2, meta = field_io.read_spec_from_field(p)
        assert basis_cache.key(s2) == basis_cache.key(rspec), \
            "reconstructed spec keys differently from the original"
        assert s2.geometry.symmetry.coords == "rz", "coords lost"
        assert s2.geometry.electrodes[1].is_grid is True, "is_grid lost"
        rt = os.path.join(tmp, "rz_boot_cache")
        field_io.load_field(p, s2, root=rt)
        b2, e2 = basis_cache.load(s2, root=rt)
        for k in rbases:
            assert np.array_equal(b2[k], rbases[k]), f"basis {k} differs"
        assert np.array_equal(e2, rele), "ele differs after open"
        # a file that does not record its deck refuses to open
        with np.load(p, allow_pickle=False) as z:
            members = {n: z[n] for n in z.files if n != "json__provenance"}
        p_bare = os.path.join(tmp, "rz", "no_deck.npz")
        field_io._atomic_savez(p_bare, **members)
        try:
            field_io.read_spec_from_field(p_bare)
            assert False, "a field without its deck opened"
        except field_io.FieldMismatch as e:
            assert "does not record the deck" in str(e), str(e)
    check("F7 r-z save -> open from the file alone; a file without its "
          "deck refuses", f7)

    # ---------------------------------------------------------------- F8
    def f8():
        # provenance sidecar: descriptive, many-to-one,
        # never keyed, never validated. Two differently-named specs of the
        # SAME geometry share one entry and BOTH appear in produced_by.
        rt = os.path.join(tmp, "prov_cache")
        s_a = _spec(name="alpha_json")
        s_b = _spec(name="beta_json")          # same geometry, other name
        assert basis_cache.key(s_a) == basis_cache.key(s_b), \
            "fixture broken: names must not key"
        ba, ea = _solved(s_a)
        basis_cache.store(s_a, ba, ea, root=rt)
        basis_cache.load(s_b, root=rt)          # a HIT records the resolver
        inv = fa_cache.entries(root=rt)
        assert len(inv) == 1, f"expected one entry, got {len(inv)}"
        names = inv[0]["produced_by"]
        assert "alpha_json" in names and "beta_json" in names, names
        # a corrupt sidecar is descriptive-only: entries() still reports
        with open(os.path.join(rt, inv[0]["key"],
                               "provenance.json"), "w") as f:
            f.write("{not json")
        inv2 = fa_cache.entries(root=rt)
        assert len(inv2) == 1 and inv2[0]["produced_by"] == [], \
            "corrupt sidecar must not hide the entry"
    check("F8 provenance: many-to-one recorded, never keyed; corrupt "
          "sidecar degrades to dash", f8)

    print("=" * 60)
    print(f"PASSED {8 - len(FAILED)}   FAILED {len(FAILED)}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
