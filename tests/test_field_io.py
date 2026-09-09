"""
Gate: field_io — field portability.

Covers:
  F1  save -> RENAME -> load round-trip (filename is cosmetic; bases and
      ele come back bit-identical; bases reinstated as a cache hit)
  F2  wrong-geometry load HARD-REFUSES, and the diagnostic NAMES the
      diverging geometry field (width_mm here)
  F3  fingerprint anchors an identical solve; a perturbed basis DIVERGES
      with the offending basis/node named
  F4  trajectory save/load: matching spec -> no warning; different
      geometry -> warning string (overlay allowed, never a refusal)
  F5  bake_label rewrites _meta.label in place (the one rewriting path);
      registry precedence: local registry > embedded label
  F6  registry locality: saving into a custom directory writes
      fields_registry.json THERE, not into the global fields dir
      (mutation guard on the save_field registry fix)

No solver run: validation/fingerprint are functions of geometry key +
array structure, so synthetic bases on a real GeometrySpec exercise every
contract deterministically in <1 s. All files under a tempdir; the repo
fields/ directory is never touched.

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
        p = field_io.save_field(spec, bases, ele,
                                path=os.path.join(tmp, "as_saved.npz"))
        renamed = os.path.join(tmp, "totally arbitrary name (sent).npz")
        os.replace(p, renamed)
        b2, e2 = field_io.load_field(renamed, spec, root=cache_root)
        assert sorted(b2) == sorted(int(k) for k in bases), "basis keys"
        for k in bases:
            assert np.array_equal(b2[int(k)], bases[k]), f"basis {k} bits"
        assert np.array_equal(e2, ele), "ele bits"
        # reinstated: the disk cache now HITS this geometry
        arrs, meta = fa_cache.load(basis_cache.geometry_key_dict(spec),
                                   root=cache_root)
        assert arrs is not None, "no cache hit after load_field"
    check("F1 save -> rename -> load round-trip + cache reinstate", f1)

    # ---------------------------------------------------------------- F2
    def f2():
        p = os.path.join(tmp, "for_refuse.npz")
        field_io.save_field(spec, bases, ele, path=p)
        other = _spec(width=25.0)          # ONE geometry field differs
        try:
            field_io.load_field(p, other, root=cache_root)
        except field_io.FieldMismatch as e:
            msg = str(e)
            assert "width_mm" in msg, f"divergence not named: {msg!r}"
            assert "Refusing" in msg or "refus" in msg.lower(), msg
            return
        raise AssertionError("wrong-geometry load did NOT refuse")
    check("F2 wrong-geometry load hard-refuses, names width_mm", f2)

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
                                       path=os.path.join(tmp, "t.traj.npz"))
        trajs, meta, warn = field_io.load_trajectories(p, spec)
        assert warn is None, f"matching spec warned: {warn}"
        assert sorted(trajs) == [0, 2] and meta["n_skipped"] == 1
        _, _, warn2 = field_io.load_trajectories(p, _spec(width=25.0))
        assert warn2 and "different geometry" in warn2, warn2
    check("F4 trajectory round-trip; overlay warns, never refuses", f4)

    # ---------------------------------------------------------------- F5
    def f5():
        p = os.path.join(tmp, "label.npz")
        field_io.save_field(spec, bases, ele, path=p, label="original")
        field_io.bake_label(p, "baked-for-send")
        meta = field_io.read_field_meta(p)
        assert meta["label"] == "baked-for-send", meta["label"]
        field_io.load_field(p, spec, root=cache_root)   # bake kept validity
        field_io.set_label(meta["key"], "local-wins", directory=tmp)
        assert field_io.display_label(meta, directory=tmp) == "local-wins"
    check("F5 bake_label rewrites in place; registry > embedded", f5)

    # ---------------------------------------------------------------- F6
    def f6():
        sub = os.path.join(tmp, "custom_dir")
        field_io.save_field(spec, bases, ele,
                            path=os.path.join(sub, "x.npz"), label="here")
        assert os.path.exists(os.path.join(sub, "fields_registry.json")), \
            "registry not written beside the saved file"
        rows = field_io.scan_fields(directory=sub, spec=spec)
        assert len(rows) == 1 and rows[0]["matches"] is True
        assert rows[0]["label"] == "here", rows[0]["label"]
    check("F6 registry lives beside a custom save dir; scan matches", f6)

    # ---------------------------------------------------------------- F7
    def f7():
        # r-z round-trip + one-file bootstrap (the cache-backed
        # r-z path): an rz-coords spec keys through the SAME canonical
        # machinery (symmetry.coords distinguishes it — no rz-special
        # branch), saves, and reopens from the file alone: spec
        # reconstructed from the embedded _meta, every integrity check
        # still firing, bases reinstated into the given cache root.
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
        p = os.path.join(tmp, "rz_session.npz")
        field_io.save_field(rspec, rbases, rele, path=p, label="rz gate")
        rt = os.path.join(tmp, "rz_boot_cache")
        s2, b2, e2 = field_io.load_field_bootstrap(p, root=rt)
        assert basis_cache.key(s2) == basis_cache.key(rspec), \
            "reconstructed spec keys differently from the original"
        assert s2.geometry.symmetry.coords == "rz", "coords lost in round-trip"
        assert s2.geometry.electrodes[1].is_grid is True, \
            "is_grid lost in round-trip (stale-hit hazard)"
        for k in rbases:
            assert np.array_equal(b2[k], rbases[k]), f"basis {k} differs"
        assert np.array_equal(e2, rele), "ele differs after bootstrap"
        db, de = basis_cache.load(s2, root=rt)
        assert db is not None, "bootstrap did not reinstate the cache root"
        # pre-bootstrap file (spec=None in _meta) must REFUSE with a reason
        import numpy as _np
        with _np.load(p, allow_pickle=False) as z:
            members = {n: z[n] for n in z.files}
        m = field_io._member_to_meta(members[field_io._META])
        m["spec"] = None
        members[field_io._META] = field_io._meta_to_member(m)
        p_old = os.path.join(tmp, "rz_prebootstrap.npz")
        field_io._atomic_savez(p_old, **members)
        try:
            field_io.read_spec_from_field(p_old)
            assert False, "pre-bootstrap save did not refuse"
        except field_io.FieldMismatch as e:
            assert "no embedded spec" in str(e), str(e)
    check("F7 r-z save -> one-file bootstrap round-trip; pre-bootstrap "
          "saves refuse", f7)

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
