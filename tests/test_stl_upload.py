"""
test_stl_upload.py — gate the STL upload core.

Layers:
  1. Sizing: known-answer AABB/pitch/grid/memory math on synthetic boxes;
     unit-suspicion and thin-feature warnings fire when they should.
  2. Manifest hash: upload-order invariant; changes on geometry bytes,
     pitch, domain, symmetry, and solver version (stale-basis protection).
  3. Bundle: save -> load round-trips bases bit-exactly and reports fresh;
     a tampered manifest reports stale.
  4. End-to-end: the quad as TWO STLs, one per RF
     phase pair (full cylinders, R = 1.148*r0, NO symmetry assumptions) ->
     spec_from_upload -> validated build_run STL path -> per-electrode
     masks exist, are disjoint, and each pair has two rods.
"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import trimesh

from ion_gym.io.stl_upload import (load_mesh_bytes, propose_sizing,
                        manifest_hash, save_bundle, load_bundle,
                        spec_from_upload, SOLVER_VERSION)


def _stl_bytes(mesh):
    return trimesh.exchange.export.export_stl(mesh)


def _box(extents, center, name):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return load_mesh_bytes(f"{name}.stl", _stl_bytes(m))


def test_sizing():
    a = _box((10, 4, 2), (0, 0, 0), "plate_a")
    b = _box((10, 4, 2), (0, 8, 0), "plate_b")
    s = propose_sizing([a, b], margin_mm=2.0)          # default pitch 0.5
    # union AABB: x -5..5, y -2..10, z -1..1; +2 margin
    assert np.allclose(s.domain_min, [-7, -4, -3])
    assert np.allclose(s.domain_max, [7, 12, 3])
    # explicit pitch drives grid dims + memory (pure cost, no shape inference)
    s2 = propose_sizing([a, b], pitch=0.5, margin_mm=2.0)
    assert s2.dims == (29, 33, 13)
    assert s2.n_voxels == 29 * 33 * 13
    assert s2.mem_bases_bytes == s2.n_voxels * 4 * 2
    assert s2.est_solve_s > 0
    assert not hasattr(s2, "proposed_pitch"), "must not infer a pitch"
    # unit-suspicion warning still fires (geometry-agnostic: extent only)
    tiny = _box((0.01, 0.004, 0.002), (0, 0, 0), "metres")
    st = propose_sizing([tiny], margin_mm=0.001)
    assert any("units" in w for w in st.warnings)
    print("  sizing: AABB/pitch/grid/memory/solve-time, no shape "
          "inference, unit warning  OK")


def test_manifest_hash():
    a = _box((10, 4, 2), (0, 0, 0), "a")
    b = _box((10, 4, 2), (0, 8, 0), "b")
    s = propose_sizing([a, b], pitch=0.5)
    h0 = manifest_hash([a, b], s.pitch, s.domain_min, s.domain_max)
    # upload-order invariant
    assert h0 == manifest_hash([b, a], s.pitch, s.domain_min, s.domain_max)
    # geometry change -> new hash
    b2 = _box((10, 4, 2.1), (0, 8, 0), "b")
    assert h0 != manifest_hash([a, b2], s.pitch, s.domain_min, s.domain_max)
    # pitch / domain / symmetry / solver version each invalidate
    assert h0 != manifest_hash([a, b], 0.25, s.domain_min, s.domain_max)
    assert h0 != manifest_hash([a, b], s.pitch, s.domain_min - 1,
                               s.domain_max)
    assert h0 != manifest_hash([a, b], s.pitch, s.domain_min, s.domain_max,
                               symmetry={"y": "mirror"})
    assert h0 != manifest_hash([a, b], s.pitch, s.domain_min, s.domain_max,
                               solver_version="other")
    print("  manifest hash: order-invariant; invalidates on "
          "geometry/pitch/domain/symmetry/solver  OK")


def test_bundle_roundtrip(tmp="/tmp/stl_bundle_test"):
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    a = _box((10, 4, 2), (0, 0, 0), "a")
    b = _box((10, 4, 2), (0, 8, 0), "b")
    s = propose_sizing([a, b], pitch=0.5)
    bases = {"a": np.random.rand(8, 9).astype(np.float32),
             "b": np.random.rand(8, 9).astype(np.float32)}
    save_bundle(tmp, '{"name":"t"}', [a, b], bases, s)
    man, meshes, loaded, fresh = load_bundle(tmp)
    assert fresh, "hash should certify an untouched bundle"
    assert sorted(loaded) == ["a", "b"]
    assert np.array_equal(loaded["a"], bases["a"])
    assert man["solver_version"] == SOLVER_VERSION
    # tamper: change recorded pitch -> stale
    import json
    from pathlib import Path
    mp = Path(tmp) / "manifest.json"
    m2 = json.loads(mp.read_text())
    m2["pitch"] = 0.99
    mp.write_text(json.dumps(m2))
    _, _, _, fresh2 = load_bundle(tmp)
    assert not fresh2, "tampered manifest must read STALE (re-solve)"
    print("  bundle: round-trip bit-exact, fresh; tamper -> stale  OK")


def test_quad_two_stl_end_to_end(tmp="/tmp/stl_quad_upload"):
    """Charter first example: quad as TWO STLs, one per RF phase pair,
    full cylinders R = 1.148*r0, no symmetry assumptions — through the
    validated build path."""
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    r0, L = 3.84, 2.0
    R = 1.148 * r0
    rod_r = R  # rod radius: r_rod = 1.148*r0 (the dodecapole-minimizing ratio)

    def rod(cx, cy):
        c = trimesh.creation.cylinder(radius=rod_r, height=L, sections=48)
        c.apply_translation([cx, cy, 0.0])
        return c

    d = r0 + rod_r                      # rod-centre distance from axis
    pair_x = trimesh.util.concatenate([rod(+d, 0), rod(-d, 0)])
    pair_y = trimesh.util.concatenate([rod(0, +d), rod(0, -d)])
    meshes = [load_mesh_bytes("quad_1.stl", _stl_bytes(pair_x)),
              load_mesh_bytes("quad_2.stl", _stl_bytes(pair_y))]

    s = propose_sizing(meshes, pitch=0.2, margin_mm=1.5)
    spec = spec_from_upload(meshes, {"quad_1": 100.0, "quad_2": -100.0},
                            s, tmp)
    errs = spec.validate()
    assert not errs, f"spec invalid: {errs}"
    assert [e.name for e in spec.geometry.electrodes] == ["quad_1", "quad_2"]
    assert spec.geometry.electrodes[0].dc == 100.0

    # DECLARED FRAME OFFSET (2a.3): the CAD->build translation is
    # first-class data (build = uploaded + frame_offset_mm), equal to
    # -domain_min from sizing, and it survives the JSON round trip
    # (G5 IO map). Pre-existing native-frame specs omit the field.
    fo = spec.geometry.frame_offset_mm
    assert fo is not None and len(fo) == 3, fo
    assert np.allclose(fo, -np.asarray(s.domain_min, float)), (
        f"declared offset {fo} != -domain_min {(-s.domain_min).tolist()}")
    import json as _json
    from ion_gym.io.sim_spec import SimSpec as _SS
    rt = _SS.from_dict(_json.loads(_json.dumps(spec.to_dict())))
    assert rt.geometry.frame_offset_mm == [float(v) for v in fo]
    print(f"  frame_offset_mm declared {[round(v,3) for v in fo]} and "
          f"JSON round-trips  OK")

    # masks through the VALIDATED stl path: present, disjoint, two rods each
    from ion_gym.physics.build_stl import stl_masks_2d
    masks = stl_masks_2d(spec)
    assert set(masks) == {1, 2}
    m1, m2 = masks[1], masks[2]
    assert m1.sum() > 0 and m2.sum() > 0
    assert not (m1 & m2).any(), "phase pairs must not overlap"
    from scipy import ndimage
    n1 = ndimage.label(m1)[1]
    n2 = ndimage.label(m2)[1]
    assert n1 == 2 and n2 == 2, f"each pair = two rods (got {n1},{n2})"
    # registration: union centroid sits at the domain centre (quad axis)
    u = m1 | m2
    cy, cx = np.array(np.nonzero(u)).mean(axis=1)
    nx, ny = u.shape
    assert abs(cy / nx - 0.5) < 0.05 and abs(cx / ny - 0.5) < 0.05
    print(f"  quad 2-STL end-to-end: masks disjoint, {n1}+{n2} rods, "
          f"centred — validated path  OK")


def test_dropper_accepts_all():
    """Regression: the FileDropper must NOT restrict by filetype — FilePond
    rejects .stl (no MIME) and the drop silently fails. Headless value-set
    bypasses the browser filter, so this asserts the widget config directly."""
    import panel as pn  # noqa
    from ion_gym.io.stl_upload import StlUploadPanel
    p = StlUploadPanel(on_commit=lambda spec: None,
                       on_clear=lambda: None)
    assert not p.dropper.accepted_filetypes, \
        "dropper must accept all files (STL has no reliable MIME)"
    # clear control present and clears staged state without error
    p.meshes = ["sentinel"]
    p._on_clear()
    assert p.meshes == [] and p.commit_btn.disabled
    print("  dropper accepts all filetypes; clear control wipes state  OK")


if __name__ == "__main__":
    test_sizing()
    test_manifest_hash()
    test_bundle_roundtrip()
    test_quad_two_stl_end_to_end()
    test_dropper_accepts_all()
    print("\nSTL UPLOAD GATE: ALL PASS")
