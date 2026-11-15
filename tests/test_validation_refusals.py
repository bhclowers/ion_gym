"""
Validation refusals: decks that used to fly a wrong answer are refused,
by name, before anything is solved or flown; every shipped deck still
validates clean.

Each section reproduces one finding of the release review with the
smallest deck change that triggered it, and asserts the refusal names
the offending deck key.
"""
import copy
import json
import math

import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec

EXAMPLES = repo_root() / "examples"
QUAD = EXAMPLES / "quadrupole_analytic_rods_2d.json"
GATED = EXAMPLES / "cdms_elit_gated_rz.json"
DRIFT = EXAMPLES / "drift_tube_mason_schamp.json"   # He, sigma_model fixed


def _deck(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _errors(d):
    return SimSpec.from_dict(copy.deepcopy(d)).validate()


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")),
                         ids=lambda p: p.name)
def test_shipped_decks_validate_clean(path):
    assert SimSpec.from_json(str(path)).validate() == []


# ---- F1: NaN or infinity anywhere in a deck ------------------------------

def _set(d, keypath, value):
    node = d
    for k in keypath[:-1]:
        node = node[k]
    node[keypath[-1]] = value


@pytest.mark.parametrize("deck,keypath,value,named", [
    (QUAD, ("geometry", "electrodes", 0, "dc"), math.nan, ".dc"),
    (QUAD, ("geometry", "electrodes", 0, "dc"), math.inf, ".dc"),
    (QUAD, ("geometry", "rf_groups", 0, "amplitude_v"), math.inf,
     ".amplitude_v"),
    (QUAD, ("geometry", "rf_groups", 0, "frequency_hz"), math.nan,
     ".frequency_hz"),
    (QUAD, ("geometry", "rf_groups", 0, "phase_deg"), math.nan,
     ".phase_deg"),
    (QUAD, ("source", "mz_list"), [math.nan], "source.mz_list[0]"),
    (GATED, ("geometry", "rf_groups", 0, "table_v"), [0.0, math.nan],
     ".table_v[1]"),
])
def test_nonfinite_number_is_refused_by_deck_path(deck, keypath, value,
                                                  named):
    d = _deck(deck)
    _set(d, keypath, value)
    errs = _errors(d)
    hits = [e for e in errs if "not a finite number" in e]
    assert hits, f"no finiteness refusal; validate() returned {errs}"
    assert any(named in e for e in hits), hits


def test_nonfinite_deck_never_reaches_a_flight():
    from ion_gym.physics.sim_build import build_run
    d = _deck(QUAD)
    d["geometry"]["electrodes"][0]["dc"] = math.nan
    with pytest.raises(ValueError, match="not a finite number"):
        build_run(SimSpec.from_dict(d))


# ---- F4: gas switched on with values that describe no gas -----------------

@pytest.mark.parametrize("key,value", [
    ("P_torr", -1.0), ("P_torr", 0.0), ("T_k", -5.0),
    ("sigma_m2", -1e-18),
])
def test_nonphysical_gas_is_refused(key, value):
    d = _deck(DRIFT)
    d["collisions"][key] = value
    errs = _errors(d)
    assert any(f"collisions.{key}" in e for e in errs), errs


def test_sds_gas_diameter_must_be_positive():
    d = _deck(DRIFT)
    d["collisions"]["model"] = "sds"
    d["collisions"]["gas_diam_nm"] = 0.0
    assert any("collisions.gas_diam_nm" in e for e in _errors(d))


def test_gas_values_are_not_checked_in_vacuum():
    d = _deck(DRIFT)
    d["collisions"]["enabled"] = False
    d["collisions"]["P_torr"] = -1.0
    assert not any("collisions.P_torr" in e for e in _errors(d))


@pytest.mark.parametrize("gas", [
    {"enabled": True, "P_torr": -1.0},
    {"enabled": True, "P_torr": math.nan},
    {"enabled": True, "T_k": math.inf},
])
def test_gas_authority_refuses_without_a_deck(gas):
    from ion_gym.physics.collision3d import resolve_collisions
    with pytest.raises(ValueError, match="invalid gas"):
        resolve_collisions(gas)


# ---- F3: an electrode no node lands in, on every route --------------------

ROUTES = [("planar", "xyz", 0.0), ("rz", "rz", 0.0), ("3d", "xyz", 2.0)]


def _plate_deck(coords, depth_mm, plate_x_mm, plate_w_mm, pitch_mm=0.25):
    """A 30 eV ion fired at a full-height plate, gas off (the review's own
    thin-plate reproduction)."""
    from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec,
                                     GeometrySpec, IntegrationSpec,
                                     ShapeSpec, SourceSpec, SymmetrySpec)

    def rect(x, y, w, h):
        return ShapeSpec("rect", {"x_mm": x, "y_mm": y,
                                  "width_mm": w, "height_mm": h})
    geo = GeometrySpec(
        width_mm=20.0, height_mm=6.0, depth_mm=depth_mm, mm_per_gu=pitch_mm,
        symmetry=SymmetrySpec(coords=coords),
        electrodes=[ElectrodeSpec(name="plate", dc=10.0,
                                  shapes=[rect(plate_x_mm, 0, plate_w_mm, 6)]),
                    ElectrodeSpec(name="ground", dc=0.0,
                                  shapes=[rect(19, 0, 1, 6)])])
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=1.0, y0_mm=2.6,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=30.0, ke_hi=30.0, mz_list=[100.0],
                     tob_span_us=0.0, seed=1)
    return SimSpec(geometry=geo, name="thin_plate", source=src,
                   collisions=CollisionSpec(enabled=False),
                   integration=IntegrationSpec(dt_ns=1.0, t_max_us=3.0,
                                               rec_every=100))


@pytest.mark.parametrize("route,coords,depth", ROUTES, ids=[r[0] for r in ROUTES])
def test_plate_thinner_than_pitch_is_refused_on_every_route(route, coords,
                                                            depth):
    from ion_gym.physics.sim_build import build_run
    with pytest.raises(ValueError, match="ZERO nodes"):
        build_run(_plate_deck(coords, depth, plate_x_mm=5.05, plate_w_mm=0.1))


def _clear_memory_caches():
    from ion_gym.physics import build_planar, build_rz, build_stl3d
    for mod in (build_planar, build_rz, build_stl3d):
        mod.clear_memory_cache()


@pytest.mark.parametrize("route,coords,depth", ROUTES, ids=[r[0] for r in ROUTES])
def test_thin_plate_is_refused_from_a_stale_cache_entry(route, coords, depth,
                                                        monkeypatch, tmp_path):
    """A basis banked by a build that skipped the zero-node rule (any build
    before the rule existed) must not be served: the cache-hit path refuses
    on the solved basis itself."""
    from ion_gym.physics import build_planar, build_shapes3d, raster2d
    from ion_gym.physics.sim_build import build_run
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "cache"))
    _clear_memory_caches()
    thin = _plate_deck(coords, depth, plate_x_mm=5.05, plate_w_mm=0.1)
    with monkeypatch.context() as m:
        # phase 1: bank the thin deck the way a build without the rule did
        if route == "3d":
            real = build_shapes3d.shapes_masks_3d
            thick = _plate_deck(coords, depth, plate_x_mm=5.0, plate_w_mm=0.3)

            def stale_masks(spec, verbose=False):
                masks = real(thick, verbose=verbose)
                masks[1] = masks[1] & False     # the plate: no nodes
                return masks
            m.setattr(build_shapes3d, "shapes_masks_3d", stale_masks)
        else:
            def no_rule(*args, **kwargs):
                return None
            m.setattr(build_planar, "refuse_empty_mask", no_rule)
            m.setattr(raster2d, "refuse_empty_mask", no_rule)
        build_run(thin)
    _clear_memory_caches()
    # phase 2: the same deck, the rule in force, the entry now on disk
    with pytest.raises(ValueError, match=r"cache hit.*ZERO nodes"):
        build_run(thin)


@pytest.mark.parametrize("route,coords,depth", ROUTES, ids=[r[0] for r in ROUTES])
def test_resolved_plate_still_builds_on_every_route(route, coords, depth):
    from ion_gym.physics.sim_build import build_run
    model, _fly, _cols, births = build_run(
        _plate_deck(coords, depth, plate_x_mm=5.0, plate_w_mm=0.3))
    assert len(births) == 1


@pytest.mark.parametrize("key", ["rx_mm", "ry_mm"])
def test_negative_ellipse_radius_is_refused(key):
    d = _deck(QUAD)
    shape = d["geometry"]["electrodes"][0]["shapes"][0]
    assert shape["type"] == "ellipse"
    shape[key] = -abs(shape[key])
    assert any(f"{key} = " in e and "must be > 0" in e for e in _errors(d))


# ---- F2: an STL that is empty, unreadable or open, on both STL routes ------

STL_2D = EXAMPLES / "quadrupole_stl_rods_transport.json"
STL_3D = EXAMPLES / "quadrupole_stl_rods_full_3-d.json"


def _open_box_bytes():
    import numpy as np
    import trimesh
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.update_faces(np.arange(len(box.faces)) >= 2)
    return trimesh.exchange.stl.export_stl(box)


def _copy_stl_deck(deck, tmp_path, *, drop_manifest=False):
    """The deck and its STL folder copied under tmp_path, unchanged."""
    import shutil
    d = _deck(deck)
    shutil.copytree(EXAMPLES / d["geometry"]["stl_dir"],
                    tmp_path / d["geometry"]["stl_dir"])
    if drop_manifest:
        # the manifest would refuse a changed file by hash before the
        # mesh itself is examined; this test examines the mesh
        d["geometry"]["stl_manifest"] = None
    p = tmp_path / deck.name
    p.write_text(json.dumps(d), encoding="utf-8")
    return p, d


@pytest.mark.parametrize("bad", ["empty", "garbage", "open_box"])
def test_unusable_stl_is_refused_on_the_2d_route(tmp_path, bad):
    from ion_gym.physics.sim_build import build_run
    p, d = _copy_stl_deck(STL_2D, tmp_path)
    rod = tmp_path / d["geometry"]["stl_dir"] / d["geometry"]["electrodes"][0]["stl"]
    rod.write_bytes({"empty": b"",
                     "garbage": b"solid x\nnot an stl\nendsolid x\n",
                     "open_box": _open_box_bytes()}[bad])
    with pytest.raises(ValueError, match="not a usable electrode mesh"):
        build_run(SimSpec.from_json(str(p)))


def test_unusable_stl_is_refused_on_the_3d_route(tmp_path):
    from ion_gym.physics.sim_build import build_run
    p, d = _copy_stl_deck(STL_3D, tmp_path, drop_manifest=True)
    rod = tmp_path / d["geometry"]["stl_dir"] / d["geometry"]["electrodes"][0]["stl"]
    rod.write_bytes(_open_box_bytes())
    with pytest.raises(ValueError, match="not a usable electrode mesh"):
        build_run(SimSpec.from_json(str(p)))


def test_stl_outside_the_domain_is_refused_on_the_3d_route(tmp_path):
    import trimesh
    from ion_gym.physics.sim_build import build_run
    p, d = _copy_stl_deck(STL_3D, tmp_path, drop_manifest=True)
    far = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    far.apply_translation([500.0, 500.0, 60.0])
    rod = tmp_path / d["geometry"]["stl_dir"] / d["geometry"]["electrodes"][0]["stl"]
    rod.write_bytes(trimesh.exchange.stl.export_stl(far))
    with pytest.raises(ValueError, match="ZERO nodes"):
        build_run(SimSpec.from_json(str(p)))


def test_upload_door_refuses_unreadable_bytes_by_name():
    from ion_gym.io.stl_upload import load_mesh_bytes
    with pytest.raises(ValueError, match="'bad.stl'"):
        load_mesh_bytes("bad.stl", b"\0" * 84)


def test_shipped_meshes_are_sound():
    import trimesh
    from ion_gym.io.stl_resolve import mesh_soundness_errors
    files = sorted(p for pat in ("*.stl", "*.STL")
                   for p in EXAMPLES.rglob(pat))
    assert files
    bad = {p.name: mesh_soundness_errors(
        trimesh.load(str(p), force="mesh", process=True)) for p in files}
    assert not any(bad.values()), bad


# ---- API-3: per-electrode RF keys the drive model never applies -----------

@pytest.mark.parametrize("key,value", [
    ("rf_phase_deg", 180.0),        # the review's anti-phase rails
    ("rf_amplitude", 241.34),       # the setup guide's old spelling
    ("rf_frequency_hz", 2.0e6),
])
def test_nonzero_legacy_electrode_drive_is_refused(key, value):
    d = _deck(QUAD)
    el = d["geometry"]["electrodes"][0]
    el[key] = value
    with pytest.raises(ValueError, match=rf"{el['name']}.*{key}"):
        SimSpec.from_dict(d)


def test_zero_legacy_electrode_drive_still_loads():
    d = _deck(QUAD)
    for el in d["geometry"]["electrodes"]:
        el.update(rf_phase_deg=0.0, rf_amplitude=0.0, rf_frequency_hz=0.0)
    assert SimSpec.from_dict(d).validate() == []


# ---- API-4: a symmetry string that is not one of the documented three ----

EINZEL_RZ = EXAMPLES / "einzel_round_r-z.json"


@pytest.mark.parametrize("bad", ["rz", "cylindircal", "Cylindrical", ""])
def test_unknown_symmetry_string_is_refused(bad):
    d = _deck(EINZEL_RZ)
    d["geometry"]["symmetry"] = bad
    with pytest.raises(ValueError, match="not a known symmetry"):
        SimSpec.from_dict(d)


@pytest.mark.parametrize("name,coords", [("cylindrical", "rz"),
                                         ("planar", "xyz"), ("none", "xyz")])
def test_documented_symmetry_strings_map_on_both_doors(name, coords):
    from ion_gym.io.sim_spec import GeometrySpec
    d = _deck(EINZEL_RZ)
    d["geometry"]["symmetry"] = name
    assert SimSpec.from_dict(d).geometry.symmetry.coords == coords
    g = GeometrySpec(width_mm=10.0, height_mm=5.0, symmetry=name)
    assert g.symmetry.coords == coords


def test_constructor_string_symmetry_round_trips():
    # the setup guide's example spelling used to fail at to_json; replace()
    # calls the constructor with the STRING, the path the guide takes
    import dataclasses
    spec = SimSpec.from_json(str(EINZEL_RZ))
    spec.geometry = dataclasses.replace(spec.geometry, symmetry="cylindrical")
    again = SimSpec.from_json(spec.to_json())
    assert again.geometry.symmetry.coords == "rz"


# ---- API-1: a schema_version this ion_gym does not read --------------------

@pytest.mark.parametrize("version", [99, "banana", None, 0, 3, True])
def test_unknown_schema_version_is_refused(version):
    d = _deck(QUAD)
    d["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version"):
        SimSpec.from_dict(d)


def test_current_and_absent_schema_version_load():
    d = _deck(QUAD)
    d["schema_version"] = 1
    assert SimSpec.from_dict(d).validate() == []
    d.pop("schema_version")
    assert SimSpec.from_dict(d).validate() == []


def test_stage_with_unknown_schema_version_is_refused(tmp_path):
    from ion_gym.physics.staged_flight import load_assembly
    src = EXAMPLES / "funnel_hexapole" / "instrument.json"
    doc = json.loads(src.read_text(encoding="utf-8"))
    doc["stages"][0]["spec"]["schema_version"] = 2
    p = tmp_path / "instrument.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version 2"):
        load_assembly(str(p))


def test_coaxial_assembly_with_unknown_schema_version_is_refused():
    from ion_gym.io.assembly import AssemblySpec
    with pytest.raises(ValueError, match="AssemblySpec: schema_version 2"):
        AssemblySpec.from_dict({"kind": "assembly", "schema_version": 2})


# ---- C-09: origin_mm on an r-z deck, which the r-z builder never applies --

@pytest.mark.parametrize("origin", [[-5.0, 0.0], [0.0, 1.0]])
def test_rz_origin_is_refused(origin):
    d = _deck(EINZEL_RZ)
    d["geometry"]["origin_mm"] = origin
    assert any("origin_mm" in e and "r-z builder does not apply" in e
               for e in _errors(d))


def test_rz_zero_origin_is_accepted():
    d = _deck(EINZEL_RZ)
    d["geometry"]["origin_mm"] = [0.0, 0.0]
    assert _errors(d) == []


def test_constructor_refuses_unknown_symmetry_string():
    from ion_gym.io.sim_spec import GeometrySpec
    with pytest.raises(ValueError, match="'rz' is a coords value"):
        GeometrySpec(width_mm=10.0, height_mm=5.0, symmetry="rz")


# ---- timeouts are not transmission unless the caller says so --------------

def _all_timed_out(n=4):
    return [{"kind": 2, "tof": 3.0, "x_end": float("nan"),
             "y_end": float("nan"), "z_end": 0.0} for _ in range(n)]


def test_transmitted_fate_must_be_stated():
    from ion_gym.physics.stats import compute_stats
    with pytest.raises(ValueError, match="transmitted_fate is required"):
        compute_stats(_all_timed_out())


def test_timeouts_do_not_count_as_transmitted_for_a_transmission_device():
    # the review's F1 reading: every ion timed out, reported 100% transmitted
    from ion_gym.physics.stats import auto_transmitted_fate, compute_stats
    spec = SimSpec.from_json(str(QUAD))
    st = compute_stats(_all_timed_out(), transmitted_fate=auto_transmitted_fate(spec))
    assert st.transmission == 0.0


def test_unknown_transmitted_fate_is_refused():
    from ion_gym.physics.stats import compute_stats
    with pytest.raises(ValueError, match="not a fate code"):
        compute_stats(_all_timed_out(), transmitted_fate=9)


# ---- C-03: a node claimed by two electrodes, on every route ---------------

def _two_plate_deck(coords, depth_mm, tops):
    """Two 50 V top plates over a grounded bottom plate (the review's
    reproduction), gas off."""
    from ion_gym.io.sim_spec import (CollisionSpec, ElectrodeSpec,
                                     GeometrySpec, IntegrationSpec,
                                     ShapeSpec, SourceSpec, SymmetrySpec)

    def rect(x, y, w, h):
        return ShapeSpec("rect", {"x_mm": x, "y_mm": y, "width_mm": w,
                                  "height_mm": h})
    els = [ElectrodeSpec(name=f"top{i}", dc=50.0, shapes=[rect(*r)])
           for i, r in enumerate(tops)]
    els.append(ElectrodeSpec(name="ground", dc=0.0, shapes=[rect(0, 0, 20, 1)]))
    geo = GeometrySpec(width_mm=20.0, height_mm=10.0, depth_mm=depth_mm,
                       mm_per_gu=0.5, symmetry=SymmetrySpec(coords=coords),
                       electrodes=els)
    src = SourceSpec(distribution="point", n_ions=1, x0_mm=10.0, y0_mm=5.0,
                     z0_mm=depth_mm / 2, direction=[1.0, 0.0, 0.0],
                     ke_lo=1.0, ke_hi=1.0, mz_list=[100.0], tob_span_us=0.0,
                     seed=1)
    return SimSpec(geometry=geo, name="two_plates", source=src,
                   collisions=CollisionSpec(enabled=False),
                   integration=IntegrationSpec(dt_ns=2.0, t_max_us=0.5,
                                               rec_every=10))


TOUCHING = [(0, 9, 10, 1), (10, 9, 10, 1)]     # share the x = 10 nodes
OVERLAPPING = [(0, 9, 12, 1), (8, 9, 12, 1)]   # 4 mm overlap


@pytest.mark.parametrize("tops", [TOUCHING, OVERLAPPING],
                         ids=["touching", "overlapping"])
@pytest.mark.parametrize("route,coords,depth", ROUTES, ids=[r[0] for r in ROUTES])
def test_shared_node_is_refused_on_every_route(route, coords, depth, tops,
                                               monkeypatch, tmp_path):
    from ion_gym.physics.sim_build import build_run
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    _clear_memory_caches()
    with pytest.raises(ValueError, match="overlaps electrode"):
        build_run(_two_plate_deck(coords, depth, tops))


@pytest.mark.parametrize("route,coords,depth", ROUTES, ids=[r[0] for r in ROUTES])
def test_separate_plates_still_build_on_every_route(route, coords, depth,
                                                    monkeypatch, tmp_path):
    import numpy as np
    from ion_gym.physics.sim_build import build_run
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    _clear_memory_caches()
    model = build_run(_two_plate_deck(coords, depth,
                                      [(0, 9, 9.5, 1), (10.5, 9, 9.5, 1)]))[0]
    A = np.asarray(model.A)
    assert A.max() <= 50.0 + 1e-9          # no node holds a summed voltage


@pytest.mark.parametrize("route,coords", [("planar", "xyz"), ("rz", "rz")])
def test_shared_node_is_refused_from_a_stale_cache_entry(route, coords,
                                                         monkeypatch,
                                                         tmp_path):
    from ion_gym.physics import build_planar, raster2d
    from ion_gym.physics.sim_build import build_run
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "cache"))
    _clear_memory_caches()
    deck = _two_plate_deck(coords, 0.0, TOUCHING)
    with monkeypatch.context() as m:
        # bank the deck the way a build without the rule did
        def no_rule(*args, **kwargs):
            return None
        m.setattr(build_planar, "refuse_shared_nodes", no_rule)
        m.setattr(raster2d, "refuse_shared_nodes", no_rule)
        build_run(deck)
    _clear_memory_caches()
    with pytest.raises(ValueError, match=r"cache hit.*overlaps electrode"):
        build_run(deck)


