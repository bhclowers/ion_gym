"""
ion_gym.spec_io
---------------
ONE loader for the JSON a user can hand ion_gym.

There are two schemas in the tree, and they are both legitimate:

  * sim_spec.SimSpec  -- top-level key "geometry".  Inline 2-D shapes
    (rect/ellipse/polygon), 2-D planar or r-z.  What the ionbench flies.
  * scene3d.GeomScene     -- top-level keys "grid" + "electrodes[].index".
    3-D analytic CSG (Box3D / Cylinder, within/notin).  What solver3d and
    voxelize consume.

Before this module the app had exactly one door: SimSpec.from_json.  Hand
it a GeomScene and it raised `KeyError: 'geometry'` -- a stack-trace fragment
that tells the user nothing about what went wrong or what to do.  A file
that IS a valid, solvable ion_gym geometry was rejected as corrupt.

So: SNIFF the schema, ROUTE it, and when routing is genuinely impossible,
REFUSE WITH A DIAGNOSTIC that names the schema seen, the schema needed,
and the way across.
"""
from __future__ import annotations

import json


def sniff(txt_or_obj):
    """Return 'simspec' | 'scene3d' | 'unknown' for a JSON string/dict."""
    d = (json.loads(txt_or_obj) if isinstance(txt_or_obj, (str, bytes))
         else txt_or_obj)
    if not isinstance(d, dict):
        return "unknown"
    if d.get("kind") == "assembly" and isinstance(d.get("units"), list):
        return "assembly"
    if d.get("schema") == "ion_gym.assembly/1" and isinstance(
            d.get("stages"), list):
        return "staged_assembly"
    if "geometry" in d and isinstance(d.get("geometry"), dict):
        return "simspec"
    if "grid" in d and "electrodes" in d:
        g = d.get("grid") or {}
        if {"nx", "ny", "nz"} <= set(g):
            return "scene3d"
    return "unknown"


def scene_to_simspec(scene, *, json_path=None, name=None, dc_group=None):
    """scene3d.GeomScene -> SimSpec, solved NATIVELY (builder='scene3d').

    Earlier this tessellated the CSG to one STL per electrode and used the
    mesh path. That was a lossy detour taken because the mesh path already
    existed -- exactly the wrong reason. rasterize3d consumes a GeomScene
    directly and exactly. See build_scene3d for the full argument.
    """
    from ion_gym.physics.build_scene3d import simspec_from_scene
    return simspec_from_scene(scene, json_path=json_path, name=name,
                              dc_group=dc_group)


def load_any_spec(txt, *, work_dir=None, stl_dir=None):
    """Load either schema.  Returns a SimSpec.

    A GeomScene is solved AS AUTHORED (analytic CSG -> rasterize3d). It needs a
    place to keep its own JSON so the spec can point at the geometry truth;
    that is all work_dir is for. No meshes are produced.
    """
    from ion_gym.io.sim_spec import SimSpec
    kind = sniff(txt)

    if kind == "simspec":
        return SimSpec.from_json(txt)

    if kind == "assembly":
        # A declarative multi-FA assembly (io.assembly): coaxial units
        # placed by offset. Flatten to ONE ordinary SimSpec so the SAME
        # JSON a user reads as a placement description also loads and
        # solves in one shot — no separate assembly build path, no
        # per-example glue. Non-coaxial units are refused inside flatten
        # with a diagnostic.
        from ion_gym.io.assembly import AssemblySpec, flatten_to_simspec
        d = json.loads(txt) if isinstance(txt, (str, bytes)) else txt
        return flatten_to_simspec(AssemblySpec.from_dict(d))

    if kind == "staged_assembly":
        # A staged multi-region instrument (physics.staged_flight): stages with
        # poses + seams, possibly NON-coaxial. Unlike the coaxial
        # 'assembly' above, this is NOT flattenable to one SimSpec — each
        # stage solves in its OWN grid and the ion is handed across the
        # seams. So it has its own entry point, not the spec loader.
        raise ValueError(
            "this is a STAGED ASSEMBLY (schema ion_gym.assembly/1), not a "
            "single spec — it lists stages that each solve separately and "
            "hand the ion across seams (a non-coaxial instrument). It "
            "cannot load as one SimSpec.\n"
            "  -> load it with ion_gym.physics.staged_flight.load_assembly(path) "
            "or fly it with fly_assembly(path).\n"
            "  (The app's example menu and single-spec loader only take "
            "individual stage specs, e.g. stageA/spec.json, not the "
            "instrument.json.)")

    if kind == "scene3d":
        from ion_gym.physics.scene3d import GeomScene
        sc = GeomScene.from_json(txt)
        try:
            sc.check()
        except ValueError as e:
            raise ValueError(f"scene3d JSON is not solvable as written: {e}")
        # No temp dir, no sidecar file: the GeomScene is embedded in the spec.
        return scene_to_simspec(sc, json_path=None)

    raise ValueError(
        "unrecognised JSON. ion_gym reads two geometry schemas:\n"
        "  • SimSpec  — needs a top-level \"geometry\" key (inline 2-D "
        "shapes; planar or r-z)\n"
        "  • scene3d  — needs top-level \"grid\" {nx,ny,nz} and "
        "\"electrodes\" with an \"index\" (3-D CSG)\n"
        "The file supplied has neither. Top-level keys seen: "
        + ", ".join(sorted(json.loads(txt).keys())[:8]
                    if isinstance(txt, (str, bytes)) else []))


def spec_summary_rows(spec):
    """Human-readable summary of a SimSpec as (label, value) rows for a
    table (shown under the Config-tab JSON so a changed
    JSON is legible at a glance). Pure/read-only; tolerant of missing
    pieces (returns '-' rather than raising) so a partial or unusual spec
    still summarises."""
    g = getattr(spec, "geometry", None)
    src = getattr(spec, "source", None)
    rows = []
    def add(k, v): rows.append((k, "-" if v is None else str(v)))
    add("name", getattr(spec, "name", None))
    add("builder", getattr(spec, "builder", None))
    if g is not None:
        n_el = len(g.electrodes)
        add("electrodes", n_el)
        # size
        w = getattr(g, "width_mm", None); h = getattr(g, "height_mm", None)
        d = getattr(g, "depth_mm", 0.0)
        if w is not None and h is not None:
            size = f"{w:.2f} x {h:.2f}" + (f" x {d:.2f}" % () if False else
                                           (f" x {d:.2f} mm" if d else " mm (2D)"))
            add("size", size)
        add("pitch (mm/gu)", getattr(g, "mm_per_gu", None))
        # symmetry
        sym = getattr(g, "symmetry", None)
        if sym is not None:
            planes = getattr(sym, "planes", {}) or {}
            active = [f"{a}:{v}" for a, v in planes.items() if v not in (None, "none")]
            add("coords", getattr(sym, "coords", None))
            add("symmetry planes", ", ".join(active) if active else "none")
        # drive (RF) groups
        rf = getattr(g, "rf_groups", []) or []
        if rf:
            parts = []
            for gr in rf:
                amp = getattr(gr, "amplitude_v", None)
                fhz = getattr(gr, "frequency_hz", None)
                ph = getattr(gr, "phase_deg", None)
                seg = gr.name
                if amp is not None:
                    seg += f" {amp:g}V"
                if fhz:
                    seg += f"@{fhz/1e3:g}kHz"
                if ph:
                    seg += f" {ph:g}deg"
                parts.append(seg)
            add("drive groups", f"{len(rf)}: " + "; ".join(parts))
        else:
            add("drive groups", "none")
        # DC groups
        dcg = getattr(g, "dc_groups", []) or []
        if dcg:
            parts = []
            for gr in dcg:
                kind = "uniform" if getattr(gr, "uniform", False) else "ladder"
                if kind == "uniform":
                    parts.append(f"{gr.name} [{kind} {gr.v_in:g}V]")
                else:
                    parts.append(f"{gr.name} [{kind} {gr.v_in:g}->{gr.v_out:g}V]")
            add("DC groups", f"{len(dcg)}: " + "; ".join(parts))
        else:
            add("DC groups", "none")
        # electrode class tally
        n_rf = sum(1 for e in g.electrodes if getattr(e, "rf_groups", None))
        n_dcg = sum(1 for e in g.electrodes if getattr(e, "dc_group", None))
        n_gnd = sum(1 for e in g.electrodes
                    if not getattr(e, "rf_groups", None)
                    and not getattr(e, "dc_group", None)
                    and abs(getattr(e, "dc", 0.0)) < 1e-12)
        add("electrode roles",
            f"{n_rf} driven, {n_dcg} in DC group, {n_gnd} grounded")
    # ions
    if src is not None:
        n_per = getattr(src, "n_ions", None)
        mz = getattr(src, "mz_list", []) or []
        if n_per is not None:
            total = n_per * max(1, len(mz))
            add("ions", f"{n_per} per m/z x {len(mz)} m/z = {total} total")
        add("m/z list", ", ".join(f"{m:g}" for m in mz) if mz else "-")
        add("distribution", getattr(src, "distribution", None))
    # integration
    integ = getattr(spec, "integration", None)
    if integ is not None:
        tmax = getattr(integ, "t_max_us", None)
        if tmax is not None:
            add("t_max", f"{tmax:g} us ({tmax/1000:g} ms)")
    return rows
