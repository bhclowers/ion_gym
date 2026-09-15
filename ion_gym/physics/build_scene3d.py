"""
ion_gym.build_scene3d
---------------------
Solve a scene3d.GeomScene DIRECTLY.  No STLs, no tessellation, no meshes.

WHY THIS EXISTS
    The first cut routed a GeomScene through the STL path: tessellate the CSG to
    35 meshes, voxelize the meshes, solve.  That is a lossy detour taken for
    no physical reason.  `rasterize3d.rasterize(scene)` already turns the
    analytic CSG into exactly the labeled grid the solver wants -- a Box3D
    becomes an exact box, a Cylinder becomes an exact cylinder, decided by
    point membership.  Tessellating first replaces those exact answers with a
    faceted approximation whose error (sagitta) depends on a section count
    nobody chose on physical grounds.  A box survives it; a bore does not.

    So the GeomScene is solved as authored.  The mesh path remains, unchanged and
    still gated, for geometry that genuinely ARRIVES as a mesh (CAD, STL).
    Neither path is privileged: both hand the SAME masks dict to the SAME
    3-D builder, so a fix in one cannot silently special-case the other.

DIMENSIONS ARE INVARIANT UNDER PITCH
    The GeomScene owns the geometry in mm.  Changing mm_per_gu re-grids it and
    changes NOTHING about where the metal is (GeomScene.at_resolution).  The
    builder derives the grid from the SPEC's pitch every time, so a pitch
    change is a pure resolution change -- which is the property that failed
    before, when a margin turned into a 2 mm offset and a clipped domain.
"""
from __future__ import annotations

from pathlib import Path


from ion_gym.io.sim_spec import SimSpec


def scene_of(spec: SimSpec):
    """The GeomScene this spec was built from, re-gridded to the spec's CURRENT
    pitch.  The GeomScene is the geometry truth; the spec carries the knobs.

    The GeomScene lives INLINE in spec.scene. It used to be a path
    (spec.import_path), which meant the spec was only loadable on the machine
    that wrote it -- move the JSON and you got
    `[Errno 2] No such file or directory: '/tmp/...'`. A path to a temp file
    is not a geometry; it is a promise about someone else's disk.

    A path is still honoured for specs written before this change, and a
    MISSING one now refuses with a diagnostic that says what to do.
    """
    from ion_gym.physics.scene3d import GeomScene
    d = getattr(spec, "scene", None)
    if d:
        sc = GeomScene.from_dict(d) if isinstance(d, dict) else GeomScene.from_json(d)
    else:
        src = str(spec.import_path or "")
        p = Path(src) if (src and len(src) < 4096 and "\n" not in src) else None
        if p is not None and p.exists():
            sc = GeomScene.from_json(p.read_text())          # legacy
        elif src.lstrip().startswith("{"):
            sc = GeomScene.from_json(src)                    # legacy inline text
        else:
            raise ValueError(
                f"builder='scene3d' but this spec carries no geometry: "
                f"spec.scene is empty and import_path={src!r} does not exist. "
                f"That path was local to whichever machine wrote the spec. "
                f"Rebuild the spec with build_scene3d.simspec_from_scene(), "
                f"which now EMBEDS the scene so the JSON travels with its "
                f"geometry.")
    return sc.at_resolution(float(spec.geometry.mm_per_gu)).check()


def scene_masks_3d(spec: SimSpec, verbose=False):
    """{electrode index -> (nx,ny,nz) bool}, straight from the analytic CSG."""
    from ion_gym.physics.rasterize3d import rasterize
    sc = scene_of(spec)
    if verbose:
        g = sc.grid
        print(f"[scene3d] rasterizing {len(sc.electrodes)} conductors on "
              f"{g.nx}x{g.ny}x{g.nz} @ {g.mm_per_gu} mm/gu "
              f"(analytic CSG — no tessellation)")
    lab = rasterize(sc)
    masks = {}
    for e in sc.electrodes:
        m = (lab == e.index)
        if not m.any():
            raise ValueError(
                f"electrode {e.index} ({e.name!r}) rasterised to ZERO voxels "
                f"at {sc.grid.mm_per_gu} mm/gu — it is thinner than one cell. "
                f"Refusing: a conductor that is not in the grid cannot hold a "
                f"boundary condition, and the solve would silently proceed "
                f"without it. Use a finer pitch.")
        masks[e.index] = m
    if verbose:
        for e in sc.electrodes:
            print(f"    e{e.index:<3d} {e.name:<8s} "
                  f"{int(masks[e.index].sum()):>9,d} voxels")
    return masks


def _geom_bytes(spec: SimSpec) -> bytes:
    """Cache key material: the SCENE ITSELF, not any file on disk."""
    sc = scene_of(spec)
    return sc.to_json().encode()


def build_scene3d_run(spec: SimSpec, verbose=False):
    """GeomScene -> (model, fly_fn, col_names, births).  Same solver, same cache,
    same flyer as the STL path; only the mask provider differs."""
    from ion_gym.physics.build_stl3d import build_stl3d_run
    return build_stl3d_run(spec, verbose=verbose, masks_fn=scene_masks_3d,
                           geom_bytes=_geom_bytes, tag="scene3d-v1")


def simspec_from_scene(scene, *, json_path=None, name=None, dc_group=None,
                       dc_group_prefix="E"):
    """scene3d.GeomScene -> SimSpec (builder='scene3d').

    THE DOMAIN IS THE GRID BOX, NOT THE METAL BOX.  This is the bug that
    clipped the Q3: the metal bbox is 10.9 mm wide but sits at 2.0..12.9 mm
    inside a 15.0 mm grid (the margin).  Declaring width_mm from the METAL
    made every electrode land 2 mm off and lopped 2 mm off the far edge --
    visibly, in the xy view.  The domain a solver must cover is the GRID box.
    """
    from ion_gym.io.sim_spec import (GeometrySpec, ElectrodeSpec, SourceSpec,
                          RFGroupSpec)
    from ion_gym.io.sim_spec import SymmetrySpec

    sc = scene.in_mm().check()
    glo, ghi = sc.grid_extent_mm()          # <- the grid, not the metal

    groups, els = {}, []
    dc_groups, ladder_idx = [], {}
    if dc_group:
        from ion_gym.io.sim_spec import DCGroupSpec
        mem = [e for e in sc.electrodes if e.name.startswith(dc_group_prefix)]
        if len(mem) >= 2:
            vs = [float(e.voltage) for e in mem
                  if not isinstance(e.voltage, str)]
            dc_groups = [DCGroupSpec(name=dc_group, v_in=max(vs),
                                     v_out=min(vs))]
            # the NUMBER is the conductor index — explicit, never parsed from
            # the name ("E10" would sort before "E9").
            ladder_idx = {e.name: e.index for e in mem}

    for e in sc.electrodes:
        dc, grp = 0.0, None
        if isinstance(e.voltage, str):
            grp = e.voltage
            groups.setdefault(grp, RFGroupSpec(name=grp))
        else:
            dc = float(e.voltage)
        els.append(ElectrodeSpec(
            name=e.name, dc=dc, rf_groups=([grp] if grp else []), basis=e.index,
            dc_group=(dc_group if e.name in ladder_idx else None),
            dc_index=ladder_idx.get(e.name)))

    planes = {a: ("mirror" if a in sc.grid.mirror else "none")
              for a in ("x", "y", "z")}
    geo = GeometrySpec(
        width_mm=float(ghi[0] - glo[0]),
        height_mm=float(ghi[1] - glo[1]),
        depth_mm=float(ghi[2] - glo[2]),
        mm_per_gu=float(sc.grid.mm_per_gu),
        symmetry=SymmetrySpec(coords="xyz", planes=planes),
        electrodes=els, rf_groups=list(groups.values()),
        dc_groups=dc_groups)

    if json_path:                     # optional sidecar, for the record only
        Path(json_path).write_text(sc.to_json())
    return SimSpec(geometry=geo, source=SourceSpec(),
                   name=(name or sc.name or "scene3d"),
                   notes=sc.notes, builder="scene3d",
                   scene=sc.to_dict())       # EMBEDDED — travels with the spec
