"""STL resource resolution + preflight for STL-backed specs.

An STL electrode stores a FILENAME (ElectrodeSpec.stl) plus the geometry's
stl_dir, and the mesh is read from disk on every solve -- so a saved STL
spec is portable only if its files travel with it and can be located and
verified. This module owns that concern in ONE place (so build_stl,
build_stl3d, the cache key, and the UI all resolve identically):

  * resolve_stl_dir(spec): the directory STL files live in, resolved
    RELATIVE TO THE SPEC FILE (spec._loaded_from) when stl_dir is
    relative -- so `spec.json + its stl folder` is a portable unit that
    works regardless of CWD. Absolute stl_dir is honored as-is.
  * preflight_stls(spec): returns (ok, report). Checks every STL
    electrode's file EXISTS and, if the spec embeds stl_manifest hashes,
    that the on-disk bytes MATCH. Refuses with a NAMED diagnostic listing
    exactly which files are missing or changed -- never a mid-solve
    trimesh stack trace.
  * build_manifest(spec): {filename: sha256} for embedding in the spec so
    a future run can verify the files are the ones it was authored with.

Doctrine: no silent fallback. A missing//changed STL is a refuse-with-
diagnostic at preflight, not a wrong-geometry solve.
"""

import hashlib
from pathlib import Path


def resolve_stl_dir(spec) -> Path:
    """Directory holding this spec's STL files. Relative stl_dir resolves
    against the spec's own location (portable spec+assets folder); absolute
    is used verbatim; CWD only as a last resort with no spec path."""
    g = spec.geometry
    raw = g.stl_dir
    if not raw:
        base = getattr(spec, "_loaded_from", None)
        return Path(base).parent if base else Path(".")
    p = Path(raw)
    if p.is_absolute():
        return p
    base = getattr(spec, "_loaded_from", None)
    if base:
        return (Path(base).parent / p)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_manifest(spec) -> dict:
    """{filename: sha256} over the STL files this spec references, for
    embedding as spec.stl_manifest so future runs can verify them."""
    d = resolve_stl_dir(spec)
    out = {}
    for el in spec.geometry.electrodes:
        if el.stl:
            fp = d / el.stl
            if fp.exists():
                out[el.stl] = _sha256(fp)
    return out


def preflight_stls(spec):
    """(ok, report). Verify every STL electrode's file is present and,
    if the spec carries stl_manifest, unchanged. report names exactly
    what is wrong so the caller can raise a clean diagnostic."""
    d = resolve_stl_dir(spec)
    manifest = getattr(spec.geometry, "stl_manifest", None) or {}
    missing, changed, ok_files = [], [], []
    for el in spec.geometry.electrodes:
        if not el.stl:
            continue
        fp = d / el.stl
        if not fp.exists():
            missing.append(el.stl)
            continue
        if el.stl in manifest and _sha256(fp) != manifest[el.stl]:
            changed.append(el.stl)
        else:
            ok_files.append(el.stl)
    ok = not missing and not changed
    lines = [f"STL preflight for {spec.name!r}: dir {d}"]
    if ok_files:
        lines.append(f"  {len(ok_files)} file(s) present"
                     + (" and matching manifest" if manifest else ""))
    if missing:
        lines.append(f"  MISSING ({len(missing)}): {', '.join(missing)}")
    if changed:
        lines.append(f"  CHANGED vs manifest ({len(changed)}): "
                     f"{', '.join(changed)}")
    return ok, "\n".join(lines)


def require_stls(spec):
    """Raise a NAMED diagnostic if preflight fails; else return the dir.
    Call this at the top of any STL build so a missing/changed file stops
    BEFORE the solve, not inside trimesh."""
    ok, report = preflight_stls(spec)
    if not ok:
        raise FileNotFoundError(
            report + "\n  -> place the listed STL files in the directory "
            "above (keep the spec and its STL folder together), or "
            "re-import the geometry.")
    return resolve_stl_dir(spec)


# --------------------------------------------------- declared placement
def placement_offset(spec):
    """The DECLARED CAD->build translation, validated. Returns a (3,) float
    array; [0,0,0] when geometry.frame_offset_mm is undeclared.

    ONE INTERPRETATION AUTHORITY: every consumer of the declared frame --
    mesh ingest below, SimSpec.validate, and (via to_dict) the basis-cache
    geometry key -- reads the offset through this function, so the mesh
    the solver voxelizes, the refusal a bad deck gets, and the identity a
    cached basis is stored under can never disagree about what the field
    means. Refuses with the value in the message: a malformed placement
    silently treated as identity would put metal in the wrong place with
    no symptom (a real notebook-breaking failure class).
    """
    import numpy as np
    raw = getattr(spec.geometry, "frame_offset_mm", None)
    if raw is None:
        return np.zeros(3)
    try:
        off = np.asarray([float(v) for v in raw], dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"geometry.frame_offset_mm must be three numbers (mm), got "
            f"{raw!r}") from e
    if off.shape != (3,):
        raise ValueError(
            f"geometry.frame_offset_mm must be [x, y, z] mm (3 values), "
            f"got {len(off)}: {raw!r}")
    if not np.all(np.isfinite(off)):
        raise ValueError(
            f"geometry.frame_offset_mm must be finite, got {raw!r}")
    return off


def load_mesh(spec, el, *, stl_dir=None):
    """Load electrode `el`'s mesh with its DECLARED placement applied --
    THE single mesh-ingest point (declare
    placement in the spec, apply it at the single mesh-ingest point').

    Every route that consumes an electrode mesh (2-D slice, 3-D voxelize,
    and anything future) loads through here, so placement has one owner
    and zero route branches. The returned mesh is in the BUILD frame, in
    mm; callers apply their own mm->gu scaling after.

    This replaces the bake-in-place pattern (build_stl.py used to
    translate fixture files by re-exporting them), under which a deck's
    correctness lived in mutated binary bytes that nothing declared and
    any innocent CAD re-export silently reverted. Placement is rigid
    translation only; there is no scale/shear/flip and none will be
    accepted here -- a resized mesh is a DIFFERENT PART, and that edit
    belongs in CAD, not in a loader.
    """
    trimesh = _require_trimesh()
    d = Path(stl_dir) if stl_dir is not None else resolve_stl_dir(spec)
    if not getattr(el, "stl", None):
        raise ValueError(
            f"load_mesh: electrode {getattr(el, 'name', '?')!r} has no stl "
            f"reference -- parametric shapes declare their own positions "
            f"and never pass through mesh ingest")
    m = trimesh.load(str(d / el.stl), force="mesh", process=True)
    off = placement_offset(spec)
    if off.any():
        m.apply_translation(off)
    return m


def _require_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise ImportError(
            "trimesh is required to load STL geometry (pip install "
            "trimesh)") from e
