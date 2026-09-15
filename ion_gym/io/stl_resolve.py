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
    # A relative dir with no recorded spec path is the CWD fallback --
    # legal, but it MUST say so: a bare "dir ." sent a user hunting file
    # placement when the real cause was a spec loaded from TEXT, which
    # has no path to anchor against (2026-09-11 field report).
    note = ""
    if not d.is_absolute() and getattr(spec, "_loaded_from", None) is None:
        note = (f" (= CWD fallback {(Path.cwd() / d).resolve()} -- the spec "
                f"records no file path, so a relative stl_dir has nothing "
                f"to anchor against; select the spec and its STL files "
                f"together, or set an absolute stl_dir)")
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
    lines = [f"STL preflight for {spec.name!r}: dir {d}{note}"]
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


# ------------------------------------------------------ upload payload
def _payload_digest(name_to_sha256):
    """Content address of a mesh set: sha256 over sorted (name, byte-hash)
    pairs. THE one derivation -- computable from raw bytes at install time
    and from a spec's stl_manifest alone at re-load time, which is what
    lets a previously installed deck resolve from its JSON with no bytes
    re-supplied."""
    h = hashlib.sha256()
    for n in sorted(name_to_sha256):
        h.update(n.encode())
        h.update(bytes.fromhex(name_to_sha256[n]))
    return h.hexdigest()


def install_stl_payload(doc, payload, workdir):
    """Make an in-memory spec DOCUMENT's STL references resolvable by
    materializing the mesh bytes that arrived with it.

    A spec and the meshes it names are ONE unit. Any transport that
    carries bytes without a filesystem path (a browser upload, a message
    body, a notebook string) strands a relative ``stl_dir``: there is no
    anchor, so resolution falls back to the process CWD and the solve
    refuses. This function is that transport's other half -- it takes the
    meshes too and writes them somewhere the spec can point at.

    doc      : the PARSED spec JSON (mutated in place: geometry.stl_dir is
               set to the absolute directory the meshes were written to).
               Operating on the document, not a SimSpec, is deliberate --
               the rewrite must happen BEFORE the spec is built and drawn,
               so no consumer ever sees the unresolvable state.
    payload  : {filename: bytes} of the STL files supplied alongside.
    workdir  : root for materialized decks (caller's choice; no path is
               baked in here). Deck bytes land in a CONTENT-ADDRESSED
               subdirectory, so re-supplying the same deck reuses the same
               directory and the bases cache key (which hashes STL bytes)
               stays stable across loads.

    Returns (directory | None, notes). None means the document declares no
    STL electrodes and nothing was written. REFUSES, by name, when a
    declared mesh is absent from the payload or when supplied bytes do not
    match the document's stl_manifest -- a deck solved from the wrong
    meshes is wrong silently, which is the failure this refuses to allow.
    """
    geom = doc.get("geometry") if isinstance(doc, dict) else None
    if not isinstance(geom, dict):
        if payload:
            raise ValueError(
                f"{len(payload)} STL file(s) supplied with a document that "
                f"has no 'geometry' section to attach them to "
                f"({', '.join(sorted(payload))}) -- a staged assembly or a "
                f"non-spec document is not handled by this door")
        return None, []
    needed = [el.get("stl") for el in geom.get("electrodes", [])
              if isinstance(el, dict) and el.get("stl")]
    if not needed:
        # Not an STL deck. Extra meshes are a visible, reported decision.
        notes = ([f"**ignored {len(payload)} STL file(s):** this deck "
                  f"declares no STL electrodes."] if payload else [])
        return None, notes

    manifest = geom.get("stl_manifest") or {}
    missing = [n for n in dict.fromkeys(needed) if n not in payload]
    if missing and not payload:
        # JSON alone. The manifest names every mesh AND its sha256, which
        # fully determines the content address below -- so a deck whose
        # meshes were installed by an earlier load resolves EXACTLY, with
        # every byte re-verified. Not a search, not a guess: either the
        # manifest-addressed directory holds the exact bytes, or refuse.
        if all(n in manifest for n in dict.fromkeys(needed)):
            cand = Path(workdir) / _payload_digest(
                {n: manifest[n] for n in dict.fromkeys(needed)})[:16]
            if cand.is_dir():
                stale = [n for n in dict.fromkeys(needed)
                         if not (cand / Path(n).name).exists()
                         or hashlib.sha256(
                             (cand / Path(n).name).read_bytes()
                         ).hexdigest() != manifest[n]]
                if not stale:
                    geom["stl_dir"] = str(cand.resolve())
                    return cand, [
                        f"**{len(set(needed))} STL file(s) found from a "
                        f"previous install** at `{cand.resolve()}` -- every "
                        f"file re-verified against the spec's manifest; "
                        f"`geometry.stl_dir` now points there."]
    if missing:
        raise FileNotFoundError(
            f"this deck references {len(set(needed))} STL file(s) and "
            f"{len(missing)} did not arrive with it: {', '.join(missing)}"
            f"\n  -> select the spec .json AND its .stl files TOGETHER in "
            f"the picker (Cmd/Ctrl-click; they are one unit -- the upload "
            f"carries bytes, not the folder they sit in). A deck loaded "
            f"this way once resolves from the .json alone afterwards.")

    bad = [n for n in dict.fromkeys(needed)
           if n in manifest
           and hashlib.sha256(payload[n]).hexdigest() != manifest[n]]
    if bad:
        raise ValueError(
            f"STL bytes do not match the spec's stl_manifest for: "
            f"{', '.join(bad)}\n  -> these are not the meshes this deck "
            f"was authored with; re-export the deck or supply the "
            f"matching files")

    # content address over (name, bytes) of exactly the meshes in use --
    # via the SAME function the manifest-only path uses, so the two can
    # never derive different addresses for the same deck
    out = Path(workdir) / _payload_digest(
        {n: hashlib.sha256(payload[n]).hexdigest()
         for n in dict.fromkeys(needed)})[:16]
    out.mkdir(parents=True, exist_ok=True)
    for n in dict.fromkeys(needed):
        fp = out / Path(n).name
        # rewrite only when content differs: an untouched file keeps its
        # mtime, and the deck stays byte-identical across reloads
        if not fp.exists() or fp.read_bytes() != payload[n]:
            fp.write_bytes(payload[n])
    geom["stl_dir"] = str(out.resolve())

    notes = [f"**{len(set(needed))} STL file(s) installed** to "
             f"`{out.resolve()}`"
             + (" and verified against the spec's manifest"
                if manifest else " (spec carries no manifest to verify "
                                 "against)")
             + "; `geometry.stl_dir` in the editor now points there, so "
               "what is displayed is what solves."]
    extra = [n for n in payload if n not in set(needed)]
    if extra:
        notes.append(f"**ignored {len(extra)} unreferenced STL file(s):** "
                     f"{', '.join(sorted(extra))}")
    return out, notes


def portable_stl_dir(doc, cache_root):
    """Make a spec DOCUMENT portable for saving: when geometry.stl_dir
    points inside the app's own mesh cache (`cache_root`, where
    install_stl_payload materializes uploaded meshes), rewrite it to
    "." in the document. The cache path is a fact about ONE machine's
    session, and a saved deck carrying it breaks everywhere else --
    while "." plus the intact stl_manifest is the portable truth: the
    deck resolves beside its meshes on any path load, and resolves from
    the JSON alone through the manifest-addressed reload on the machine
    that installed it. A user-managed absolute stl_dir OUTSIDE the cache
    is a deliberate declaration and is left untouched.

    Mutates doc in place; returns a note string when it rewrote,
    None otherwise. Never touches spec objects -- the LIVE session keeps
    its absolute dir (it must keep solving); only the saved copy is
    normalized.
    """
    geom = doc.get("geometry") if isinstance(doc, dict) else None
    if not isinstance(geom, dict):
        return None
    raw = geom.get("stl_dir")
    if not raw:
        return None
    d = Path(raw)
    if not d.is_absolute():
        return None
    try:
        d.resolve().relative_to(Path(cache_root).resolve())
    except ValueError:
        return None            # user-managed absolute dir: theirs, kept
    geom["stl_dir"] = "."
    return ('stl_dir rewritten to "." for portability (it pointed into '
            'this machine\'s mesh cache); keep the .stl files with the '
            'saved JSON, or reload the JSON alone on this machine and '
            'the meshes resolve by manifest')


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
