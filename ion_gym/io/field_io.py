"""
ion_gym.io.field_io — portable solved fields (SPEC_field_portability.md)
========================================================================
Three separately-loadable, plainly-named user artifacts:

  * geometry  — the spec JSON (spec_io, already exists);
  * FIELD     — ONE human-named .npz holding the solved per-electrode bases
                + electrode labels + an embedded `_meta` JSON member that
                makes the file SELF-VALIDATING (this module);
  * TRAJECTORIES — an .npz of decimated ion paths + `_meta` frame (this
                module).  Overlay-friendly: geometry mismatch WARNS, never
                refuses.

Identity contract: a field belongs to a geometry iff
    fa_cache.spec_key(basis_cache.geometry_key_dict(spec)) == _meta["key"].
That is the SAME discrimination the disk cache uses (gate B4 asserts it),
so a filename is purely cosmetic — users may rename files freely; loading
hard-validates on the embedded key and REFUSES on any mismatch, naming the
diverging geometry fields.  A field applied to the wrong geometry is
invisible catastrophic physics; there is no warn-and-proceed path for
fields.

Cross-machine transfer: the efficient unit is the tiny
geometry JSON + a FINGERPRINT (this module), not the megabytes of bases —
bases are a deterministic function of geometry.  Receiver re-solves and
ANCHORS its own solve to the sender's fingerprint (probe potentials agree
to tolerance) or surfaces the divergence loudly.  The full-bases .npz is
the fallback for receivers who cannot re-solve.

Fingerprint probes index the basis ARRAYS at deterministic grid nodes —
exact (no interpolation) and layering-clean (the field samplers live under
projects/, which core io must not import).

Labels: display labels are LOCAL by default (fields_registry.json in
paths.fields_dir()); `_meta["label"]` written once at save travels with
the file as a default.  Relabeling edits the registry only — it never
rewrites a multi-hundred-MB npz.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

import numpy as np

from ion_gym.io import fa_cache
from ion_gym.io import basis_cache
from ion_gym.io import paths

# Bump when the FILE layout/semantics change (the basis_cache CACHE_FORMAT
# lesson: a version that cannot express "the stored thing is shaped
# differently now" serves yesterday's bug forever).
FIELD_FORMAT = 1
TRAJ_FORMAT = 1

_META = "_meta"                       # the embedded-JSON npz member name
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class FieldMismatch(ValueError):
    """Loaded field does not belong to the loaded geometry (or is a version
    /format this code cannot honor).  Message names what diverged."""


class TrajectoryFormatError(ValueError):
    """Trajectory npz is not a recognized ion_gym trajectory file."""


# ------------------------------------------------------------------ helpers
def _slug(name):
    s = _SAFE.sub("_", (name or "field").strip()) or "field"
    return s[:80]


def _meta_to_member(meta: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)


def _member_to_meta(arr) -> dict:
    return json.loads(bytes(np.asarray(arr, np.uint8)).decode())


def _atomic_savez(path, **members):
    """Write an npz atomically: temp file in the SAME directory, fsync,
    os.replace.  A crashed half-write is never visible under `path`."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".field_io.", suffix=".npz", dir=d)
    ok = False
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **members)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        ok = True
    finally:
        if not ok and os.path.exists(tmp):
            os.remove(tmp)


def _geometry_diff(a, b, prefix=""):
    """Named divergence between two canonical geometry dicts, for the
    refuse-with-diagnostic message.  Returns a list of 'path: A vs B'.

    THE ONLY geometry-diff in this module, serving validate_field,
    resolve_cache_entry and import_field_bundle alike. A second, shallow
    `_geometry_diff(g1, g2)` was once defined further down the file and
    SILENTLY SHADOWED this one at import: mismatch messages degraded from
    naming the diverging path to naming bare top-level keys ("electrodes;
    width_mm" instead of which electrode field differs), and
    validate_field raised TypeError instead of refusing whenever a field
    npz carried no `geometry` in its `_meta` (set(None) is not iterable).
    Both callers of the shallow version only test the result for
    emptiness and print it, which this satisfies. Do not add a second
    one: one name, one authority."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{prefix}{k}: <absent> vs present")
            elif k not in b:
                out.append(f"{prefix}{k}: present vs <absent>")
            else:
                out.extend(_geometry_diff(a[k], b[k], f"{prefix}{k}."))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{prefix.rstrip('.')}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(_geometry_diff(x, y, f"{prefix}[{i}]."))
    elif a != b:
        out.append(f"{prefix.rstrip('.')}: {a!r} vs {b!r}")
    return out


# ================================================================== FIELDS
def field_meta(spec, bases, ele, label=None):
    """The embedded `_meta` for a solved field.  Everything a receiver
    needs to validate and reinstate without any sidecar."""
    gd = basis_cache.geometry_key_dict(spec)
    key = fa_cache.spec_key(gd)
    arrays = {f"b{int(k)}": np.ascontiguousarray(v, float)
              for k, v in bases.items()}
    arrays["ele"] = np.ascontiguousarray(ele)
    manifest = {n: dict(shape=list(a.shape), dtype=str(a.dtype),
                        sha_head=fa_cache._sha_head(a))
                for n, a in arrays.items()}
    spec_name = getattr(spec, "name", "") or "field"
    meta = dict(
        kind="ion_gym_field",
        field_format=FIELD_FORMAT,
        key=key,
        cache_version=fa_cache.CACHE_VERSION,
        cache_format=basis_cache.CACHE_FORMAT,
        geometry=gd,
        arrays=manifest,
        electrode_order=sorted(int(k) for k in bases),
        grid_dims=list(np.asarray(ele).shape),
        label=label or spec_name,
        spec_name=spec_name,
        # full spec (voltages included) = the operating point, inline.
        spec=spec.to_dict() if hasattr(spec, "to_dict") else None,
    )
    return meta, arrays


def default_field_filename(spec):
    gd = basis_cache.geometry_key_dict(spec)
    key = fa_cache.spec_key(gd)
    return f"{_slug(getattr(spec, 'name', '') or 'field')}__{key[:8]}.npz"


def save_field(spec, bases, ele, path=None, label=None):
    """Save a solved field as ONE named, self-validating npz.  Returns the
    written path.  `path=None` -> paths.fields_dir()/<name>__<key8>.npz."""
    meta, arrays = field_meta(spec, bases, ele, label=label)
    if path is None:
        path = str(paths.fields_dir() / default_field_filename(spec))
    _atomic_savez(path, **arrays, **{_META: _meta_to_member(meta)})
    # seed the local label registry BESIDE the saved file (the registry
    # indexes the folder it lives in; a custom save dir gets its own)
    if meta["label"]:
        set_label(meta["key"], meta["label"],
                  directory=os.path.dirname(os.path.abspath(path)))
    return path


def read_field_meta(path):
    """Read ONLY the embedded `_meta` (npz is a zip: one small member is
    read, never the base arrays).  Raises FieldMismatch if the file is not
    an ion_gym field."""
    with np.load(path) as z:
        if _META not in z.files:
            raise FieldMismatch(
                f"{os.path.basename(path)}: no embedded _meta — not an "
                f"ion_gym field npz (members: {sorted(z.files)[:6]}...)")
        meta = _member_to_meta(z[_META])
    if meta.get("kind") != "ion_gym_field":
        raise FieldMismatch(f"{os.path.basename(path)}: _meta.kind is "
                            f"{meta.get('kind')!r}, not 'ion_gym_field'")
    return meta


def validate_field(meta, spec):
    """HARD-REFUSE chain (SPEC §Load field).  Raises FieldMismatch naming
    the divergence; returns the recomputed key on success."""
    # 1) format/version compatibility first — a wrong-format file must not
    #    reach the geometry comparison with undefined semantics.
    if meta.get("field_format") != FIELD_FORMAT:
        raise FieldMismatch(
            f"field_format {meta.get('field_format')} != supported "
            f"{FIELD_FORMAT} — file from an incompatible ion_gym")
    if meta.get("cache_version") != fa_cache.CACHE_VERSION or \
       meta.get("cache_format") != basis_cache.CACHE_FORMAT:
        raise FieldMismatch(
            f"cache version/format {meta.get('cache_version')}/"
            f"{meta.get('cache_format')} != local "
            f"{fa_cache.CACHE_VERSION}/{basis_cache.CACHE_FORMAT} — "
            f"solved under an incompatible cache scheme; re-solve locally")
    # 2) PRIMARY: identity by geometry key (the cache's own discrimination)
    gd = basis_cache.geometry_key_dict(spec)
    key = fa_cache.spec_key(gd)
    if meta.get("key") != key:
        diffs = _geometry_diff(meta.get("geometry"), gd)
        head = "; ".join(diffs[:6]) or "geometry dicts differ"
        more = f" (+{len(diffs)-6} more)" if len(diffs) > 6 else ""
        raise FieldMismatch(
            f"field was solved for a DIFFERENT geometry "
            f"({meta.get('spec_name','?')}, key {str(meta.get('key'))[:8]}) "
            f"than the loaded one (key {key[:8]}). Diverging: {head}{more}. "
            f"Refusing — a wrong-geometry field is silently wrong physics.")
    # 3) structural cross-checks (defense in depth + better diagnostics)
    n_ele = len((gd or {}).get("electrodes", []))
    order = meta.get("electrode_order") or []
    if len(order) != n_ele:
        raise FieldMismatch(f"basis count {len(order)} != electrode count "
                            f"{n_ele} of the loaded geometry")
    grid = meta.get("grid_dims")
    for name, m in (meta.get("arrays") or {}).items():
        if name.startswith("b") and list(m["shape"]) != list(grid):
            raise FieldMismatch(f"array {name} shape {m['shape']} != "
                                f"grid_dims {grid} — inconsistent file")
    return key


def load_field(path, spec, root=fa_cache.DEFAULT_ROOT):
    """Validate `path` against the CURRENT spec (hard-refuse), verify array
    integrity against the embedded manifest, reinstate the bases into the
    local disk cache under the key, and return (bases, ele) exactly as
    basis_cache.load would.  After this, the geometry is a cache HIT."""
    meta = read_field_meta(path)
    validate_field(meta, spec)
    with np.load(path) as z:
        arrays = {}
        for name, m in meta["arrays"].items():
            if name not in z.files:
                raise FieldMismatch(f"{os.path.basename(path)}: manifest "
                                    f"array {name!r} missing from file")
            a = np.ascontiguousarray(z[name])
            if list(a.shape) != m["shape"] or str(a.dtype) != m["dtype"]:
                raise FieldMismatch(
                    f"{name}: stored {a.shape}/{a.dtype} != manifest "
                    f"{m['shape']}/{m['dtype']} — truncated or altered file")
            if fa_cache._sha_head(a) != m["sha_head"]:
                raise FieldMismatch(f"{name}: content hash != manifest — "
                                    f"altered or corrupt file")
            arrays[name] = a
    # reinstate: same publish path the solver uses (atomic, deduplicating)
    key = fa_cache.store(basis_cache.geometry_key_dict(spec), arrays,
                         root=root)
    fa_cache.annotate(key, meta.get("spec_name", ""),
                      label=meta.get("label"), root=root)
    bases = {int(k[1:]): np.ascontiguousarray(v, float)
             for k, v in arrays.items() if k.startswith("b")}
    ele = np.ascontiguousarray(arrays["ele"])
    return bases, ele.astype(np.int16)


def read_spec_from_field(path):
    """Reconstruct the SimSpec EMBEDDED in a saved field's `_meta["spec"]`
    (the full operating point — voltages included — written at save).

    This is the bootstrap direction: a saved
    field is the whole run bundle, so work can start FROM the
    file with no JSON on hand.  Reconstruction goes through load_any_spec
    — the same single door every spec enters by — so the result is
    byte-for-byte the spec the loaders would produce, not a private
    deserialization.  Refuses with a diagnostic on a pre-bootstrap save
    (older files whose `_meta` carries spec=None): those remain loadable
    by the validate-against-loaded-geometry path only.
    """
    from ion_gym.io.spec_io import load_any_spec
    meta = read_field_meta(path)
    sd = meta.get("spec")
    if not sd:
        raise FieldMismatch(
            f"{os.path.basename(path)}: no embedded spec in _meta — this "
            f"is a pre-bootstrap save. Load its geometry JSON first, then "
            f"use the validate-and-load path.")
    spec = load_any_spec(json.dumps(sd))
    return spec, meta


def load_field_bootstrap(path, root=fa_cache.DEFAULT_ROOT):
    """One-file session open: reconstruct the spec from the file itself,
    then run the UNCHANGED hard-refuse load against it.  Every integrity
    check load_field performs (manifest shapes/dtypes/content hashes,
    geometry-key match) still fires — against the reconstructed spec the
    validation is a self-consistency proof of the file, not a bypass.
    Returns (spec, bases, ele); the bases are reinstated into the disk
    cache exactly as an ordinary load would."""
    spec, _meta_ = read_spec_from_field(path)
    bases, ele = load_field(path, spec, root=root)
    return spec, bases, ele


# ============================================================= FINGERPRINT
def _probe_nodes(shape, per_axis=5):
    """Deterministic interior probe lattice: per_axis fractional positions
    on each axis, clipped inside the grid.  Depends only on the grid shape,
    so sender and receiver agree by construction."""
    fracs = np.linspace(0.15, 0.85, per_axis)
    axes = [np.unique(np.clip((fracs * (n - 1)).round().astype(int), 0, n - 1))
            for n in shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)   # (n_probe, ndim)


def fingerprint(spec, bases, ele, per_axis=5):
    """Few-KB portable anchor for a solve: geometry key + per-basis
    potentials at deterministic grid nodes.  Receiver re-solves and calls
    verify_fingerprint on ITS solve."""
    gd = basis_cache.geometry_key_dict(spec)
    shape = np.asarray(ele).shape
    pts = _probe_nodes(shape, per_axis)
    values = {}
    for k in sorted(int(i) for i in bases):
        b = np.ascontiguousarray(bases[k], float)
        if b.shape != tuple(shape):
            raise FieldMismatch(f"basis b{k} shape {b.shape} != electrode "
                                f"grid {tuple(shape)} — cannot fingerprint")
        values[f"b{k}"] = [float(b[tuple(p)]) for p in pts]
    return dict(kind="ion_gym_field_fingerprint",
                field_format=FIELD_FORMAT,
                key=fa_cache.spec_key(gd),
                cache_version=fa_cache.CACHE_VERSION,
                cache_format=basis_cache.CACHE_FORMAT,
                grid_dims=list(shape),
                probe_nodes=[[int(i) for i in p] for p in pts],
                values=values,
                spec_name=getattr(spec, "name", "") or "field")


def save_transfer(spec, bases, ele, path=None, per_axis=5):
    """The efficient cross-machine unit: ONE small JSON = full spec (the
    geometry travels) + fingerprint.  Receiver: load spec, re-solve, then
    verify_fingerprint(transfer['fingerprint'], spec, bases, ele)."""
    if not hasattr(spec, "to_dict"):
        raise FieldMismatch("spec has no to_dict(); cannot export a "
                            "self-contained transfer")
    doc = dict(kind="ion_gym_field_transfer",
               spec=spec.to_dict(),
               fingerprint=fingerprint(spec, bases, ele, per_axis))
    if path is None:
        gd = basis_cache.geometry_key_dict(spec)
        key = fa_cache.spec_key(gd)
        path = str(paths.fields_dir() /
                   f"{_slug(getattr(spec, 'name', '') or 'field')}"
                   f"__{key[:8]}.transfer.json")
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".field_io.", suffix=".json", dir=d)
    ok = False
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        ok = True
    finally:
        if not ok and os.path.exists(tmp):
            os.remove(tmp)
    return path


def verify_fingerprint(fp, spec, bases, ele, atol_v=1e-3, rtol=1e-6):
    """Anchor the LOCAL solve to a sender's fingerprint.  Raises
    FieldMismatch naming the worst deviation; returns max |dV| on success.
    Tolerances default well above solver tol (1e-6 relative on ~1e4 V
    bases) but far below anything physical."""
    if fp.get("kind") != "ion_gym_field_fingerprint":
        raise FieldMismatch("not a fingerprint document")
    gd = basis_cache.geometry_key_dict(spec)
    key = fa_cache.spec_key(gd)
    if fp.get("key") != key:
        raise FieldMismatch(
            f"fingerprint is for geometry key {str(fp.get('key'))[:8]}, "
            f"local geometry is {key[:8]} — different geometry, not a "
            f"solver difference")
    if fp.get("cache_version") != fa_cache.CACHE_VERSION or \
       fp.get("cache_format") != basis_cache.CACHE_FORMAT:
        raise FieldMismatch("fingerprint from an incompatible cache "
                            "version/format — cannot anchor; use the "
                            "full-bases npz fallback")
    shape = tuple(np.asarray(ele).shape)
    if tuple(fp.get("grid_dims", ())) != shape:
        raise FieldMismatch(f"fingerprint grid {fp.get('grid_dims')} != "
                            f"local grid {list(shape)}")
    pts = [tuple(p) for p in fp["probe_nodes"]]
    worst = (0.0, None)
    for name, vals in fp["values"].items():
        k = int(name[1:])
        if k not in bases:
            raise FieldMismatch(f"local solve has no basis {name}")
        b = np.ascontiguousarray(bases[k], float)
        for p, v_ref in zip(pts, vals):
            dv = abs(float(b[p]) - float(v_ref))
            if dv > worst[0]:
                worst = (dv, (name, p, float(v_ref), float(b[p])))
        # fail fast per-basis with the named worst point
    dv, w = worst
    if w is not None:
        name, p, v_ref, v_loc = w
        tol = atol_v + rtol * max(abs(v_ref), abs(v_loc))
        if dv > tol:
            raise FieldMismatch(
                f"local solve diverges from fingerprint: {name} at node "
                f"{list(p)}: sender {v_ref:.6g} V vs local {v_loc:.6g} V "
                f"(|dV|={dv:.3g} > tol {tol:.3g}) — solver/version "
                f"mismatch; do not quote against the sender's results; "
                f"request the full-bases npz")
    return dv


# ============================================================ TRAJECTORIES
def _mz_per_ion(spec, ion_indices):
    """m/z (Da) for each saved ion index, or the string "unknown" where the
    spec cannot say (e.g. a respec'd loaded run) — recorded, not omitted."""
    from ion_gym.physics.sim_build import mz_of
    out = []
    for i in ion_indices:
        try:
            out.append(float(mz_of(spec, int(i))))
        except (ValueError, TypeError, IndexError, AttributeError):
            out.append("unknown")
    return out


def save_trajectories(results, spec, path=None, label=None):
    """Save decimated ion paths (IonResult list or (index, traj, summary)
    tuples) + a `_meta` frame.  Overlay-friendly on load: geometry
    mismatch is a WARNING, never a refusal."""
    gd = basis_cache.geometry_key_dict(spec)
    members, index, summaries = {}, [], []
    for r in results:
        i = getattr(r, "index", None)
        traj = getattr(r, "traj", None)
        summ = getattr(r, "summary", None)
        if i is None and isinstance(r, (tuple, list)):
            i, traj = r[0], r[1]
            summ = r[2] if len(r) > 2 else {}
        if traj is None:
            continue                      # never flown/not recorded: skip,
        index.append(int(i))              # and say so in _meta.n_skipped
        summaries.append(summ or {})
        members[f"t{int(i)}"] = np.ascontiguousarray(traj)
    n_skipped = len(list(results)) - len(index)
    cols = spec.column_names() if hasattr(spec, "column_names") else None
    meta = dict(kind="ion_gym_trajectories",
                traj_format=TRAJ_FORMAT,
                columns=cols,
                frame="Convention A: axial coordinate horizontal (z axial)",
                units=dict(position="mm", time="us", velocity="mm/us"),
                geometry_key=fa_cache.spec_key(gd),
                spec_name=getattr(spec, "name", "") or "simulation",
                label=label or (getattr(spec, "name", "") or "trajectories"),
                ion_index=index,
                summaries=summaries,
                # m/z per SAVED ion (for per-axis KE):
                # directional KE is derivable from stored channels alone
                # (KE_i = ke_ev * v_i^2/speed^2), but absolute momenta and
                # cross-file comparisons need the mass, so record it.
                # "unknown" is a recorded value, not an absent line (P-doc).
                mz_da=_mz_per_ion(spec, index),
                n_skipped=n_skipped)
    if path is None:
        path = str(paths.fields_dir() /
                   f"{_slug(meta['spec_name'])}__"
                   f"{meta['geometry_key'][:8]}.traj.npz")
    # PLAIN-ARRAY MIRROR of the display metadata (a
    # reader must work on the npz members ALONE, no JSON decode). `_meta`
    # stays for the GUI's rich loader; these plain members let numpy-only
    # code read columns/m-z/summaries/label without touching `_meta`.
    # Mirrors the Export-run file's own plain `columns`/`summary` layout,
    # so one json-free reader handles both files.
    plain = {}
    if cols is not None:
        plain["columns"] = np.array([str(c) for c in cols])
    plain["label"] = np.array([str(meta["label"])])
    # m/z as a float array (NaN = "unknown"), the numeric twin of the
    # string-tolerant _meta list.
    plain["mz_da"] = np.array(
        [np.nan if m == "unknown" else float(m) for m in meta["mz_da"]],
        dtype=float)
    # a flat summary table with a fixed, self-describing column set — the
    # same shape the Export-run file uses, so the reader treats them alike.
    s_cols = ["ion", "fate", "tof", "x_end", "y_end", "z_end"]
    s_rows = [[float(i),
               float(s.get("kind", -1)),
               float(s.get("tof", np.nan)),
               float(s.get("x_end", np.nan)),
               float(s.get("y_end", np.nan)),
               float(s.get("z_end", np.nan))]
              for i, s in zip(index, summaries)]
    plain["summary"] = (np.array(s_rows, float) if s_rows
                        else np.zeros((0, len(s_cols))))
    plain["summary_cols"] = np.array(s_cols)
    _atomic_savez(path, **members, **plain,
                  **{_META: _meta_to_member(meta)})
    return path


def read_trajectory_meta(path):
    with np.load(path) as z:
        if _META not in z.files:
            raise TrajectoryFormatError(
                f"{os.path.basename(path)}: no embedded _meta — not an "
                f"ion_gym trajectory npz")
        meta = _member_to_meta(z[_META])
    if meta.get("kind") != "ion_gym_trajectories":
        raise TrajectoryFormatError(f"_meta.kind {meta.get('kind')!r} != "
                                    f"'ion_gym_trajectories'")
    return meta


def load_trajectories(path, spec=None):
    """Returns (trajs, meta, warning).  trajs = {ion_index: (n,k) array}.
    warning is a string if `spec` is given and the geometry keys differ
    (overlaying is legitimate — the caller SURFACES it, never hides it),
    else None."""
    meta = read_trajectory_meta(path)
    if meta.get("traj_format") != TRAJ_FORMAT:
        raise TrajectoryFormatError(
            f"traj_format {meta.get('traj_format')} != supported "
            f"{TRAJ_FORMAT}")
    trajs = {}
    with np.load(path) as z:
        for i in meta.get("ion_index", []):
            name = f"t{int(i)}"
            if name not in z.files:
                raise TrajectoryFormatError(f"{name} listed in _meta but "
                                            f"missing from file")
            trajs[int(i)] = np.ascontiguousarray(z[name])
    warning = None
    if spec is not None:
        key = fa_cache.spec_key(basis_cache.geometry_key_dict(spec))
        if meta.get("geometry_key") != key:
            warning = (f"trajectories were flown in a different geometry "
                       f"({meta.get('spec_name','?')}, key "
                       f"{str(meta.get('geometry_key'))[:8]}) than the one "
                       f"loaded ({key[:8]}) — overlaying for comparison")
    return trajs, meta, warning


# ================================================================ REGISTRY
def _registry_path(directory=None):
    d = paths.fields_dir() if directory is None else directory
    return os.path.join(str(d), "fields_registry.json")


def _read_registry(directory=None):
    p = _registry_path(directory)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def get_label(key, directory=None):
    """Local display label for a geometry key, or None."""
    return _read_registry(directory).get(key)


def set_label(key, label, directory=None):
    """Relabel LOCALLY (cheap; never rewrites the npz).  Atomic write."""
    p = _registry_path(directory)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    reg = _read_registry(directory)
    reg[str(key)] = str(label)
    fd, tmp = tempfile.mkstemp(prefix=".fields_reg.", suffix=".json",
                               dir=os.path.dirname(p))
    ok = False
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(reg, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        ok = True
    finally:
        if not ok and os.path.exists(tmp):
            os.remove(tmp)
    return reg[str(key)]


def display_label(meta, directory=None):
    """Picker precedence: local registry > embedded _meta.label > filename
    stem (the caller supplies the filename fallback)."""
    return (get_label(meta.get("key") or meta.get("geometry_key"),
                      directory)
            or meta.get("label") or None)


def bake_label(path, label):
    """The ONE path that rewrites the npz: embed `label` into _meta so it
    travels with the file (explicit pre-send action).  Atomic."""
    with np.load(path) as z:
        members = {n: np.ascontiguousarray(z[n]) for n in z.files
                   if n != _META}
        meta = _member_to_meta(z[_META])
    meta["label"] = str(label)
    _atomic_savez(path, **members, **{_META: _meta_to_member(meta)})
    key = meta.get("key") or meta.get("geometry_key")
    if key:
        set_label(key, label)
    return path


def scan_fields(directory=None, spec=None):
    """Match-annotated picker feed.  Scans `directory` (default
    paths.fields_dir()) for field npzs; reads ONLY each file's _meta.
    Returns a list of dicts: path, label, spec_name, key, matches (bool or
    None if no spec given), problem (str for unreadable files — surfaced,
    never silently dropped)."""
    d = str(paths.fields_dir() if directory is None else directory)
    if not os.path.isdir(d):
        return []
    cur = (fa_cache.spec_key(basis_cache.geometry_key_dict(spec))
           if spec is not None else None)
    rows = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".npz") or name.endswith(".traj.npz"):
            continue
        p = os.path.join(d, name)
        try:
            meta = read_field_meta(p)
        except (FieldMismatch, OSError, ValueError, KeyError) as e:
            rows.append(dict(path=p, label=None, spec_name=None, key=None,
                             matches=None, problem=str(e)))
            continue
        rows.append(dict(
            path=p,
            label=display_label(meta, directory) or
                  os.path.splitext(name)[0],
            spec_name=meta.get("spec_name"),
            key=meta.get("key"),
            matches=(meta.get("key") == cur) if cur else None,
            problem=None))
    return rows


# --------------------------------------------------------------------------
# Cache-entry bundles: the array-level pair above
# (save_field/load_field) speaks the planar/rz dialect (b{k}/ele) and
# demands raw arrays that nothing downstream of build_run holds. These two
# operate one level up — on the DISK CACHE ENTRY, verbatim — so they are
# dialect-agnostic by construction (planar/rz b{k}+ele and 3-D
# basis_N+labels ride identically) and reachable from a plain solved
# solved deck. Export writes ONE npz (every entry file embedded) plus the
# spec JSON beside it; import validates the geometry key against the
# CURRENT spec (hard refusal on mismatch — a field must never be flown
# on a geometry it was not solved for), republishes the entry verbatim,
# and the next build_run is a cache HIT with no re-solve.
# --------------------------------------------------------------------------

def _entry_spec_json(key, root=None):
    """The stored spec_json of a cache entry (latest provenance record
    carrying one), or None."""
    import json as _json
    from pathlib import Path as _P
    from ion_gym.io import fa_cache as _pc
    pp = _P(root or _pc.DEFAULT_ROOT).expanduser() / key / "provenance.json"
    if not pp.exists():
        return None
    try:
        recs = _json.loads(pp.read_text())
    except (OSError, ValueError):
        return None
    for r in reversed(recs if isinstance(recs, list) else []):
        if isinstance(r, dict) and r.get("spec_json"):
            return r["spec_json"]
    return None


def resolve_cache_entry(spec, root=None):
    """Find the cache entry whose STORED spec geometry equals this
    spec's — route-agnostic (no key derivation, which is route-owned).
    Returns the key. Refuses when no entry matches (solve first) and
    when the entry predates descriptors (no spec_json to compare)."""
    from ion_gym.io import fa_cache as _pc
    want = spec.to_dict()["geometry"]
    hits, legacy = [], 0
    for e in _pc.entries(root=root or _pc.DEFAULT_ROOT):
        sj = _entry_spec_json(e["key"], root=root)
        if sj is None:
            legacy += 1
            continue
        if not _geometry_diff(want, sj.get("geometry", {})):
            hits.append(e["key"])
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"resolve_cache_entry: no cache entry matches the geometry of "
            f"{getattr(spec, 'name', '?')!r} ({legacy} legacy entries "
            f"without descriptors were skipped). Solve first (build_run); "
            f"legacy entries gain descriptors on their next solve.")
    raise ValueError(
        f"resolve_cache_entry: {len(hits)} entries share this geometry "
        f"({[h[:8] for h in hits]}) — export by key instead.")


def export_field_bundle(spec_or_key, out_dir, root=None):
    """Export a solved field from the disk cache: ONE npz with every
    entry file embedded verbatim (dialect-agnostic: planar/rz b{k}+ele
    and 3-D basis_N+labels ride identically) + the producing spec JSON
    beside it. Accepts a key from entries()/the Cache tab, or a spec
    (resolved by stored-geometry equality). Never solves."""
    import json as _json
    import numpy as _np
    from pathlib import Path as _P
    from ion_gym.io import fa_cache as _pc
    if isinstance(spec_or_key, str):
        key = spec_or_key
    else:
        key = resolve_cache_entry(spec_or_key, root=root)
    src = _P(root or _pc.DEFAULT_ROOT).expanduser() / key
    if not src.is_dir():
        raise FileNotFoundError(
            f"export_field_bundle: cache entry {key} does not exist.")
    sj = _entry_spec_json(key, root=root)
    name = ((sj or {}).get("name") or "field").replace(" ", "_")[:40]
    out = _P(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{key[:8]}"
    payload = {"__bundle_version": _np.array([1]),
               "__key": _np.frombuffer(key.encode(), _np.uint8)}
    for f in sorted(src.iterdir()):
        if f.suffix == ".npy":
            payload[f"arr__{f.stem}"] = _np.load(f)
        elif f.suffix == ".json":
            payload[f"json__{f.stem}"] = _np.frombuffer(
                f.read_bytes(), _np.uint8)
    npz = out / f"{stem}.field.npz"
    with open(npz, "wb") as fh:
        _np.savez_compressed(fh, **payload)
    spec_path = None
    if sj is not None:
        spec_path = out / f"{stem}.spec.json"
        spec_path.write_text(_json.dumps(sj, indent=1))
    else:
        print(f"[field export] NOTE: entry {key[:8]} predates descriptors "
              f"— no spec JSON to write beside it")
    print(f"[field export] {key[:8]} -> {npz.name}"
          + (f" + {spec_path.name}" if spec_path else ""))
    return str(npz), (str(spec_path) if spec_path else None)


def import_field_bundle(path, spec=None):
    """Reinstate an exported field bundle into the disk cache. If `spec`
    is given, HARD-REFUSES unless its geometry EQUALS the bundle's
    stored producing-spec geometry, naming every differing field — a
    field is never reinstated for geometry it was not solved for. On
    success the next build_run of the matching spec is a cache HIT."""
    import numpy as _np
    from pathlib import Path as _P
    from ion_gym.io import fa_cache as _pc
    z = _np.load(path, allow_pickle=False)
    if "__bundle_version" not in z:
        raise ValueError(
            f"import_field_bundle: {path} is not a field bundle (no "
            f"version tag) — array-level fields go through load_field.")
    key = bytes(z["__key"]).decode()
    if spec is not None:
        import json as _json
        sj = None
        for k in z.files:
            if k == "json__provenance":
                recs = _json.loads(bytes(z[k]).decode())
                for r in reversed(recs if isinstance(recs, list) else []):
                    if isinstance(r, dict) and r.get("spec_json"):
                        sj = r["spec_json"]; break
        if sj is None:
            raise ValueError(
                "import_field_bundle: a spec was given for validation but "
                "the bundle carries no producing-spec record (pre-"
                "descriptor entry) — import by trust is refused; pass "
                "spec=None only if you know the provenance.")
        diff = _geometry_diff(spec.to_dict()["geometry"],
                              sj.get("geometry", {}))
        if diff:
            raise ValueError(
                f"import_field_bundle: geometry mismatch — the bundle was "
                f"solved for different geometry. Differing fields: "
                f"{diff}. Refusing; load the spec JSON exported beside "
                f"the bundle, or re-solve.")
    dest = _P(_pc.DEFAULT_ROOT).expanduser() / key
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for k in z.files:
        if k.startswith("arr__"):
            _np.save(dest / f"{k[5:]}.npy", z[k]); n += 1
        elif k.startswith("json__"):
            (dest / f"{k[6:]}.json").write_bytes(bytes(z[k]))
    print(f"[field import] entry {key[:8]} reinstated ({n} arrays) — "
          f"next build of this geometry is a cache HIT")
    return key
