"""
ion_gym.io.field_io — portable solved fields (SPEC_field_portability.md)
========================================================================
Three separately-loadable, plainly-named user artifacts, written and read
by the SAME functions in the dashboard and in notebooks:

  * geometry  — the spec JSON (spec_io, already exists);
  * FIELD     — the deck's solve-cache entry exported verbatim as one
                <deck>__<key8>.field.npz, with the deck's JSON beside it,
                on every route (planar, r-z, 3-D): save_field, load_field,
                read_spec_from_field, scan_fields (this module);
  * TRAJECTORIES — an .npz of decimated ion paths + `_meta` frame (this
                module).  Overlay-friendly: geometry mismatch WARNS, never
                refuses.

Identity contract: a field belongs to a deck iff its cache key equals
field_key(deck) -- the geometry, the STL contents and the solver code,
the same discrimination the disk cache uses -- so a filename is purely
cosmetic; users may rename files freely. Loading REFUSES on any
mismatch. A field applied to the wrong geometry is invisible
catastrophic physics; there is no warn-and-proceed path for fields.

Cross-machine transfer: the efficient unit is the tiny
geometry JSON + a FINGERPRINT (this module), not the megabytes of bases —
bases are a deterministic function of geometry.  Receiver re-solves and
ANCHORS its own solve to the sender's fingerprint (probe potentials agree
to tolerance) or surfaces the divergence loudly.  The full-bases .npz is
the fallback for receivers who cannot re-solve.

Fingerprint probes index the basis ARRAYS at deterministic grid nodes —
exact (no interpolation) and layering-clean (the field samplers live under
projects/, which core io must not import).

Labels: display labels are LOCAL (fields_registry.json in the fields
folder, keyed by the field's key); without one a field is listed by its
deck's name. Relabeling edits the registry only — it never rewrites a
multi-hundred-MB npz.
"""

from __future__ import annotations

# The public names (docs/PUBLIC_API.md); everything else is internal.
__all__ = [
    "save_field", "load_field", "read_field_meta", "read_field_arrays",
    "read_spec_from_field",
    "scan_fields", "field_key", "is_solved", "is_field_bundle",
    "FieldMismatch", "FieldNotSolved", "get_label", "set_label",
    "export_field_bundle", "import_field_bundle",
    "fingerprint", "verify_fingerprint", "save_transfer",
    "save_trajectories", "trajectories_npz", "load_trajectories",
    "read_trajectory_meta", "TrajectoryFormatError", "TRAJ_DTYPES",
]

import io
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
TRAJ_FORMAT = 2         # 2: precision option, mass sources, time<i>
FINGERPRINT_FORMAT = 1

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


def _atomic_write_text(path, text):
    """Write a UTF-8 text file atomically, as _atomic_savez does."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".field_io.", suffix=".tmp", dir=d)
    ok = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
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
# ONE FORMAT ON EVERY ROUTE. A saved field is the deck's solve-cache entry
# exported verbatim (export_field_bundle: <deck>__<key8>.field.npz, every
# array and JSON record of the entry inside), with the deck's JSON beside
# it. Planar, r-z and 3-D fields are all saved, listed, matched and
# loaded the same way, and the dashboard's Save solved field / Load field
# / Open saved field call exactly these functions, as notebooks do.
#
# IDENTITY: a field belongs to a deck iff its cache key equals the deck's
# (field_key): the geometry, the STL contents and the solver code. Loading
# HARD-REFUSES anything else -- a field from an edited mesh or from other
# solve code is never served. The older array-level field files (a single
# npz with a `_meta` member) are refused by name: solve and save again.

FIELD_SUFFIX = ".field.npz"


class FieldNotSolved(LookupError):
    """The deck has no solved field in the cache to save: solve it first."""


def field_key(spec):
    """The cache key (hex) of `spec`'s solved field, on any route: what a
    saved field is identified and matched by."""
    from ion_gym.physics.sim_build import solved_field_key
    return solved_field_key(spec)


def is_solved(spec, root=None):
    """True when `spec`'s solved field is in the cache (a corrupt entry
    raises OSError rather than reading as unsolved)."""
    return fa_cache.load(field_key(spec), root=root)[0] is not None


def save_field(spec, directory=None, label=None, root=None):
    """Save `spec`'s solved field from the cache (never a re-solve) into
    `directory` (default paths.fields_dir()): the field file and the deck
    JSON beside it, both written atomically. `label` is kept in that
    folder's label registry. Returns (field_path, deck_path). Raises
    FieldNotSolved when the deck has not been solved."""
    key = field_key(spec)
    if fa_cache.load(key, root=root)[0] is None:
        raise FieldNotSolved(
            f"{getattr(spec, 'name', '') or 'this deck'}: no solved field in "
            f"the cache (entry {key[:8]}). Solve it first, then save.")
    out = paths.fields_dir() if directory is None else directory
    npz, deck = export_field_bundle(key, out, root=root)
    if label:
        set_label(key, label, directory=out)
    return npz, deck


def is_field_bundle(path):
    """True for a field file in the current format; False for anything
    else (an older array-level field file, another npz)."""
    with np.load(path) as z:
        return "__bundle_version" in z.files


def read_field_meta(path):
    """{key, spec, spec_name} of a saved field, reading only its small
    members. `spec` is the producing deck as a dict (None for an entry
    solved before decks were recorded). Refuses (FieldMismatch) a file in
    the older array-level format, or one that is not a field."""
    name = os.path.basename(path)
    with np.load(path) as z:
        if "__bundle_version" not in z.files:
            if _META in z.files:
                raise FieldMismatch(
                    f"{name} is in the older field format, which this "
                    f"version no longer reads. Solve the deck and save the "
                    f"field again.")
            raise FieldMismatch(f"{name} is not an ion_gym field file "
                                f"(members: {sorted(z.files)[:6]})")
        key = bytes(z["__key"]).decode()
        sj = None
        if "json__provenance" in z.files:
            recs = json.loads(bytes(z["json__provenance"]).decode())
            for r in reversed(recs if isinstance(recs, list) else []):
                if isinstance(r, dict) and r.get("spec_json"):
                    sj = r["spec_json"]
                    break
    return {"key": key, "spec": sj, "spec_name": (sj or {}).get("name")}


def load_field(path, spec, root=None):
    """Reinstate a saved field for `spec`, so its next build is a cache
    hit with no re-solve. HARD-REFUSES (FieldMismatch) unless the file's
    key equals field_key(spec). Returns the key."""
    meta = read_field_meta(path)
    want = field_key(spec)
    if meta["key"] != want:
        raise FieldMismatch(
            f"{os.path.basename(path)} was solved for a different geometry, "
            f"STL contents or solver code (cache entry {meta['key'][:8]}, "
            f"deck {meta['spec_name']!r}) than the loaded deck (entry "
            f"{want[:8]}). Open it with its own deck (Open saved field / "
            f"read_spec_from_field), or re-solve.")
    # The key equality IS the provenance check, and the strict one:
    # import_field_bundle's own comparison of recorded geometry blocks
    # refuses the same solve when two decks merely spell a default
    # differently.
    return import_field_bundle(path, spec=None, root=root)


def read_field_arrays(path):
    """The solved field in a saved file, the same way on every route:
    (bases, labels, meta). `bases` maps basis number (1, 2, ...; an
    electrode's `basis`, else its position) to its potential PER VOLT on
    the grid, as float64 on every route (the planar and r-z caches keep
    each basis as solved, with its electrode at V_BASIS = 1e4 V, which
    this divides out; the 3-D caches keep it per volt). The grid pitch is
    the deck's
    geometry.mm_per_gu. `labels` gives each node's basis number (0 =
    vacuum), and `meta` is read_field_meta's. The potential at voltages
    V_k is sum_k V_k * bases[k]."""
    from ion_gym.physics.build_planar import V_BASIS
    meta = read_field_meta(path)
    bases, labels = {}, None
    with np.load(path) as z:
        for name in z.files:
            if not name.startswith("arr__"):
                continue
            arr = name[5:]
            if arr in ("ele", "labels"):          # 2-D / 3-D entry names
                labels = z[name]
            # the 3-D routes cache basis_<k> already per volt (float32);
            # planar and r-z cache b<k> as solved, at V_BASIS
            elif arr.startswith("basis_") and arr[6:].isdigit():
                bases[int(arr[6:])] = np.asarray(z[name], np.float64)
            elif arr.startswith("b") and arr[1:].isdigit():
                bases[int(arr[1:])] = z[name] / V_BASIS
    if not bases or labels is None:
        raise FieldMismatch(f"{os.path.basename(path)}: no bases or "
                            f"labels in the file")
    return dict(sorted(bases.items())), labels, meta


def read_spec_from_field(path):
    """The deck a saved field was solved for, as a SimSpec, with the
    file's meta: (spec, meta). The bootstrap direction -- work can start
    from the field file alone. Goes through load_any_spec, the one door
    every deck enters by."""
    from ion_gym.io.spec_io import load_any_spec
    meta = read_field_meta(path)
    if not meta["spec"]:
        raise FieldMismatch(
            f"{os.path.basename(path)} does not record the deck it was "
            f"solved for. Load the deck first, then load the field.")
    return load_any_spec(json.dumps(meta["spec"])), meta


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
                field_format=FINGERPRINT_FORMAT,
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
# ONE FORMAT for every trajectory file. The dashboard's Save trajectories
# and Download NPZ write it (save_trajectories / trajectories_npz, the
# same members), Load as run reads it (load_trajectories), and notebooks
# call the same functions. Members:
#   t<i>          one array per saved ion: rows = recorded steps, columns
#                 named by `columns`;
#   time<i>       float32 files only: that ion's time column at float64;
#   columns, label, mz_da, mz_source, summary, summary_cols
#                 plain arrays (a numpy-only reader needs no JSON);
#   _meta         the JSON frame: columns, units, ion_index, summaries,
#                 masses, precision, provenance (deck as flown, seed,
#                 code versions).
# PRECISION: "float64" (default) is exactly what was flown; "float32" is
# about half the size with ~7 significant digits. A float32 file keeps the
# time column at float64 too: late in a long flight float32 cannot resolve
# a 1 ns step, and timing analysis works on time differences.

TRAJ_DTYPES = ("float64", "float32")
_TRAJ_READABLE = (1, TRAJ_FORMAT)    # format 1 = the same layout, float64
_SUMMARY_COLS = ["ion", "fate", "tof", "x_end", "y_end", "z_end", "mz_da"]


def _source_name(source):
    return (os.path.basename(source) if isinstance(source, (str, os.PathLike))
            else getattr(source, "name", "the uploaded file"))


def _mass_and_source(spec, index, summary):
    """An ion's mass (Da) and where it came from: the run's own summary
    first, then the deck; NaN with the reason when neither says."""
    from ion_gym.physics.ion_envelope import summary_value
    from ion_gym.physics.sim_build import mz_of
    mz = summary_value(summary, "mz")
    if mz is not None:
        return float(mz), "summary"
    try:
        return float(mz_of(spec, int(index))), "spec"
    except (ValueError, TypeError, IndexError, AttributeError) as e:
        return float("nan"), f"unknown ({type(e).__name__})"


def _trajectory_members(results, spec, label, provenance, dtype, columns):
    """Every member of a trajectory file (see the section note)."""
    from ion_gym.io.run_provenance import run_provenance
    from ion_gym.physics.ion_envelope import summary_fate, summary_value
    if dtype not in TRAJ_DTYPES:
        raise ValueError(f"trajectory precision {dtype!r}: choose one of "
                         f"{TRAJ_DTYPES}")
    if provenance is None:
        provenance = run_provenance(spec)
    cols = list(columns) if columns is not None else (
        spec.column_names() if hasattr(spec, "column_names") else None)
    tcol = cols.index("t") if cols and "t" in cols else None
    if dtype == "float32" and tcol is None:
        raise ValueError("a float32 trajectory file keeps its time column "
                         "at float64, but these columns name no 't'")
    gd = basis_cache.geometry_key_dict(spec)
    # one pass over a list: a generator would be spent by the loop and the
    # skipped count taken from it afterwards would be wrong
    results = list(results)
    members, index, summaries, masses, sources = {}, [], [], [], []
    for r in results:
        # tuples first: a tuple HAS an .index (the sequence method), so the
        # attribute test would read every tuple as an IonResult without a
        # trajectory and skip it
        if isinstance(r, (tuple, list)):
            i, traj = r[0], r[1]
            summ = r[2] if len(r) > 2 else {}
        else:
            i = r.index
            traj = getattr(r, "traj", None)
            summ = getattr(r, "summary", None)
        if traj is None:
            continue                      # never flown/not recorded: skip,
        i, summ = int(i), summ or {}      # and say so in _meta.n_skipped
        traj = np.asarray(traj)
        if cols is not None and (traj.ndim != 2
                                 or traj.shape[1] != len(cols)):
            raise ValueError(f"ion {i}: trajectory shape {traj.shape} does "
                             f"not match the {len(cols)} columns {cols}")
        members[f"t{i}"] = np.ascontiguousarray(traj, dtype)
        if dtype == "float32":
            members[f"time{i}"] = np.ascontiguousarray(traj[:, tcol],
                                                       np.float64)
        mz, src = _mass_and_source(spec, i, summ)
        index.append(i)
        summaries.append(summ)
        masses.append(mz)
        sources.append(src)
    spec_name = getattr(spec, "name", "") or "simulation"
    meta = dict(kind="ion_gym_trajectories",
                traj_format=TRAJ_FORMAT,
                precision=dtype,
                columns=cols,
                frame="Convention A: axial coordinate horizontal (z axial)",
                units=dict(position="mm", time="us", velocity="mm/us"),
                geometry_key=fa_cache.spec_key(gd),
                spec_name=spec_name,
                label=label or spec_name,
                ion_index=index,
                summaries=summaries,
                # mass per SAVED ion: absolute KE, momenta and cross-file
                # comparisons need it; "unknown" is a recorded value
                mz_da=["unknown" if np.isnan(m) else m for m in masses],
                mz_source=sources,
                n_skipped=len(results) - len(index),
                provenance=provenance)
    plain = {"label": np.array([str(meta["label"])]),
             "mz_da": np.array(masses, float),
             "mz_source": np.array(sources),
             "summary_cols": np.array(_SUMMARY_COLS)}
    if cols is not None:
        plain["columns"] = np.array([str(c) for c in cols])

    def _num(v):
        return np.nan if v is None else float(v)
    rows = [[float(i), float(summary_fate(s)),
             _num(summary_value(s, "tof")), _num(summary_value(s, "x_end")),
             _num(summary_value(s, "y_end")), _num(summary_value(s, "z_end")),
             m]
            for i, s, m in zip(index, summaries, masses)]
    plain["summary"] = (np.array(rows, float) if rows
                        else np.zeros((0, len(_SUMMARY_COLS))))
    return {**members, **plain, _META: _meta_to_member(meta)}, meta


def trajectories_npz(results, spec, label=None, provenance=None,
                     dtype="float64", columns=None):
    """The trajectory file as bytes (for a download): exactly what
    save_trajectories writes to disk."""
    members, _ = _trajectory_members(results, spec, label, provenance,
                                     dtype, columns)
    buf = io.BytesIO()
    np.savez_compressed(buf, **members)
    return buf.getvalue()


def save_trajectories(results, spec, path=None, label=None, provenance=None,
                      dtype="float64", columns=None):
    """Save decimated ion paths (IonResult list or (index, traj, summary)
    tuples) as a trajectory file, atomically; returns its path (default
    paths.trajectories_dir()/<deck>__<key8>.traj.npz).

    dtype: "float64" (default, exactly what was flown) or "float32" (half
    the size; the time column is still kept at float64).
    columns: the trajectory column names; default the deck's
    (spec.column_names()). Pass the run's own when it recorded others.
    provenance: the run's io.run_provenance record (deck as flown, seed,
    code versions). Pass the one captured when the run was built; when
    None it is taken from `spec` now, which is right only if `spec` is the
    spec that just flew."""
    members, meta = _trajectory_members(results, spec, label, provenance,
                                        dtype, columns)
    if path is None:
        path = str(paths.trajectories_dir() /
                   f"{_slug(meta['spec_name'])}__"
                   f"{meta['geometry_key'][:8]}.traj.npz")
    _atomic_savez(path, **members)
    return path


def read_trajectory_meta(source):
    """The `_meta` frame of a trajectory file (a path or an open file).
    Refuses, naming the remedy, a Download NPZ from an older version
    (which had no frame) and anything that is not a trajectory file."""
    name = _source_name(source)
    with np.load(source, allow_pickle=False) as z:
        if _META not in z.files:
            if any(re.fullmatch(r"ion_\d+", k) for k in z.files):
                raise TrajectoryFormatError(
                    f"{name} is a Download NPZ from an older version, which "
                    f"this version no longer reads. Download the run again "
                    f"(or use its CSV).")
            raise TrajectoryFormatError(
                f"{name}: no embedded _meta — not an ion_gym trajectory npz")
        meta = _member_to_meta(z[_META])
    if meta.get("kind") != "ion_gym_trajectories":
        raise TrajectoryFormatError(f"_meta.kind {meta.get('kind')!r} != "
                                    f"'ion_gym_trajectories'")
    return meta


def load_trajectories(source, spec=None):
    """Read a trajectory file (a path, or an open file such as an upload).
    Returns (trajs, meta, warning): trajs = {ion_index: (n, k) float64
    array}, with a float32 file's time column restored from its float64
    copy. warning is a string if `spec` is given and the geometry keys
    differ (overlaying is legitimate — the caller SURFACES it, never hides
    it), else None."""
    meta = read_trajectory_meta(source)
    if meta.get("traj_format") not in _TRAJ_READABLE:
        raise TrajectoryFormatError(
            f"traj_format {meta.get('traj_format')} is not one this version "
            f"reads ({list(_TRAJ_READABLE)})")
    cols = meta.get("columns") or []
    tcol = cols.index("t") if "t" in cols else None
    if hasattr(source, "seek"):
        source.seek(0)
    trajs = {}
    with np.load(source, allow_pickle=False) as z:
        for i in meta.get("ion_index", []):
            name = f"t{int(i)}"
            if name not in z.files:
                raise TrajectoryFormatError(f"{name} listed in _meta but "
                                            f"missing from file")
            arr = np.array(z[name], dtype=np.float64)
            if f"time{int(i)}" in z.files and tcol is not None:
                arr[:, tcol] = z[f"time{int(i)}"]
            trajs[int(i)] = arr
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
    with open(p, encoding="utf-8") as f:
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


def scan_fields(directory=None, spec=None):
    """The saved-fields list: every field file in `directory` (default
    paths.fields_dir()), reading only each file's small members. Returns
    dicts: path, label, spec_name, key, matches (True when the file's key
    is the deck's field_key; None when no spec is given), problem (why a
    file cannot be loaded -- an older format, an unreadable file --
    listed, never silently dropped). Temporary files of a save in
    progress (atomic writes' dot-prefixed names) are skipped."""
    d = str(paths.fields_dir() if directory is None else directory)
    if not os.path.isdir(d):
        return []
    cur = field_key(spec) if spec is not None else None
    rows = []
    for name in sorted(os.listdir(d)):
        if (not name.endswith(".npz") or name.endswith(".traj.npz")
                or name.startswith(".")):
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
            label=(get_label(meta["key"], directory) or meta["spec_name"]
                   or os.path.splitext(name)[0]),
            spec_name=meta["spec_name"], key=meta["key"],
            matches=(meta["key"] == cur) if cur is not None else None,
            problem=None))
    return rows


# --------------------------------------------------------------------------
# Cache entries as files: export_field_bundle writes a solve-cache entry
# verbatim (every array and JSON record) into ONE npz plus the producing
# deck's JSON beside it; import_field_bundle republishes it atomically.
# save_field and load_field above are these two, with the deck's key
# checked. Entries of every route have the same layout, so one pair
# serves planar, r-z and 3-D.
# --------------------------------------------------------------------------

def _entry_spec_json(key, root=None):
    """The stored spec_json of a cache entry (latest provenance record
    carrying one), or None when the entry has no provenance record (a
    legacy, pre-descriptor entry). A provenance file that EXISTS but
    cannot be read is REFUSED by name: it used to return None and
    be counted as legacy, so a corrupt record looked like an old one."""
    import json as _json
    from pathlib import Path as _P
    pp = _P(root or paths.cache_root()).expanduser() / key / "provenance.json"
    if not pp.exists():
        return None
    try:
        recs = _json.loads(pp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(
            f"cache entry {key}: provenance record {pp} is unreadable "
            f"({type(e).__name__}: {e}) -- the entry's producing spec is "
            f"unknown; remove the entry (Cache tab) and re-solve") from e
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
    for e in _pc.entries(root=root):
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
    if isinstance(spec_or_key, str):
        key = spec_or_key
    else:
        key = resolve_cache_entry(spec_or_key, root=root)
    src = _P(root or paths.cache_root()).expanduser() / key
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
    spec_path = None
    # Both files are written atomically (temporary name, then rename), the
    # deck first: an interrupted export leaves no partial file, and a
    # field file never appears without the deck beside it.
    if sj is not None:
        spec_path = out / f"{stem}.spec.json"
        _atomic_write_text(spec_path, _json.dumps(sj, indent=1))
    _atomic_savez(str(npz), **payload)
    if sj is None:
        print(f"[field export] NOTE: entry {key[:8]} predates descriptors "
              f"— no spec JSON to write beside it")
    print(f"[field export] {key[:8]} -> {npz.name}"
          + (f" + {spec_path.name}" if spec_path else ""))
    return str(npz), (str(spec_path) if spec_path else None)


def import_field_bundle(path, spec=None, root=None):
    """Reinstate an exported field bundle into the disk cache under `root`
    (default the cache root). If `spec` is given, HARD-REFUSES unless its
    geometry EQUALS the bundle's stored producing-spec geometry, naming
    every differing field — a field is never reinstated for geometry it
    was not solved for. On success the next build_run of the matching
    spec is a cache HIT.

    ATOMIC: the entry is staged in a temporary directory under
    the root, meta.json written LAST, then published by one rename -- as
    fa_cache.store does. It used to write in place (meta.json could land
    before the arrays) under the default root whatever `root` said, and a
    crash left a partial entry the cache then served as complete. A
    complete existing entry is kept; a partial leftover is replaced."""
    import numpy as _np
    from pathlib import Path as _P
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
    import os as _os
    import tempfile as _tf
    from ion_gym.io.fa_cache import remove_tree, CachePurgeIncomplete
    if "json__meta" not in z.files:
        raise ValueError(
            f"import_field_bundle: {path} carries no meta.json -- the "
            f"entry's completeness marker; refusing a partial bundle")
    base = _P(root or paths.cache_root()).expanduser()
    dest = base / key
    if (dest / "meta.json").exists():
        print(f"[field import] entry {key[:8]} is already in the cache at "
              f"{base}; kept as is")
        return key
    base.mkdir(parents=True, exist_ok=True)
    tmp = _P(_tf.mkdtemp(prefix=f".{key}.import.", dir=base))
    try:
        n = 0
        for k in z.files:
            if k.startswith("arr__"):
                _np.save(tmp / f"{k[5:]}.npy", z[k]); n += 1
        for k in z.files:
            if k.startswith("json__") and k != "json__meta":
                (tmp / f"{k[6:]}.json").write_bytes(bytes(z[k]))
        with open(tmp / "meta.json", "wb") as fh:    # LAST: completeness
            fh.write(bytes(z["json__meta"]))
            fh.flush()
            _os.fsync(fh.fileno())
        if dest.exists():
            # a directory without meta.json is a crash-partial leftover
            fail = remove_tree(dest)
            if fail:
                raise CachePurgeIncomplete(0, 0, fail)
            print(f"[field import] replaced a PARTIAL entry {key[:8]} "
                  f"(no meta.json)")
        _os.replace(tmp, dest)
    finally:
        if tmp.exists():
            remove_tree(tmp)
    print(f"[field import] entry {key[:8]} reinstated ({n} arrays) under "
          f"{base} — next build of this geometry is a cache HIT")
    return key
