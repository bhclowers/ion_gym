"""
ion_gym.fa_cache
----------------
Persistence for solved node-centred field grids (build scope item 2).

Design:
  * Big arrays (phi, and the derived V/m fields Ez, Eu the kernel reads) are
    stored as raw .npy so they can be loaded ZERO-COPY via np.load(mmap_mode)
    and indexed directly by the Numba sampler — no deserialisation, shared
    page cache across processes.
  * Metadata (geometry, voltages, symmetry, grid spacing, solver conventions,
    array shapes/dtypes) lives in a JSON sidecar. The CACHE KEY is a hash of
    the canonicalised metadata, so any config change misses the cache
    automatically.
  * Writes are ATOMIC: arrays + sidecar are written to a temp dir and renamed
    into place only once all are complete, so a crashed/half-written entry is
    never visible. Load asserts shape/dtype/key match the sidecar — a
    truncated .npy (which memmaps as silent garbage) is caught, not flown.

Layout:
    <root>/<key>/
        meta.json      # spec + array manifest (shape, dtype, sha256 head)
        phi.npy        # solved potential (nz, nu)  -- for plotting
        Ez.npy, Eu.npy # V/m on the mirror-extended grid -- for the kernel
        Z.npy, Ue.npy  # axes (Ue is the extended transverse axis)

Nothing here imports numba; it just produces memmappable arrays. FieldNumba
gains a .from_cache()/.cached() path that consumes them.
"""

import hashlib
import json
import os
import shutil
import tempfile
import numpy as np

CACHE_VERSION = 2                       # bump to invalidate all entries
# v2: planar bases now solved by the DIRECT engine
# (sparse LU, exact) instead of multigrid (tol-approximate, residual
# ~7e-4 of V_BASIS). Banked mg bases must not be served next to
# direct solves — engine change is a result change.
DEFAULT_ROOT = os.environ.get("ION_GYM_CACHE",
                              os.path.expanduser("~/.ion_gym_cache"))


# ---------------------------------------------------------------- key / spec
def _canonical(obj):
    """Deterministic JSON for hashing: sorted keys, rounded floats so trivial
    fp noise doesn't fork the cache, arrays -> nested lists."""
    def norm(x):
        if isinstance(x, float):
            return round(x, 10)
        if isinstance(x, (np.floating,)):
            return round(float(x), 10)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return [norm(v) for v in x.tolist()]
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, dict):
            return {k: norm(x[k]) for k in sorted(x)}
        return x
    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"))


def spec_key(spec):
    """spec: any JSON-able dict fully describing the solve (geometry, voltages,
    h, symmetry, edge_ghost, solver version...). Returns a short hex key."""
    payload = _canonical({"v": CACHE_VERSION, "spec": spec})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- write / read
def _sha_head(a, n=4096):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)[:n].tobytes()).hexdigest()[:16]


def store(spec, arrays, root=DEFAULT_ROOT):
    """Atomically write {name: ndarray} under the spec's key. Returns the key."""
    key = spec_key(spec)
    dest = os.path.join(root, key)
    if os.path.isdir(dest) and os.path.exists(os.path.join(dest, "meta.json")):
        return key                                   # already cached
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{key}.", dir=root)
    ok = False
    try:
        manifest = {}
        for name, a in arrays.items():
            a = np.ascontiguousarray(a)
            np.save(os.path.join(tmp, name + ".npy"), a)
            manifest[name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                  sha_head=_sha_head(a))
        meta = dict(key=key, version=CACHE_VERSION, spec=spec, arrays=manifest)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        # atomic publish: rename the temp dir into place. os.replace onto an
        # existing dir fails, so only the first writer wins; a concurrent
        # writer's rename raises and we treat the existing entry as valid.
        try:
            os.replace(tmp, dest)
            ok = True
        except OSError:
            if os.path.exists(os.path.join(dest, "meta.json")):
                ok = True                            # someone else finished it
            else:
                raise
    finally:
        if not ok and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    return key


def load(spec_or_key, root=DEFAULT_ROOT, mmap=True, names=None):
    """Load a cache entry. Returns (arrays_dict, meta) or (None, None) on miss.
    arrays are memmapped read-only by default (zero-copy). Integrity: the
    sidecar's shape/dtype/sha_head must match each loaded array or it raises."""
    key = spec_or_key if isinstance(spec_or_key, str) and len(spec_or_key) == 16 \
        else spec_key(spec_or_key)
    dest = os.path.join(root, key)
    mpath = os.path.join(dest, "meta.json")
    if not os.path.exists(mpath):
        return None, None
    meta = json.load(open(mpath))
    if meta.get("key") != key or meta.get("version") != CACHE_VERSION:
        return None, None
    out = {}
    want = names or list(meta["arrays"])
    for name in want:
        p = os.path.join(dest, name + ".npy")
        if not os.path.exists(p):
            raise IOError(f"cache {key}: missing array {name}")
        try:
            a = np.load(p, mmap_mode="r" if mmap else None)
        except (ValueError, OSError, EOFError) as e:
            # numpy raises here if the .npy header promises more bytes than the
            # file holds (truncated write) -- normalise to the cache error type
            raise IOError(f"cache {key}: {name} unreadable ({e}) "
                          f"-- truncated or corrupt entry")
        m = meta["arrays"][name]
        if list(a.shape) != m["shape"] or str(a.dtype) != m["dtype"]:
            raise IOError(f"cache {key}: {name} shape/dtype mismatch "
                          f"(got {a.shape}/{a.dtype}, meta {m['shape']}/{m['dtype']}) "
                          f"-- truncated or corrupt entry")
        out[name] = a
    return out, meta


def verify(spec_or_key, root=DEFAULT_ROOT):
    """Full integrity check (reads arrays, recomputes head sha). Returns bool;
    a missing, truncated, or tampered entry returns False rather than raising."""
    try:
        arrs, meta = load(spec_or_key, root=root, mmap=False)
    except (IOError, OSError):
        return False
    if arrs is None:
        return False
    for name, a in arrs.items():
        if _sha_head(a) != meta["arrays"][name]["sha_head"]:
            return False
    return True


def annotate(spec_or_key, name, label=None, root=DEFAULT_ROOT,
             spec_json=None):
    """DESCRIPTIVE provenance: record which named spec
    produced or resolved a cache entry, in a `provenance.json` SIDECAR —
    deliberately not meta.json, which is the validation document and is
    never mutated after publish, and deliberately not the key: names are
    not geometry.  The mapping is many-to-one BY DESIGN (same geometry,
    different voltages or names share one entry), so records accumulate
    as a list; a repeat (name, label) refreshes its timestamp instead of
    duplicating.  Purely cosmetic: nothing reads it for correctness, a
    corrupt sidecar is reported and rebuilt, and an unwritable one warns
    without failing the store/load that triggered it."""
    import time
    key = (spec_or_key if isinstance(spec_or_key, str)
           else spec_key(spec_or_key))
    name = (name or "").strip()
    if not name:
        return                      # nothing to record — not an error
    dest = os.path.join(os.path.expanduser(root), key)
    if not os.path.isdir(dest):
        return                      # entry gone (cleared mid-flight) — moot
    ppath = os.path.join(dest, "provenance.json")
    recs = []
    if os.path.exists(ppath):
        try:
            with open(ppath) as f:
                recs = json.load(f)
            if not isinstance(recs, list):
                raise ValueError(f"expected a list, got {type(recs).__name__}")
        except (OSError, ValueError) as e:
            print(f"fa_cache: provenance sidecar unreadable for {key} "
                  f"({e}) — rebuilding it (descriptive data only).")
            recs = []
    now = time.strftime("%Y-%m-%d %H:%M")
    for r in recs:
        if r.get("name") == name and r.get("label") == label:
            r["when"] = now
            if spec_json is not None:
                r["spec_json"] = spec_json
            break
    else:
        rec = dict(name=name, label=label, when=now)
        if spec_json is not None:
            rec["spec_json"] = spec_json
        recs.append(rec)
    try:
        fd, tmp = tempfile.mkstemp(dir=dest, prefix=".prov.")
        with os.fdopen(fd, "w") as f:
            json.dump(recs, f, indent=1)
        os.replace(tmp, ppath)
    except OSError as e:
        print(f"fa_cache: could not write provenance for {key} ({e}) — "
              f"the entry works; only the Cache-tab 'produced by' column "
              f"is affected.")


def entries(root=DEFAULT_ROOT):
    """Inventory of the disk basis cache: a list of dicts, one per entry,
    with key, size in bytes, array count, mtime, and the entry's stored
    spec name/shape when available. Read-only — powers the Cache tab's
    per-item view so a user can see what is on disk and how big each item
    is before deciding what to remove. Never raises on a malformed entry;
    it is reported with what could be read."""
    root = os.path.expanduser(root)
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or name.startswith("."):
            continue
        size = 0
        n_arr = 0
        for dirpath, _dn, fn in os.walk(p):
            for f in fn:
                fp = os.path.join(dirpath, f)
                try:
                    size += os.path.getsize(fp)
                except OSError:
                    pass
                if f.endswith(".npy"):
                    n_arr += 1
        rec = dict(key=name, bytes=size, arrays=n_arr,
                   mtime=os.path.getmtime(p), spec_name=None, shape=None,
                   produced_by=[])
        ppath = os.path.join(p, "provenance.json")
        if os.path.exists(ppath):
            try:
                with open(ppath) as f:
                    precs = json.load(f)
                if isinstance(precs, list):
                    rec["produced_by"] = [
                        r.get("name") for r in precs
                        if isinstance(r, dict) and r.get("name")]
                    # descriptor + exportability:
                    # surface the store-time label and whether a full
                    # spec JSON rides in provenance (the Export path).
                    labels = [r.get("label") for r in precs
                              if isinstance(r, dict) and r.get("label")]
                    rec["descriptor"] = labels[-1] if labels else None
                    rec["has_spec_json"] = any(
                        isinstance(r, dict) and r.get("spec_json")
                        for r in precs)
            except (OSError, ValueError):
                pass                # descriptive only; row reported anyway
        meta_p = os.path.join(p, "meta.json")
        if os.path.exists(meta_p):
            try:
                with open(meta_p) as f:
                    meta = json.load(f)
                spec = meta.get("spec", {}) or {}
                rec["spec_name"] = (spec.get("name")
                                    if isinstance(spec, dict) else None)
                arrs = meta.get("arrays", {}) or {}
                if arrs:
                    any_a = next(iter(arrs.values()))
                    rec["shape"] = any_a.get("shape")
            except (OSError, ValueError, KeyError):
                pass                                 # report the rest anyway
        out.append(rec)
    return out


def remove(key, root=DEFAULT_ROOT):
    """Remove ONE cache entry by key. Returns (removed_bool, bytes_freed).
    Selective counterpart to clear_all for the Cache tab. A missing key is
    reported (removed=False), not an error."""
    import shutil
    root = os.path.expanduser(root)
    p = os.path.join(root, key)
    if not os.path.isdir(p):
        return False, 0
    freed = 0
    for dirpath, _dn, fn in os.walk(p):
        for f in fn:
            try:
                freed += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    shutil.rmtree(p, ignore_errors=True)
    return True, freed


def clear_all(root=DEFAULT_ROOT):
    """Remove every cached basis set under `root`. Returns (n_entries,
    bytes_freed). Used by the app's 'clear cache' control."""
    import shutil
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return 0, 0
    n = bytes_freed = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            for dirpath, _dn, fn in os.walk(p):
                for f in fn:
                    try:
                        bytes_freed += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
            shutil.rmtree(p, ignore_errors=True)
            n += 1
        except OSError:
            pass
    return n, bytes_freed
