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

from ion_gym.io.paths import cache_root

CACHE_VERSION = 2                       # bump to invalidate all entries
# v2: planar bases now solved by the DIRECT engine
# (sparse LU, exact) instead of multigrid (tol-approximate, residual
# ~7e-4 of V_BASIS). Banked mg bases must not be served next to
# direct solves — engine change is a result change.


def _root(root):
    """The cache root a call uses: the caller's, else paths.cache_root()
    resolved NOW (never frozen at import -- L-551)."""
    return str(cache_root()) if root is None else root


def __getattr__(name):
    # DEFAULT_ROOT is kept as a public name (notebooks 01-05 and V01-V06
    # and internal tools read it) but resolved on every access, so it
    # always agrees with what store/load use.
    if name == "DEFAULT_ROOT":
        return str(cache_root())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _entry_complete(dest):
    """True when every array meta.json lists is on disk. Unreadable
    meta.json counts as incomplete (the entry cannot be served)."""
    try:
        with open(os.path.join(dest, "meta.json")) as f:
            names = list((json.load(f).get("arrays") or {}))
    except (OSError, ValueError, AttributeError):
        return False
    return all(os.path.exists(os.path.join(dest, n + ".npy"))
               for n in names)


def store(spec, arrays, root=None):
    """Atomically write {name: ndarray} under the spec's key. Returns the key."""
    root = _root(root)
    key = spec_key(spec)
    dest = os.path.join(root, key)
    if os.path.isdir(dest) and os.path.exists(os.path.join(dest, "meta.json")):
        if _entry_complete(dest):
            return key                               # already cached
        # meta.json present but an array it lists is not: a partial entry
        # (e.g. a pre-fix in-place import that crashed). Served as complete
        # it would fail every load; replaced by this solve instead (L-537)
        fail = remove_tree(dest)
        if fail:
            raise CachePurgeIncomplete(0, 0, fail)
        print(f"fa_cache: entry {key} was INCOMPLETE (arrays listed in "
              f"meta.json missing); replacing it with this solve")
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


def load(spec_or_key, root=None, mmap=True, names=None):
    """Load a cache entry. Returns (arrays_dict, meta) or (None, None) on miss.
    arrays are memmapped read-only by default (zero-copy). Integrity: the
    sidecar's shape/dtype/sha_head must match each loaded array or it raises."""
    root = _root(root)
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


def verify(spec_or_key, root=None):
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


def annotate(spec_or_key, name, label=None, root=None,
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
    dest = os.path.join(os.path.expanduser(_root(root)), key)
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


def entries(root=None):
    """Inventory of the disk basis cache: a list of dicts, one per entry,
    with key, size in bytes, array count, mtime, and the entry's stored
    spec name/shape when available. Read-only — powers the Cache tab's
    per-item view so a user can see what is on disk and how big each item
    is before deciding what to remove. Never raises on a malformed entry;
    it is reported with what could be read."""
    root = os.path.expanduser(_root(root))
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or name.startswith("."):
            continue
        size, unsized = _tree_bytes(p)
        n_arr = sum(f.endswith(".npy") for _dp, _dn, fn in os.walk(p)
                    for f in fn)
        # unsized_files > 0 means `bytes` is a LOWER BOUND (L-543: an
        # unreadable file used to vanish from the count silently)
        rec = dict(key=name, bytes=size, unsized_files=unsized,
                   arrays=n_arr, mtime=os.path.getmtime(p), spec_name=None,
                   shape=None, produced_by=[])
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


class CachePurgeIncomplete(OSError):
    """A purge that could not delete everything (L-543). Raised AFTER
    every deletable entry is gone, carrying what WAS removed and each path
    that was not, so the caller reports a partial purge as partial."""

    def __init__(self, n_removed, bytes_freed, failures):
        self.n_removed = n_removed
        self.bytes_freed = bytes_freed
        self.failures = dict(failures)        # {path: reason}
        shown = "; ".join(f"{p} ({w})" for p, w in
                          list(self.failures.items())[:5])
        more = (f" (+{len(self.failures) - 5} more)"
                if len(self.failures) > 5 else "")
        super().__init__(
            f"cache purge incomplete: {n_removed} entries removed "
            f"({bytes_freed / 1e6:.1f} MB); could not delete "
            f"{len(self.failures)} path(s): {shown}{more}")


def remove_tree(path):
    """Delete a directory tree; return {path: reason} for every file or
    directory that could NOT be deleted ({} = fully gone; a missing tree
    is already gone). The reporting counterpart of
    shutil.rmtree(ignore_errors=True), which made failed deletes look
    like successes (L-543)."""
    path = str(path)
    failures = {}
    if not os.path.lexists(path):
        return failures
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                os.remove(fp)
            except OSError as e:
                failures[fp] = f"{type(e).__name__}: {e.strerror or e}"
        for d in dirnames:
            dp = os.path.join(dirpath, d)
            if os.path.islink(dp):
                try:
                    os.remove(dp)
                except OSError as e:
                    failures[dp] = f"{type(e).__name__}: {e.strerror or e}"
            elif os.path.isdir(dp) and dp not in failures:
                try:
                    os.rmdir(dp)
                except OSError as e:
                    if not any(k.startswith(dp + os.sep) for k in failures):
                        failures[dp] = (f"{type(e).__name__}: "
                                        f"{e.strerror or e}")
    try:
        os.rmdir(path)
    except OSError as e:
        if not any(k.startswith(path + os.sep) for k in failures):
            failures[path] = f"{type(e).__name__}: {e.strerror or e}"
    return failures


def _tree_bytes(path):
    """(bytes, n_unsized): total size of the files under `path`, and how
    many could not be sized -- reported, never silently dropped."""
    size = unsized = 0
    for dirpath, _dn, fn in os.walk(path):
        for f in fn:
            try:
                size += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                unsized += 1
    return size, unsized


def remove(key, root=None):
    """Remove ONE cache entry by key. Returns (removed_bool, bytes_freed).
    Selective counterpart to clear_all for the Cache tab. A missing key is
    reported (removed=False), not an error; an entry that cannot be fully
    deleted RAISES CachePurgeIncomplete naming each path (it used to
    report success after rmtree(ignore_errors=True))."""
    root = os.path.expanduser(_root(root))
    p = os.path.join(root, key)
    if not os.path.isdir(p):
        return False, 0
    freed, unsized = _tree_bytes(p)
    failures = remove_tree(p)
    if failures:
        raise CachePurgeIncomplete(0, 0, failures)
    if unsized:
        print(f"fa_cache: {unsized} file(s) in {key} could not be sized; "
              f"{freed / 1e6:.1f} MB freed is a lower bound")
    return True, freed


def clear_all(root=None):
    """Remove every cached basis set under `root`. Returns (n_entries,
    bytes_freed), counting ONLY entries actually deleted. If any entry
    could not be fully deleted, every other entry is still removed and
    then CachePurgeIncomplete (an OSError) is raised with the counts and
    each failed path -- a failed delete is never counted as a purge
    (L-543). Used by the app's 'clear cache' control."""
    root = os.path.expanduser(_root(root))
    if not os.path.isdir(root):
        return 0, 0
    n = bytes_freed = unsized = 0
    failures = {}
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or os.path.islink(p):
            # an ENTRY is a directory; loose files at the root are not
            # cache entries and are kept (e.g. sizing's per-machine
            # solve_rates.json) -- the old rmtree failed on them silently
            continue
        size, n_uns = _tree_bytes(p)
        fail = remove_tree(p)
        if fail:
            failures.update(fail)
            continue
        n += 1
        bytes_freed += size
        unsized += n_uns
    if failures:
        raise CachePurgeIncomplete(n, bytes_freed, failures)
    if unsized:
        print(f"fa_cache: {unsized} file(s) could not be sized; "
              f"{bytes_freed / 1e6:.1f} MB freed is a lower bound")
    return n, bytes_freed
