"""
ion_gym.basis_cache  --  disk field-array cache (gate M-0 scope item)
=========================================================
Solved per-electrode BASIS potentials, persisted to disk, keyed on the
GEOMETRY ALONE.

Why geometry alone: Laplace is linear in the electrode voltages, so a
voltage change is a re-weight of the same bases (the fast-adjust
invariant).  Keying on the whole spec would re-solve every time a deck
voltage moved by a millivolt; keying on geometry means the 40-60 s solve
happens ONCE per geometry, ever, on this machine -- which is what makes a
multi-FA bench (N solves) affordable, and what makes a fresh process
reproduce a certified run without re-solving.

AGNOSTIC BY CONSTRUCTION: the key is a hash of the canonicalised
GeometrySpec dict with the voltage/appearance fields (dc, rf_*, color,
name) stripped, the DC-ladder placement (dc_group/dc_index/dc_weight)
held at None, and nothing else touched.  It knows no instrument, no plate
count, no frame constant.  Any change that can move a boundary node --
width/height/depth, mm_per_gu, symmetry declaration, a shape parameter,
an STL reference, the basis ordering -- changes the key and misses the
cache.  Any change that cannot -- a voltage, a colour, a name -- does not.
`test_bench_M_PA0.py` gate B4 asserts exactly that discrimination.

Storage is fa_cache (atomic write, memmapped read, sha-head integrity), so
a half-written or truncated entry is refused rather than flown.
"""

from __future__ import annotations

import numpy as np

from ion_gym.io import fa_cache

CACHE_KIND = "planar_bases_v1"

# Fields on an ElectrodeSpec that CANNOT change the solved bases.
_VOLTAGE_FIELDS = ("dc", "rf_groups", "color", "name")
# DC-ladder placement: these only place a member's
# tap -- resolve_dc_groups writes dc = v_in + (v_out - v_in) * w, a
# voltage -- and no builder reads them. Keying their VALUES made a ladder
# re-weight re-solve every basis (and set_flight_conditions refuse it).
# They are held at None in the key rather than removed: to_dict emits all
# three on every electrode (None when unset), so removing them would have
# re-keyed every deck; nulling them re-keys only decks that set them.
_LADDER_FIELDS = ("dc_group", "dc_index", "dc_weight")


# The cache key describes the GEOMETRY. It says nothing about the FORMAT of
# the arrays stored under it -- so when `ele` changed from a boolean OR-fold to
# an int16 label array, every existing on-disk entry kept serving the old bool
# and the fix silently did nothing. A cache whose key cannot express "the thing
# I stored is shaped differently now" will hand you yesterday's bug forever.
#
# Bump this whenever the LAYOUT or SEMANTICS of a cached array changes.
#   1 -> ele was a bool metal mask
#   2 -> ele is an int16 label array (0 = vacuum, i = electrode i)
CACHE_FORMAT = 2


def geometry_key_dict(spec):
    """Canonical, voltage-free description of what the solver will see.

    Includes CACHE_FORMAT: the key must change when the stored array format
    changes, or a code fix cannot invalidate the entries it invalidates."""
    g = spec.geometry.to_dict()
    # EXTRAS ARE INERT BY CONSTRUCTION: to_dict re-emits retained
    # foreign keys for faithful saves, but an ANNOTATION MUST NEVER FORK
    # A BASIS. The subtraction is SURGICAL: it removes exactly the keys
    # each object's _extras retained -- nothing else -- by walking the
    # spec objects beside their dicts. (A dataclass-field filter was
    # tried first and REKEYED every shapes deck, because ShapeSpec's
    # to_dict is a custom projection whose emitted keys are not the field
    # set; subtracting only what passthrough added is exact for every
    # to_dict style, present and future.) For extra-less decks this is a
    # no-op and every existing key is byte-identical (B2 baseline check).
    def _sans_extras(obj, d):
        ex = getattr(obj, "_extras", None)
        return {k: v for k, v in d.items() if not ex or k not in ex} \
            if isinstance(d, dict) else d
    g = _sans_extras(spec.geometry, g)
    # THE SYMMETRY AS THE SOLVER READS IT, not as the deck spells it:
    # `planes: {}` and `planes: {x: none, y: none, z: none}` are the same
    # deck, and keyed apart, a field solved from a deck as written (a
    # notebook, a script) was not found by the app, whose synced copy
    # spells every plane out. normalized() fills absent axes with 'none'
    # (and z for r-z); its dict carries no annotation extras.
    g["symmetry"] = spec.geometry.symmetry.normalized().to_dict()
    els = []
    _el_objs = list(spec.geometry.electrodes)
    for i, e in enumerate(g.get("electrodes", [])):
        _eo = _el_objs[i] if i < len(_el_objs) else None
        if _eo is not None:
            e = _sans_extras(_eo, e)
            if isinstance(e.get("shapes"), list):
                e["shapes"] = [
                    _sans_extras(so, sh) for so, sh in
                    zip(list(_eo.shapes) + [None] * len(e["shapes"]),
                        e["shapes"])]
        e = {k: (None if k in _LADDER_FIELDS else v)
             for k, v in e.items() if k not in _VOLTAGE_FIELDS}
        # basis index defaults to position; make it explicit so a reorder
        # (which DOES change the basis->electrode mapping) changes the key.
        e["_basis"] = e.get("basis") if e.get("basis") is not None else i + 1
        els.append(e)
    d = {"raster": 3,  # v3 = closed-edge tolerance raster:
                       # _POLY_EDGE_TOL on rect/ellipse —
                       # boundary nodes classify noise-independently.
                       # Invalidates EVERY cached basis (global raster
                       # change). v2 was node-at-i*h
                       # sampling.
            "kind": CACHE_KIND,
            "_fmt": CACHE_FORMAT,
            "width_mm": g["width_mm"], "height_mm": g["height_mm"],
            "depth_mm": g.get("depth_mm", 0.0),
            "mm_per_gu": g["mm_per_gu"],
            "symmetry": g.get("symmetry"),
            "stl_dir": g.get("stl_dir"),
            "builder": getattr(spec, "builder", ""),
            "electrodes": els}
    # ANCHORED-RASTER MARKER: a spec DECLARING plane_mm
    # rasterizes on the plane-anchored lattice, a different raster from
    # its un-anchored entries under the same symmetry dict — key it, or a
    # warm cache serves un-anchored bases for an anchored build. Added
    # ONLY when a plane is declared: every undeclaring spec's key is
    # byte-identical (the is_grid lesson: every geometry-affecting
    # derivation is IN the key, and only when it can differ).
    # DECLARED PLACEMENT: frame_offset_mm MOVES METAL (applied at
    # stl_resolve.load_mesh), so it is geometry and must fork the key --
    # two decks differing only in placement must never share a basis.
    # Keyed ONLY when declared (geometry.to_dict omits None; plane_mm
    # precedent), so every undeclaring spec's key is byte-identical.
    if g.get("frame_offset_mm") is not None:
        d["frame_offset_mm"] = g["frame_offset_mm"]
    # SIGNED FRAME: origin_mm places the node lattice relative to the
    # shapes (raster2d.anchored_grid), so two decks with the same shapes
    # and different origins rasterize different metal. Unkeyed, the
    # second deck was served the first one's bases -- 369 of 1,025 label
    # nodes wrong, depending only on which deck had been built first.
    # Keyed ONLY when declared and not [0, 0] (the unshifted lattice), so
    # every other spec's key is byte-identical; every basis banked for a
    # shifted deck under the old key is never served again.
    _org = g.get("origin_mm")
    if _org is not None and any(float(v) != 0.0 for v in _org):
        d["origin_mm"] = [float(v) for v in _org]
    sym = d.get("symmetry") or {}
    if isinstance(sym, dict) and sym.get("plane_mm"):
        d["anchor_raster"] = 1
    # FOLD-BC REVISION (a real y-fold defect): rev 2 = the
    # folded planar solve applies the TRUE even-reflection ghost
    # (phi_ghost = phi_inner) on fold-plane edges. Rev 1 (implicit)
    # applied the open-boundary ghost_linear (phi_ghost = phi_edge)
    # there — an O(h^2 * phi_yy) plane BC error, measured 38 V on the
    # a real MRT median plane. Keyed ONLY for specs declaring a foldable
    # plane kind, so every undeclaring spec's key is byte-identical;
    # every previously banked FOLDED basis cold-invalidates, which is
    # exactly the point — a code fix must invalidate what it fixes.
    if isinstance(sym, dict) and any(
            k != "none" for k in (sym.get("planes") or {}).values()):
        d["fold_bc_rev"] = 2
    # STL CONTENT: an STL electrode IS its file's bytes, not its name. The
    # key named the file only, so moving a rod in the STL left a saved
    # field matching the new geometry. Keyed only when the deck has STL
    # electrodes, so every other key is byte-identical. A file that cannot
    # be found is recorded as "missing" (which matches no real bytes), so
    # the key can still be computed for display; the build refuses it.
    stl = [e.stl for e in spec.geometry.electrodes if getattr(e, "stl", None)]
    if stl:
        from ion_gym.io.stl_resolve import resolve_stl_dir
        sdir = resolve_stl_dir(spec)
        d["stl_content"] = {name: _file_digest(sdir / name) for name in stl}
    return d


# (path, size, mtime_ns) -> sha256; a cache key is computed often and an
# STL can be large, so a file is hashed again only when it changes
_DIGESTS = {}


def _file_digest(path):
    """sha256 of the file's bytes, or "missing"."""
    import hashlib
    import os
    if not os.path.isfile(path):
        return "missing"
    st = os.stat(path)
    sig = (str(path), st.st_size, st.st_mtime_ns)
    if sig not in _DIGESTS:
        with open(path, "rb") as fh:
            _DIGESTS[sig] = hashlib.sha256(fh.read()).hexdigest()
    return _DIGESTS[sig]


def cache_key_dict(spec):
    """What a cached basis is stored under: the geometry AND the code that
    solves it (physics.solve_code). The geometry alone (geometry_key_dict)
    stays a field's identity -- field files and the editor compare it --
    while the cache must also miss when the solver or a rasterizer
    changes, or a fix reaches new geometries and never the cached ones."""
    from ion_gym.physics.solve_code import solve_code_key
    return dict(geometry_key_dict(spec), _solve_code=solve_code_key())


def key(spec):
    return fa_cache.spec_key(cache_key_dict(spec))


def load(spec, root=None):
    """(bases dict {basis_index: (nx,ny) float64}, ele (nx,ny) int16) or
    (None, None) on a miss.  Arrays are copied out of the memmap so callers
    can hand them to numba kernels without a read-only surprise.

    A bool `ele` is a PRE-FORMAT-2 entry. CACHE_FORMAT should already have
    excluded it, but if one reaches here it is treated as a MISS and said out
    loud -- serving it would collapse every electrode label to #1 and the
    display would lie while looking perfectly healthy."""
    arrs, meta = fa_cache.load(cache_key_dict(spec), root=root)
    if arrs is None:
        return None, None
    bases = {int(k[1:]): np.ascontiguousarray(v, float)
             for k, v in arrs.items() if k.startswith("b")}
    # NOT .astype(bool). This cast was the real root cause: a builder could
    # store a perfectly good int16 label array and the cache would hand it back
    # as a boolean mask, collapsing every electrode to #1. Cold runs looked
    # right; the bug appeared the moment the cache warmed -- which is why it
    # survived being "fixed" in build_rz and looked unfixed in build_planar.
    ele = np.ascontiguousarray(arrs["ele"])
    if ele.dtype == bool:
        # pre-FORMAT-2 entry. CACHE_FORMAT should have excluded it; if one
        # still reaches here, MISS loudly rather than serve a lying display.
        print("basis_cache: ignoring a pre-format-2 (bool) `ele` entry — "
              "labels would collapse to electrode #1; re-solving.")
        return None, None
    # a HIT is provenance too: a differently-named spec resolving this
    # geometry belongs in the many-to-one "produced by" record.
    fa_cache.annotate(cache_key_dict(spec), getattr(spec, "name", ""),
                      root=root)
    return bases, ele.astype(np.int16)



def describe_spec(spec):
    """One-line human descriptor for cache listings (
    bare hex keys made entries unidentifiable). Route + grid + pitch +
    electrode count — everything needed to tell entries apart at a
    glance, derived from the spec so it can never drift from it."""
    g = spec.geometry
    dims = [g.width_mm, g.height_mm] + (
        [g.depth_mm] if getattr(g, "depth_mm", None) else [])
    grid = "x".join(f"{int(round(d / g.mm_per_gu)) + 1}" for d in dims)
    route = getattr(spec, "route", None) or (
        "3-D" if getattr(g, "depth_mm", None) else "2-D/rz")
    return (f"{route} | {grid} @ {g.mm_per_gu:g} mm/gu | "
            f"{len(g.electrodes)} electrodes")


def store(spec, bases, ele, root=None):
    """Atomic publish of the solved bases + the electrode mask."""
    arrays = {f"b{int(k)}": np.ascontiguousarray(v, float)
              for k, v in bases.items()}
    # labels are stored in 8 bits (the format is kept as is); a label
    # past the limit would WRAP silently, so it is
    # refused here at the data level too
    from ion_gym.io.sim_spec import electrode_limit_error
    ele_a = np.ascontiguousarray(ele)
    _lim = electrode_limit_error(int(ele_a.max()) if ele_a.size else 0)
    if _lim:
        raise ValueError(f"basis_cache.store: {_lim}")
    arrays["ele"] = ele_a.astype(np.uint8)
    key = fa_cache.store(cache_key_dict(spec), arrays, root=root)
    # descriptive provenance: the key dict is rightly
    # name-free, so the "which json produced this" record is written here,
    # the last layer that still knows the spec's name.
    fa_cache.annotate(key, getattr(spec, "name", ""),
                      label=describe_spec(spec),
                      spec_json=spec.to_dict(), root=root)
    return key
