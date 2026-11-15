"""
run_provenance.py -- what a saved run needs in order to be flown again.

A run is reproducible from three things: the deck exactly as it flew, the
seed its births and collisions were drawn from, and the code that flew it.
A deck in random mode (source.seed null) draws its seed at build time
(ion_envelope.resolve_run_seed) and keeps it only on the live spec, so a
saved run that stores the deck alone stores `seed: null` and cannot be
repeated. run_provenance() captures all three right after the build;
replay_spec() turns a captured record back into a deck with the seed
pinned.

Every value is recorded, never omitted: a seed that was never drawn is
recorded as None with seed_mode "not drawn", and a version that cannot be
read is recorded with the reason, so a reader can tell "unknown" from
"absent".
"""
# The public names (docs/PUBLIC_API.md); everything else is internal.
__all__ = ["run_provenance", "replay_spec", "run_seed", "versions",
           "unrecorded", "deck_location", "portable_spec_dict",
           "with_location", "PROVENANCE_FORMAT"]

import platform
import subprocess
import time
from pathlib import Path

import ion_gym

PROVENANCE_FORMAT = 1


def _git_state():
    """Commit and dirty flag of the checkout this ion_gym runs from, or the
    reason there is none (an installed wheel has no checkout)."""
    root = Path(ion_gym.__file__).resolve().parent.parent
    if not (root / ".git").exists():
        return {"commit": None, "note": "not a git checkout"}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=no"], check=True, capture_output=True,
            text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as e:
        return {"commit": None,
                "note": f"git unavailable ({type(e).__name__}: {e})"}
    return {"commit": commit, "dirty": bool(dirty)}


def versions():
    """The code that flies: ion_gym, its checkout, and the numerical stack."""
    import numba
    import numpy
    return {"ion_gym": ion_gym.__version__, "git": _git_state(),
            "python": platform.python_version(),
            "numpy": numpy.__version__, "numba": numba.__version__,
            "platform": platform.platform()}


def run_seed(spec):
    """(seed, mode) the run used: the deck's own seed ("stated"), the seed
    drawn for this build ("drawn"), or (None, "not drawn") when the spec
    has not been built since its last draw was cleared."""
    s = spec.source
    if s.seed is not None:
        return int(s.seed), "stated"
    drawn = getattr(s, "_drawn_seed", None)
    if drawn is not None:
        return int(drawn), "drawn"
    return None, "not drawn"


def deck_location(spec):
    """The absolute path the deck was loaded from, or None. Relative
    resources (an STL folder, a births file) resolve against it, and
    to_dict() does not carry it."""
    loaded = getattr(spec, "_loaded_from", None)
    return str(Path(loaded).resolve()) if loaded else None


def portable_spec_dict(spec):
    """spec.to_dict(), with an stl_manifest (sha256 per referenced STL)
    embedded when the deck has STL electrodes and declares none, so a
    re-run verifies the meshes are the ones that flew (stl_resolve.
    require_stls checks it before any STL build)."""
    d = spec.to_dict()
    geo = d.get("geometry") or {}
    stl = [e.stl for e in spec.geometry.electrodes if getattr(e, "stl", None)]
    if stl and not geo.get("stl_manifest"):
        from ion_gym.io.stl_resolve import build_manifest, resolve_stl_dir
        sdir = resolve_stl_dir(spec)
        # only when every mesh is on disk (a save must not fail for want
        # of one); without a manifest a re-run still checks they exist
        if all((sdir / name).is_file() for name in stl):
            geo["stl_manifest"] = build_manifest(spec)
    return d


def with_location(spec, loaded_from):
    """Re-attach a deck's original location to a spec rebuilt from a dict,
    so its relative STL folder and births file resolve where they did."""
    if loaded_from:
        spec._loaded_from = str(loaded_from)
    return spec


def run_provenance(spec):
    """Capture, right after a build, everything needed to fly the run
    again. JSON-serializable."""
    seed, mode = run_seed(spec)
    return {"format": PROVENANCE_FORMAT, "seed": seed, "seed_mode": mode,
            "spec": portable_spec_dict(spec),
            "spec_loaded_from": deck_location(spec),
            "versions": versions(),
            "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())}


def unrecorded(reason):
    """The record for a run whose provenance is not known (e.g. loaded
    from a file saved before provenance was written). Says so, rather
    than describing the run with whatever spec happens to be live."""
    return {"format": PROVENANCE_FORMAT, "seed": None,
            "seed_mode": "not recorded", "spec": None, "versions": None,
            "note": str(reason)}


def replay_spec(provenance):
    """The deck a captured run flew, with its seed pinned, so building it
    draws the same births and collisions. Refuses a record without a
    seed: it cannot be repeated exactly, and flying it would draw anew."""
    from ion_gym.io.sim_spec import SimSpec
    seed = provenance.get("seed")
    if seed is None:
        raise ValueError(
            f"replay_spec: this run's seed was not recorded (seed_mode "
            f"{provenance.get('seed_mode')!r}); it cannot be flown again "
            f"exactly")
    spec = with_location(SimSpec.from_dict(provenance["spec"]),
                         provenance.get("spec_loaded_from"))
    spec.source.seed = int(seed)
    return spec
