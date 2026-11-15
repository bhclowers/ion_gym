"""ion_gym.physics — subpackage.

One deliberate effect at import, before any kernel module is loaded: numba's
on-disk kernel cache is pointed at a directory keyed by the SOURCE of this
package (every .py file under ion_gym/physics, and the version).

Why. numba invalidates a cached kernel on the stamp of the file that defines
it, not on the files whose functions it calls. A kernel in tracer_rz.py that
calls collision3d.hs_collision_step kept the old compiled collision physics
after collision3d.py changed (a git pull, a local edit, an upgrade), with no
message, until the .nbi/.nbc files were deleted by hand. With the key, any
change to any physics module selects a fresh directory, so every kernel is
recompiled from the current source; an unchanged tree reuses its cache.

The directory is <cache root>/.numba/<key>, the cache root being
ION_GYM_CACHE or ~/.ion_gym_cache (io.paths.cache_root); the leading dot
keeps it out of the field-cache inventory (fa_cache.entries), and the app's
"clear ALL" removes it with the fields (kernels then recompile). A NUMBA_CACHE_DIR
the user set is respected and left alone, with a warning, because a single
user-chosen directory carries the stale-kernel hazard above. A cache root
that cannot be created is reported, not silently replaced.
"""
import hashlib
import os
import warnings
from pathlib import Path


def kernel_source_key():
    """16 hex digits over the version and every .py file under this package
    (path and bytes), so any source change gives a new key."""
    from ion_gym import __version__
    here = Path(__file__).resolve().parent
    h = hashlib.sha256(__version__.encode())
    for p in sorted(here.rglob("*.py")):
        h.update(str(p.relative_to(here)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _key_numba_cache():
    if os.environ.get("NUMBA_CACHE_DIR"):
        warnings.warn(
            "NUMBA_CACHE_DIR is set, so ion_gym's numba kernels are cached "
            "there unkeyed: after any change to an ion_gym physics module, "
            "clear that directory, or kernels that call the changed module "
            "can keep running the old physics. Unset it to let ion_gym key "
            "the cache by source.", stacklevel=3)
        return
    from ion_gym.io.paths import cache_root
    target = cache_root() / ".numba" / kernel_source_key()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        warnings.warn(
            f"ion_gym could not create its numba kernel cache {target} "
            f"({e}); numba falls back to caching beside the source, where an "
            f"edit to a called module can leave stale kernels. Set "
            f"ION_GYM_CACHE to a writable folder.", stacklevel=3)
        return
    import numba.core.config as _numba_config
    _numba_config.CACHE_DIR = str(target)


_key_numba_cache()
