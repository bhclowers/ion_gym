"""ion_gym.projects — project packages.

An instrument project (a lab's own design: certified voltages, geometry
constants, trade tables)
may live PHYSICALLY under internal/projects/ — dev checkouts only, deleted
from every shareable cut. This __path__ extension is the ONE sanctioned
bridge, and it is OPT-IN (PI ruling 2026-10-14, L-550): it runs only when
the environment sets ION_GYM_INTERNAL=1. Without the variable -- every
public checkout, and a dev checkout that has not asked -- nothing from
internal/ is reachable, and an import such as ion_gym.projects.oaTOF fails
loudly with the standard ModuleNotFoundError, exactly as it will in the
published package. Internal gates and studies run with the variable set.
Its gate corpus lives beside it in the same internal tree.
"""
import os

from ion_gym.io.paths import repo_root as _repo_root

if os.environ.get("ION_GYM_INTERNAL") == "1":
    _internal = _repo_root() / "internal" / "projects"   # THE repo root (L-551)
    if _internal.is_dir():
        __path__.append(str(_internal))
