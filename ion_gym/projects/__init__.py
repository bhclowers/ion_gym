"""ion_gym.projects — project packages.

An instrument project (a lab's own design: certified voltages, geometry
constants, trade tables) can be kept outside the published package, in a
development checkout's private projects folder. This __path__ extension
is the one bridge to it, and it is OPT-IN: it runs only when the
environment sets ION_GYM_INTERNAL=1 and that folder exists. Without both
-- every installed package and every public checkout -- nothing outside
the package is reachable, and an import of a project that is not here
fails with the standard ModuleNotFoundError.
"""
import os

from ion_gym.io.paths import repo_root as _repo_root

if os.environ.get("ION_GYM_INTERNAL") == "1":
    _internal = _repo_root() / "internal" / "projects"   # the repo root
    if _internal.is_dir():
        __path__.append(str(_internal))
