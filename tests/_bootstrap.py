"""_bootstrap.py -- put the repo ROOT on sys.path for a SCRIPT-style gate.

conftest.py does this for pytest, but conftest is a pytest mechanism: it is not
consulted when a script is run directly.  The script-style gates
(`python3 tests/test_ui_smoke.py`) have sys.path[0] == tests/, so after the
flat -> tests/ move every one of them died with ModuleNotFoundError on the
first core import.  They were dark twice over: excluded from collection AND
unrunnable by hand.

Rooted on __file__, never cwd: a gate must give the same answer from anywhere.
"""
import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- layout-aware module resolution -----------------------------------------
# A gate that names a core module by FILENAME (e.g. "build_planar.py") used to
# find it at ROOT/<file>, a flat-repo assumption.  The flat -> ion_gym package
# migration invalidates that: modules live under ion_gym/<subpkg>/ (or, for a
# "tests.*" map key, under tests/).  `locate` follows migration_map.json so a
# gate resolves a module wherever it CURRENTLY sits -- root before its slice
# moves it, the mapped package dir after -- with no per-slice path baked in.
# Map-driven, so it is configuration-agnostic (one fix, all configs).
@functools.lru_cache(maxsize=1)
def _module_index():
    """{filename: path} for every .py under the package, the tests, and any
    trees.

    REPLACES the migration_map.json lookup: the map was a
    root-litter artifact of the flat -> package migration and was deleted.
    A SEARCH is also strictly better than a map -- a map goes stale silently
    every time a module moves and the JSON is not updated, which is exactly
    how a gate ends up scanning nothing while reporting PASS.
    """
    index = {}
    # the dev tree adds an internal directory to the search; a public checkout has
    # no such directory and the guard makes that a clean no-op.
    _bases = [ROOT / "ion_gym", ROOT / "tests"]
    _dev = ROOT / "internal"
    if _dev.exists():
        _bases.append(_dev)
    for base in _bases:
        if not base.exists():
            continue
        for f in base.rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            index.setdefault(f.name, f)      # first hit wins: package before
    return index                             # tests take precedence


def locate(filename):
    """Path to a core module given its bare FILENAME, wherever it lives now.

    Raises FileNotFoundError -- never returns a wrong or guessed path -- if the
    module is not found, so a caller sees a genuine absence as an error
    to record, never a silent skip (refuse with a diagnostic)."""
    fname = filename if filename.endswith(".py") else filename + ".py"
    hit = _module_index().get(fname)
    if hit is not None:
        return hit
    root_copy = ROOT / fname
    if root_copy.exists():
        return root_copy
    raise FileNotFoundError(
        f"locate({fname!r}): module not found anywhere under ion_gym/, "
        f"the searched trees. If it was renamed or removed, "
        f"update the caller's list; this is not skipped silently.")
