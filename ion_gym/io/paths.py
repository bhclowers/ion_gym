"""
ion_gym.io.paths
----------------
Single source of truth for every path ion_gym reads or writes (example
decks, the user data root and uploads, per-machine output, caches),
regardless of where the package lives; docs/DATA_LAYOUT.md describes it
for users. No module should hardcode
an absolute path; they call data_path(...) / data_dir(...) instead.

Resolution order for the data ROOT (first that exists wins):
  1. an explicit path passed to set_data_root(...) at runtime;
  2. the ION_GYM_DATA environment variable;
  3. a 'geometry' directory at the repo root (post-migration data home);
  4. the repo root itself (data dirs sit beside the ion_gym package);
  5. a 'data' directory next to the package (installed layout);
  6. the package directory itself (legacy flat layout — reference folders
     sit beside the .py files, which is how the early v-zips shipped);
  7. the current working directory.

Named datasets are registered once here (their folder/file names inside
the root), so a validator says data_path("funnel_csv") and stays
location-independent. Unknown names fall through to a literal join, so
data_path("funnel/ionfunnel") also works.
"""

# The public names (docs/PUBLIC_API.md); everything else is internal.
__all__ = ["set_data_root", "data_root", "data_dir", "data_path",
           "fields_dir", "trajectories_dir", "panels_dir", "repo_root",
           "examples_dir", "is_shipped_example", "uploads_dir",
           "cache_root", "default_deck", "outputs_dir", "pkg_dir"]

import os
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_OVERRIDE = None

# Logical name -> path relative to the data root. Kept flat and explicit
# so moving the data only means moving one folder, not editing modules.
_REGISTRY = {
    # EXTERNAL REFERENCE DATASETS REMOVED. Every key
    # that used to live here named a folder of imported reference geometry
    # and field arrays; that tree is deleted, and the keys go with it -- a
    # registry key resolving to a deleted tree is a trap, not a
    # convenience. The gates that read them are retired with their
    # recorded results intact.
    # uploads (STL round-trip inputs)
    "uploads": "uploads",
}


def set_data_root(path):
    """Override the data root programmatically (highest priority). A
    folder that does not exist is refused: it used to be skipped, and the
    data root silently became another folder (review API-15). None
    clears the override."""
    global _OVERRIDE
    if path:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"set_data_root({str(path)!r}): {p} is not a folder. Create "
                f"it first; an explicit data root is never skipped in "
                f"favour of another folder.")
        _OVERRIDE = p
    else:
        _OVERRIDE = None


def _checkout_root():
    """The source checkout this package was imported from, or None.

    This module sits at <root>/ion_gym/io/paths.py, so the checkout root is
    exactly two levels above this directory, and it is a checkout only when
    pyproject.toml sits there. The position is fixed rather than searched
    for: walking up to the nearest pyproject.toml would make an installed
    copy (site-packages/ion_gym) adopt whatever Python project happens to
    contain the virtual environment as its "repo"."""
    root = _PKG_DIR.parents[1]
    return root if (root / "pyproject.toml").is_file() else None


def _repo_root():
    """Repo root: the checkout holding the ion_gym package as a sibling of
    the data dirs. An installed package has no checkout; it returns
    _PKG_DIR, as it always has (per-machine output does not use this; see
    _output_root)."""
    root = _checkout_root()
    return root if root is not None else _PKG_DIR


def _output_root():
    """Where per-machine output (outputs/, fields/, trajectories/) goes:
    the checkout root in a clone, the current working folder for an
    installed package. An installed package has no repo to write into,
    and writing under site-packages puts a user's results inside the
    Python installation (or fails where that is read-only). Resolved at
    every call, so a process that changes folder writes where it now is."""
    root = _checkout_root()
    return root if root is not None else Path.cwd()


def _candidates():
    if _OVERRIDE is not None:
        yield _OVERRIDE
    env = os.environ.get("ION_GYM_DATA")
    if env:
        yield Path(env).expanduser().resolve()
    root = _repo_root()
    # and is DELETED. No candidate root points there any more.
    yield root / "geometry"   # pre-cleanup layout (older checkouts)
    yield root                # data dirs sit at the repo root, beside the package
    yield _PKG_DIR / "data"   # installed-package data dir
    yield _PKG_DIR            # legacy flat layout (paths.py beside the data)
    yield Path.cwd()


def data_root():
    """The resolved data root: the first candidate that exists. Falls
    back to the package dir so callers always get a usable Path even
    before any data is placed. An ION_GYM_DATA that names a missing
    folder is refused, as set_data_root refuses one: an explicit choice
    is never skipped in favour of another folder (review API-15)."""
    env = os.environ.get("ION_GYM_DATA")
    if _OVERRIDE is None and env and not Path(env).expanduser().is_dir():
        raise FileNotFoundError(
            f"ION_GYM_DATA={env!r} is not a folder. Create it, or unset the "
            f"variable to use the default data root.")
    for c in _candidates():
        if c and c.exists():
            return c
    return _PKG_DIR


def data_dir(name):
    """Directory for a registered dataset name (or a literal subpath)."""
    rel = _REGISTRY.get(name, name)
    return data_root() / rel


def fields_dir():
    """Home for USER-SAVED solved-field NPZs and the label
    registry (fields_registry.json). Distinct from data_root(): that is the
    reference-data corpus (geometry/), this is user output that must be easy
    to find, rename, and send. Resolution: ION_GYM_FIELDS env override, else
    <output root>/fields (the checkout root in a clone, the working folder
    for an installed package). Created on first use by the writer.

    Trajectory NPZs used to land here too; they now have their own home
    (trajectories_dir) so a folder full of solved fields is not intermixed
    with flight output. Old .traj.npz files left in fields/ still load: the
    picker scans both directories."""
    env = os.environ.get("ION_GYM_FIELDS")
    if env:
        return Path(env).expanduser().resolve()
    return _output_root() / "fields"


def trajectories_dir():
    """Home for USER-SAVED trajectory NPZs (<name>__<key8>.traj.npz), split
    out from fields_dir so flight output and solved fields do not share one
    folder. Resolution: ION_GYM_TRAJECTORIES env override, else
    <output root>/trajectories (see fields_dir). Created on first use by
    the writer, not here."""
    env = os.environ.get("ION_GYM_TRAJECTORIES")
    if env:
        return Path(env).expanduser().resolve()
    return _output_root() / "trajectories"


def panels_dir(*parts):
    """Home for the BANKED notebook instrument panels (panel_<family>.png).

    SHIPPED CONTENT, not regenerable state, and that is the whole point:
    every teaching notebook opens by displaying its instrument, and when
    the bank is absent the panel is rebuilt live -- a solve plus a short
    flight -- before the reader sees anything. The bank used to sit under
    notebooks/out/_panels, which no zip and no checkout carries, so the
    fast path never once fired outside the tree that built it and every
    notebook paid the rebuild on open.

    Both the writer (a development-tree batch tool) and the reader
    (ion_gym.viz.nb_panels) resolve here, so the location is stated once
    rather than as matching literals in two files.

    PURE -- no mkdir; the writer creates what it needs."""
    p = _repo_root() / "notebooks" / "panels"
    for q in parts:
        p = p / q
    return p


def data_path(name, *parts):
    """File path: a registered name (optionally with extra parts joined)
    or a literal relative path. data_path('quad') / 'quad_field.npz'
    is spelled data_path('quad', 'quad_field.npz')."""
    base = data_dir(name)
    return base.joinpath(*parts) if parts else base


def repo_root():
    """The repo root: the directory holding the ion_gym package and the data
    dirs / examples/ as siblings. Public wrapper over the checkout resolver,
    for callers that must anchor a repo-root-relative path independently of
    the imported data root. Example decks are NOT located through this: an
    installed package has no repo, so use examples_dir()."""
    return _repo_root()


def examples_dir():
    """The shipped example decks and meshes.

    In a checkout: <root>/examples. In an installed package: the copy the
    wheel carries at ion_gym/examples (pyproject maps the root folder onto
    that package path). Refuses, naming both places, when an installed
    package has no examples, rather than handing back a path that does
    not exist."""
    root = _checkout_root()
    if root is not None:
        return root / "examples"
    packaged = _PKG_DIR.parent / "examples"
    if packaged.is_dir():
        return packaged
    raise FileNotFoundError(
        f"ion_gym has no example decks: not a source checkout (no "
        f"pyproject.toml at {_PKG_DIR.parents[1]}) and the installed "
        f"package carries no {packaged}. Reinstall from a wheel built by "
        f"this version's pyproject, which packages examples/.")


def is_shipped_example(path):
    """True when `path` resolves inside examples_dir(). Shipped examples
    are read-only to the app's writers: a user edit saved
    over one is lost on the next install and changes what every later
    user of the example sees. Paths are resolved (symlinks, '..') before
    the comparison, so a relative or indirect path cannot slip past."""
    target = Path(path).expanduser().resolve()
    return target.is_relative_to(examples_dir().resolve())


def uploads_dir(*parts):
    """Home for files the app's upload doors install (STL meshes under
    'stl', decks with their meshes under 'decks'). The registered
    'uploads' dataset, so it moves with the data root (set_data_root /
    ION_GYM_DATA) like every other registered name -- callers used to
    rebuild repo_root()/'uploads' by hand beside this entry. PURE -- no
    mkdir; the writer creates what it needs."""
    return data_path("uploads", *parts)


def cache_root():
    """Home of the solved-basis cache (fa_cache, basis_cache, field_io):
    the ION_GYM_CACHE environment variable if set, else ~/.ion_gym_cache.
    Resolved at CALL time, every call: fa_cache used to read it once at
    import and freeze it into default arguments, so setting the variable
    after import (a notebook, a test, a tool) silently had no effect.
    PURE -- no mkdir."""
    env = os.environ.get("ION_GYM_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".ion_gym_cache"


# The deck SimApp() opens when given no spec, relative to examples_dir().
# Registered here, as a registry entry rather than a user config value, so
# the app does not carry its own path literal.
_DEFAULT_DECK = "ion_funnel_rz.json"


def default_deck():
    """Path of the deck SimApp() opens when given no spec."""
    return examples_dir() / _DEFAULT_DECK


def outputs_dir(*parts):
    """<output root>/outputs (optionally a subpath) -- the ONE home for
    where generated figures/reports go: the checkout root in a clone, the
    working folder for an installed package. Committed code once
    carried absolute `/mnt/user-data/outputs` literals (a per-machine
    staging dir), which is forbidden; every producer now routes
    here. PURE -- no mkdir at import time; the consumer that writes
    creates what it needs."""
    p = _output_root() / "outputs"
    for q in parts:
        p = p / q
    # Create the CONTAINING directory (not the file). Import-time purity is
    # preserved -- nothing happens until someone actually asks for a path --
    # while a caller that writes to what we return cannot fail on a missing
    # parent. A notebook once died here on a fresh clone.
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def study_outputs_dir(anchor, *parts):
    """The `outputs/` folder BESIDE a study, not the repo-wide one.

    Not public: studies live only in the development tree (the
    `studies` folder of its development area), so this refuses
    everywhere else. It is kept here because the development studies
    import it.

    A study is a self-contained project: its decks, its notebook, its
    banked JSON and its figures travel together, so a reader can copy or
    archive one folder and have the whole thing. The repo-wide
    outputs_dir() remains correct for cross-cutting artifacts that
    belong to no single study.

    `anchor` is any path inside the study -- typically `__file__` from
    a study module, or the notebook's own directory. The study root is
    the nearest ancestor study directory; passing something
    outside a study REFUSES rather than silently writing to the
    repo-wide bucket, because a misfiled artifact is worse than a
    stopped run.

        OUTDIR = study_outputs_dir(__file__)      # .../mrtof/outputs
    """
    p = Path(anchor).resolve()
    if p.is_file():
        p = p.parent
    root = _repo_root().resolve()
    studies = root / "internal" / "studies"
    try:
        rel = p.relative_to(studies)
    except ValueError:
        raise ValueError(
            f"study_outputs_dir({anchor!r}): {p} is not inside "
            f"{studies}. A study's outputs live beside the study; for "
            f"an artifact that belongs to no study use outputs_dir()."
        ) from None
    if not rel.parts:
        raise ValueError(
            f"study_outputs_dir({anchor!r}) resolved to the studies "
            f"root itself; name the individual study folder.")
    out = studies / rel.parts[0] / "outputs"
    for q in parts:
        out = out / q
    (out.parent if out.suffix else out).mkdir(parents=True, exist_ok=True)
    return out


def pkg_dir():
    return _PKG_DIR
