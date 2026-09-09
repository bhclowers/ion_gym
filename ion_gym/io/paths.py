"""
ion_gym.paths
-------------
Single source of truth for locating reference data (reference projects, CSVs,
uploads) regardless of where the package lives. No module should hardcode
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
    """Override the data root programmatically (highest priority)."""
    global _OVERRIDE
    _OVERRIDE = Path(path).expanduser().resolve() if path else None


def _repo_root():
    """Repo root: the ancestor that holds the ion_gym package as a sibling of
    the data dirs.
    Identified by the repo marker pyproject.toml so it is
    independent of how deeply the package is nested. In a flat checkout, or an
    installed package with no marker, this returns _PKG_DIR -- the pre-
    migration behavior is preserved unchanged."""
    for anc in (_PKG_DIR, *_PKG_DIR.parents):
        if (anc / "pyproject.toml").exists():
            return anc
    return _PKG_DIR


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
    before any data is placed."""
    for c in _candidates():
        if c and c.exists():
            return c
    return _PKG_DIR


def data_dir(name):
    """Directory for a registered dataset name (or a literal subpath)."""
    rel = _REGISTRY.get(name, name)
    return data_root() / rel


def fields_dir():
    """Home for USER-SAVED solved-field / trajectory NPZs and the label
    registry (fields_registry.json). Distinct from data_root(): that is the
    reference-data corpus (geometry/), this is user output that must be easy
    to find, rename, and send. Resolution: ION_GYM_FIELDS env override, else
    <repo_root>/fields. Created on first use by the writer, not here."""
    env = os.environ.get("ION_GYM_FIELDS")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "fields"


def data_path(name, *parts):
    """File path: a registered name (optionally with extra parts joined)
    or a literal relative path. data_path('quad') / 'quad_field.npz'
    is spelled data_path('quad', 'quad_field.npz')."""
    base = data_dir(name)
    return base.joinpath(*parts) if parts else base


def repo_root():
    """The repo root: the directory holding the ion_gym package and the data
    dirs / examples/ as siblings. Public wrapper over the marker-based
    resolver, for callers that must anchor a repo-root-relative path (e.g. the
    app's examples/ dir) independently of the imported data root."""
    return _repo_root()


def outputs_dir(*parts):
    """repo_root()/outputs (optionally a subpath) -- the ONE home for where
    generated figures/reports go. Committed code once
    carried absolute `/mnt/user-data/outputs` literals (a per-machine
    staging dir), which is forbidden; every producer now routes
    here. PURE -- no mkdir at import time; the consumer that writes
    creates what it needs."""
    p = _repo_root() / "outputs"
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

    BY DESIGN, output files from studies go
    into an output folder within the study... keep all project files,
    inputs and outputs, in the project folder"). A study is a
    self-contained project: its decks, its notebook, its banked JSON
    and its figures travel together, so a reader can copy or archive
    one folder and have the whole thing. The repo-wide outputs_dir()
    remains correct for cross-cutting artifacts that belong to no
    single study -- and it is the reason study results were previously
    scattered into a shared bucket where they lost their provenance.

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
