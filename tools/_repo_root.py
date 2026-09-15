"""Repo root by MARKER, not by position.

Relocating this directory one level deeper once silently
broke every script here that computed Path(__file__).parent.parent: the
sys.path inserts missed ion_gym, dependency roots pointed inside
the wrong tree, and the notebook authors would have written their output
to the wrong place. The harvester was the first measured
casualty (UI reference not regenerated for a month of UI changes).

ONE authority: every script in this directory imports repo_root() from
here (the script's own directory is on sys.path when it is run directly,
so this import needs no path bootstrap of its own). Walking parents for
the repo's own markers -- pyproject.toml beside the ion_gym package --
is location-agnostic: the tools keep working wherever this directory is
moved next, and refuse loudly, with the searched path, when run from a
copy that is not inside the repo at all, instead of importing some OTHER
ion_gym off sys.path and operating on the wrong tree.
"""
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file() and (p / "ion_gym").is_dir():
            return p
    raise FileNotFoundError(
        "repo_root: no repo root (pyproject.toml + ion_gym/) found in any "
        f"parent of {here} -- run this tool from inside the ion_gym repo "
        "tree")
