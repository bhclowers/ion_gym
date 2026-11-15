"""
PACKAGE MANIFEST: everything the installed package needs is declared to
setuptools, so a regular (non-editable) pip install carries it.

A file setuptools is not told about is silently absent from the wheel,
and nothing fails until a pip user reaches the code that needs it. Two
such absences broke a pip install outright: the example decks (the app
opens a default deck at start-up, so it could not start) and the
vendored three.js (the 3-D viewer fell back to a CDN, or raised
offline). A clone never notices, because everything is on disk beside
the code.

Checked against pyproject.toml, with no build:
  * packages are an explicit list, and it names every directory under
    ion_gym/ that has an __init__.py;
  * every listed package (or its package-dir mapping) exists, and the
    example decks are mapped in as ion_gym.examples;
  * every .py file under ion_gym/ sits in a listed package;
  * every other file in a listed package's folder (dot-files and
    caches aside) is matched by one of that package's package-data
    patterns, using the same recursive glob setuptools applies.

Run: python tests/test_package_manifest.py [repo_root]
"""
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES_PACKAGE = "ion_gym.examples"
CACHE_SUFFIXES = {".pyc", ".nbi", ".nbc"}


def _is_cache_or_hidden(rel_parts):
    return any(p == "__pycache__" or p.startswith(".") for p in rel_parts)


def check(root):
    """List of failure messages for the tree at `root` (empty = pass)."""
    cfg = tomllib.loads((root / "pyproject.toml").read_text(
        encoding="utf-8"))
    st = cfg["tool"]["setuptools"]
    packages = st["packages"] if "packages" in st else None
    if not isinstance(packages, list):
        return [f"packages are discovered ({packages!r}), not listed: "
                f"discovery cannot map examples/ into the wheel, so the "
                f"installed package ships without its example decks"]
    pkg_dir_map = st["package-dir"] if "package-dir" in st else {}
    data = st["package-data"] if "package-data" in st else {}
    fails = []

    def folder(pkg):
        if pkg in pkg_dir_map:
            return root / pkg_dir_map[pkg]
        return root.joinpath(*pkg.split("."))

    if EXAMPLES_PACKAGE not in packages:
        fails.append(f"{EXAMPLES_PACKAGE} is not a listed package: the "
                     f"example decks would not ship")
    for pkg in packages:
        d = folder(pkg)
        if not d.is_dir():
            fails.append(f"listed package {pkg} has no folder at {d}")
        elif pkg not in pkg_dir_map and not (d / "__init__.py").is_file():
            fails.append(f"listed package {pkg} has no __init__.py in {d}")

    listed_dirs = {folder(p).resolve(): p for p in packages}
    pkg_root = root / "ion_gym"
    for init in sorted(pkg_root.rglob("__init__.py")):
        rel = init.parent.relative_to(root)
        if _is_cache_or_hidden(rel.parts):
            continue
        if init.parent.resolve() not in listed_dirs:
            fails.append(f"subpackage {'.'.join(rel.parts)} is not in the "
                         f"packages list, so it would not ship")

    def owner(path):
        """Deepest listed package whose folder contains `path`."""
        for anc in path.resolve().parents:
            if anc in listed_dirs:
                return listed_dirs[anc]
        return None

    covered = {}
    for pkg in packages:
        d = folder(pkg)
        hits = set()
        for pattern in data.get(pkg, []):
            hits.update(p.resolve() for p in d.glob(pattern) if p.is_file())
        covered[pkg] = hits

    scan = [pkg_root]
    if EXAMPLES_PACKAGE in pkg_dir_map:
        scan.append(folder(EXAMPLES_PACKAGE))
    for base in scan:
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root)
            if _is_cache_or_hidden(rel.parts) or f.suffix in CACHE_SUFFIXES:
                continue
            pkg = owner(f)
            if f.suffix == ".py":
                if pkg is None or f.parent.resolve() not in listed_dirs:
                    fails.append(f"{rel}: module outside every listed "
                                 f"package, so it would not ship")
                continue
            if pkg is None:
                fails.append(f"{rel}: inside no listed package, so it "
                             f"would not ship")
            elif f.resolve() not in covered[pkg]:
                fails.append(f"{rel}: no package-data pattern of {pkg} "
                             f"matches it, so it would not ship")
    return fails


def main(argv):
    root = Path(argv[0]).resolve() if argv else HERE.parent
    fails = check(root)
    for msg in fails:
        print(f"  FAIL {msg}")
    print(f"PACKAGE MANIFEST ({root}): "
          + ("PASS" if not fails else f"FAIL, {len(fails)} problem(s)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
