"""
PACKAGE METADATA: the two install routes declare the same environment,
and the package metadata is complete.

README offers `pip install -e ".[all]"` and `pip install -r
requirements.txt` as equivalent routes. Two package lists drift: a
package added to one and not the other installs on one route only. And a
requirements file that lists dependencies installs everything except
ion_gym itself. So requirements.txt holds no list. Checked:

  * requirements.txt is the one requirement `-e .[all]`: this checkout,
    with the `all` extra, and nothing else;
  * `all` contains every other extra's packages, except the extras
    named in ALL_EXCLUDES with their reason;
  * the `notebooks` extra brings a server to open the notebooks in;
  * project.readme names a file that exists, so the published package
    has a description;
  * numpy is bounded below at 2 while the package calls a numpy-2-only
    name (np.trapezoid has no numpy 1 spelling), so a numpy 1 install
    is refused by pip rather than failing at the first figure.

Run: python tests/test_package_metadata.py [repo_root]
"""
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Extras deliberately kept out of `all`, with the reason.
ALL_EXCLUDES = {
    "cad": "cadquery pulls the OpenCascade wheels, a heavy install that no "
           "solve or flight path needs",
}

_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*$")
# the only requirement requirements.txt may hold: this checkout + `all`
THIS_PACKAGE_ALL = "-e .[all]"
# a server to open the notebooks with (any one of these)
NOTEBOOK_SERVERS = {"jupyterlab", "notebook"}
# numpy attributes that exist only from numpy 2 on
NUMPY2_ONLY = re.compile(r"\bnp\.(trapezoid)\b")


def _parse(req):
    """(normalised name, specifier text) of one requirement string."""
    m = _REQ.match(req)
    if m is None:
        raise ValueError(f"cannot parse requirement {req!r}")
    return re.sub(r"[-_.]+", "-", m.group(1)).lower(), m.group(2).replace(" ", "")


def _requirement_lines(path):
    """The requirement lines of a requirements file, comments dropped."""
    lines = (raw.split("#", 1)[0].strip()
             for raw in path.read_text(encoding="utf-8").splitlines())
    return [" ".join(line.split()) for line in lines if line]


def check(root):
    fails = []
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    proj = cfg["project"]
    extras = {k: dict(_parse(r) for r in v)
              for k, v in proj["optional-dependencies"].items()}
    if "all" not in extras:
        return ["pyproject has no `all` extra"]

    reqs = _requirement_lines(root / "requirements.txt")
    if reqs != [THIS_PACKAGE_ALL]:
        fails.append(f"requirements.txt must hold exactly "
                     f"{THIS_PACKAGE_ALL!r} (this package with `all`; the "
                     f"dependency list lives in pyproject only), found "
                     f"{reqs}")
    if not NOTEBOOK_SERVERS & set(extras.get("notebooks", {})):
        fails.append(f"the `notebooks` extra has no notebook server (one "
                     f"of {sorted(NOTEBOOK_SERVERS)})")
    for extra, pkgs in sorted(extras.items()):
        if extra == "all" or extra in ALL_EXCLUDES:
            continue
        for name in sorted(set(pkgs) - set(extras["all"])):
            fails.append(f"extra `{extra}` has {name}, which `all` lacks")

    if "readme" not in proj:
        fails.append("project.readme is not set: the published package "
                     "would have no description")
    elif not (root / proj["readme"]).is_file():
        fails.append(f"project.readme names {proj['readme']}, which does "
                     f"not exist")

    users = sorted(str(p.relative_to(root))
                   for p in (root / "ion_gym").rglob("*.py")
                   if NUMPY2_ONLY.search(p.read_text(encoding="utf-8")))
    numpy_spec = dict(_parse(r) for r in proj["dependencies"]).get("numpy")
    if users and numpy_spec != ">=2":
        fails.append(f"{users} call numpy-2-only names but the numpy "
                     f"requirement is {numpy_spec!r}, not '>=2'")
    return fails


def main(argv):
    root = Path(argv[0]).resolve() if argv else HERE.parent
    fails = check(root)
    for msg in fails:
        print(f"  FAIL {msg}")
    print(f"PACKAGE METADATA ({root}): "
          + ("PASS" if not fails else f"FAIL, {len(fails)} problem(s)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
