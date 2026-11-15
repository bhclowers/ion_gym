"""
RUNTIME IMPORTS: every third-party module ion_gym imports is declared.

An import that no install route declares works on the developer's machine
(where everything is installed) and fails for a pip user the first time
the code path runs -- often inside a callback, where it becomes a
traceback in the server log behind a page that still returns 200.
Measured: the dashboard's geometry preview imports contourpy, which was
declared only in an optional extra, so a plain install served a preview
that failed on every build.

Every third-party top-level module imported anywhere under ion_gym/ (at
module level or inside a function) must be exactly one of:
  * a core dependency in pyproject;
  * in VIA_CORE: installed by a named core dependency;
  * in OPTIONAL: supplied by a named extra, with the reason the core
    never needs it. Its import must sit inside a function, so importing
    ion_gym never requires it.
An import in none of these fails by name. The vendored three.js folder is
not Python and is skipped.

Run: python tests/test_runtime_imports.py [repo_root]
"""
import ast
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# import name -> distribution name, where they differ
IMPORT_TO_DIST = {"PIL": "pillow"}

# Installed by a core dependency; not imported at top level by ion_gym on
# its own account.
VIA_CORE = {
    "pandas": "panel",
    "param": "panel",
    "pillow": "bokeh",
    "tornado": "bokeh",          # the dashboard server's HTTP export route
    "cycler": "matplotlib",
    "mpl-toolkits": "matplotlib",
}

# Supplied by an extra; the core never needs them.
OPTIONAL = {
    "cma": ("notebooks", "CMA-ES optimiser used by study notebooks"),
    "IPython": ("notebooks", "notebook display helpers; arrives with the "
                             "notebook extra's jupyter_bokeh"),
    "cadquery": ("cad", "STEP export of generated CAD"),
    "jupytercad_core": ("cad", "schema check of emitted .jcad documents"),
}


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _imports(pkg):
    """{module: [(file, line, top_level)]} for third-party imports."""
    own = pkg.name
    stdlib = set(sys.stdlib_module_names)
    out = {}
    for f in sorted(pkg.rglob("*.py")):
        rel = f.relative_to(pkg.parent)
        if "__pycache__" in rel.parts or "vendor" in rel.parts:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        top = set(map(id, tree.body))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif (isinstance(node, ast.ImportFrom) and node.level == 0
                  and node.module):
                mods = [node.module.split(".")[0]]
            else:
                continue       # not an absolute import statement
            for m in mods:
                if m in stdlib or m == own or m == "__future__":
                    continue
                out.setdefault(m, []).append(
                    (rel.as_posix(), node.lineno, id(node) in top))
    return out


def check(root):
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    proj = cfg["project"]
    core = {_norm(re.match(r"[A-Za-z0-9._-]+", r).group(0))
            for r in proj["dependencies"]}
    extras = {k: {_norm(re.match(r"[A-Za-z0-9._-]+", r).group(0)) for r in v}
              for k, v in proj["optional-dependencies"].items()}
    fails = []
    for mod, sites in sorted(_imports(root / "ion_gym").items()):
        dist = _norm(IMPORT_TO_DIST.get(mod, mod))
        if dist in core:
            continue
        if dist in VIA_CORE:
            if _norm(VIA_CORE[dist]) not in core:
                fails.append(f"{mod}: VIA_CORE names {VIA_CORE[dist]}, which "
                             f"is not a core dependency")
            continue
        if mod in OPTIONAL:
            extra = OPTIONAL[mod][0]
            if extra not in extras:
                fails.append(f"{mod}: OPTIONAL names extra `{extra}`, which "
                             f"pyproject does not define")
            for f, ln, is_top in sites:
                if is_top:
                    fails.append(f"{mod}: optional, but imported at module "
                                 f"top level in {f}:{ln}, so importing that "
                                 f"module requires the `{extra}` extra")
            continue
        where = ", ".join(f"{f}:{ln}" for f, ln, _ in sites[:3])
        fails.append(f"{mod} is imported ({where}) but is neither a core "
                     f"dependency nor declared in VIA_CORE or OPTIONAL")
    return fails


def main(argv):
    root = Path(argv[0]).resolve() if argv else HERE.parent
    fails = check(root)
    for msg in fails:
        print(f"  FAIL {msg}")
    print(f"RUNTIME IMPORTS ({root}): "
          + ("PASS" if not fails else f"FAIL, {len(fails)} problem(s)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
