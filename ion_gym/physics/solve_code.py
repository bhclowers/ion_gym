"""
solve_code.py -- the identity of the code that produces a solved field.

A cached basis is reused whenever its key matches. The key used to
describe only the geometry and some version markers set by hand, so a fix
to the solver or a rasterizer reached an already-cached geometry only if
someone remembered to bump a marker; more than one fix silently did
nothing that way, and new geometries got the fix while cached ones did
not. solve_code_key() hashes the code on the solve path, and the field
caches put it in their keys: any change to that code gives every cached
field a new key, and each re-solves once, on first use.

What is hashed:
  * SOLVE_MODULES, whole: the solvers, rasterizers and the grid helpers
    they read.
  * SOLVE_FUNCTIONS: the solve part of each builder (the rest of a builder
    flies ions, and editing it must not re-solve fields), together with
    the module-level constants those functions read.
Code is compared as a syntax tree with docstrings removed, so a comment
or docstring edit changes nothing; any code change does. A listed module
or function that no longer exists raises, so a rename cannot drop code
from the key unnoticed (tests/test_solve_code_key.py also checks that the
listed functions call no unlisted builder function).

The key is NOT part of a field's geometry identity: a field file saved
by another version still matches its geometry (field_io records the key
it was solved with and says when it differs).
"""
import ast
import hashlib
from functools import lru_cache
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent          # the ion_gym package

SOLVE_MODULES = (
    "physics/solver2d.py", "physics/solver3d.py", "physics/multigrid3d.py",
    "physics/raster2d.py", "physics/rasterize3d.py", "physics/voxelize.py",
    "physics/scene3d.py", "physics/symmetry.py",
    "io/lattice.py", "io/stl_resolve.py",
)
SOLVE_FUNCTIONS = {
    "physics/build_planar.py": ("build_planar_model",
                                "_assert_planar_fold_symmetry"),
    "physics/build_rz.py": ("build_rz_model",),
    "physics/build_stl3d.py": ("_grid_from_spec", "stl_masks_3d",
                               "_declared_mirror_axes", "_solve_stl3d_bases"),
    "physics/build_shapes3d.py": ("_inplane_axes", "_shape_volume",
                                  "electrode_mask_3d", "shapes_masks_3d"),
    "physics/build_scene3d.py": ("scene_of", "scene_masks_3d"),
}


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _module_part(source):
    """The whole module, as a docstring-free syntax tree."""
    return _dump(_strip_docstrings(ast.parse(source)))


def _functions_part(source, names, where):
    """The named top-level functions, and the module-level assignments of
    every name they read, as docstring-free syntax trees."""
    tree = _strip_docstrings(ast.parse(source))
    top = {n.name: n for n in tree.body
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [n for n in names if n not in top]
    if missing:
        raise LookupError(
            f"solve_code: {where} has no top-level function(s) {missing}; "
            f"update SOLVE_FUNCTIONS so the field-cache key still covers "
            f"the solve path")
    parts, read = [], set()
    for name in names:
        fn = top[name]
        parts.append(_dump(fn))
        read |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign) else
                   [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id in read for t in targets):
            parts.append(_dump(node))
    return "\n".join(parts)


def key_of_sources(modules, functions):
    """16 hex digits over {path: source} for whole modules and
    {path: (source, names)} for builders (the arithmetic of
    solve_code_key, open to tests)."""
    h = hashlib.sha256()
    for path in sorted(modules):
        h.update(path.encode())
        h.update(_module_part(modules[path]).encode())
    for path in sorted(functions):
        source, names = functions[path]
        h.update(path.encode())
        h.update(_functions_part(source, names, path).encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def solve_code_key():
    """The key of the solve-path code in this installation."""
    modules = {p: (PKG / p).read_text(encoding="utf-8")
               for p in SOLVE_MODULES}
    functions = {p: ((PKG / p).read_text(encoding="utf-8"), names)
                 for p, names in SOLVE_FUNCTIONS.items()}
    return key_of_sources(modules, functions)
