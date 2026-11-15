"""
GATE IMPORT ORDER: a script gate must put the repo root on sys.path
before its first ion_gym import.

A script gate is run as `python tests/<gate>.py`, so sys.path[0] is
tests/, not the repo root. If it imports ion_gym before `import
_bootstrap` (or its own sys.path.insert of the root), Python finds
whatever ion_gym is INSTALLED -- a wheel in site-packages, or an
editable install of a different checkout -- and the gate certifies that
tree instead of the one it lives in, with nothing in its output to say
so. Measured: test_pe_view and test_pe_surface_tab imported ion_gym
from another checkout's editable install and reported on its code.

Which files are script gates follows the runner's structural rule (a
test_*.py with no `def test_` in it, plus every validate_*.py), the
same rule conftest.py and run_gates.py each state. It is restated here
rather than imported, because no test module may import another
(tools/dependency_audit.py, K9). pytest modules are out of scope:
conftest.py puts the root on sys.path before collection.

Run: python tests/test_gate_import_order.py
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _imports_ion_gym(node):
    if isinstance(node, ast.Import):
        return any(a.name == "ion_gym" or a.name.startswith("ion_gym.")
                   for a in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.level == 0 and node.module is not None
                and (node.module == "ion_gym"
                     or node.module.startswith("ion_gym.")))
    return False


def _roots_path(node):
    """`import _bootstrap`, or a top-level `sys.path.insert(...)` call."""
    if isinstance(node, ast.Import):
        return any(a.name == "_bootstrap" for a in node.names)
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        f = node.value.func
        return (isinstance(f, ast.Attribute) and f.attr == "insert"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "path"
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == "sys")
    return False


def check(path):
    """None when the order is right, else a message naming the lines."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    first_ig = next((n.lineno for n in tree.body if _imports_ion_gym(n)),
                    None)
    if first_ig is None:
        return None            # imports no ion_gym at top level: nothing to order
    first_root = next((n.lineno for n in tree.body if _roots_path(n)), None)
    if first_root is None:
        return (f"line {first_ig} imports ion_gym and nothing at top level "
                f"puts the repo root on sys.path")
    if first_ig < first_root:
        return (f"line {first_ig} imports ion_gym before line {first_root} "
                f"puts the repo root on sys.path")
    return None


def script_gates():
    """The runner's structural rule (run_gates.script_gates)."""
    out = []
    for p in sorted(list(HERE.glob("test_*.py"))
                    + list(HERE.glob("validate_*.py"))):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if not any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
                   for n in ast.walk(tree)):
            out.append(p)
    return out


def main():
    gates = script_gates()
    if not gates:
        print("GATE IMPORT ORDER: FAIL -- no script gates found; refusing to "
              "pass an empty check")
        return 1
    bad = [(p.name, msg) for p in gates if (msg := check(p)) is not None]
    for name, msg in bad:
        print(f"  FAIL {name}: {msg}")
    print(f"GATE IMPORT ORDER: checked {len(gates)} script gates, "
          f"PASSED {len(gates) - len(bad)}   FAILED {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
