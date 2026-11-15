"""
The public API is declared, and the declaration matches the code.

ion_gym had __all__ in 6 of 104 modules and no statement of what was
public; `from ion_gym.io.sim_spec import *` exported json, math, Path,
Optional and the rest of its imports; the package docstring named an
`examples` subpackage that does not import. docs/PUBLIC_API.md now lists
the public modules, and each declares its names in __all__. Checked:

  * every module the document lists imports and has __all__, and every
    name in it exists;
  * a star import gives exactly __all__ (no imported helpers leak out);
  * every subpackage the package docstring names imports;
  * every command the document shows is a command of the CLI;
  * no public function reaches into the development tree, which no
    installed package has (study_outputs_dir accepted only paths there
    and was listed as public).
"""
import ast
import importlib
import inspect
import re
import textwrap

import pytest

import ion_gym
from ion_gym import cli
from ion_gym.io.paths import repo_root

DOC = repo_root() / "docs" / "PUBLIC_API.md"
MODULES = re.findall(r"^\| `(ion_gym(?:\.\w+)+)` \|", DOC.read_text(),
                     flags=re.M)


def test_the_document_lists_modules():
    assert len(MODULES) >= 10, MODULES


@pytest.mark.parametrize("name", MODULES)
def test_each_listed_module_declares_its_names(name):
    mod = importlib.import_module(name)
    declared = getattr(mod, "__all__", None)
    assert declared, f"{name} has no __all__"
    assert len(declared) == len(set(declared)), "a name is listed twice"
    missing = [n for n in declared if not hasattr(mod, n)]
    assert not missing, f"{name}.__all__ names what it lacks: {missing}"


@pytest.mark.parametrize("name", MODULES)
def test_a_star_import_gives_exactly_the_public_names(name):
    ns = {}
    exec(f"from {name} import *", ns)
    got = {k for k in ns if k != "__builtins__"}
    assert got == set(importlib.import_module(name).__all__)


def test_the_spec_module_no_longer_exports_its_imports():
    ns = {}
    exec("from ion_gym.io.sim_spec import *", ns)
    assert not {"json", "math", "Path", "Optional", "field", "asdict"} & set(ns)


def test_every_subpackage_the_docstring_names_imports():
    m = re.search(r"Subpackages \(([^)]*)\)", ion_gym.__doc__)
    names = [s.strip() for s in m.group(1).split(",")]
    assert names
    for sub in names:
        importlib.import_module(f"ion_gym.{sub}")


@pytest.mark.parametrize("name", MODULES)
def test_no_public_function_needs_the_development_tree(name):
    mod = importlib.import_module(name)
    bad = []
    for n in mod.__all__:
        obj = getattr(mod, n)
        if not inspect.isfunction(obj) or obj.__module__ != name:
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        if any(isinstance(c, ast.Constant) and c.value == "internal"
               for c in ast.walk(tree)):
            bad.append(n)
    assert not bad, f"{name}: public functions using the dev tree: {bad}"


def test_every_documented_command_exists():
    cmds = set(re.findall(r"^ion-gym (\w+)", DOC.read_text(), flags=re.M))
    assert {"dashboard", "validate", "edit"} <= cmds
    for c in cmds:
        with pytest.raises(SystemExit) as e:
            cli.main([c, "--help"])
        assert e.value.code == 0
