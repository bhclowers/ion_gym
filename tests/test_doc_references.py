"""
The README and docs/ name only things a user's checkout has.

The module map cited tests and a module that do not ship (test_W3d,
edit_gate, quad_scene_build), UI_REFERENCE named a regeneration tool
that does not ship, and 55 modules were missing from the map. Checked:

  * every backticked test_*/validate_* name is a file in tests/;
  * every backticked repository path exists (paths under outputs/ are
    written at run time and are exempt);
  * every module named in a MODULES.md table exists, and every public
    module (not underscore-prefixed) is named somewhere in MODULES.md;
  * SimSpec.from_json on the shipped staged assembly names what it is
    and how to load it, instead of blaming a missing block.
"""
import re

import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec

ROOT = repo_root()
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
TICK = re.compile(r"`([^`\n]+)`")
PATHY = re.compile(r"^[\w./-]+\.(py|md|json|ipynb|dat|txt|toml|stl)$")
RUNTIME_DIRS = ("outputs/",)


def _modules():
    return {p.stem for p in (ROOT / "ion_gym").rglob("*.py")
            if "vendor" not in p.parts}


def _ticks():
    for doc in DOCS:
        for i, line in enumerate(doc.read_text(encoding="utf-8")
                                 .splitlines(), 1):
            for t in TICK.findall(line):
                yield doc.name, i, t.strip()


def test_cited_tests_ship():
    tests = {p.stem for p in (ROOT / "tests").glob("*.py")}
    bad = [(d, i, t) for d, i, t in _ticks()
           if re.fullmatch(r"(test|validate)_\w+", t) and t not in tests]
    assert not bad, f"docs cite tests that do not ship: {bad}"


def test_cited_paths_exist():
    bad = [(d, i, t) for d, i, t in _ticks()
           if PATHY.match(t) and "/" in t and "*" not in t
           and not t.startswith(RUNTIME_DIRS)
           and not ((ROOT / t).exists() or (ROOT / "ion_gym" / t).exists())]
    assert not bad, f"docs cite paths that do not exist: {bad}"


def test_module_map_names_real_modules_and_all_of_them():
    mods = _modules()
    pkgs = {p.name for p in (ROOT / "ion_gym").iterdir() if p.is_dir()}
    text = (ROOT / "docs" / "MODULES.md").read_text(encoding="utf-8")
    in_tables = set()
    for line in text.splitlines():
        if line.startswith("| `"):
            in_tables |= set(re.findall(r"[A-Za-z_]\w*",
                                        line.split("|")[1].replace("`", " ")))
    unknown = sorted(n for n in in_tables - pkgs - {"py"} if n not in mods)
    assert not unknown, f"MODULES.md tables name missing modules: {unknown}"
    named = set(re.findall(r"`([A-Za-z_]\w*)", text))
    named |= set(re.findall(r"[A-Za-z_]\w*", text))
    missing = sorted(m for m in mods if not m.startswith("_")
                     and m not in named and m not in pkgs)
    assert not missing, f"modules absent from MODULES.md: {missing}"


def test_from_json_names_a_staged_assembly():
    with pytest.raises(ValueError, match="staged assembly") as e:
        SimSpec.from_json(str(ROOT / "examples" / "funnel_hexapole"
                              / "instrument.json"))
    assert "staged_flight.load_assembly" in str(e.value)
    assert "no 'source' block" not in str(e.value)
