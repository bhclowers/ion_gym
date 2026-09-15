"""conftest.py -- put the repo ROOT on sys.path for every gate in this folder.

The gates live in tests/ so a release can drop the folder wholesale.  That only
works because nothing in the core imports anything from here: the dependency
arrow points tests -> core, never the reverse.

Rooting on __file__, not cwd: a gate must give the same answer from anywhere.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- two kinds of file wear the test_ prefix, and only one is pytest's ------
# Some test_*.py here are pytest modules (they define `def test_...`).  Others
# are STANDALONE SCRIPTS that run on import and sys.exit() with a pass/fail
# count -- test_ui_smoke, test_polygon_raster, test_scene_resolution,
# test_ele_labels.  In the old flat layout that never mattered, because they
# were only ever invoked one file at a time.  Collecting a folder surfaces it:
# pytest imports the script, the script calls sys.exit, and the run dies with
# INTERNALERROR before a single test executes.
#
# Detected STRUCTURALLY, not from a hardcoded list: a hardcoded list rots the
# first time someone adds a script, and it would fail exactly the way this did.
# A file with no `def test_` in it is not a pytest module.  It is still a gate
# and is still run -- as a script, by run_gates.py.
import ast

collect_ignore = []
for _p in Path(__file__).parent.glob("test_*.py"):
    _tree = ast.parse(_p.read_text())
    _has = any(isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
               for n in ast.walk(_tree))
    if not _has:
        collect_ignore.append(_p.name)
