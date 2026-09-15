"""
PYTEST BRIDGE.

The battery is MIXED: ~35 script-style gates (check()/main(), explicit
ALL-PASS lines, exit codes) plus pytest-native files. Bare `pytest
tests/` used to collect ONLY the pytest subset and report green while
silently skipping most of the battery — the tw2d-Gate-B failure class
at suite scale. This bridge closes that hole: every `__main__`-guarded,
non-pytest gate is discovered automatically and subprocess-run as its
own pytest case, so

    python -m pytest tests/          # runs EVERYTHING, one exit code
    python -m pytest tests/ -k lint  # select as usual

Script gates stay runnable standalone (`python tests/test_x.py`) — the
bridge subprocess-runs them exactly that way, so there is ONE execution
convention per gate, not two. Discovery is dynamic: a new script gate
is bridged the day it lands; a gate that grows pytest functions drops
out of the bridge automatically (pytest then owns it natively).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TIMEOUT_S = 900


def _script_gates():
    out = []
    for p in sorted(HERE.glob("test_*.py")):
        if p.name == Path(__file__).name:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if "__main__" not in src:
            continue                 # module-level files: pytest imports run them
        if re.search(r"^def test_", src, re.M):
            # pytest collects any top-level `def test_*` whether or not
            # the file imports pytest (the old import-based
            # heuristic double-ran test_stl3d — once collected, once as
            # a bridged script — against this bridge's own ONE-execution
            # bridge). Files with collectible tests are pytest's; the
            # bridge carries only pure script gates.
            continue
        out.append(p.name)
    return out


@pytest.mark.parametrize("gate", _script_gates())
def test_script_gate(gate):
    env = dict(os.environ)
    r = subprocess.run([sys.executable, str(HERE / gate)], cwd=ROOT,
                       env=env, capture_output=True, text=True,
                       timeout=TIMEOUT_S)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + "\n" + r.stderr).splitlines()[-25:])
        pytest.fail(f"{gate} exited {r.returncode}\n--- tail ---\n{tail}",
                    pytrace=False)
