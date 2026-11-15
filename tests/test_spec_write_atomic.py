"""
SimSpec.to_json(path) never leaves a truncated deck (review F8).

It wrote in place, so a write that failed part-way (full disk, killed
process) left a half-written file under the good deck's name: the error
was raised, but the deck was gone. The review stood RLIMIT_FSIZE in for a
full disk; so does this test, in a subprocess.
"""
import json
import os
import stat
import subprocess
import sys

import pytest

from ion_gym.io.sim_spec import SimSpec, reflectron_vacuum_spec

CODE = r'''
import json, resource, signal, sys
sys.path.insert(0, {root!r})
from ion_gym.io.sim_spec import reflectron_vacuum_spec
spec = reflectron_vacuum_spec()
spec.notes = "x" * 20000                   # bigger than the limit below
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (4000, 4000))
try:
    spec.to_json({path!r})
except OSError as e:
    print("RAISED", type(e).__name__)
else:
    print("WROTE")
'''


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE is POSIX")
def test_a_failed_write_keeps_the_previous_deck(tmp_path):
    deck = tmp_path / "deck.json"
    good = reflectron_vacuum_spec()
    good.to_json(str(deck))
    before = deck.read_text(encoding="utf-8")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c", CODE.format(root=root, path=str(deck))],
        capture_output=True, text=True, timeout=120)
    assert "RAISED" in out.stdout, (out.stdout, out.stderr[-1500:])
    assert deck.read_text(encoding="utf-8") == before
    SimSpec.from_json(str(deck))                 # still loads
    assert [p.name for p in tmp_path.iterdir()] == ["deck.json"]


def test_a_rewrite_keeps_the_file_mode(tmp_path):
    deck = tmp_path / "deck.json"
    reflectron_vacuum_spec().to_json(str(deck))
    os.chmod(deck, 0o640)
    reflectron_vacuum_spec().to_json(str(deck))
    assert stat.S_IMODE(deck.stat().st_mode) == 0o640
    assert json.loads(deck.read_text(encoding="utf-8"))["name"]
