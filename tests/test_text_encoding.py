"""
Text files are read and written as UTF-8, whatever the system locale.

About forty reads and writes used the platform's default encoding: a deck
named "einzel 5 µm Å" failed with UnicodeDecodeError under an ASCII
locale, and became mojibake under Windows' older cp1252 default. Every
text read and write in the package now states encoding="utf-8"; this
gate keeps it that way.
"""
import ast
import json
import os
import subprocess
import sys

from ion_gym.io.paths import repo_root

PKG = repo_root() / "ion_gym"


def _bare_text_io(path):
    out = []
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(n, ast.Call) or "encoding" in {
                k.arg for k in n.keywords}:
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in ("read_text",
                                                       "write_text"):
            out.append(n.lineno)
        elif isinstance(f, ast.Name) and f.id == "open":
            mode = (n.args[1].value if len(n.args) > 1
                    and isinstance(n.args[1], ast.Constant) else None)
            for k in n.keywords:
                if k.arg == "mode" and isinstance(k.value, ast.Constant):
                    mode = k.value.value
            if mode is None or "b" not in str(mode):
                out.append(n.lineno)
    return out


def test_every_text_read_and_write_states_its_encoding():
    bare = {str(p.relative_to(PKG)): _bare_text_io(p)
            for p in sorted(PKG.rglob("*.py"))}
    bare = {k: v for k, v in bare.items() if v}
    assert not bare, f"text I/O without encoding=: {bare}"


def test_a_non_ascii_deck_loads_and_saves_under_an_ascii_locale(tmp_path):
    d = json.loads((repo_root() / "examples" / "einzel_round_r-z.json")
                   .read_text(encoding="utf-8"))
    d["name"] = "einzel 5 µm Å"
    deck = tmp_path / "deck.json"
    deck.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    code = (
        "import sys, warnings; warnings.simplefilter('ignore');"
        f"sys.path.insert(0, {str(repo_root())!r});"
        "from ion_gym.io.sim_spec import SimSpec;"
        f"s = SimSpec.from_json({str(deck)!r});"
        f"s.to_json({str(tmp_path / 'out.json')!r});"
        "sys.stdout.buffer.write(s.name.encode('utf-8'))")
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0",
               PYTHONCOERCECLOCALE="0", PYTHONIOENCODING="ascii")
    run = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True)
    assert run.returncode == 0, run.stderr.decode("utf-8", "replace")[-800:]
    assert run.stdout.decode("utf-8") == "einzel 5 µm Å"
    saved = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert saved["name"] == "einzel 5 µm Å"
