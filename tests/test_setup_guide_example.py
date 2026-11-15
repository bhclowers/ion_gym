"""
The worked example in docs/SETUP_GUIDE.md runs as written.

The guide's only Python example once could not run (review NU-04): it
passed a symmetry string the constructor kept until to_json failed, used
the deprecated P_pa, and left the electrodes as a literal "...". This
executes every python block under "## Worked example" in a scratch
directory, so the guide cannot drift from the API again.
"""
import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "SETUP_GUIDE.md"


def _worked_example_blocks():
    text = GUIDE.read_text(encoding="utf-8")
    start = text.index("## Worked example")
    nxt = text.find("\n## ", start + 1)
    section = text[start:] if nxt < 0 else text[start:nxt]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    if not blocks:
        raise AssertionError(f"{GUIDE}: no python block under "
                             f"'## Worked example'")
    return blocks


def test_the_worked_example_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    for i, block in enumerate(_worked_example_blocks()):
        exec(compile(block, f"{GUIDE.name}[block {i}]", "exec"),
             {"__name__": "__main__"})
    out = capsys.readouterr().out
    assert "builder='rz'" in out
    assert (tmp_path / "drift_cell.json").is_file()


def test_the_guide_teaches_torr_not_pa():
    text = GUIDE.read_text(encoding="utf-8")
    code = "\n".join(_worked_example_blocks())
    assert "P_pa" not in code
    assert "mz_list` holds ion masses in Da" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
