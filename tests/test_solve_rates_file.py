"""
Rewriting the solve-rate records (sizing._store_rates) over an unreadable
file says so.

The store read the existing file to keep other machines' records; an
unreadable or non-mapping file was replaced by {} silently, so every
other machine's records vanished with no message. It is still replaced
(the file is broken), but by name.
"""
import json

import pytest

from ion_gym.physics import sizing


@pytest.mark.parametrize("content,said", [
    ("{not json", "unreadable"),
    ("[1, 2, 3]", "not a mapping"),
], ids=["corrupt", "list"])
def test_rewriting_a_broken_rates_file_is_announced(content, said, tmp_path,
                                                    monkeypatch, capsys):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    (tmp_path / sizing.SOLVE_RATES_FILE).write_text(content,
                                                    encoding="utf-8")
    monkeypatch.setattr(sizing, "_RATES", {"lu": [], "mg": []})
    sizing._store_rates()
    out = capsys.readouterr().out
    assert said in out and sizing.SOLVE_RATES_FILE in out
    stored = json.loads((tmp_path / sizing.SOLVE_RATES_FILE).read_text())
    assert isinstance(stored, dict) and len(stored) == 1


def test_a_good_file_keeps_other_machines(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ION_GYM_CACHE", str(tmp_path))
    (tmp_path / sizing.SOLVE_RATES_FILE).write_text(
        json.dumps({"other-machine": {"lu": [[1, 2]], "mg": []}}),
        encoding="utf-8")
    monkeypatch.setattr(sizing, "_RATES", {"lu": [], "mg": []})
    sizing._store_rates()
    assert capsys.readouterr().out == ""
    stored = json.loads((tmp_path / sizing.SOLVE_RATES_FILE).read_text())
    assert "other-machine" in stored and len(stored) == 2
