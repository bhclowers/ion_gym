"""
The cache inventory (fa_cache.entries, the Cache tab's rows) says when an
entry's sidecar files cannot be read.

An unreadable provenance.json or meta.json was skipped with `pass`, so
the row looked exactly like an entry that predates descriptors, and the
Cache tab told the user it predated descriptors ("re-solve to describe")
about a corrupt file. A meta.json that parsed to a list crashed the inventory.
Each row now carries `problems`, naming the file and the error.
"""
import json

from ion_gym.io.fa_cache import entries


def test_unreadable_sidecars_are_named(tmp_path):
    bad = tmp_path / "badentry"
    bad.mkdir()
    (bad / "provenance.json").write_text("{not json", encoding="utf-8")
    (bad / "meta.json").write_text("[1, 2]", encoding="utf-8")
    good = tmp_path / "goodentry"
    good.mkdir()
    (good / "provenance.json").write_text(
        json.dumps([{"name": "deck", "label": "lbl"}]), encoding="utf-8")
    (good / "meta.json").write_text(
        json.dumps({"spec": {"name": "deck"},
                    "arrays": {"phi": {"shape": [3, 4]}}}), encoding="utf-8")
    rows = {r["key"]: r for r in entries(root=str(tmp_path))}
    probs = rows["badentry"]["problems"]
    assert any(p.startswith("provenance.json unreadable") for p in probs)
    assert any(p.startswith("meta.json unreadable") for p in probs)
    assert rows["goodentry"]["problems"] == []
    assert rows["goodentry"]["descriptor"] == "lbl"
    assert rows["goodentry"]["shape"] == [3, 4]
