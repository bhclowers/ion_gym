"""
A refused editor edit is rolled back.

The editor queued an edit, then checked it with verify_commit(); a
refusal said "REFUSED" but left the op queued, so a later save wrote it.
EditSession.verified() makes edit + verify one step: on refusal every op
the edit queued is removed and the redo stack restored.
"""
import pytest

from ion_gym.edit.outline import shift_shape
from ion_gym.edit.policy import EditRefusal
from ion_gym.edit.session import EditSession
from ion_gym.io.paths import repo_root

DECK = repo_root() / "examples" / "ion_funnel_rz.json"


def _session():
    s = EditSession.load(DECK)
    first = next(i for i, e in enumerate(s.committed_electrodes())
                 if e.get("shapes"))
    return s, first


def test_a_refused_verification_leaves_nothing_queued(monkeypatch):
    s, el = _session()
    before = s.to_document_bytes()
    shape = s.committed_electrodes()[el]["shapes"][0]

    def _refuse():
        raise EditRefusal("verification refused (staged)")
    monkeypatch.setattr(s, "verify_commit", _refuse)
    with pytest.raises(EditRefusal, match="staged"):
        s.verified(lambda: s.set_shape(el, 0, shift_shape(shape, dx=0.5)))
    assert s.to_document_bytes() == before
    with pytest.raises(EditRefusal, match="nothing to undo"):
        s.undo_last()


def test_a_refused_edit_keeps_the_redo_stack(monkeypatch):
    s, el = _session()
    shape = s.committed_electrodes()[el]["shapes"][0]
    s.verified(lambda: s.set_shape(el, 0, shift_shape(shape, dx=0.5)))
    s.undo_last()                         # one op on the redo stack
    monkeypatch.setattr(s, "verify_commit",
                        lambda: (_ for _ in ()).throw(EditRefusal("no")))
    with pytest.raises(EditRefusal):
        s.verified(lambda: s.set_shape(el, 0, shift_shape(shape, dx=1.0)))
    monkeypatch.undo()
    assert s.redo_last()["op"] == "set"         # the undone edit, still redoable


def test_an_accepted_edit_stays(tmp_path):
    s, el = _session()
    before = s.to_document_bytes()
    shape = s.committed_electrodes()[el]["shapes"][0]
    rep = s.verified(lambda: s.set_shape(el, 0, shift_shape(shape, dx=0.5)))
    assert rep["geometry_key_changed"] is True
    assert s.to_document_bytes() != before
