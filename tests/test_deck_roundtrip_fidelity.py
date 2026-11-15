"""
Loading a deck and saving it keeps what the deck says.

Two losses: a deck that sets dt_ns to null (derive the step from the
drives) was saved with the derived number, so it stopped following its
drives; and the menu metadata (_display_name, _ui_hidden) was accepted on
load but dropped on save. The census test walks every shipped deck: the
save may ADD keys (defaults written out in full), but may drop or change
an authored value only through a listed normalisation.
"""
import json
import re
import warnings

import pytest

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec

EXAMPLES = repo_root() / "examples"
NULL_DT_DECK = EXAMPLES / "reflectron_tof_oa_refined_wm_matched.json"

# authored values the save may rewrite, each meaning the same thing:
# (kind, generic path) -> test on the authored value
NORMALISED = {
    # legacy per-electrode RF keys, migrated into rf_groups on load
    ("dropped", ".geometry.electrodes[].rf_group"): lambda v: True,
    ("dropped", ".geometry.electrodes[].rf_phase_deg"): lambda v: v == 0,
    # omitted when it equals the default ("all")
    ("dropped", ".collisions.ccs_fit_class"): lambda v: v == "all",
    # the legacy model alias
    ("changed", ".collisions.model"): lambda v: v == ("hs1", "hs"),
}
_MISSING = object()


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SimSpec.from_json(str(path))


def _saved(spec):
    return json.loads(json.dumps(spec.to_dict()))


def _walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            _walk(a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
    elif b is _MISSING:
        out.append(("dropped", path, a))
    elif a is not _MISSING and (a != b or type(a) is not type(b)):
        out.append(("changed", path, (a, b)))


def _decks():
    out = []
    for p in sorted(EXAMPLES.glob("*.json")):
        raw = json.loads(p.read_text())
        if isinstance(raw, dict) and "geometry" in raw:
            out.append(p)
    return out


@pytest.mark.parametrize("path", _decks(), ids=lambda p: p.stem)
def test_a_save_keeps_every_authored_value(path):
    raw = json.loads(path.read_text())
    diffs = []
    _walk(raw, _saved(_load(path)), "", diffs)
    lost = [(kind, p, v) for kind, p, v in diffs
            if not NORMALISED.get((kind, re.sub(r"\[\d+\]", "[]", p)),
                                  lambda v: False)(v)]
    assert not lost, lost


def test_a_null_step_is_saved_as_null_and_flies_derived():
    spec = _load(NULL_DT_DECK)
    assert spec.integration.dt_ns == pytest.approx(spec.derive_dt_ns())
    assert _saved(spec)["integration"]["dt_ns"] is None


def test_writing_the_same_step_back_keeps_it_derived():
    spec = _load(NULL_DT_DECK)
    spec.integration.dt_ns = float(spec.integration.dt_ns)   # app sync
    assert _saved(spec)["integration"]["dt_ns"] is None


def test_a_different_step_is_saved_as_stated():
    spec = _load(NULL_DT_DECK)
    spec.integration.dt_ns = 0.25
    assert _saved(spec)["integration"]["dt_ns"] == 0.25


def test_menu_metadata_survives_a_save():
    d = json.loads(NULL_DT_DECK.read_text())
    d["_display_name"], d["_ui_hidden"] = "Shown as", True
    out = _saved(SimSpec.from_dict(d))
    assert out["_display_name"] == "Shown as" and out["_ui_hidden"] is True
