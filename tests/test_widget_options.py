"""
test_widget_options.py -- a selector's value is always one of its options.

WHY THIS EXISTS. `pn.widgets.Select` does not re-point `value` when
`options` is reassigned; a value that is no longer in the list simply
stays there while the browser renders the first real option as though it
were chosen. Measured on v523, three controls were dead in exactly that
way -- Cache-tab export ("nothing selected" after its own refresh),
Cache-tab remove ("nothing to remove (refresh first)"), and Fields-tab
Load field ("no field selected -- pick one") with a matching field
listed in the picker.

Hand-repairing each call site is what produced the split: of the app's
22 option assignments some restored the value, some restored it only
when it was still valid (leaving the invalid case -- the one that
matters -- unhandled), and some did nothing at all. So the rule lives in
ion_gym.ui.widget_options.set_options, and this gate asserts the
resulting invariant rather than checking 22 call sites one at a time:
after every refresh the app offers, every selector it owns holds a value
its own options contain. That single assertion covers all 22 and any
site added later.

Run:  pytest tests/test_widget_options.py     (pytest module, not a script)
"""
import pytest

import panel as pn

from ion_gym.io.paths import repo_root
from ion_gym.io.sim_spec import SimSpec
from ion_gym.ui.widget_options import (set_options, selectable_values,
                                       _selector_classes)

# The app's own refresh entry points -- the methods that repopulate a
# selector. Named rather than discovered so a rename SURFACES here
# instead of silently reducing coverage to nothing (a gate that quietly
# exercises zero code still reports PASS).
REFRESHERS = ("_refresh_cache_tab", "_refresh_field_picker",
              "_sync_planes", "_sync_analysis_mz",
              "_station_sync_pick", "_clear_assembly_state")

# The shipped r-z gate fixture: a real deck, so no geometry numbers are
# invented here. Nothing in this module solves -- repopulating a picker
# reads the spec and the cache inventory, never the solver.
FIXTURE = "tests/fixtures/einzel_rz_gate_fixture.json"


def _app():
    from ion_gym.ui.sim_app import SimApp
    app = SimApp(SimSpec.from_json(str(repo_root() / FIXTURE)))
    app.panel()          # forces the lazily-built tabs (Cache among them)
    return app


def _selectors(app):
    """Every option-constrained selector the app holds, by attribute name."""
    return [(n, w) for n, w in vars(app).items()
            if isinstance(w, _selector_classes())]


def _violations(app):
    """Selectors whose value is not among their own options.

    Returned rather than asserted so the mutation check below can prove
    this detector actually fires on the defect it is written to catch.
    """
    bad = []
    for name, w in _selectors(app):
        vals = selectable_values(w.options)
        held = w.value if isinstance(w.value, list) else [w.value]
        for v in held:
            if v is None and not vals:
                continue          # empty selector holding nothing: valid
            if v not in vals:
                bad.append(f"{name}: value {v!r} not in options {vals[:4]}")
    return bad


# ----------------------------------------------------------- set_options
def test_preserves_a_surviving_selection():
    w = pn.widgets.Select(options=["a", "b", "c"], value="b")
    set_options(w, ["c", "b", "a"])
    assert w.value == "b"


def test_falls_to_first_when_the_selection_dies():
    # the reported defect: value stayed '(refresh first)', a string absent
    # from the new options, and every reader saw "nothing selected"
    w = pn.widgets.Select(options=["(refresh first)"])
    set_options(w, ["entry-1", "entry-2"])
    assert w.value == "entry-1"


def test_populating_an_empty_selector_selects_something():
    # w_fieldpick's case: constructed options=[] so value was None, and
    # None survived every later repopulation
    w = pn.widgets.Select(options=[])
    set_options(w, {"label A": "/tmp/a.npz", "label B": "/tmp/b.npz"})
    assert w.value == "/tmp/a.npz"      # dict -> the VALUES are selectable


def test_emptying_a_selector_clears_its_value():
    w = pn.widgets.Select(options=["x", "y"], value="y")
    set_options(w, [])
    assert w.value is None


def test_prefer_beats_a_surviving_selection():
    w = pn.widgets.Select(options=["a", "b"], value="a")
    set_options(w, ["a", "b", "new"], prefer="new")
    assert w.value == "new"


def test_prefer_absent_from_options_is_ignored_not_forced():
    # a caller asking for something the list does not contain is asking
    # for a state the widget cannot represent; it must not be written
    w = pn.widgets.Select(options=["a", "b"], value="b")
    set_options(w, ["a", "b"], prefer="nonexistent")
    assert w.value == "b"


def test_multi_value_selector_keeps_only_surviving_entries():
    w = pn.widgets.MultiChoice(options=["115", "322", "622"],
                               value=["322", "622"])
    set_options(w, ["115", "322"])
    assert w.value == ["322"]


def test_value_is_not_rewritten_when_it_need_not_change():
    """A repopulation that changes nothing must not fire watchers."""
    w = pn.widgets.Select(options=["a", "b"], value="a")
    fired = []
    w.param.watch(lambda e: fired.append(e), "value")
    set_options(w, ["a", "b"])
    assert w.value == "a"
    assert fired == []


# --------------------------------------------------- adversarial refusal
def test_refuses_an_unsupported_widget_by_name():
    """AutocompleteInput carries `options` but is a free-text field whose
    empty value is legitimate; forcing it to options[0] would invent a
    selection the user never made. The guard must refuse, and must name
    what it knows so the message is actionable."""
    w = pn.widgets.AutocompleteInput(options=["alpha", "beta"])
    with pytest.raises(TypeError) as exc:
        set_options(w, ["alpha", "beta"])
    msg = str(exc.value)
    assert "AutocompleteInput" in msg
    assert "Select" in msg          # names the known set, not just "bad type"
    assert w.value == ""            # and changed nothing on the way out


# ------------------------------------------------- the invariant, in situ
def test_every_refresh_leaves_every_selector_valid():
    app = _app()
    assert _selectors(app), "no selectors found -- the walk is broken"
    assert not _violations(app), (
        f"invalid before any refresh: {_violations(app)}")
    for meth in REFRESHERS:
        fn = getattr(app, meth, None)
        assert fn is not None, (
            f"{meth} is gone -- this gate's coverage is only as good as "
            f"this list; re-point it at the method that replaced it")
        fn()
        bad = _violations(app)
        assert not bad, f"after {meth}(): " + "; ".join(bad)


def test_the_detector_fires_on_the_defect_it_is_written_for():
    """MUTATION CHECK. A green invariant test proves nothing unless the
    detector fails on the broken case. Reproduce the v523 defect exactly
    -- a raw `.options =` assignment, no re-point -- and assert
    _violations reports it."""
    app = _app()
    name, w = next((n, x) for n, x in _selectors(app)
                   if isinstance(x, pn.widgets.Select))
    w.options = ["__not_the_held_value__"]        # the defect, verbatim
    bad = _violations(app)
    assert any(name in b for b in bad), (
        f"the detector missed a raw options assignment on {name} -- it "
        f"would not have caught the shipped defect")
