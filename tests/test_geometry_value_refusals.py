"""
validate() names geometry and mass values that cannot describe a flight
(review F7). Each used to surface far downstream with an error pointing
elsewhere: no electrodes -> a bare StopIteration, a zero width -> "Factor
is exactly singular", a zero mass -> ZeroDivisionError.
"""
import pytest

from ion_gym.io.sim_spec import reflectron_vacuum_spec

CASES = [
    ("geometry.electrodes is empty",
     lambda s: setattr(s.geometry, "electrodes", [])),
    ("geometry.width_mm = 0.0", lambda s: setattr(s.geometry, "width_mm",
                                                  0.0)),
    ("geometry.height_mm = -1.0", lambda s: setattr(s.geometry, "height_mm",
                                                    -1.0)),
    ("geometry.depth_mm = -2.0", lambda s: setattr(s.geometry, "depth_mm",
                                                   -2.0)),
    ("source.mz_list holds [0.0]", lambda s: setattr(s.source, "mz_list",
                                                     [0.0])),
    ("source.mz_list holds [-100.0]",
     lambda s: setattr(s.source, "mz_list", [100.0, -100.0])),
]


@pytest.mark.parametrize("said,mutate", CASES, ids=[c[0] for c in CASES])
def test_refused_by_name(said, mutate):
    s = reflectron_vacuum_spec()
    mutate(s)
    assert any(e.startswith(said) for e in s.validate()), s.validate()


def test_a_good_deck_is_untouched():
    from ion_gym.io.paths import repo_root
    from ion_gym.io.sim_spec import SimSpec
    s = SimSpec.from_json(str(repo_root() / "examples/ion_funnel_rz.json"))
    errs = s.validate()
    assert not [e for e in errs if e.startswith(("geometry.", "source.mz"))]


def test_the_electrode_free_skeleton_says_so():
    errs = reflectron_vacuum_spec().validate()
    assert "geometry.electrodes is empty -- a field needs at least one " \
           "electrode" in errs
