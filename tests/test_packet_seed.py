"""
A staged packet takes a seed, so a gas packet can be repeated
independently.

fly_packet seeded ion i with i + 1 and had no seed input, so every repeat
of a staged gas flight was identical. It now takes seed= (or the beam's
"seed"); base 0 keeps the old seeds exactly, so an unseeded packet is
unchanged, and any other base gives independent streams.
"""
import pytest

from ion_gym.io.paths import repo_root
from ion_gym.physics.staged_flight import (fly_packet, load_assembly,
                                           packet_ion_seed)

DEMO = repo_root() / "examples" / "funnel_hexapole" / "instrument.json"


@pytest.fixture(scope="module")
def slice2():
    regions, beam = load_assembly(str(DEMO))
    small = dict(beam, ions=beam["ions"][:2],
                 mz_per_ion=list(beam["mz_per_ion"][:2]))
    small.pop("seed", None)
    return regions, small


def _times(out):
    return [(p.get("i"), p.get("t_us"), p.get("x"), p.get("why"))
            for p in out["per_ion"]]


def test_base_zero_is_the_old_seeding():
    assert [packet_ion_seed(0, i) for i in range(4)] == [1, 2, 3, 4]
    a = {packet_ion_seed(5, i) for i in range(50)}
    b = {packet_ion_seed(6, i) for i in range(50)}
    assert len(a) == 50 and not a & b


def test_an_unseeded_packet_repeats_and_another_base_differs(slice2):
    regions, beam = slice2
    first = fly_packet(regions, beam, check_seams=False)
    again = fly_packet(regions, beam, check_seams=False)
    zero = fly_packet(regions, beam, check_seams=False, seed=0)
    other = fly_packet(regions, beam, check_seams=False, seed=7)
    assert first["seed"] == 0 and other["seed"] == 7
    assert _times(first) == _times(again) == _times(zero)
    assert _times(other) != _times(first)        # gas: a new stream


def test_the_beam_can_carry_the_seed(slice2):
    regions, beam = slice2
    out = fly_packet(regions, dict(beam, seed=7), check_seams=False)
    ref = fly_packet(regions, beam, check_seams=False, seed=7)
    assert out["seed"] == 7 and _times(out) == _times(ref)
