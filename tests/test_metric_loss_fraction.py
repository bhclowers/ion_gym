"""Gate: metric_loss_fraction honors the deck's own transmitted fate.

Regression for a defect: the metric called _transmitted
without the spec, so the fate-1 literal was used, and on any deck whose
transmission is another fate (bounding-plane decks are fate 3) every
candidate scored 1.0, CMA saw flat fitness, and the search stopped on an
arbitrary point. The audit/flown-check contradiction exposed it. This
test evaluates the metric on a fully transmitting non-fate-1 deck and
refuses the 1.0 score that the broken path returns.
"""
from pathlib import Path

from ion_gym.io.sim_spec import SimSpec
from ion_gym.physics.optimize import OptimizeSpec, _evaluate
from ion_gym.physics.stats import auto_transmitted_fate

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "examples" / "einzel_round_r-z.json"  # vacuum, fast, and its
#     transmission is a declared-plane fate, exactly the class the broken
#     metric scored as total loss
LENS_PATH = "geometry.electrodes[1].dc"


def test_loss_fraction_respects_deck_fate():
    spec = SimSpec.from_json(str(DECK))
    spec.source.seed = 0       # the deck declares seed null; the gate
    #                            needs a reproducible flight
    spec.stations = []         # the transparent detector station alters
    #                            no flight and is not under test
    errs = spec.validate()
    assert not errs, "deck refused:\n  " + "\n  ".join(errs)

    # the fixture must transmit on a non-default fate, or this gate no
    # longer exercises the regression at all
    fate = auto_transmitted_fate(spec)
    assert fate != 1, (
        f"fixture {DECK.name} now transmits on fate {fate}: pick a deck "
        f"whose transmission is not the fate-1 literal, or this gate "
        f"cannot catch the regression it exists for")

    v_stated = float(spec.geometry.electrodes[1].dc)
    osp = OptimizeSpec(params=[LENS_PATH],
                       lo=[v_stated - 1.0], hi=[v_stated + 1.0],
                       metric="loss_fraction", n_ions=6, max_evals=1,
                       seed=1)
    value, _ = _evaluate(spec, osp, [v_stated], seed_base=1)

    # at the deck's stated tune the ensemble transmits in full, so the
    # loss fraction is exactly zero; the broken metric returned 1.0 here
    assert value == 0.0, (
        f"loss_fraction scored {value} on a fully transmitting fate-{fate} "
        f"deck: the metric is not reaching the deck's transmitted fate "
        f"(fate-1 literal regression)")
    print(f"  loss_fraction on {DECK.name} (fate {fate}) at "
          f"{v_stated:+.0f} V: {value:.3f}  OK")
