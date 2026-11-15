"""
latest_bank returns the last bank in name order, with digit runs
compared as numbers.

It sorted names as plain strings, so with run_9 and then run_10 on disk
it returned run_9: a re-plot meant for the newest result silently drew
the previous one. Timestamped names must keep sorting chronologically.
"""
import numpy as np
import pytest

from ion_gym.io import results_bank as RB


def _bank(d, name):
    RB.bank_run(str(d / name), meta={"operating_point": name},
                arrays={"a": np.zeros(1)})


@pytest.mark.parametrize("names, newest", [
    (["run_1", "run_2", "run_9", "run_10"], "run_10"),
    (["sweep_v2_n10", "sweep_v10_n2", "sweep_v10_n9"], "sweep_v10_n9"),
    (["einzel_20261105T0930", "einzel_20261106T0815"],
     "einzel_20261106T0815"),
])
def test_latest_bank_compares_digit_runs_as_numbers(tmp_path, names, newest):
    for n in names:
        _bank(tmp_path, n)
    prefix = newest.split("_")[0] + "_"
    assert RB.latest_bank(tmp_path, prefix).endswith(f"/{newest}.npz")


def test_latest_bank_still_refuses_an_empty_match(tmp_path):
    _bank(tmp_path, "run_1")
    with pytest.raises(RB.BankError, match="no bank matching"):
        RB.latest_bank(tmp_path, "other_")
