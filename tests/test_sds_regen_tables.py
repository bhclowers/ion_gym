"""SDS shipped-table gate.

The ion_gym-native jump statistics (sds_jump_icdf.dat, made by
tools/regen_sds_tables.py) exist, parse, and are sane -- 5 x 1002
monotone-nondecreasing quantile curves with the provenance header --
and no mobility table ships. The tool itself is tested in
test_regen_sds_tables_tool.py; the table's effect on flights is checked
by test_sds_transport.py (Einstein diffusion on every route) and V06.
"""
import pathlib

import numpy as np

from ion_gym.physics import sds

PHYS = pathlib.Path(sds.__file__).resolve().parent


def _load_curves(path):
    vals = []
    for line in open(path):
        body = line.split(";")[0].strip()
        if body:
            vals.append(float(body))
    a = np.array(vals)
    assert a.size == 5 * 1002, f"{path}: {a.size} values (want 5x1002)"
    return a.reshape(5, 1002)


def test_shipped_jump_icdf_sane():
    p = PHYS / sds.JUMP_ICDF_FILE
    assert p.exists(), f"shipped SDS table missing: {p}"
    head = open(p).read(400)
    assert "REGENERATED" in head and "Appelhans" in head, \
        "provenance header missing from shipped SDS table"
    curves = _load_curves(p)
    for k in range(5):
        q = curves[k]
        assert np.all(np.diff(q) >= -1e-12), f"curve {k} not monotone"
        assert q[-1] > q[501] > 0, f"curve {k} degenerate"
    # heavier ions walk farther at equal collision count
    med = curves[:, 501]
    assert np.all(np.diff(med) > 0), f"median ordering broken: {med}"


def test_no_mobility_table_ships():
    """No mobility table is shipped: K0 and the diameter come from the
    published estimators (the old table's rows were another program's
    distributed values)."""
    assert not (PHYS / "sds_mobility.dat").exists()
    assert not hasattr(sds, "MOBILITY_FILE")


# test_matches_licensed_original_within_accepted_tolerance REMOVED
# It compared the regenerated tables against the
# licensed reference originals, which are not distributed. The
# test could only ever skip, and a permanently-skipping test is noise
# pretending to be coverage. THE RESULT IT CERTIFIED IS NOT WITHDRAWN:
# the central max |dlog10| tolerance was met on a dated, recorded run;
# re-running it requires a licensed reference install and is a
# local, out-of-tree exercise.
