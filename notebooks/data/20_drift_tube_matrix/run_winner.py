"""Checkpointed our-side flight of the Stage-2 spec (scratch runner, not
repo code). Flies ions [start, stop) from packC2/funnel_stage2_spec.json,
appending one summary row per ion to the checkpoint CSV; already-present
ions are skipped, so repeated foreground invocations accumulate to 500.

Usage: python3 run_stage2.py START STOP
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "ion_gym"))

from ion_gym.io.sim_spec import SimSpec                    # noqa: E402
from ion_gym.physics.sim_build import build_run            # noqa: E402
from ion_gym.physics import ensemble_driver as ed          # noqa: E402

PACK = Path(__file__).parent / "packC2"
import os as _os
CKPT = Path(__file__).parent / ("winner_ckpt_dt%s.csv" % _os.environ.get("DT_NS", "1"))
PLANES = (32.0, 53.0)
FIELDS = ["ion", "fate", "tof_us", "x_end", "y_end", "ke_last",
          "t32", "y32", "t53", "y53"]


def row_plate(m):
    return m["x_end"] > 59.0


def done_ions():
    if not CKPT.exists():
        return set()
    with open(CKPT) as f:
        return {int(r["ion"]) for r in csv.DictReader(f)}


def main(start, stop):
    have = done_ions()
    todo = [i for i in range(start, stop) if i not in have]
    if not todo:
        print(f"[{start},{stop}) already checkpointed ({len(have)} total)")
        return
    t0 = time.time()
    import json as _j, os
    d = _j.load(open(Path(__file__).parent / "drift_winner_spec.json"))
    d["integration"]["dt_ns"] = float(os.environ.get("DT_NS", "1"))
    d["integration"]["rec_every"] = max(1, int(round(1000 / d["integration"]["dt_ns"])))
    spec = SimSpec.from_dict(d)
    model, fly_fn, cols, births = build_run(spec)
    ix, iy = cols.index("x"), cols.index("y")
    ik = cols.index("ke_ev") if "ke_ev" in cols else None
    if ik is None:
        raise SystemExit("ke_ev channel absent from cols=%r" % (cols,))
    print(f"build+compile {time.time()-t0:.0f}s | flying {len(todo)} ions "
          f"({todo[0]}..{todo[-1]})")
    new = CKPT.exists() is False
    t1 = time.time()
    with open(CKPT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, i in enumerate(todo):
            prog = ed.run(1, lambda _j, i=i: fly_fn(i), keep_full=True,
                          decimate=1)
            r = prog.results[0]
            m = r.summary
            t, x, y = r.full[:, 0], r.full[:, ix], r.full[:, iy]
            row = dict(ion=i, tof_us=m["tof"], x_end=m["x_end"],
                       y_end=m["y_end"], ke_last=float(r.full[-1, ik]),
                       fate=("thru" if row_plate(m) else
                             "timeout" if m["kind"] == 2 else "wall"))
            for X in PLANES:
                j = np.where((x[:-1] < X) & (x[1:] >= X))[0]
                key = str(int(X))
                if len(j):
                    a = j[0]
                    fr = (X - x[a]) / (x[a + 1] - x[a])
                    row["t" + key] = t[a] + fr * (t[a + 1] - t[a])
                    row["y" + key] = abs(y[a] + fr * (y[a + 1] - y[a]))
                else:
                    row["t" + key] = ""
                    row["y" + key] = ""
            w.writerow(row)
            f.flush()
            if k == min(4, len(todo) - 1):
                rate = (time.time() - t1) / (k + 1)
                print(f"rate ~{rate:.1f} s/ion -> "
                      f"{rate*len(todo):.0f}s for this chunk")
    print(f"chunk done in {time.time()-t1:.0f}s | "
          f"checkpointed {len(done_ions())} total")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
