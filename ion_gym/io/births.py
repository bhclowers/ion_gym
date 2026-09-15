"""ion_gym.io.births -- write the birth table that `distribution="file"`
reads back.

WHY THIS EXISTS: SourceSpec has read the CSV since the schema was written,
but nothing in the tree ever WROTE one, so the round trip was half a
contract. Any two-stage instrument -- an accelerator handing a packet to
an analyzer, a source stage handing to a transport stage -- needs it, and
each such study was otherwise going to invent its own format.

UNITS ARE THE WHOLE POINT AND ARE NOT NEGOTIABLE. The kernel works in
mm and mm/us; `generate_births` documents "(N,7) = [x,y,z,vx,vy,vz,tob],
velocities in mm/us; positions mm; tob us". A file written in m/s reads
back as a beam a thousand times too slow and simply misses the detector,
which looks like a transmission result rather than a units bug. So the
writer takes SI-free mm/us and refuses anything it can detect as wrong.

The station-hit dict from `ion_gym.physics.stations` is already in these
units, which is why `from_station_hits` is the intended entry point.
"""
from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ("x", "y", "z", "vx", "vy", "vz", "tob")


def write_births(path, rows, *, overwrite=False):
    """Write an (N,7) iterable of [x,y,z,vx,vy,vz,tob] to `path`.

    Refuses an empty table and a short/long row: a birth table with a
    missing velocity component would be read back with that component
    defaulted to zero, which is a physically meaningful and completely
    wrong beam rather than an error.
    """
    p = Path(path)
    rows = [list(map(float, r)) for r in rows]
    if not rows:
        raise ValueError(
            f"write_births: refusing to write an empty birth table to {p}. "
            f"An empty file reads back as zero ions, which downstream looks "
            f"like total transmission loss rather than an authoring mistake.")
    bad = [i for i, r in enumerate(rows) if len(r) != len(COLUMNS)]
    if bad:
        raise ValueError(
            f"write_births: rows {bad[:5]}{'...' if len(bad) > 5 else ''} "
            f"have the wrong length; every row must be exactly "
            f"{len(COLUMNS)} values {COLUMNS}. Padding would silently set a "
            f"velocity component to zero.")
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"write_births: {p} exists. Pass overwrite=True to replace it -- "
            f"a birth table is an experimental record and clobbering one "
            f"silently would make a flown result unreproducible.")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)
    return p


def from_station_hits(path, hits, *, t_zero="first", overwrite=False):
    """Write a birth table from station-hit dicts (t_us, x, y, z, vx, vy, vz).

    `t_zero` sets the time origin carried into `tob`:
      "first" -- subtract the earliest hit, so the downstream run starts at
                 t = 0 and `tob` carries only the packet's internal time
                 structure. This is what a handoff wants.
      "keep"  -- carry absolute arrival times.
    The distinction matters because `tob` is added to the downstream flight
    time; keeping absolute times silently offsets every reported T by the
    upstream transit.
    """
    if t_zero not in ("first", "keep"):
        raise ValueError(
            f"from_station_hits: t_zero must be 'first' or 'keep', got "
            f"{t_zero!r}. There is no safe default for a time origin -- "
            f"'keep' offsets every downstream flight time by the upstream "
            f"transit, and that offset is invisible in the result.")
    hits = list(hits)
    if not hits:
        raise ValueError(
            "from_station_hits: no hits supplied. If the upstream run "
            "transmitted nothing, that is the result to report, not a file "
            "to write.")
    t0 = min(h["t_us"] for h in hits) if t_zero == "first" else 0.0
    rows = [[h["x"], h["y"], h["z"], h["vx"], h["vy"], h["vz"],
             h["t_us"] - t0] for h in hits]
    return write_births(path, rows, overwrite=overwrite)
