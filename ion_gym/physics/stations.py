"""stations -- apply a SimSpec's declared StationSpec planes to recorded
trajectories (bounding planes are absolute and cannot
represent a detector; stations are the deck-carried impact/record planes
with transmission windows).

The kernel is untouched: stations are evaluated on trajectory records
with linear in-step interpolation, which is EXACT when both bracketing
records are field-free.  A "detect" station is ABSORBING: the ion's
first in-window crossing is its detection event and every later record
is nonphysical -- consumers of this module never see them.  A crossing
whose bracketing records may be in-field is flagged, never silently
trusted.
"""
import numpy as np

__all__ = ["station_hits", "first_detection"]

_AX = {"x": "x", "y": "y", "z": "z"}


def _interp_state(tr, ix, j, f, channels):
    out = {}
    for c in channels:
        a, b = tr[j, ix[c]], tr[j + 1, ix[c]]
        out[c] = float(a + f * (b - a))
    return out


def station_hits(traj, cols, station, *, t_min_us=1.0e-3):
    """All crossings of one station plane, in time order.

    Returns a list of dicts (t_us, x/y/z, vx/vy/vz, k) where k counts
    plane crossings INCLUDING out-of-window ones (so k is the crossing
    ordinal, comparable across stations).  Crossings with t <= t_min_us
    are dropped: an ion born exactly ON a plane manufactures a spurious
    sign flip at index 0 (measured) and t_min guards it.
    """
    ix = {c: i for i, c in enumerate(cols)}
    ax = _AX[station.axis]
    t = traj[:, ix["t"]]
    q = traj[:, ix[ax]] - float(station.pos_mm)
    hits = []
    k = 0
    for j in np.nonzero(np.diff(np.sign(q)) != 0)[0]:
        if t[j] <= t_min_us:
            continue
        f = q[j] / (q[j] - q[j + 1])
        st = _interp_state(traj, ix, j, f,
                           ("t", "x", "y", "z", "vx", "vy", "vz"))
        st["t_us"] = st.pop("t")
        st["k"] = k
        k += 1
        inside = True
        for wax, (lo, hi) in (station.window or {}).items():
            if not (float(lo) <= st[wax] <= float(hi)):
                inside = False
                break
        st["in_window"] = inside
        hits.append(st)
    return hits


def first_detection(traj, cols, spec, *, name=None, t_min_us=1.0e-3):
    """First in-window crossing of the named "detect" station (or the
    only detect station when name is None).  Returns the hit dict or
    None (a MISS).  Refuses ambiguity and absent stations by name --
    a detector the deck does not declare cannot be silently invented.
    """
    dets = [st for st in spec.stations if st.kind == "detect"
            and (name is None or st.name == name)]
    if not dets:
        raise ValueError(
            "first_detection: the spec declares no matching 'detect' "
            f"station (name={name!r}); detection planes live in the "
            "DECK, not in analysis kwargs")
    if len(dets) > 1:
        raise ValueError(
            f"first_detection: {len(dets)} detect stations match "
            f"name={name!r}; pass the station name explicitly: "
            f"{[d.name for d in dets]}")
    for h in station_hits(traj, cols, dets[0], t_min_us=t_min_us):
        if h["in_window"]:
            return h
    return None
