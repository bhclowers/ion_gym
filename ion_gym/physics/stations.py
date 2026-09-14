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

__all__ = ["station_hits", "first_detection",
           "kernel_stations", "unsupported_stations_notice",
           "station_planes", "bound_planes", "kernel_planes"]

_AX = {"x": "x", "y": "y", "z": "z"}

_INF = float("inf")

# fate codes the kernels assign to a terminating station crossing. These
# are the wire contract between this module and every tracer that
# consumes its output; they are named here so no route re-declares them.
FATE_IMPACT_PLANE = 5
FATE_DETECT = 6


def station_planes(spec, world_off_mm=(0.0, 0.0, 0.0)):
    """A spec's stations as kernel planes: (axis, value_mm, sign, window4,
    fate).

    THE single implementation of the ruled station contract, shared by
    every route, so a fix cannot land on one builder and drift on the
    others (the defect that left planar/r-z/stl2d silently ignoring
    stations while stl3d honoured them).

    sign is 0.0: a station terminates on a crossing in EITHER direction.
    window4 is (t1_lo, t1_hi, t2_lo, t2_hi) over the two axes transverse
    to the station, in "xyz" order.

    The two kinds that reach a kernel, with symmetric window senses:
      impact_plane (fate 5): a PLATE with an aperture — crossings INSIDE
        the window pass, OUTSIDE splat. Empty window = a wall (nothing
        passes).
      detect + on_hit="splat" (fate 6): a DETECTOR PATCH — crossings
        INSIDE the window ABSORB (the detection event), outside pass.
        Empty window = full-plane detector.
    detect + on_hit="pass" and record never reach a kernel: they are
    post-hoc trajectory bookkeeping (station_hits), and are skipped here.

    `world_off_mm` converts the CANONICAL frame the deck is authored in
    (and that the display and records use) to the frame the kernel flies:
    field = canonical - offset. Routes that unfold a mirrored axis pass
    their offset; routes that fly the authored frame pass zeros.

    Refuses, rather than silently misreading:
      - a window key naming the station's OWN axis (a plane cannot be
        constrained along its own normal, and treating it as
        "unconstrained" silently turns an impact_plane wall into a
        full-aperture pass-through)
      - a window pair that is not (lo, hi) with lo <= hi
    """
    off = dict(zip("xyz", [float(v) for v in world_off_mm]))
    out = []
    for st in (getattr(spec, "stations", None) or []):
        kind = getattr(st, "kind", "")
        if kind == "detect" and getattr(st, "on_hit", None) != "splat":
            continue                      # pass-detect: post-hoc only
        if kind not in ("impact_plane", "detect"):
            continue                      # record: post-hoc only
        sax = st.axis
        name = getattr(st, "name", "<unnamed>")
        t1, t2 = [a for a in "xyz" if a != sax]
        wd = st.window or {}
        if sax in wd:
            raise ValueError(
                f"station {name!r}: window names its own axis "
                f"{sax!r}. A plane cannot be windowed along its own "
                f"normal — a window constrains the two TRANSVERSE axes "
                f"({t1!r}, {t2!r}). Remove the {sax!r} entry (an empty "
                f"window means a full-plane detector, or a solid wall "
                f"for impact_plane).")
        for a, w in wd.items():
            if len(w) != 2 or float(w[0]) > float(w[1]):
                raise ValueError(
                    f"station {name!r}: window[{a!r}] = {list(w)!r} is "
                    f"not a (lo, hi) pair with lo <= hi.")

        def _pair(a):
            w = wd.get(a)
            if w is None:
                return (-_INF, _INF)          # unconstrained axis
            return (float(w[0]) - off[a], float(w[1]) - off[a])

        if wd:
            win = (*_pair(t1), *_pair(t2))
        elif kind == "impact_plane":
            win = (_INF, -_INF, _INF, -_INF)  # wall: nothing inside
        else:
            win = (-_INF, _INF, -_INF, _INF)  # full-plane detector
        out.append((sax, float(st.pos_mm) - off[sax], 0.0, win,
                    FATE_IMPACT_PLANE if kind == "impact_plane"
                    else FATE_DETECT))
    return out


def kernel_stations(spec):
    """The stations that WOULD become kernel planes — i.e. the ones whose
    effect on a flight is real (detect+splat absorbs, impact_plane
    splats), not the post-hoc bookkeeping ones.

    A route that cannot hand planes to its kernel uses this to report
    what it is unable to honour; detect+on_hit="pass" is deliberately
    NOT listed, because it is fully served post-hoc on every route and
    warning about it would train users to ignore the notice.
    """
    out = []
    for st in (getattr(spec, "stations", None) or []):
        kind = getattr(st, "kind", "")
        if kind == "detect" and getattr(st, "on_hit", None) != "splat":
            continue
        if kind not in ("impact_plane", "detect"):
            continue
        out.append(st)
    return out


def unsupported_stations_notice(spec, route, fix):
    """A LOUD, specific notice for stations a route cannot honour, or
    None when there is nothing to report.

    Never a silent skip: a station that changes a flight everywhere else
    but does nothing here is exactly the failure that made a working
    splat look broken for a week (L-435). The message names each
    station, says what it would have done, and states the fix — it does
    not merely announce that something was ignored.
    """
    affected = kernel_stations(spec)
    if not affected:
        return None
    lines = []
    for st in affected:
        nm = getattr(st, "name", "<unnamed>")
        kind = getattr(st, "kind", "")
        what = ("would ABSORB ions crossing inside its window "
                "(detector patch)" if kind == "detect"
                else "would SPLAT ions crossing outside its window "
                     "(plate with an aperture)")
        lines.append(f"  - {nm!r} ({kind} at {st.axis}="
                     f"{float(st.pos_mm):g} mm) {what}")
    return (f"[stations] NOT HONOURED on the {route} route — "
            f"{len(affected)} station(s) below will have NO effect on "
            f"this flight:\n" + "\n".join(lines) + f"\n  {fix}\n"
            f"  Arrival-time stations (detect with on_hit='pass') are "
            f"unaffected: those are post-hoc and work on every route.")

def bound_planes(bounds, world_off_mm=(0.0, 0.0, 0.0)):
    """Enabled bounding planes as (axis, value_mm, sign) for a kernel.

    sign = -1 terminates on a crossing DOWNWARD through the plane (a min
    plane), +1 upward (a max plane). Bounding planes are absolute
    whole-plane kills and carry no window — a route whose kernel takes
    per-face bound flags does not need this and should not call it.
    """
    off = dict(zip("xyz", [float(v) for v in world_off_mm]))
    out = []
    for ax in ("x", "y", "z"):
        if getattr(bounds, f"{ax}_min_on", False):
            out.append((ax, float(getattr(bounds, f"{ax}_min"))
                        - off[ax], -1.0))
        if getattr(bounds, f"{ax}_max_on", False):
            out.append((ax, float(getattr(bounds, f"{ax}_max"))
                        - off[ax], +1.0))
    return out


def kernel_planes(spec, world_off_mm=(0.0, 0.0, 0.0)):
    """Stations THEN bounds, the order the tracers have always received.

    Station planes are 5-tuples and bounding planes 3-tuples; the tracer
    distinguishes them by length.
    """
    return (station_planes(spec, world_off_mm)
            + bound_planes(spec.bounds, world_off_mm))



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
