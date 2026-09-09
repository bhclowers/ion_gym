"""Analytic ballistic drift beyond the solved domain.

WHY THIS EXISTS: a detector station or terminating bound placed
downstream of the solved domain (the einzel focal-plane case) was
previously handled by accident or not at all. The planar kernel
terminates at the grid edge, so the declaration was silently dead; the
r-z kernel had no axial high-side termination and kept integrating a
LINEAR EXTRAPOLATION of the last two field columns -- a
fabricated field that grows without bound with distance. Neither was a
design.

THE DESIGN: field-free flight is a straight line, and a straight
line's crossing of a plane is exact algebra on the ion's exit state.
So when a flight ends in grid escape (kind = 1) and the deck DECLARES
something downstream -- an enabled bound or a station plane reachable
along the exit ray -- the trajectory is extended by ONE analytically
exact record. The existing station machinery then finds its crossings
by its own interpolation, which is exact between field-free records.
No kernel is touched, no route is dispatched on, no field is
resampled: the same function serves planar, r-z and 3-D results
because it reads only the shared (t, x, y, z, vx, vy, vz) contract.

WHAT IT REFUSES, BY NAME:
  * gas on -- field-free is not collision-free; a drift through a
    declared background gas needs the kernel, not algebra.
  * a hot domain edge -- the extension ASSERTS zero field beyond the
    domain. The caller measures the escape face's residual field and
    this module compares it to EDGE_FIELD_MAX_V_PER_MM, refusing with
    the measured value rather than silently blessing a fringe that has
    not decayed.
  * a recorded channel with no exact field-free extension rule and no
    caller-supplied rule -- extending it wrongly would be a silent lie
    in the record.

WHAT IT DELIBERATELY DOES NOT DO: extend to t_max for its own sake.
The extension honors DECLARATIONS. With a reachable enabled bound the
drift ends exactly ON the bound plane (kind becomes 3 -- and note this
endpoint is exact, where the kernel's own bound crossing overshoots by
up to one step). With only stations declared, the drift ends at the
furthest reachable station crossing and the fate stays kind = 1: the
ion did leave the solved domain; the appended record exists so the
declared observations are honored, not to simulate empty space. A deck
that declares nothing downstream is byte-identical to before.
"""

import math

import numpy as np

# Residual-field ceiling on the escape face, V/mm. NAMED DEFAULT,
# deliberately movable: 1e-3 V/mm acting over a 100 mm drift
# changes a 1 eV ion's energy by <= 0.1 meV (1e-4 relative) and deflects
# a 10 mm/us ion by ~5 um -- below the pitch of any shipped deck. The
# right response to a refusal is padding the domain until the fringe
# decays, not raising this ceiling.
EDGE_FIELD_MAX_V_PER_MM = 1e-3

_BASE = ("t", "x", "y", "z", "vx", "vy", "vz")

# Exact field-free extension rules by channel name. Values:
#   "const"  -- invariant under zero-field drift (velocities constant;
#               the vacuum guard forbids collisions).
#   "zero"   -- the extension DECLARES zero field; recording 0 is that
#               declaration, not a measurement.
#   callable -- (row_prev_value, state, dt_us) -> value.
# Route-anchored channels (radius) have no universal rule here: the
# hook that owns the route's convention supplies it via `derived`.
_CHANNEL_RULES = {
    "speed": "const", "ke_ev": "const",
    "ke_x": "const", "ke_y": "const", "ke_z": "const",
    "n_col": "const", "wrap_passes": "const",
    "e_field": "zero", "e_axial": "zero", "e_radial": "zero",
    "e_x": "zero", "e_y": "zero", "e_z": "zero",
    "e_axial_tint": "const",       # integral of E dt: +0 under E = 0
    "path_mm": lambda prev, s, dt: prev + s["sp"] * dt,
    "ke_tint": lambda prev, s, dt: prev + s["ke_ev"] * dt,
}


def edge_field_max(components, axis, side):
    """Conservative max |E| (V/mm) over one boundary face of a solved
    field pack.

    `components`: list of vector components; each component is a list
    of (array, weight) pairs whose worst-case magnitude is
    sum(|weight * array|) -- the drive-channel bound. `axis` 0/1(/2)
    selects the array axis of the face, `side` "lo"/"hi" the face.
    Returns the max over face nodes of the component-wise norm.
    """
    acc = None
    for comp in components:
        mag = None
        for arr, w in comp:
            a = np.asarray(arr, float)
            sl = [slice(None)] * a.ndim
            sl[axis] = 0 if side == "lo" else -1
            face = np.abs(float(w) * a[tuple(sl)])
            mag = face if mag is None else mag + face
        if mag is None:
            continue
        acc = mag ** 2 if acc is None else acc + mag ** 2
    if acc is None:
        raise ValueError("edge_field_max: no field components supplied "
                         "-- a guard that measures nothing guards "
                         "nothing.")
    return float(np.sqrt(acc).max())


def extend_ballistic(traj, col_names, kind, spec, *,
                     edge_e_vpermm, face_axis, face_side, edge_face="",
                     derived=None, label=""):
    """Extend a grid-escape flight along its exact ballistic ray to
    honor declared downstream bounds and stations.

    `face_axis` (0/1/2, trajectory axes) and `face_side` ('lo'|'hi')
    name the face the ion LEFT THROUGH; the route supplies them (r-z
    maps its radial edge to axis 1, per the BoundsSpec r-z convention).
    ENGAGEMENT RULE (root-caused on a real funnel deck):
    the extension engages only when the escape face's OWN axis has a
    declared target beyond it. A ballistic ray can only assert
    field-free space beyond the face it exited; chasing a target on a
    DIFFERENT axis flies the ion obliquely through unsolved, possibly
    structured space — measured on the funnel: a radially lost ion
    "reaching" the axial detector bound tripped the hot-edge guard at
    32.1 V/mm and killed the run, for an extension nothing needed. Once
    engaged, ALL declared bounds still clip the ray (nearest crossing
    wins), so a downstream radial aperture keeps limiting an axial
    drift.

    Returns (traj, kind, info) where info is None when nothing was done
    (kind != 1, no target beyond the escape face, no reachable declared
    target) and otherwise a dict {dt_us, bnd_face (int or None),
    end_mm}. The input array is never mutated; extension appends exactly
    one row."""
    if kind != 1:
        return traj, kind, None
    if face_axis not in (0, 1, 2) or face_side not in ("lo", "hi"):
        raise ValueError(
            f"drift extension{' (' + label + ')' if label else ''}: "
            f"face_axis/face_side must name the escape face (axis 0/1/2, "
            f"'lo'/'hi'); got {face_axis!r}/{face_side!r}. The route owns "
            f"the escape classification and must supply it.")
    idx = {c: i for i, c in enumerate(col_names)}
    for c in _BASE:
        if c not in idx:
            raise ValueError(
                f"drift extension{' (' + label + ')' if label else ''}: "
                f"trajectory lacks base channel {c!r} -- the shared "
                f"record contract is broken.")
    last = traj[-1]
    t0 = float(last[idx["t"]])
    pos = np.array([last[idx["x"]], last[idx["y"]], last[idx["z"]]], float)
    vel = np.array([last[idx["vx"]], last[idx["vy"]], last[idx["vz"]]],
                   float)
    budget = float(spec.integration.t_max_us) - (t0 - float(traj[0, idx["t"]]))
    if budget <= 1e-12:
        return traj, kind, None

    # --- does anything declared lie BEYOND the escape face? -----------
    bflags, bvals = spec.bounds.as_tuple()
    _face_k = 2 * face_axis + (1 if face_side == "hi" else 0)
    _own_bound = bool(bflags[_face_k])
    _own_station = any(
        {"x": 0, "y": 1, "z": 2}.get(st.axis) == face_axis
        and ((face_side == "hi" and float(st.pos_mm) >= pos[face_axis])
             or (face_side == "lo" and float(st.pos_mm) <= pos[face_axis]))
        for st in (spec.stations or []))
    if not _own_bound and not _own_station:
        # the ion left through a face with nothing declared beyond it
        # (e.g. a radial loss between funnel rings): its flight record
        # is complete as flown; there is nothing to extend TO. kind
        # stays 1, which the fate report names as leaving the grid.
        return traj, kind, None

    # --- declared targets along the ray ------------------------------
    tb, bface = math.inf, None
    for k in range(6):
        if not bflags[k]:
            continue
        a, hi = k // 2, (k % 2 == 1)
        v, p, val = vel[a], pos[a], float(bvals[k])
        if v == 0.0:
            continue
        dt = (val - p) / v
        if dt <= 1e-12:
            continue
        crosses = (hi and v > 0.0) or ((not hi) and v < 0.0)
        if crosses and dt < tb:
            tb, bface = dt, k
    st_dts = []
    _ax = {"x": 0, "y": 1, "z": 2}
    for st in (spec.stations or []):
        a = _ax.get(st.axis)
        if a is None or vel[a] == 0.0:
            continue
        dt = (float(st.pos_mm) - pos[a]) / vel[a]
        if dt > 1e-12:
            st_dts.append(dt)

    bound_reachable = tb <= budget
    st_reachable = [d for d in st_dts if d <= min(tb, budget)]
    if not bound_reachable and not st_reachable:
        return traj, kind, None            # nothing declared is reachable

    # --- refusals, only once the extension would matter --------------
    col = getattr(spec, "collisions", None)
    if col is not None and getattr(col, "enabled", False):
        raise ValueError(
            f"drift extension{' (' + label + ')' if label else ''}: the "
            f"deck declares background gas (collisions.enabled), and "
            f"field-free is not collision-free -- an analytic drift "
            f"through gas would silently drop the scattering the deck "
            f"asked for. Cover the drift with solved domain, or turn "
            f"the gas off if it is truly vacuum downstream.")
    if float(edge_e_vpermm) > EDGE_FIELD_MAX_V_PER_MM:
        raise ValueError(
            f"drift extension{' (' + label + ')' if label else ''}: the "
            f"escape face{' ' + edge_face if edge_face else ''} carries "
            f"a residual field of {float(edge_e_vpermm):.3g} V/mm, above "
            f"the field-free ceiling {EDGE_FIELD_MAX_V_PER_MM:g} V/mm. "
            f"The extension asserts ZERO field beyond the domain; a hot "
            f"edge means the deck's fringe has not decayed there. Pad "
            f"the domain until it does.")

    if bound_reachable:
        dt_end, kind_out = tb, 3
    else:
        dt_end, kind_out = max(st_reachable), kind
    # --- one exact appended record -----------------------------------
    end = pos + vel * dt_end
    sp = float(np.linalg.norm(vel))
    state = {"sp": sp,
             "ke_ev": float(last[idx["ke_ev"]]) if "ke_ev" in idx else None}
    row = np.array(last, float, copy=True)
    row[idx["t"]] = t0 + dt_end
    row[idx["x"]], row[idx["y"]], row[idx["z"]] = end
    derived = derived or {}
    for c in col_names[7:]:
        if c in derived:
            row[idx[c]] = float(derived[c](float(last[idx[c]]),
                                           row, idx, dt_end))
            continue
        rule = _CHANNEL_RULES.get(c)
        if rule == "const":
            continue                        # copied with the row
        if rule == "zero":
            row[idx[c]] = 0.0
            continue
        if callable(rule):
            if c == "ke_tint" and state["ke_ev"] is None:
                raise ValueError(
                    "drift extension: ke_tint is recorded without "
                    "ke_ev; the extension cannot derive the integrand.")
            row[idx[c]] = rule(float(last[idx[c]]), state, dt_end)
            continue
        raise ValueError(
            f"drift extension{' (' + label + ')' if label else ''}: "
            f"recorded channel {c!r} has no exact field-free extension "
            f"rule and the caller supplied none -- extending it by "
            f"guesswork would put a silent lie in the record. Add its "
            f"rule where the route's convention lives.")
    return np.vstack([traj, row[None, :]]), kind_out, {
        "dt_us": float(dt_end), "bnd_face": bface,
        "end_mm": tuple(float(v) for v in end)}
