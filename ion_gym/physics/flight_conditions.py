"""flight_conditions.py -- set the VOLTAGES and the ION after the field
solve, the way the UI does it.

THE WORKFLOW RULE: build the geometry,
solve the fields, and THEN specify the flying conditions. Voltages and
source parameters are not geometry: the solver stores one basis per
electrode and per drive, so changing a voltage or an ion is a
re-weighting of cached bases, never a re-solve. A notebook that buries
those settings in the spec it constructs before solving forces the
reader to conceptually restart the whole pipeline to answer "what if I
raised this electrode 20 V" -- exactly what the UI's ordering exists to
prevent.

Use:
    model, fly, cols, births = build_run(spec)      # geometry + solve
    ...                                             # look at the fields
    report = set_flight_conditions(spec, voltages={...}, ion={...})
    model, fly, cols, births = build_run(spec)      # REWEIGHT, no re-solve
                                                    # (asserted below)

`set_flight_conditions` refuses anything that would change the geometry
key -- so a "voltage change" that would silently trigger a re-solve is
reported as the error it is, rather than costing minutes and quietly
invalidating the figures above it.
"""
from __future__ import annotations

from ion_gym.io.basis_cache import key as _basis_key

__all__ = ["set_flight_conditions", "solved_key", "FlightConditionError"]


def solved_key(spec):
    """The geometry key the fields were just solved for. Capture this
    right after build_run and hand it to set_flight_conditions, which
    then guarantees the conditions did not move the geometry out from
    under those fields -- including a change made OUTSIDE the call,
    which a before/after comparison inside the call cannot see."""
    return _basis_key(spec)


class FlightConditionError(ValueError):
    """A requested condition names something the deck does not have, or
    would move the geometry out from under the solved fields."""


def _electrodes(spec):
    return {e.name: e for e in spec.geometry.electrodes}


def _groups(spec):
    return {g.name: g for g in getattr(spec.geometry, "rf_groups", [])}


def set_flight_conditions(spec, *, voltages=None, drives=None, ion=None,
                          gas=None, integration=None, dc_groups=None,
                          solved_key=None, show=True):
    """Apply post-solve conditions and report them.

    voltages   {electrode_name: volts}          -- DC on named electrodes.
               An electrode in a UNIFORM dc_group is set through the group
               (equivalent, and it survives resolve_dc_groups); one in a
               non-uniform LADDER refuses and names dc_groups instead,
               because its dc is derived and would be overwritten.
    dc_groups  {group_name: {v_in=, v_out=, interp=, uniform=}}
                                                -- drive a DC ladder
    drives     {group_name: {amplitude_v=, frequency_hz=, phase_deg=,
                             offset_v=, duty=}} -- RF/TW settings
    ion        {mz_list=, charge=, ke_ev=(lo,hi), temperature_k=,
                n_ions=, x0_mm=, y0_mm=, z0_mm=, direction=, r_mm=,
                box_mm=, seed=}                 -- the source
    gas        {enabled=, gas=, p_torr=, T_k=}  -- buffer gas
    integration{dt_ns=, t_max_us=, rec_every=, max_records=}

    Returns the report lines. Raises FlightConditionError on an unknown
    name or on any change that would alter the geometry cache key.
    """
    before = solved_key if solved_key is not None else _basis_key(spec)
    L = []

    if voltages:
        els = _electrodes(spec)
        grps_dc = {g.name: g for g in
                   getattr(spec.geometry, "dc_groups", []) or []}
        members = {}
        for e in spec.geometry.electrodes:
            if e.dc_group:
                members.setdefault(e.dc_group, []).append(e)
        for name, v in voltages.items():
            if name not in els:
                raise FlightConditionError(
                    f"voltage for unknown electrode {name!r}; this deck has "
                    f"{sorted(els)}")
            el = els[name]
            grp = grps_dc.get(el.dc_group) if el.dc_group else None
            if grp is None:
                # ungrouped: the electrode's own dc IS what the solver reads
                L.append(f"V[{name}]: {el.dc:g} -> {float(v):g} V")
                el.dc = float(v)
                continue
            # GROUPED. resolve_dc_groups() runs at the top of every
            # build_run and rewrites every member's dc from the group, so
            # writing el.dc here would be discarded before the solve and
            # the report above would be a lie -- the run would quietly fly
            # the ORIGINAL tune while claiming the new one. Measured
            # on a mirror re-null: 600 candidate voltage sets
            # all returned bit-identical objectives.
            mem = members.get(el.dc_group, [])
            if getattr(grp, "uniform", False) or len(mem) == 1:
                # every member sits at v_in, so setting the group is
                # exactly equivalent and unambiguous
                L.append(f"V[{name}] via uniform dc_group "
                         f"{el.dc_group!r}: {grp.v_in:g} -> {float(v):g} V"
                         + (f" (also moves {len(mem) - 1} other member(s))"
                            if len(mem) > 1 else ""))
                grp.v_in = float(v)
                grp.v_out = float(v)
                for m in mem:
                    m.dc = float(v)
                continue
            raise FlightConditionError(
                f"voltage for {name!r} cannot be set per-electrode: it is a "
                f"member of the non-uniform dc ladder {el.dc_group!r} "
                f"({len(mem)} members), whose taps are DERIVED from the "
                f"group's v_in/v_out by resolve_dc_groups() at every build. "
                f"Setting one member's dc would be silently overwritten. "
                f"Drive the ladder instead: "
                f"set_flight_conditions(spec, dc_groups={{'{el.dc_group}': "
                f"{{'v_in': ..., 'v_out': ...}}}}).")

    if dc_groups:
        grps_dc = {g.name: g for g in
                   getattr(spec.geometry, "dc_groups", []) or []}
        for name, settings in dc_groups.items():
            if name not in grps_dc:
                raise FlightConditionError(
                    f"dc_group settings for unknown group {name!r}; this "
                    f"deck has {sorted(grps_dc)}")
            g = grps_dc[name]
            for attr, val in settings.items():
                if not hasattr(g, attr):
                    raise FlightConditionError(
                        f"dc_group {name!r} has no field {attr!r}; settable: "
                        f"{sorted(f for f in g.__dataclass_fields__)}")
                L.append(f"{name}.{attr}: {getattr(g, attr)!r} -> {val!r}")
                setattr(g, attr, val)

    if drives:
        grps = _groups(spec)
        for name, settings in drives.items():
            if name not in grps:
                raise FlightConditionError(
                    f"drive settings for unknown group {name!r}; this deck "
                    f"has {sorted(grps)}")
            g = grps[name]
            for attr, val in settings.items():
                if not hasattr(g, attr):
                    raise FlightConditionError(
                        f"drive group {name!r} has no field {attr!r}")
                L.append(f"{name}.{attr}: {getattr(g, attr)!r} -> {val!r}")
                setattr(g, attr, val)

    if ion:
        s = spec.source
        for attr, val in ion.items():
            if attr == "ke_ev":
                L.append(f"ion KE: {s.ke_lo:g}-{s.ke_hi:g} -> "
                         f"{float(val[0]):g}-{float(val[1]):g} eV")
                s.ke_lo, s.ke_hi = float(val[0]), float(val[1])
                continue
            if not hasattr(s, attr):
                raise FlightConditionError(
                    f"source has no field {attr!r}; settable: "
                    f"{sorted(f for f in s.__dataclass_fields__)}")
            L.append(f"ion {attr}: {getattr(s, attr)!r} -> {val!r}")
            setattr(s, attr, val)

    if gas:
        c = spec.collisions
        if "p_torr" in gas:
            L.append(f"gas pressure: {c.P_torr!r} -> {float(gas['p_torr']):g} Torr")
            # ATOMIC: setting P_torr and nulling P_pa does NOT re-derive, and
            # a None P_pa reaches the kernel as an untypeable argument. The
            # spec owns the one door through that hazard.
            c.set_pressure_torr(float(gas["p_torr"]))
        for k, attr in (("enabled", "enabled"), ("gas", "gas"),
                        ("T_k", "T_k"), ("model", "model")):
            if k in gas:
                L.append(f"gas {attr}: {getattr(c, attr)!r} -> {gas[k]!r}")
                setattr(c, attr, gas[k])

    if integration:
        i = spec.integration
        for attr, val in integration.items():
            if not hasattr(i, attr):
                raise FlightConditionError(
                    f"integration has no field {attr!r}")
            L.append(f"{attr}: {getattr(i, attr)!r} -> {val!r}")
            setattr(i, attr, val)

    after = _basis_key(spec)
    if after != before:
        raise FlightConditionError(
            "the geometry key does not match the solved fields "
            f"({before} -> {after}), so the fields shown above no longer "
            "describe this spec and the next build_run would RE-SOLVE. "
            "Flight conditions are voltage/ion/gas/integration only. If the "
            "geometry really must change, change it in the geometry cell and "
            "re-solve deliberately."
            + ("" if solved_key is not None else
               "  (No solved_key was passed, so this call could only compare "
               "against the key at entry -- pass solved_key=solved_key(spec) "
               "captured right after build_run to catch changes made "
               "outside this call too.)"))

    if show:
        print("FLIGHT CONDITIONS (applied to the SOLVED model; the next "
              "build_run re-weights cached bases, it does not re-solve)")
        for line in L:
            print("  " + line)
        if not L:
            print("  (none changed -- flying the deck's own conditions)")
    return L
