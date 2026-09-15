"""deck_params.py -- make a deck's inherited physics VISIBLE, and let a
notebook override it by name.

The problem this solves: a notebook that loads a
deck inherits the confining RF, the ion, the gas and the integration
settings from a JSON the reader never opens. Those are exactly the
knobs a reader wants to change, so they must be reachable from the
notebook's parameter cell -- and, just as important, whatever is
actually in force must be PRINTED, because a parameter that appears
editable but silently fails to take effect is worse than no parameter.

Two calls:
    describe_deck(spec)                 -> report only
    apply_deck_overrides(spec, ...)     -> apply named overrides, report

Both return the report lines, so a caller can log instead of print.
Every override is applied to the SPEC OBJECT, after construction, so it
cannot be defeated by deck defaults; None always means "inherit".
"""
from __future__ import annotations

__all__ = ["describe_deck", "apply_deck_overrides", "spec_table",
           "DeckParamError"]


class DeckParamError(ValueError):
    """An override that names something the deck does not have."""


def _rf_groups(spec, tw_prefix="TW"):
    """Drive groups that are NOT the transport wave. Identified by NAME:
    a frequency comparison is a guess that breaks the day the transport
    drive runs faster than the confinement drive."""
    return [g for g in getattr(spec.geometry, "rf_groups", [])
            if not str(g.name).startswith(tw_prefix)]


def describe_deck(spec, *, tw_prefix="TW", show=True):
    """Report the physics this spec inherited from its deck: drives, the
    ion, the gas, and the integration settings."""
    L = []
    rf = _rf_groups(spec, tw_prefix)
    tw = [g for g in getattr(spec.geometry, "rf_groups", [])
          if str(g.name).startswith(tw_prefix)]
    if rf:
        L.append("RF drive:   " + ", ".join(
            f"{g.name} {g.amplitude_v:g} V @ {g.frequency_hz:g} Hz"
            f" ({g.waveform})" for g in rf))
    if tw:
        L.append(f"TW drive:   {len(tw)} phases, {tw[0].amplitude_v:g} V "
                 f"@ {tw[0].frequency_hz:g} Hz ({tw[0].waveform})")
    if not rf and not tw:
        L.append("drives:     none (static device)")
    dcs = [(e.name, e.dc) for e in getattr(spec.geometry, "electrodes", [])
           if getattr(e, "dc", None)]
    if dcs:
        lo = min(v for _, v in dcs)
        hi = max(v for _, v in dcs)
        L.append(f"DC bias:    {len(dcs)} biased electrode(s), "
                 f"{lo:g} to {hi:g} V")
    s = spec.source
    L.append(f"ion:        m/z {list(s.mz_list)}, charge {s.charge}, "
             f"KE {s.ke_lo:g}-{s.ke_hi:g} eV, source T {s.temperature_k:g} K, "
             f"{s.n_ions} ion(s)")
    c = spec.collisions
    if getattr(c, "enabled", False):
        L.append(f"gas:        {c.gas} at {c.P_torr:g} Torr, {c.T_k:g} K "
                 f"(model {c.model})")
    else:
        L.append("gas:        collisions OFF (vacuum)")
    i = spec.integration
    L.append(f"integration: dt {i.dt_ns:g} ns, t_max {i.t_max_us:g} us, "
             f"rec_every {i.rec_every}")
    if show:
        print("INHERITED FROM THE DECK")
        for line in L:
            print("  " + line)
    return L


def apply_deck_overrides(spec, *, rf_v=None, rf_f=None, tw_v=None, tw_f=None,
                         mz_list=None, charge=None, ke_ev=None,
                         source_t_k=None, n_ions=None,
                         gas_on=None, gas=None, p_torr=None, gas_t_k=None,
                         dt_ns=None, t_max_us=None, rec_every=None,
                         dc=None, tw_prefix="TW", show=True):
    """Apply named overrides to a spec built from a deck. None = inherit.

    `dc` is a {electrode_name: volts} mapping; an unknown name RAISES
    (DeckParamError) rather than being ignored -- a silently dropped
    override is the failure mode this module exists to prevent.

    Returns the report lines: every parameter appears as inherited or
    OVERRIDE, so the notebook's output states its own operating point.
    """
    L = []

    def note(label, value, current):
        if value is None:
            L.append(f"{label}: inherited {current!r}")
        else:
            L.append(f"{label}: OVERRIDE -> {value!r}")

    rf = _rf_groups(spec, tw_prefix)
    tw = [g for g in getattr(spec.geometry, "rf_groups", [])
          if str(g.name).startswith(tw_prefix)]
    for grp, (v, f, tag) in ((rf, (rf_v, rf_f, "RF")),
                             (tw, (tw_v, tw_f, "TW"))):
        if not grp:
            if v is not None or f is not None:
                raise DeckParamError(
                    f"{tag} override requested but this deck has no {tag} "
                    f"group (drives: "
                    f"{[g.name for g in spec.geometry.rf_groups]})")
            continue
        for g in grp:
            if v is not None:
                g.amplitude_v = float(v)
            if f is not None:
                g.frequency_hz = float(f)
        L.append(f"{tag} drive: {grp[0].amplitude_v:g} V @ "
                 f"{grp[0].frequency_hz:g} Hz"
                 + ("" if (v is None and f is None) else "   (OVERRIDDEN)"))

    if dc is not None:
        byname = {e.name: e for e in spec.geometry.electrodes}
        for k, v in dc.items():
            if k not in byname:
                raise DeckParamError(
                    f"dc override for unknown electrode {k!r}; this deck has "
                    f"{sorted(byname)}")
            byname[k].dc = float(v)
            L.append(f"dc[{k}]: OVERRIDE -> {float(v):g} V")

    s = spec.source
    note("ion m/z", mz_list, list(s.mz_list))
    if mz_list is not None:
        s.mz_list = list(mz_list)
    note("charge", charge, s.charge)
    if charge is not None:
        s.charge = int(charge)
    if ke_ev is None:
        L.append(f"initial KE: inherited {s.ke_lo:g}-{s.ke_hi:g} eV")
    else:
        s.ke_lo, s.ke_hi = float(ke_ev[0]), float(ke_ev[1])
        L.append(f"initial KE: OVERRIDE -> {s.ke_lo:g}-{s.ke_hi:g} eV")
    note("source T", source_t_k, s.temperature_k)
    if source_t_k is not None:
        s.temperature_k = float(source_t_k)
    note("n_ions", n_ions, s.n_ions)
    if n_ions is not None:
        s.n_ions = int(n_ions)

    c = spec.collisions
    note("collisions", gas_on, c.enabled)
    if gas_on is not None:
        c.enabled = bool(gas_on)
    note("gas", gas, c.gas)
    if gas is not None:
        c.gas = str(gas)
    if p_torr is None:
        L.append(f"pressure: inherited {c.P_torr!r} Torr")
    else:
        # atomic: assigning P_torr and nulling P_pa does not re-derive, and a
        # None P_pa reaches the kernel as an untypeable argument.
        c.set_pressure_torr(float(p_torr))
        L.append(f"pressure: OVERRIDE -> {c.P_torr:g} Torr")
    note("gas T", gas_t_k, c.T_k)
    if gas_t_k is not None:
        c.T_k = float(gas_t_k)

    i = spec.integration
    note("dt_ns", dt_ns, i.dt_ns)
    if dt_ns is not None:
        i.dt_ns = float(dt_ns)
    note("t_max_us", t_max_us, i.t_max_us)
    if t_max_us is not None:
        i.t_max_us = float(t_max_us)
    note("rec_every", rec_every, i.rec_every)
    if rec_every is not None:
        i.rec_every = int(rec_every)

    if show:
        print("PARAMETERS IN EFFECT")
        for line in L:
            print("  " + line)
    return L


def spec_table(spec, *, sections=None, show=True):
    """Render a SimSpec as a DATATABLE for a notebook.

    A deck JSON is hundreds of lines and unreadable in a cell; this gives
    the same content as a flat, scannable (section, parameter, value,
    units) table. In Jupyter it returns a pandas DataFrame so the notebook
    renders it natively; without pandas it falls back to an aligned text
    table and returns the rows, so the function works headless too --
    never a silent no-op.

    sections : optional iterable restricting output, e.g. ("drives",
        "ion"). Unknown names are REFUSED rather than quietly ignored,
        because a typo that silently drops a section is how a reader ends
        up trusting an incomplete table.

    Values are read off the SPEC, so this shows what the solver will be
    handed -- not what a deck file on disk once said.
    """
    known = ("geometry", "electrodes", "drives", "ion", "gas",
             "integration", "bounds")
    if sections is None:
        sections = known
    else:
        sections = tuple(sections)
        bad = [s for s in sections if s not in known]
        if bad:
            raise DeckParamError(
                f"spec_table: unknown section(s) {bad} -- known sections are "
                f"{list(known)}. Refusing rather than silently omitting.")
    g = getattr(spec, "geometry", None)
    src = getattr(spec, "source", None)
    col = getattr(spec, "collisions", None)
    it = getattr(spec, "integration", None)
    rows = []

    def add(sec, name, val, unit=""):
        rows.append(dict(section=sec, parameter=name, value=val, units=unit))

    if "geometry" in sections and g is not None:
        add("geometry", "name", getattr(spec, "name", "") or "(unnamed)")
        add("geometry", "width", f"{getattr(g, 'width_mm', float('nan')):g}", "mm")
        add("geometry", "height", f"{getattr(g, 'height_mm', float('nan')):g}", "mm")
        if float(getattr(g, "depth_mm", 0.0) or 0.0):
            add("geometry", "depth", f"{g.depth_mm:g}", "mm")
        add("geometry", "pitch", f"{getattr(g, 'mm_per_gu', float('nan')):g}", "mm/gu")
        sym = getattr(g, "symmetry", None)
        if sym is not None:
            planes = getattr(sym, "planes", None) or {}
            add("geometry", "symmetry",
                f"{getattr(sym, 'coords', '?')}; " +
                ", ".join(f"{k}:{v}" for k, v in dict(planes).items()))
    if "electrodes" in sections and g is not None:
        for e in getattr(g, "electrodes", []) or []:
            drives = list(getattr(e, "rf_groups", []) or [])
            what = (f"DC {getattr(e, 'dc', 0.0):g} V" if not drives
                    else "drive " + "+".join(drives))
            add("electrodes", getattr(e, "name", "?"), what,
                f"{len(getattr(e, 'shapes', []) or [])} shape(s)")
    if "drives" in sections and g is not None:
        for grp in getattr(g, "rf_groups", []) or []:
            add("drives", getattr(grp, "name", "?"),
                f"{getattr(grp, 'amplitude_v', float('nan')):g} V @ "
                f"{getattr(grp, 'frequency_hz', float('nan')):g} Hz "
                f"({getattr(grp, 'waveform', '?')}, phase "
                f"{getattr(grp, 'phase_deg', 0.0):g} deg)", "0-pk")
    if "ion" in sections and src is not None:
        add("ion", "m/z", ", ".join(f"{m:g}" for m in
                                    (getattr(src, "mz_list", []) or [])), "Th")
        add("ion", "charge", f"{getattr(src, 'charge', 1)}", "e")
        add("ion", "count", f"{getattr(src, 'n_ions', 0)}", "ions")
        add("ion", "kinetic energy",
            f"{getattr(src, 'ke_lo', float('nan')):g} - "
            f"{getattr(src, 'ke_hi', float('nan')):g}", "eV")
        add("ion", "source temperature",
            f"{getattr(src, 'temperature_k', float('nan')):g}", "K")
        add("ion", "birth position",
            f"({getattr(src, 'x0_mm', float('nan')):g}, "
            f"{getattr(src, 'y0_mm', float('nan')):g}, "
            f"{getattr(src, 'z0_mm', float('nan')):g})", "mm")
        add("ion", "distribution", f"{getattr(src, 'distribution', '?')}")
    if "gas" in sections and col is not None:
        on = bool(getattr(col, "enabled", False))
        add("gas", "collisions", "ON" if on else "OFF")
        if on:
            add("gas", "species", f"{getattr(col, 'gas', '?')}")
            add("gas", "pressure", f"{getattr(col, 'P_torr', float('nan')):g}", "Torr")
            add("gas", "temperature", f"{getattr(col, 'T_k', float('nan')):g}", "K")
            add("gas", "model", f"{getattr(col, 'model', '?')}")
    if "integration" in sections and it is not None:
        add("integration", "dt", f"{getattr(it, 'dt_ns', float('nan')):g}", "ns")
        add("integration", "t_max", f"{getattr(it, 't_max_us', float('nan')):g}", "us")
        add("integration", "rec_every", f"{getattr(it, 'rec_every', 0)}", "steps")
        add("integration", "channels",
            ", ".join(getattr(it, "record_channels", []) or []) or "(none)")
    if "bounds" in sections and getattr(spec, "bounds", None) is not None:
        from ion_gym.physics.stats import enabled_planes
        pl = enabled_planes(spec)
        add("bounds", "enabled planes",
            ", ".join(f"{lbl}={mm:g}" for lbl, _ax, mm in pl) or "(none)", "mm")

    try:
        import pandas as pd
    except ImportError:
        # No pandas: print an aligned table rather than returning nothing.
        if show:
            w = [max(len(str(r[k])) for r in rows + [{"section": "section",
                 "parameter": "parameter", "value": "value", "units": "units"}])
                 for k in ("section", "parameter", "value", "units")]
            hdr = ("section", "parameter", "value", "units")
            print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
            print("  ".join("-" * w[i] for i in range(4)))
            for r in rows:
                print("  ".join(str(r[k]).ljust(w[i]) for i, k in
                                enumerate(("section", "parameter", "value", "units"))))
        return rows
    df = pd.DataFrame(rows, columns=["section", "parameter", "value", "units"])
    if show:
        try:
            from IPython.display import display as _display
            _display(df)
        except ImportError:
            print(df.to_string(index=False))
    return df
