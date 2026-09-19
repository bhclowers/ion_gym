"""ion_gym.io.field_cycle_io — export the APPLIED E-field over a time window.

`field_io.save_field` stores the per-electrode BASES (geometry, one field per
electrode, time-independent) — the portable-solve artifact. This module
writes the other thing people ask for: the COMPOSED vector field E(x, t) at a
sequence of instants over a window the caller chooses, with EVERY drive the
flight applies (RF, travelling-wave squares, tables, offsets, gates), exactly
as the route's kernel composes it.

WHAT "APPLIED" MEANS (PI ruling 2026-09-16). Each frame is the field an ion
would feel at that instant. Exporting one RF period of a SLIM therefore
exports the RF swinging with the travelling wave HELD at its state over that
window — not removed. The window start is a real parameter: which TW state is
captured depends on it. Channels that switch inside the window are named in
the metadata.

ONE COMPOSITION AUTHORITY PER ROUTE. Frames are built from the arrays the
kernel flies, with the kernel's own waveform evaluator where one exists:

  * 3-D (stl3d / scene3d / shapes3d): model.fly_fields channel pack,
        E = EA + sum_k w_k(t) E_k,  w_k = tracer3d._wave_eval (amp, offset,
        duty, tables); arrays already V/mm.
  * planar / stl2d: PlanarModel ExA/EyA + sum_k w_k(t) ExK/EyK,
        w_k = build_planar._wave_eval (unit waveform, amplitude baked in);
        arrays V/m, exported as V/mm.
  * r-z: RZModel EzA/EuA + sum_k w_k(t) EzK/EuK + gate(t) EzG/EuG — the
        same per-group channels tracer_rz._fly_rec_full integrates, with
        the same unit evaluator; arrays V/m on the mirror-extended radial
        grid, exported as V/mm.

The field method (electrode-aware or plain gradient) is therefore the one the
build applied, not a re-derivation here.

REFUSE, DON'T DROP. Each builder module names the declared drive features its
kernel does not apply (unsupported_drive_features). If any are present the
export refuses with that list: a file claiming to hold the applied field must
not silently disagree with the deck.

COORDINATES are in the deck's own frame: 3-D axes carry the pack's
world_off_mm (a z-mirror plane sits at z = 0), planar axes carry anchor_mm,
r-z axes are (axial x, radial r) with r spanning the mirror-extended grid.
"""
import os
import numpy as np


_META = "_meta"
_FORMAT = "field_window/2"
_3D_BUILDERS = ("stl3d", "scene3d", "shapes3d")
_PLANAR_BUILDERS = ("planar", "stl2d")
_WAVE_NAMES = {0: "sin", 1: "cos", 2: "square", 3: "table_hold",
               4: "table_linear"}
# the 2-D kernels store fields in V/m (build_planar._grad2d and
# ionbench.build_field_aware scale the mm-grid gradient by 1e3); the export
# contract is V/mm, as recorded e_field channels are
_V_PER_M_TO_V_PER_MM = 1e-3


def _builder(spec):
    from ion_gym.physics.sim_build import build_route  # io stands below physics
    return build_route(spec).builder


def _refuse_unsupported(spec, builder):
    """Refuse when the deck declares drives this route's kernel drops."""
    if builder in _3D_BUILDERS:
        from ion_gym.physics.build_stl3d import unsupported_drive_features
    elif builder == "planar":
        from ion_gym.physics.build_planar import unsupported_drive_features
    elif builder == "stl2d":
        from ion_gym.physics.build_stl import unsupported_drive_features
    elif builder == "rz":
        from ion_gym.physics.build_rz import unsupported_drive_features
    else:
        raise ValueError(f"field export: no composition authority for route "
                         f"{builder!r}; supported: "
                         f"{_3D_BUILDERS + _PLANAR_BUILDERS + ('rz',)}")
    found = unsupported_drive_features(spec)
    if found:
        raise ValueError(
            f"field export refused: the {builder} kernel does not apply "
            f"every drive this deck declares, so no file can hold the "
            f"applied field:\n  - " + "\n  - ".join(found))


def _pack_3d(model):
    f = getattr(model, "fly_fields", None)
    if not isinstance(f, dict) or f.get("route") != "3d":
        raise ValueError("3-D field export needs the built model's fly_fields "
                         "channel pack (route '3d'); this model has none")
    return f


def field_at(spec, model, t_us):
    """The applied field at ONE lab-clock instant, V/mm.

    Returns (Ex, Ey, Ez): each (nx, ny, nz) on 3-D routes; (nx, ny) on the
    2-D routes, where Ez is zeros (planar: the plane-normal field is zero;
    r-z: Ex is AXIAL, Ey is RADIAL, the azimuthal field is zero).
    """
    builder = _builder(spec)
    t = float(t_us)
    if builder in _3D_BUILDERS:
        from ion_gym.physics.tracer3d import _wave_eval
        f = _pack_3d(model)
        Ex = np.asarray(f["EAx"], np.float64).copy()
        Ey = np.asarray(f["EAy"], np.float64).copy()
        Ez = np.asarray(f["EAz"], np.float64).copy()
        kinds = np.asarray(f["ch_kind"])
        duty = np.asarray(f.get("ch_duty", np.full(len(kinds), 0.5)))
        tab_t, tab_v = np.asarray(f["tab_t"]), np.asarray(f["tab_v"])
        tab_off = np.asarray(f["tab_off"])
        for k in range(len(kinds)):
            w = _wave_eval(int(kinds[k]), float(f["ch_om"][k]),
                           float(f["ch_ph"][k]), float(f["ch_amp"][k]),
                           float(f["ch_off"][k]), float(duty[k]), tab_t, tab_v,
                           int(tab_off[k]), int(tab_off[k + 1]), t)
            if w != 0.0:
                Ex += w * f["ExK"][k]
                Ey += w * f["EyK"][k]
                Ez += w * f["EzK"][k]
        return Ex, Ey, Ez
    elif builder in _PLANAR_BUILDERS:
        from ion_gym.physics.build_planar import _wave_eval
        Ex = np.asarray(model.ExA, np.float64).copy()
        Ey = np.asarray(model.EyA, np.float64).copy()
        for k in range(len(model.ch_kind)):
            w = _wave_eval(int(model.ch_kind[k]), float(model.ch_om[k]),
                           float(model.ch_ph[k]), float(model.ch_duty[k]),
                           model.tab_t, model.tab_v,
                           int(model.tab_off[k]), int(model.tab_off[k + 1]), t)
            if w != 0.0:
                Ex += w * model.ExK[k]
                Ey += w * model.EyK[k]
        s = _V_PER_M_TO_V_PER_MM
        return Ex * s, Ey * s, np.zeros_like(Ex)
    elif builder == "rz":
        # tracer_rz._fly_rec_full: A + sum_k w_k(t) E_k + step(t - tau) G
        from ion_gym.physics.build_planar import _wave_eval
        from ion_gym.physics.build_rz import _TAB_T0, _TAB_V0
        Ex = np.asarray(model.EzA, np.float64).copy()
        Ey = np.asarray(model.EuA, np.float64).copy()
        for k in range(len(model.ch_kind)):
            w = _wave_eval(int(model.ch_kind[k]), float(model.ch_om[k]),
                           float(model.ch_ph[k]), float(model.ch_duty[k]),
                           _TAB_T0, _TAB_V0, 0, 0, t)
            if w != 0.0:
                Ex += w * model.EzK[k]
                Ey += w * model.EuK[k]
        if model.tau_gate >= 0.0 and t >= model.tau_gate:
            Ex += model.EzG
            Ey += model.EuG
        s = _V_PER_M_TO_V_PER_MM
        return Ex * s, Ey * s, np.zeros_like(Ex)
    raise ValueError(f"field export: no composition authority for route "
                     f"{builder!r}")


def field_axes_mm(spec, model):
    """Node coordinates in the DECK frame, mm: (x_mm, y_mm, z_mm, axis_names).

    3-D: world_off_mm + i*h per axis. planar/stl2d: anchor_mm + i*h, z (0,).
    r-z: x axial i*h, y radial u0 + j*h over the mirror-extended grid, z (0,).
    """
    builder = _builder(spec)
    if builder in _3D_BUILDERS:
        f = _pack_3d(model)
        h = float(f["h_mm"])
        off = np.asarray(f.get("world_off_mm", (0.0, 0.0, 0.0)), float)
        nx, ny, nz = np.asarray(f["EAx"]).shape
        return (off[0] + np.arange(nx) * h, off[1] + np.arange(ny) * h,
                off[2] + np.arange(nz) * h, ("x", "y", "z"))
    elif builder in _PLANAR_BUILDERS:
        h = float(model.mm_per_gu)
        ax, ay = model.anchor_mm
        nx, ny = np.asarray(model.ExA).shape
        return (ax + np.arange(nx) * h, ay + np.arange(ny) * h,
                np.zeros(1), ("x", "y", "z (planar: one plane)"))
    elif builder == "rz":
        h = float(model.mm_per_gu)
        nx, nu = np.asarray(model.EzA).shape
        return (np.arange(nx) * h, float(model.u0) + np.arange(nu) * h,
                np.zeros(1), ("x = axial", "y = radial r (mirror-extended)",
                              "z (r-z: none)"))
    raise ValueError(f"field export: no coordinate authority for route "
                     f"{builder!r}")


def _switch_count(kind, om, ph, duty, t0, t1, tab_t=None):
    """How many times a channel's waveform changes value in [t0, t1).
    None for continuous waveforms (sin/cos at non-zero frequency)."""
    if kind in (0, 1):
        return None if om != 0.0 else 0
    if kind == 2:
        if om == 0.0:
            return 0
        # sign(sin) switches where om t + ph = n pi (duty 0.5); otherwise
        # where the phase fraction crosses 0 or duty
        edges = (0.0, 0.5) if duty == 0.5 else (0.0, float(duty))
        n = 0
        for e in edges:
            a, b = sorted(((om * t0 + ph) / (2 * np.pi) - e,
                           (om * t1 + ph) / (2 * np.pi) - e))
            # integers strictly inside (a, b): a switch exactly at the
            # window start is the state the window opens in, not a change
            n += max(0, int(np.ceil(b) - np.floor(a) - 1))
        return n
    tt = np.asarray(tab_t if tab_t is not None else [], float)
    return int(np.count_nonzero((tt > t0) & (tt < t1)))


def channel_table(spec, model, t0_us, span_us):
    """Every drive the export applies, as plain dicts, with how often each
    changes inside the window (null = continuously varying)."""
    builder = _builder(spec)
    t1 = t0_us + span_us
    out = []
    if builder in _3D_BUILDERS:
        f = _pack_3d(model)
        kinds = np.asarray(f["ch_kind"])
        names = f.get("ch_name") or [f"channel_{k}" for k in range(len(kinds))]
        duty = np.asarray(f.get("ch_duty", np.full(len(kinds), 0.5)))
        for k in range(len(kinds)):
            o0, o1 = int(f["tab_off"][k]), int(f["tab_off"][k + 1])
            om = float(f["ch_om"][k])
            out.append(dict(
                name=str(names[k]), waveform=_WAVE_NAMES[int(kinds[k])],
                freq_hz=om / (2 * np.pi) * 1e6,
                phase_deg=float(np.degrees(f["ch_ph"][k])),
                amplitude_v=float(f["ch_amp"][k]),
                offset_v=float(f["ch_off"][k]), duty=float(duty[k]),
                table_points=o1 - o0,
                switches_in_window=_switch_count(
                    int(kinds[k]), om, float(f["ch_ph"][k]), float(duty[k]),
                    t0_us, t1, np.asarray(f["tab_t"])[o0:o1])))
    elif builder in _PLANAR_BUILDERS:
        for k in range(len(model.ch_kind)):
            o0, o1 = int(model.tab_off[k]), int(model.tab_off[k + 1])
            om = float(model.ch_om[k])
            duty = float(model.ch_duty[k])
            out.append(dict(
                name=f"channel_{k}", waveform=_WAVE_NAMES[int(model.ch_kind[k])],
                freq_hz=om / (2 * np.pi) * 1e6,
                phase_deg=float(np.degrees(model.ch_ph[k])),
                amplitude_v="baked into the channel basis",
                table_points=o1 - o0, duty=duty,
                switches_in_window=_switch_count(
                    int(model.ch_kind[k]), om, float(model.ch_ph[k]), duty,
                    t0_us, t1, model.tab_t[o0:o1])))
    elif builder == "rz":
        for k, (_B, gr) in enumerate(model.drives):
            om = float(model.ch_om[k])
            duty = float(model.ch_duty[k])
            out.append(dict(
                name=gr.name, waveform=_WAVE_NAMES[int(model.ch_kind[k])],
                freq_hz=om / (2 * np.pi) * 1e6,
                phase_deg=float(np.degrees(model.ch_ph[k])),
                amplitude_v=float(gr.amplitude_v),
                offset_v=float(getattr(gr, "offset_v", 0.0)), duty=duty,
                switches_in_window=_switch_count(
                    int(model.ch_kind[k]), om, float(model.ch_ph[k]), duty,
                    t0_us, t1)))
        if model.tau_gate >= 0.0:
            out.append(dict(name="gate", waveform="step",
                            t_step_us=float(model.tau_gate),
                            switches_in_window=int(t0_us < model.tau_gate
                                                   < t1)))
    else:
        raise ValueError(f"field export: no channel authority for route "
                         f"{builder!r}")
    return out


def estimate_window_bytes(spec, model, *, n_samples, dtype="float32"):
    """Uncompressed size of the export BEFORE composing — a safe upper bound.

    n_samples * nodes * 3 stored components * itemsize, plus axes and a flat
    metadata allowance. Nodes are the kernel's own grid (the r-z grid is the
    mirror-extended one the kernel flies).
    """
    x, y, z, _ = field_axes_mm(spec, model)
    n_nodes = int(len(x) * len(y) * len(z))
    itemsize = {"float32": 4, "float64": 8}.get(str(dtype))
    if itemsize is None:
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
    field_bytes = int(n_samples) * n_nodes * 3 * itemsize
    return {"n_nodes": n_nodes, "grid": (len(x), len(y), len(z)),
            "n_samples": int(n_samples), "dtype": str(dtype),
            "field_bytes": field_bytes,
            "total_bytes": field_bytes + (len(x) + len(y) + len(z)) * 8 + 65536}


def save_field_window(spec, model, path, *, t_start_us, total_time_us,
                      n_samples, dtype="float32", label=None, extra_meta=None):
    """Write the applied field over [t_start_us, t_start_us + total_time_us).

    spec/model: the deck and the model build_run returned for it (the model
        carries the arrays the kernel flies; the spec names the route and
        is checked for drives the kernel does not apply — refused if any).
    t_start_us: lab-clock start of the window (µs). Which state a slow drive
        is captured in depends on this.
    total_time_us: window length (µs). n_samples frames, endpoint excluded,
        so frames sit at t_start + k * total/n.
    dtype: 'float32' or 'float64' for the stored field.

    File: t_us (nt,), Ex/Ey/Ez (nt, nx, ny[, nz]) V/mm, x_mm/y_mm/z_mm in the
    deck frame, and a JSON _meta with the route, field method, units, axis
    meaning, window, and every applied channel with its switch count.
    """
    import json
    if not (float(total_time_us) > 0.0):
        raise ValueError(f"total_time_us must be > 0, got {total_time_us}")
    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    npdt = {"float32": np.float32, "float64": np.float64}.get(str(dtype))
    if npdt is None:
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")
    builder = _builder(spec)
    _refuse_unsupported(spec, builder)

    t0, span, n = float(t_start_us), float(total_time_us), int(n_samples)
    t_us = t0 + np.arange(n) * (span / n)
    x_mm, y_mm, z_mm, axis_names = field_axes_mm(spec, model)
    first = field_at(spec, model, t_us[0])
    shape = first[0].shape
    comp = [np.empty((n,) + shape, npdt) for _ in range(3)]
    for c in range(3):
        comp[c][0] = first[c]
    for i in range(1, n):
        e = field_at(spec, model, t_us[i])
        for c in range(3):
            comp[c][i] = e[c]

    channels = channel_table(spec, model, t0, span)
    switching = [c["name"] for c in channels if c.get("switches_in_window")]
    meta = {
        "format": _FORMAT, "route": builder,
        "field_method": getattr(spec.geometry, "field_method", None),
        "units": {"E": "V/mm", "t": "us", "xyz": "mm"},
        "axes": list(axis_names),
        "frame": "deck coordinates (x_mm/y_mm/z_mm are node positions)",
        "window": {"t_start_us": t0, "total_time_us": span, "n_samples": n,
                   "frame_dt_us": span / n, "endpoint": "excluded"},
        "grid": {"shape": [int(s) for s in shape],
                 "h_mm": float(x_mm[1] - x_mm[0]) if len(x_mm) > 1 else None},
        "channels": channels,
        "held_channels": [c["name"] for c in channels
                          if c.get("switches_in_window") == 0],
        "switching_in_window": switching,
        "dtype": str(dtype), "label": label, "deck": spec.name,
    }
    if extra_meta:
        meta.update(extra_meta)

    tmp_base = str(path) + ".tmp"
    np.savez_compressed(
        tmp_base, t_us=t_us, Ex=comp[0], Ey=comp[1], Ez=comp[2],
        x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
        **{_META: np.frombuffer(json.dumps(meta).encode("utf-8"),
                                dtype=np.uint8)})
    written = tmp_base if os.path.exists(tmp_base) else tmp_base + ".npz"
    os.replace(written, path)
    return str(path), meta


def read_field_window(path):
    """Load an exported window. Returns (data_dict, meta_dict)."""
    import json
    with np.load(path, allow_pickle=False) as d:
        if _META not in d.files:
            raise ValueError(f"{os.path.basename(str(path))}: no {_META} "
                             f"member — not an ion_gym field-window export")
        meta = json.loads(bytes(d[_META]).decode("utf-8"))
        data = {k: d[k] for k in d.files if k != _META}
    return data, meta
