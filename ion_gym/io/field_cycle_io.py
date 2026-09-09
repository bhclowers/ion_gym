"""ion_gym.io.field_cycle_io — export the INSTANTANEOUS E-field over one drive cycle.

`field_io.save_field` stores the per-electrode BASES (geometry, one field per
electrode, time-independent) — the portable-solve artifact. That is not what
you want when you ask for "the field over one RF cycle": you want the COMPOSED
vector field E(x, t) at a sequence of instants, i.e. the bases already weighted
by each channel's waveform and summed. This module builds exactly that from a
tracer3d channel field-pack and writes it as one self-describing npz.

The composition is the tracer's own, not a re-derivation:

    E(x, t) = EA(x)  +  Σ_k  w_k(t) · E_k(x)

with EA the static (guard/DC) field, E_k the per-volt Laplace basis of channel
k, and w_k the SAME scalar waveform the tracer evaluates (ion_gym.physics.
tracer3d._wave_eval). Because every E_k is a Laplace basis and the field is
linear in electrode voltages, this superposition is EXACT — it is the field the
tracer flies, sampled on the grid, not a pseudopotential approximation.

Zeroing a drive (e.g. the SLIM travelling wave) is done by setting that
channel's amplitude to 0 in the pack BEFORE calling here, so the exported field
is composed without it and the exported metadata records amp=0 for those
channels — the file states what it contains.
"""
# PROVENANCE
#   waveform  : ion_gym.physics.tracer3d._wave_eval (kinds 0 sin,1 cos,
#               2 square,3 tab-hold,4 tab-linear) — imported, not copied.
#   field pack: ion_gym.physics.tracer3d field-pack contract
#               (EAx/EAy/EAz, ExK/EyK/EzK, ch_kind/ch_om/ch_ph/ch_amp/
#               ch_off[/ch_duty], tab_t/tab_v/tab_off, h_mm).
import os
import numpy as np


_META = "_meta"
_FORMAT = "field_cycle/1"


def _pack_get(fields, key, default=None):
    """Field packs are plain dicts; some optional members may be absent."""
    v = fields.get(key, default)
    if v is None and default is None:
        raise KeyError(
            f"field pack is missing required member {key!r}; this does not "
            "look like a tracer3d channel pack (expected EAx/ExK/ch_* keys)")
    return v


def cycle_times_us(fields, *, cycle_of="rf", n_samples=64, tob_us=0.0):
    """The sample times (absolute/lab clock, µs) spanning ONE period.

    cycle_of selects WHICH period to span, because a SLIM pack carries two:
      * "rf"  — one RF-rail period (the fast confinement cycle);
      * "tw"  — one travelling-wave period (the slow transport cycle);
      * a bare channel index — that channel's period.
    The waveform clock includes the ion birth time in the tracer, so tob_us
    is carried here too and defaults to 0 (a field snapshot has no ion).

    Endpoint is EXCLUDED (t0 .. t0+T with n_samples points, so sample 0 and
    a would-be sample n are identical and only one is kept) — the natural
    convention for a closed cycle you intend to loop or FFT.
    """
    kinds = np.asarray(_pack_get(fields, "ch_kind"))
    oms = np.asarray(_pack_get(fields, "ch_om"))
    if cycle_of == "rf":
        idx = int(np.flatnonzero(kinds == 0)[0])   # first sin channel = RF
    elif cycle_of == "tw":
        idx = int(np.flatnonzero(kinds == 2)[0])    # first square = TW pad
    else:
        idx = int(cycle_of)
    om = float(oms[idx])                              # rad/µs
    if om == 0.0:
        raise ValueError(
            f"channel {idx} has zero angular frequency; it has no period to "
            "span. Pick a driven channel, or pass cycle_of='rf'/'tw'.")
    period_us = 2.0 * np.pi / om
    return tob_us + np.linspace(0.0, period_us, n_samples, endpoint=False), \
        idx, period_us


def model_drives(model):
    """Enumerate a built model's RF/AC drives, whatever route produced it.

    Different builders store the same physics in different shapes, and an
    exporter that pretended they were identical would be lying about at
    least one of them. This reads each honestly and returns a common
    description: a list of (name, freq_hz, phase_rad) plus a callable
    `phi_at(t_us)` giving the composed potential at any instant.

      * r-z / funnel model: scalar rf_V, om_rad_us, single basis B, static
        A. phi(t) = A + sin(om*t)*rf_V*B.
      * STL / scene model: A plus Bk = [(B_k, f0_k, phase_k), ...], each a
        sin group. phi(t) = A + sum_k sin(2pi f0_k t + phase_k) * B_k.
      * channel-pack model (SLIM): compose_field_at handles the pack
        directly (it carries ch_*); this function is for A/B(k) models and
        reports a pack so the caller routes to compose_cycle instead.

    Returns dict(kind=, drives=[(name,f_hz,ph_rad)], phi_at=callable|None,
                 A=, h_mm=).
    """
    if isinstance(model, dict) and "ch_kind" in model:
        return {"kind": "channel_pack", "drives": None, "phi_at": None}

    h_mm = float(getattr(model, "mm_per_gu", None) or getattr(model, "h_mm", 0.0))
    A = np.asarray(getattr(model, "A"), np.float64)

    Bk = getattr(model, "Bk", None)
    if Bk:
        drives, bases = [], []
        for i, (B, f0, ph_deg) in enumerate(Bk):
            drives.append((f"rf{i}", float(f0), float(np.radians(ph_deg))))
            bases.append(np.asarray(B, np.float64))

        def phi_at(t_us):
            phi = A.copy()
            for (B, (_n, f_hz, ph)) in zip(bases, drives):
                om = 2.0 * np.pi * f_hz * 1e-6
                phi = phi + np.sin(om * t_us + ph) * B
            return phi
        return {"kind": "bk_list", "drives": drives, "phi_at": phi_at,
                "A": A, "h_mm": h_mm}

    B = getattr(model, "B", None)
    if B is not None and getattr(model, "rf_V", 0.0):
        B = np.asarray(B, np.float64)
        rf_V = float(model.rf_V)
        om = float(getattr(model, "om_rad_us", 0.0))
        f_hz = om / (2.0 * np.pi) / 1e-6 if om else 0.0

        def phi_at(t_us):
            return A + np.sin(om * t_us) * rf_V * B
        return {"kind": "rz_scalar",
                "drives": [("rf", f_hz, 0.0)], "phi_at": phi_at,
                "A": A, "h_mm": h_mm}

    return {"kind": "static", "drives": [], "phi_at": lambda t_us: A.copy(),
            "A": A, "h_mm": h_mm}


def compose_model_field_at(model, t_us):
    """Composed E-field components at one instant for an A/B(k) model, V/mm.

    Returns a TUPLE of components, one per solved axis: (Ex, Ey) for a 2-D
    solve, (Ex, Ey, Ez) for a 3-D solve. Gradient of the composed potential
    at the model's own pitch — the same quantity viz_core draws, so a panel
    and an export cannot disagree. The caller must not assume a fixed count;
    a 3-D route (e.g. STL full-3-D, SLIM transport) genuinely has three.
    """
    info = model_drives(model)
    if info["phi_at"] is None:
        raise ValueError("compose_model_field_at is for A/B(k) SimSpec "
                         "models; this looks like a channel pack — use "
                         "compose_field_at(pack, t_us) instead.")
    h = info["h_mm"]
    phi = info["phi_at"](t_us)
    grads = np.gradient(-phi, h)                  # list len == phi.ndim
    if phi.ndim == 1:
        return (np.asarray(grads),)
    return tuple(grads)


def compose_field_at(fields, t_us):
    """The composed E-field (Ex, Ey, Ez) at ONE instant, each (nx, ny, nz).

    E = EA + Σ_k w_k(t) E_k, with w_k = tracer3d._wave_eval. A channel whose
    amplitude is 0 contributes its OFFSET only (w_k = off_k), so a zeroed TW
    pad with off=0 contributes nothing — exactly the intended semantics.
    """
    from ion_gym.physics.tracer3d import _wave_eval  # deferred to call time (io stands below physics)
    EAx = np.asarray(_pack_get(fields, "EAx"), np.float64)
    EAy = np.asarray(_pack_get(fields, "EAy"), np.float64)
    EAz = np.asarray(_pack_get(fields, "EAz"), np.float64)
    ExK = np.asarray(_pack_get(fields, "ExK"), np.float64)
    EyK = np.asarray(_pack_get(fields, "EyK"), np.float64)
    EzK = np.asarray(_pack_get(fields, "EzK"), np.float64)
    kinds = np.asarray(_pack_get(fields, "ch_kind"))
    oms = np.asarray(_pack_get(fields, "ch_om"))
    phs = np.asarray(_pack_get(fields, "ch_ph"))
    amps = np.asarray(_pack_get(fields, "ch_amp"))
    offs = np.asarray(_pack_get(fields, "ch_off"))
    duty = np.asarray(fields.get("ch_duty", np.full(len(kinds), 0.5)))
    tab_t = np.asarray(fields.get("tab_t", np.zeros(0)))
    tab_v = np.asarray(fields.get("tab_v", np.zeros(0)))
    tab_off = np.asarray(fields.get("tab_off", np.zeros(len(kinds) + 1,
                                                        np.int64)))

    Ex = EAx.copy(); Ey = EAy.copy(); Ez = EAz.copy()
    for k in range(len(kinds)):
        o0 = int(tab_off[k]) if k < len(tab_off) else 0
        o1 = int(tab_off[k + 1]) if k + 1 < len(tab_off) else o0
        w = _wave_eval(int(kinds[k]), float(oms[k]), float(phs[k]),
                       float(amps[k]), float(offs[k]), float(duty[k]),
                       tab_t, tab_v, o0, o1, float(t_us))
        if w == 0.0:
            continue
        Ex += w * ExK[k]; Ey += w * EyK[k]; Ez += w * EzK[k]
    return Ex, Ey, Ez


def compose_cycle(fields, *, cycle_of="rf", n_samples=64, tob_us=0.0):
    """Stack the composed field over one cycle.

    Returns (t_us, Ex, Ey, Ez) where t_us is (nt,) and each E is
    (nt, nx, ny, nz), plus the channel state actually used (so the caller
    can see which channels were live). float32 to keep the file reasonable;
    the composition is done in float64 and cast on the way out.
    """
    t_us, drive_idx, period_us = cycle_times_us(
        fields, cycle_of=cycle_of, n_samples=n_samples, tob_us=tob_us)
    nx, ny, nz = np.asarray(fields["EAx"]).shape
    Ex = np.empty((len(t_us), nx, ny, nz), np.float32)
    Ey = np.empty_like(Ex); Ez = np.empty_like(Ex)
    for i, t in enumerate(t_us):
        ex, ey, ez = compose_field_at(fields, float(t))
        Ex[i] = ex; Ey[i] = ey; Ez[i] = ez
    return t_us, Ex, Ey, Ez, drive_idx, period_us


def save_field_cycle(fields, frame=None, path=None, *, cycle_of="rf",
                     n_samples=64, tob_us=0.0, label=None, extra_meta=None):
    """Compose one drive cycle and write it as ONE self-describing npz.

    Arrays in the file:
      t_us   (nt,)              sample times, absolute clock
      Ex,Ey,Ez (nt,nx,ny,nz)   composed field components, V/mm
      x_mm,y_mm,z_mm           axis coordinates (from h_mm and the frame)
      ch_amp,ch_om,ch_ph,ch_kind  the channel state used (amp=0 = off)
    plus a JSON `_meta` member recording format, cycle, period, grid, the
    zeroed channels, and any provided label / extra_meta.

    To EXPORT WITH THE TW OFF: build the pack with tw_amp_v=0 (or set the
    TW channels' ch_amp to 0 before calling). The eight square channels
    then contribute nothing and the metadata shows them at amp 0.
    """
    import json

    t_us, Ex, Ey, Ez, drive_idx, period_us = compose_cycle(
        fields, cycle_of=cycle_of, n_samples=n_samples, tob_us=tob_us)

    h_mm = float(fields.get("h_mm", 1.0))
    nx, ny, nz = np.asarray(fields["EAx"]).shape
    y0_mm = float((frame or {}).get("y0_mm", 0.0))
    z_center = int((frame or {}).get("z_center_gu", (nz - 1) // 2))
    x_mm = np.arange(nx) * h_mm
    y_mm = y0_mm + np.arange(ny) * h_mm
    z_mm = (np.arange(nz) - z_center) * h_mm      # mirror plane at z=0

    amps = np.asarray(fields["ch_amp"], float)
    zeroed = [int(k) for k in np.flatnonzero(amps == 0.0)]
    meta = {
        "format": _FORMAT,
        "cycle_of": cycle_of,
        "drive_channel": int(drive_idx),
        "period_us": float(period_us),
        "n_samples": int(n_samples),
        "tob_us": float(tob_us),
        "grid": {"nx": nx, "ny": ny, "nz": nz, "h_mm": h_mm},
        "units": {"E": "V/mm", "t": "us", "xyz": "mm"},
        "n_channels": int(len(amps)),
        "zeroed_channels": zeroed,
        "label": label,
    }
    if extra_meta:
        meta.update(extra_meta)

    if path is None:
        path = f"field_cycle_{cycle_of}_{n_samples}.npz"
    # np.savez_compressed appends '.npz' if the name lacks it; write to a
    # temp basename, then rename the file it actually produced to `path`.
    tmp_base = path + ".tmp"
    np.savez_compressed(
        tmp_base, t_us=t_us, Ex=Ex, Ey=Ey, Ez=Ez,
        x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
        ch_amp=amps, ch_om=np.asarray(fields["ch_om"], float),
        ch_ph=np.asarray(fields["ch_ph"], float),
        ch_kind=np.asarray(fields["ch_kind"]),
        **{_META: np.frombuffer(json.dumps(meta).encode("utf-8"),
                                dtype=np.uint8)})
    written = tmp_base if os.path.exists(tmp_base) else tmp_base + ".npz"
    os.replace(written, path)
    return path, meta


# export_slim_rf_cycle is RETIRED: it was a thin
# wrapper over slim3d.build_slim3d_fields, deleted with the module. The
# generic exporter above serves any channel pack (scene3d/shapes3d/stl3d).

def estimate_cycle_bytes(model_or_pack, *, n_samples, dtype="float32",
                         n_components=None):
    """Estimate the exported .npz payload size in bytes BEFORE composing.

    Field arrays dominate: n_samples * n_nodes * n_stored_components *
    itemsize. n_components defaults to 3 (the file always stores Ex,Ey,Ez
    for a uniform shape, with unused components zero-filled); pass an
    explicit count only to size a hypothetical. Axes and metadata are
    kilobytes, added as flat overhead. The figure is the UNCOMPRESSED array
    size — npz compression on smooth fields lands well under it — so it is a
    safe upper bound to show before committing.
    """
    if isinstance(model_or_pack, dict) and "EAx" in model_or_pack:
        shape = np.asarray(model_or_pack["EAx"]).shape
    else:
        shape = np.asarray(getattr(model_or_pack, "A")).shape
    n_nodes = int(np.prod(shape))
    itemsize = 4 if str(dtype) == "float32" else 8
    stored_components = 3 if n_components is None else int(n_components)
    field_bytes = int(n_samples) * n_nodes * stored_components * itemsize
    axes_bytes = int(sum(shape)) * 8
    return {"n_nodes": n_nodes, "grid": tuple(int(s) for s in shape),
            "n_samples": int(n_samples), "dtype": str(dtype),
            "n_components": stored_components,
            "field_bytes": field_bytes,
            "total_bytes": field_bytes + axes_bytes + 4096}


def save_model_cycle(model, path=None, *, freq_hz=None, n_samples=64,
                     total_time_us=None, dtype="float32", tob_us=0.0,
                     label=None, extra_meta=None):
    """Compose and save an A/B(k) model's field over a time span.

    freq_hz: the drive whose period defines "one cycle". Default: the
        model's first drive. Ignored if total_time_us is given.
    total_time_us: export this many µs instead of exactly one period (for
        several cycles, or a fixed window). n_samples spans it.
    dtype: 'float32' (half the size, ample for display/most transport) or
        'float64' (exports at solver precision).

    File layout matches save_field_cycle: t_us, Ex/Ey/Ez, x_mm/y_mm/z_mm,
    drive table, JSON _meta. Ez is written as zeros for a 2-D solve (the
    plane-normal field is zero in the solved plane) so every export has the
    same shape regardless of route.
    """
    import json

    info = model_drives(model)
    if info["phi_at"] is None:
        raise ValueError("save_model_cycle is for A/B(k) SimSpec models; "
                         "for a SLIM channel pack use save_field_cycle.")
    h_mm = info["h_mm"]

    if total_time_us is not None:
        span_us = float(total_time_us)
        period_us = span_us
    else:
        f = freq_hz
        if f is None:
            if not info["drives"]:
                raise ValueError(
                    "this model has no RF/AC drive, so there is no cycle to "
                    "span. Pass total_time_us to export a fixed window "
                    "(the field is static — every frame is identical).")
            f = info["drives"][0][1]
        if f <= 0:
            raise ValueError(f"drive frequency {f} Hz has no period.")
        period_us = 1e6 / f
        span_us = period_us

    t_us = tob_us + np.linspace(0.0, span_us, int(n_samples), endpoint=False)
    npdt = np.float32 if str(dtype) == "float32" else np.float64
    A_shape = np.asarray(info["A"]).shape
    ndim = len(A_shape)

    # Component count follows the SOLVE dimensionality: 2 for a plane, 3 for
    # a volume. A 3-D route (STL full-3-D, SLIM transport) genuinely has Ez;
    # a 2-D route has none and we write zeros so every file has (Ex,Ey,Ez).
    comp = {ax: np.empty((len(t_us),) + A_shape, npdt) for ax in range(ndim)}
    for i, t in enumerate(t_us):
        grads = compose_model_field_at(model, float(t))
        for ax in range(ndim):
            comp[ax][i] = grads[ax].astype(npdt)
    Ex = comp[0]
    Ey = comp[1] if ndim >= 2 else np.zeros_like(Ex)
    Ez = comp[2] if ndim >= 3 else np.zeros_like(Ex)

    # axis coordinates, one per solved axis (mm)
    axes_mm = [np.arange(n) * h_mm for n in A_shape]
    x_mm = axes_mm[0]
    y_mm = axes_mm[1] if ndim >= 2 else np.zeros(1)
    z_mm = axes_mm[2] if ndim >= 3 else np.zeros(1)

    grid = {"nx": int(A_shape[0]),
            "ny": int(A_shape[1]) if ndim >= 2 else 1,
            "nz": int(A_shape[2]) if ndim >= 3 else 1,
            "h_mm": h_mm}
    meta = {
        "format": _FORMAT, "compose": info["kind"], "ndim": ndim,
        "period_us": float(period_us), "span_us": float(span_us),
        "n_samples": int(n_samples), "tob_us": float(tob_us),
        "grid": grid,
        "units": {"E": "V/mm", "t": "us", "xyz": "mm"},
        "drives": [{"name": n, "freq_hz": f, "phase_rad": p}
                   for (n, f, p) in info["drives"]],
        "dtype": str(dtype), "label": label,
    }
    if extra_meta:
        meta.update(extra_meta)

    if path is None:
        path = f"field_cycle_{info['kind']}_{n_samples}.npz"
    tmp_base = path + ".tmp"
    np.savez_compressed(
        tmp_base, t_us=t_us, Ex=Ex, Ey=Ey, Ez=Ez,
        x_mm=x_mm, y_mm=y_mm, z_mm=z_mm,
        drive_freq_hz=np.array([d[1] for d in info["drives"]], float),
        drive_phase_rad=np.array([d[2] for d in info["drives"]], float),
        **{_META: np.frombuffer(json.dumps(meta).encode("utf-8"),
                                dtype=np.uint8)})
    written = tmp_base if os.path.exists(tmp_base) else tmp_base + ".npz"
    os.replace(written, path)
    return path, meta


def read_field_cycle(path):
    """Load an exported cycle. Returns (data_dict, meta_dict)."""
    import json
    d = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(d[_META]).decode("utf-8")) if _META in d else {}
    data = {k: d[k] for k in d.files if k != _META}
    return data, meta
