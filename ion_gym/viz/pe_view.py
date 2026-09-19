"""
pe_view.py — node-centred potential-energy landscape ("rubber sheet").

The v51 PE mode drew the effective potential as a flat 60%-opacity heatmap
under the trajectories, which is hard to read. This module renders the
classic PE view instead:

  * a 3-D surface at height z = PE(x, y) with a fine grid-mesh look
    (the green graph-paper sheet),
  * red equipotential contour lines draped ON the surface,
  * ion trajectories draped at the local surface height (the ion "rolls
    on the landscape"), coloured by fate like the 2-D view,
  * electrode plateaus outlined (they are the flat tops — equipotentials
    at the electrode voltage).

Figure-builder only (no Panel): sim_app wires it as a display mode; tests
call it headless. Works for any model exposing pe_surface(mz) — the same
DC + RF-Dehmelt reduction validated in validate_pe_surface.py — so the
adiabatic caveat annotation is carried onto the 3-D view too.
"""
from __future__ import annotations
from ion_gym.viz.viz_core import world_off as _world_off
import numpy as np
import plotly.graph_objects as go
from ion_gym.viz import viz_core as V
from ion_gym.viz import viz_core as _VIZ

# Fate colours from THE single authority (physics.ion_envelope); the
# local copy stopped at 3, so station fates drew in the fallback blue.
from ion_gym.physics.ion_envelope import FATE_COLOR  # noqa: F401

# classic palette: pale green sheet, darker green mesh, red contours,
# blue default trajectories.

# trajectory CHANNELS that lie IN each plane, in the same order pe_surface
# returns its (a, b) in-plane coordinate arrays; and the plane's normal
# channel. Name-based — resolved through TrajRecord over the
# guaranteed BASE_CHANNELS prefix (column_names() always puts
# [t,x,y,z,vx,vy,vz] first, so the prefix schema holds for every run).
_PLANE_TRAJ_AXES = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
_PLANE_NORMAL_AXIS = {"xy": "z", "xz": "y", "yz": "x"}

# THE default 3-D view every PE figure has always used; camera=None
# resolves to exactly this (the UI's view checkbox off, the API default).
_PE_CAMERA_DEFAULT = dict(eye=dict(x=1.5, y=-1.7, z=0.9))


def camera_eye(azim_deg=-48.6, elev_deg=21.7, dist=2.44, ortho=False):
    """Plotly scene camera from viewing angles: azimuth about +z from +x
    toward +y, elevation above the xy plane, dist = |eye| in plotly's
    normalized units. The DOCUMENTED defaults reproduce the historical
    eye (1.5, -1.7, 0.9) to 3 significant figures; camera=None in
    pe_figure_3d bypasses this and uses that eye exactly."""
    import math as _m
    az, el = _m.radians(float(azim_deg)), _m.radians(float(elev_deg))
    d = float(dist)
    eye = dict(x=d * _m.cos(el) * _m.cos(az),
               y=d * _m.cos(el) * _m.sin(az),
               z=d * _m.sin(el))
    cam = dict(eye=eye)
    if ortho:
        cam["projection"] = dict(type="orthographic")
    return cam


def plane_axis_names(plane):
    """(a_name, b_name) for a plane's in-plane axes, matching the order
    the surface accessors return (a, b). NOT simply plane[0]/plane[1]:
    the r-z accessors return (z, r_full) — axial first, per Convention A
    (axial horizontal) — so 'rz' maps to ('z', 'r')."""
    if plane == "rz":
        return ("z", "r")
    return (plane[0], plane[1])
_SHEET = [[0.0, "#d8f3dc"], [0.5, "#b7e4c7"], [1.0, "#74c69d"]]
_MESH = "#2d6a4f"
_CONTOUR = "#c1121f"


def _pe_sampler(x, y, PE):
    """Bilinear PE(x, y) sampler on the model grid (mm in, eV out)."""
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator((np.asarray(x), np.asarray(y)), PE,
                                   bounds_error=False, fill_value=None)


def scale_pe(PE, mode="linear", factor=1.0, quantity=("PE", "eV")):
    """Transform PE for the vertical axis.
      linear: PE * factor            (uniform exaggeration)
      asinh : factor*arcsinh(PE/factor)  — ~linear for |PE|<factor,
              logarithmic above, so a deep well stops flattening the
              shallow structure around it. `factor` is the eV knee.
    Returns (PEs, axis_label)."""
    PE = np.asarray(PE, float)
    _q, _u = quantity
    if mode == "asinh":
        s = max(float(factor), 1e-9)
        return s * np.arcsinh(PE / s), f"asinh({_q} / {s:g} {_u})"
    f = float(factor)
    return PE * f, (f"{_q} [{_u}]" if abs(f - 1.0) < 1e-9
                    else f"{_q} x{f:g} [{_u}]")


def slice_index(model, plane, slice_mm):
    """Node index along `plane`'s NORMAL axis for a slice position in mm
    (canonical/world frame, mirror planes at 0 -- the frame the UI's slice
    control and every figure axis use). The ONE mm -> node rule: both PE
    tabs and the API call it.

    Refuses, naming what did not line up, rather than guessing: a plane
    with no normal (a 2-D or r-z model has nothing to slice), a model
    without a grid pitch, and a position outside the solved grid (the
    valid mm range is stated)."""
    ax = {"xy": 2, "xz": 1, "yz": 0}.get(plane)
    A = getattr(model, "A", None)
    if ax is None or A is None or np.ndim(A) != 3:
        raise ValueError(
            f"slice_mm={slice_mm!r}: {type(model).__name__} plane {plane!r} "
            f"has no normal axis to slice (slicing needs a 3-D model and a "
            f"plane in ['xy', 'xz', 'yz'])")
    h = float(getattr(model, "h_mm", 0.0) or 0.0)
    if not h > 0.0:
        raise ValueError(f"{type(model).__name__} has no positive grid "
                         f"pitch h_mm ({h!r}); cannot map mm to a node")
    off = float(_world_off(model)[ax])
    n = int(np.shape(A)[ax])
    k = int(round((float(slice_mm) - off) / h))
    if not (0 <= k < n):
        raise ValueError(
            f"slice {'xyz'[ax]} = {slice_mm!r} mm is outside the solved "
            f"grid [{off:g}, {off + (n - 1) * h:g}] mm ({n} nodes at "
            f"{h:g} mm)")
    return k


def _slice_label(model, plane, index):
    """'z = 1.2 mm (node 12)' for a 3-D slice (auto slices resolved to the
    node the model actually cut), or None where there is no slice."""
    ax = {"xy": 2, "xz": 1, "yz": 0}.get(plane)
    if ax is None or not hasattr(model, "_plane_slice"):
        return None
    _ax, k = model._plane_slice(plane, index)
    pos = float(_world_off(model)[ax]) + k * float(model.h_mm)
    how = "electrode-dense auto" if index is None else "requested"
    return f"slice {'xyz'[ax]} = {pos:.4g} mm (node {k}, {how})"


def model_instant_drives(model):
    """Names of drive groups the PE view evaluates at a LAB TIME (their
    resolved pe_mode is 'instant' — slow waves the ions surf, not
    average). Empty means t_us does not move the surface. Reads the same
    channel-aligned group lists every route's pe_surface reduces:
    planar/r-z `drives`, 3-D `chan_groups`."""
    pairs = getattr(model, "drives", None)
    if pairs is not None:
        groups = [g for _B, g in pairs]
    else:
        groups = list(getattr(model, "chan_groups", []) or [])
    return [g.name for g in groups if g.resolved_pe_mode() == "instant"]


def model_has_rf(model):
    """True if the model carries any drive: a planar/r-z `drives` list, a
    3-D channel-group list, a sin Bk list, or the 3-D display pair's
    rf_V. The old rf_V-only fallback was a hidden branch: a SQUARE-only
    r-z deck (sin Bk empty, scalar rf_V removed in L-455) read as
    drive-free."""
    if getattr(model, "drives", None) or getattr(model, "chan_groups", None):
        return True
    if getattr(model, "Bk", None):
        return True
    om = getattr(model, "om_rad_us", None)
    if om is None:
        om = getattr(model, "rf_om", 0.0)
    return getattr(model, "rf_V", 0.0) != 0.0 and (om or 0.0) > 0.0


def pe_figure_3d(model=None, mz=None, results=None, *, charge=1,
                 traj_color=None, traj_width=3,
                 n_contours=14, max_traj=60, decimate=1,
                 z_exaggerate=1.0, height=560, stride=1, surface=None,
                 title=None, pe_scale_mode="linear",
                 show_adiabatic_caveat=False, z_aspect=0.55,
                 metal_mode="barrier", electrode_dc=None, plane=None,
                 trust_cells=2, quantity=("PE", "eV"),
                 normal_mm=None, h_mm=0.1, t_us=0.0,
                 component="effective", index=None, slice_mm=None,
                 subtitles=(), camera=None):
    """node-centred 3-D PE landscape.

    model: planar/r-z model exposing pe_surface(mz, charge)->(x,y,PE,ele).
    surface: optional precomputed (x, y, PE, ele) — lets a caller pick the
        component (DC / RF-pseudo / combined) and reuse a cached reduction.
    stride: decimate the surface grid (>1) for a responsive 3-D render.
    z_exaggerate / pe_scale_mode: vertical scaling (see scale_pe) —
        'linear' uses z_exaggerate as a factor, 'asinh' as an eV knee that
        tames deep wells so they don't obscure shallow detail.
    traj_color / traj_width: appearance of the overlaid ion paths.
        traj_color=None (default) colours each path by its FATE, using the
        same FATE_COLOR map as every other figure in the toolkit; pass a
        single colour string to draw them all alike, or a {fate: colour}
        dict to override selected fates. Display preference only — it
        changes no value, position or fate.
    title: figure title (be specific: device · component · m/z).
    plane: None (default) = the MODEL's own first declared plane ("rz"
        for an r-z model), so a bare call works on every route; an
        explicit plane the model does not declare refuses. With a
        precomputed `surface`, None labels axes as "xy".
    metal_mode: "barrier" (default) draws electrodes AT the barrier
        landmark so the sheet stays INTACT -- no NaN holes or ragged
        tears where metal cuts the slice; "mask" keeps the honest NaN
        holes; "dc" draws q*V_DC.
    camera: None = the default view (eye 1.5,-1.7,0.9, perspective) --
        BYTE-IDENTICAL to every figure before this knob existed. Else a
        dict(azim_deg=, elev_deg=, dist=, ortho=False): azimuth about
        +z from +x toward +y, elevation above the xy plane, dist in
        plotly eye units (~2.4 matches the default), ortho=True for an
        orthographic projection. Built by camera_eye().
    component: 'effective' (DC + RF pseudo), 'dc', 'rf' (pseudo only) or
        'efield' (|E|) -- exactly the PE tab's component selector, via
        compute_component. Ignored when `surface` is given.
    index / slice_mm: the slice on a 3-D model's plane NORMAL, as a node
        index or as a position in mm (the tab's slice control; mapped by
        slice_index, which refuses off-grid positions). Give one, not
        both; neither = the electrode-dense auto slice. The resolved
        slice is stamped on the figure.
    subtitles: extra header lines; placed with every other piece of
        figure text by viz_core.layout_plotly_text, so nothing overlaps.
    t_us: LAB TIME the surface's 'instant' drives are evaluated at (a
        slow travelling wave is drawn where it actually is at t_us, so
        stepping t_us animates the wave). Ignored by surfaces that carry
        no instant drive; the figure is stamped with the time whenever
        the model has one, so a frame lifted into a report says WHEN it
        is.
    show_adiabatic_caveat: only True for RF devices — the adiabatic-
    approximation note is
        meaningless (and misleading) on a purely DC lens.
    """
    header = [str(t) for t in subtitles]
    footer = []
    if surface is not None:
        x, y, PE, ele = surface
        if plane is None:
            plane = "xy"                      # display labels only
    else:
        if plane is None:
            plane = V.model_planes(model)[0]
        if index is not None and slice_mm is not None:
            raise ValueError(f"give index OR slice_mm, not both (index="
                             f"{index!r}, slice_mm={slice_mm!r})")
        if slice_mm is not None:
            index = slice_index(model, plane, slice_mm)
        # compute_component owns the plane check and the component split;
        # 'effective' is the very pe_surface call this used to make
        x, y, PE, ele = compute_component(model, mz, charge, component,
                                          plane=plane, index=index,
                                          t_us=t_us)
        if component == "efield" and tuple(quantity) == ("PE", "eV"):
            quantity = ("|E|", "V/mm")   # the label follows the quantity
    _sl = _slice_label(model, plane, index) if model is not None else None
    if _sl is not None:
        header.append(_sl)
        if normal_mm is None:
            _ax = {"xy": 2, "xz": 1, "yz": 0}[plane]
            _k = model._plane_slice(plane, index)[1]
            normal_mm = (float(_world_off(model)[_ax])
                         + _k * float(model.h_mm))
            h_mm = float(model.h_mm)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    PE = np.asarray(PE, float).copy()
    ele = np.asarray(ele)
    if stride and stride > 1:
        x, y = x[::stride], y[::stride]
        PE, ele = PE[::stride, ::stride], ele[::stride, ::stride]

    # The Dehmelt pseudopotential diverges at electrode surfaces (|E|^2 large)
    # where the adiabatic approximation is invalid anyway (Mathieu q >> 0.4).
    # Mask the metal AND a thin adjacent ring (the divergence extends a couple
    # of nodes into vacuum) so the CONFINING WELL sets the colour/height scale.
    metal = np.asarray(ele) > 0.5
    # ImportError ONLY.  `except Exception` also caught binary_dilation FAILING,
    # and then silently used the undilated mask -- which changes the trust region,
    # which changes the colour/height scale of the whole surface.  A different
    # picture, unannounced.
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        _mask = metal          # no scipy: the ring is not dilated
    else:
        _mask = binary_dilation(metal, iterations=int(trust_cells))

    # ---- WHERE DOES AN ELECTRODE SIT ON THE PE SCALE? --------------------
    #
    # The PE surface is not a potential: it is an effective energy for the
    # SECULAR motion,  PSI = q*phi_DC + q*V_pseudo,  V_pseudo = q E0^2/(4 m W^2).
    # So an RF rod's instantaneous +/-300 V is NOT commensurable with this
    # axis, and V_pseudo ~ 1/m makes an RF electrode's effective height
    # MASS-DEPENDENT while a DC electrode's is not. The two kinds of conductor
    # genuinely do not live on the same footing, and a single "drape at the
    # assigned voltage" rule silently pretends they do.
    #
    #   'mask'    : punch a hole. Promises nothing.
    #   'dc'      : plateau at q*V_DC -- the conductor's TIME-AVERAGED
    #               potential (<V sin Wt> = 0). Right for a DC ladder. For an
    #               RF rod it reads 0 V, i.e. a VALLEY -- but |E_RF| is
    #               LARGEST at the rods, so they are the WALLS of the trap.
    #               On an RF device this convention draws the physics upside
    #               down.
    #   'barrier' : plateau at PSI evaluated on the nearest TRUSTWORTHY vacuum
    #               cells around that conductor -- the height of the wall the
    #               secular ion must actually climb. One rule, no special
    #               casing: at a DC surface PSI -> q*phi, so a DC electrode
    #               reduces to its own ladder voltage automatically, while an
    #               RF rod becomes a wall of the correct (mass-dependent)
    #               height.
    #
    # THE CATCH, STATED PLAINLY: V_pseudo DIVERGES at the metal surface, and
    # that is exactly where the adiabatic approximation has already failed
    # (Mathieu q >> 0.4). There is therefore NO well-defined finite wall
    # height: the barrier value depends on how far out you stop trusting it
    # (trust_cells). That is an arbitrary parameter, not a physical one, so it
    # is reported, never hidden. The plateau remains a LANDMARK, not a PE
    # value -- but 'barrier' at least puts the landmark on the right side of
    # the ion.
    if metal_mode in ("drape", "dc", "barrier"):
        if metal_mode == "drape":
            metal_mode = "dc"                     # legacy alias
        PE[_mask & ~metal] = np.nan               # drop the divergent ring
        trusted = np.isfinite(PE) & ~_mask        # vacuum we still believe

        plateau = {}
        if metal_mode == "barrier":
            try:
                from scipy.ndimage import binary_dilation as _dil
                for idx in np.unique(np.asarray(ele)):
                    idx = int(idx)
                    if idx <= 0:
                        continue
                    sel = np.asarray(ele) == idx
                    ring = _dil(sel, iterations=trust_cells + 1) & trusted
                    vals = PE[ring]
                    if vals.size:
                        # MEDIAN of the trusted ring: robust to the one cell
                        # nearest a corner, where E0 is largest and the
                        # adiabatic assumption is weakest.
                        plateau[idx] = float(np.median(vals))
            except (ValueError, IndexError, KeyError) as e:
                # Not a silent {}.  Falling back here means the plateau heights
                # shown are the RAW DC, not the measured trusted-ring medians --
                # a different quantity on the same axis, and the reader cannot
                # tell.  Announce the substitution.
                print(f"[pe_view] WARNING: electrode plateau measurement failed "
                      f"({e}); falling back to raw DC for the plateau heights")
                plateau = {}
            if not plateau and electrode_dc:      # no ring survived
                plateau = {int(k): float(v) * float(charge)
                           for k, v in electrode_dc.items()}
        elif electrode_dc:
            plateau = {int(k): float(v) * float(charge)
                       for k, v in electrode_dc.items()}

        if plateau:
            # the SKIRT (the dropped divergent ring, _mask & ~metal) is
            # part of the same wall landmark: leaving it NaN tore the
            # sheet into stripes wherever rings cut the slice (the PI's
            # "holes and rips", L-463). Each electrode's skirt fills at
            # its plateau; where two skirts overlap the WALL is the
            # higher one (max). Metal cells then get their exact
            # plateau, as before.
            try:
                from scipy.ndimage import binary_dilation as _dil2
                wall = np.full(PE.shape, -np.inf)
                for idx, v in plateau.items():
                    sel = np.asarray(ele) == idx
                    if sel.any():
                        reach = _dil2(sel, iterations=int(trust_cells))
                        wall[reach] = np.maximum(wall[reach], v)
                skirt = _mask & ~metal & np.isfinite(wall)
                PE[skirt] = wall[skirt]
            except ImportError:
                pass          # no scipy: _mask is undilated, no skirt exists
            for idx, v in plateau.items():
                sel = np.asarray(ele) == idx
                if sel.any():
                    PE[sel] = v
            left = int(np.isnan(PE[_mask | metal]).sum())
            if left:
                # an electrode whose trusted ring yielded no value (fully
                # enclosed) leaves its cells NaN -- fill at the tallest
                # measured wall and SAY so, never a silent hole
                top = max(plateau.values())
                PE[np.isnan(PE) & (_mask | metal)] = top
                footer.append(f"{left} masked cell(s) had no trusted ring; "
                              f"drawn at the tallest wall ({top:.3g})")
        else:
            PE[metal] = np.nan
    else:
        PE[_mask] = np.nan
    # The cap tames the divergence so the CONFINING WELL sets the scale. It
    # must be computed from the VACUUM only, and applied to the vacuum only:
    # a draped electrode plateau is a landmark at a known voltage, and
    # clipping it would make the landmark report the wrong voltage (a 20 V
    # electrode was rendering at 18.4 V).
    _vac = PE[np.isfinite(PE) & ~metal]
    if _vac.size:
        _cap = float(np.percentile(_vac, 90.0))
        PE = np.where(metal, PE, np.minimum(PE, _cap))

    PEs, z_label = scale_pe(PE, pe_scale_mode, z_exaggerate,
                            quantity=quantity)          # scaled sheet
    Zs = PEs.T                                    # Surface wants (ny, nx)

    lo, hi = float(np.nanmin(PEs)), float(np.nanmax(PEs))
    size = max((hi - lo) / max(n_contours, 1), 1e-9)

    fig = go.Figure()
    fig.add_surface(
        x=np.asarray(x), y=np.asarray(y), z=Zs,
        colorscale=_SHEET, showscale=False, opacity=1.0,
        lighting=dict(ambient=0.85, diffuse=0.4, specular=0.05),
        contours=dict(
            x=dict(show=True, color=_MESH, width=1,
                   start=float(np.min(x)), end=float(np.max(x)),
                   size=max((float(np.max(x)) - float(np.min(x))) / 40, 1e-9)),
            y=dict(show=True, color=_MESH, width=1,
                   start=float(np.min(y)), end=float(np.max(y)),
                   size=max((float(np.max(y)) - float(np.min(y))) / 40, 1e-9)),
            z=dict(show=True, color=_CONTOUR, width=2,
                   start=lo, end=hi, size=size)),
        hovertemplate="x %{x:.2f} mm<br>y %{y:.2f} mm<br>"
                      "PE(scaled) %{z:.3f}<extra></extra>")

    # ---- electrode outlines, PER ELECTRODE, at local (scaled) height
    #
    # This block had three defects stacked, and together they ARE the reported
    # symptom ("electrode numbering and shading is largely absent"):
    #
    #   1. `np.asarray(ele, float) > 0.5` CAST THE int16 LABELS TO A BOOLEAN
    #      MASK.  ele carries the electrode INDEX per node -- that is the
    #      electrode's identity -- and collapsing it to "metal / not metal"
    #      threw the identity away.  This is the same class of defect as the
    #      basis_cache `.astype(bool)` that made every electrode draw at #1's
    #      voltage (doctrine C).
    #   2. `if ii.size < 20000` SILENTLY SKIPPED the whole overlay on a fine
    #      grid.  Not "drew fewer points" -- drew NONE, and said nothing.
    #   3. every electrode was one grey (#333): no shading, no legend.
    #
    # ...and the lot was wrapped in `except Exception: pass`, so if any of it
    # failed the electrodes simply were not there and nobody was told.
    lab = np.asarray(ele)
    if lab.dtype == bool:
        raise ValueError(
            "pe_view got a BOOLEAN ele; it must carry int16 electrode LABELS. "
            "A boolean mask cannot say WHICH electrode a node belongs to, so "
            "the overlay would draw every electrode identically -- the exact "
            "defect this code used to have.")
    ids = [int(v) for v in np.unique(lab) if int(v) > 0]
    dcs = dict(electrode_dc or {})
    vs = [dcs[i] for i in ids if i in dcs]
    lim = max((abs(v) for v in vs), default=0.0)
    spread = (max(vs) - min(vs)) if len(vs) > 1 else 0.0

    import matplotlib.pyplot as _plt
    _cm = _plt.get_cmap(_VIZ.STYLE["cmap"])

    def _col(i):
        if spread > 1e-12 and i in dcs and lim > 0:
            r, g, b, _a = _cm(0.5 + 0.5 * float(dcs[i]) / lim)
            return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"
        return _VIZ.STYLE["metal"]

    xs, ys = np.asarray(x), np.asarray(y)
    for i in ids:
        m = (lab == i)
        edge = np.zeros_like(m, bool)
        edge[1:, :] |= m[1:, :] ^ m[:-1, :]
        edge[:, 1:] |= m[:, 1:] ^ m[:, :-1]
        ii, jj = np.where(edge)
        if not ii.size:
            continue
        # DECIMATE, never omit.  A cap that silently drops the overlay is a
        # picture that lies by absence; a cap that thins it is a picture that
        # is still true.  The stride is announced in the trace name.
        step = max(1, int(np.ceil(ii.size / 6000)))
        sl = slice(None, None, step)
        v = dcs.get(i)
        nm = f"E{i}" + (f"  {v:+.4g} V" if v is not None else "")
        if step > 1:
            nm += f"  (1/{step})"
        fig.add_scatter3d(
            x=xs[ii[sl]], y=ys[jj[sl]],
            z=PEs[ii[sl], jj[sl]] + 0.01 * (hi - lo + 1e-9),
            mode="markers", marker=dict(size=1.8, color=_col(i)),
            name=nm, hoverinfo="skip", showlegend=True)

    # trajectories draped on the SCALED sheet (sampler on scaled PE)
    if results:
        samp = _pe_sampler(x, y, PEs)
        lift = 0.02 * (hi - lo + 1e-9)
        # NORMAL-axis column + the slice's position, so a SLICED plane only
        # drapes ions that actually live near it. The PE surface is ONE
        # slice at a fixed normal index; an ion far from that slice has a
        # completely different local PE, so draping it on this slice lifted
        # it onto whatever the slice happened to cut (a barrier wall ->
                # an 80 eV spike on a yz cut). We keep only ions
        # within HALF A SLAB of the slice and drape them on it; the slab is
        # generous (10% of the normal extent, min a few cells) so a thin
        # confined cloud still shows.
        _norm_ax = _PLANE_NORMAL_AXIS.get(plane)
        _norm_pos = None
        if _norm_ax is not None and normal_mm is not None:
            _norm_pos = float(normal_mm)
            _span = float(max(np.ptp(x), np.ptp(y)))
            _slab = max(0.10 * _span, 3.0 * float(h_mm))
        shown = 0
        skipped_far = 0
        for r in results:
            if r.traj is None or shown >= max_traj:
                continue
            tr = r.traj[::max(int(decimate), 1)]
            # IN-PLANE channels for THIS plane, by NAME. The record
            # wraps the guaranteed BASE prefix [t,x,y,z,vx,vy,vz]; the
            # earlier hard-coded (x, y)-for-every-plane bug drew xz/yz
            # drapes at the wrong coordinate.
            from ion_gym.io.records import TrajRecord
            from ion_gym.io.sim_spec import BASE_CHANNELS
            rec = TrajRecord(tr[:, :len(BASE_CHANNELS)], BASE_CHANNELS)
            ca, cb = _PLANE_TRAJ_AXES.get(plane, ("x", "y"))
            xs, ys = rec[ca], rec[cb]
            if _norm_pos is not None:
                near = np.abs(rec[_norm_ax] - _norm_pos) <= _slab
                if not near.any():
                    skipped_far += 1
                    continue
                xs, ys = xs[near], ys[near]
            zs = samp(np.column_stack([xs, ys])) + lift
            kind = r.summary.get("kind", 2)
            # TRAJECTORY COLOUR: fate-coloured by
            # DEFAULT -- the fate is a declared property of the flight and
            # colouring by it costs the reader nothing. But the default is
            # not always the point: on a landscape where every ion shares a
            # fate the palette carries no information, and a figure going
            # into a document may need to match its neighbours. So:
            #   traj_color=None   -> by fate (unchanged)
            #   traj_color="#0ff" -> one colour for every path
            #   traj_color={0: "#d62728", 2: "#00e5ff"} -> per-fate
            #                        override, falling back to FATE_COLOR
            if traj_color is None:
                colour = FATE_COLOR.get(kind, "#1f77b4")
            elif isinstance(traj_color, dict):
                colour = traj_color.get(kind, FATE_COLOR.get(kind, "#1f77b4"))
            else:
                colour = traj_color
            fig.add_scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=colour, width=traj_width),
                showlegend=False, hoverinfo="skip")
            shown += 1
        if _norm_pos is not None and skipped_far:
            # a skipped item is a REPORTED item: say how many
            # ions fell outside the slice's slab, so an empty-looking slice
            # is an explained decision, not a silent drop.
            footer.append(f"{skipped_far} trajectory(ies) outside the "
                          f"{plane} slab at {_norm_pos:.3g} mm — not draped")

    _inst = model_instant_drives(model) if model is not None else []
    if _inst:
        header.append(f"instant drive(s) {', '.join(_inst)} at lab time "
                      f"t = {float(t_us):g} µs")
    # the adiabatic caveat describes a pseudopotential: it is attached only
    # when the plotted component contains one
    if show_adiabatic_caveat and component in ("effective", "rf"):
        footer.append("effective RF pseudopotential (adiabatic "
                      "approximation; the full RF tracer is exact)")
    fig.update_layout(
        height=height,
        scene=dict(
            xaxis_title=f"{plane[0]} [mm]" if len(plane) == 2 else "x [mm]",
            yaxis_title=f"{plane[1]} [mm]" if len(plane) == 2 else "y [mm]",
            zaxis_title=z_label,
            aspectmode="manual",
            aspectratio=dict(
                x=(float(np.ptp(x)) / max(float(np.ptp(y)), 1e-9))
                if np.ptp(y) > 0 else 1.6,
                y=1.0, z=max(float(z_aspect), 0.05)),
            camera=camera_eye(**camera) if camera else _PE_CAMERA_DEFAULT))
    V.layout_plotly_text(fig, title=title, subtitles=header, footers=footer)
    return fig


def pe_overlay_2d(fig, model, mz=None, *, charge=1, n_contours=12,
                  t_us=0.0):
    """Improved 2-D PE overlay: heatmap of PE plus brick-red PE
    equipotentials (the v51 overlay contoured the raw potential in faint
    blue over a 60%-opacity heatmap — low contrast). Adds traces to an
    existing 2-D figure."""
    x, y, PE, _ele = model.pe_surface(mz=mz, charge=charge, t_us=t_us)
    PE = np.asarray(PE, float).copy()
    try:
        from scipy.ndimage import binary_dilation
        _m = binary_dilation(np.asarray(_ele) > 0.5, iterations=2)
    except ImportError:
        # scipy is optional: without it the near-surface ring is not
        # dilated (metal itself still masked). AUDITED:
        # narrowed from Exception — a dilation failure WITH scipy
        # present is a bug and must raise.
        _m = np.asarray(_ele) > 0.5
    PE[_m] = np.nan                              # mask metal + near-surface ring
    _finite = PE[np.isfinite(PE)]
    if _finite.size:
        PE = np.minimum(PE, float(np.percentile(_finite, 90.0)))
    fig.add_heatmap(x=x, y=y, z=PE.T, colorscale="Viridis",
                    opacity=0.75, showscale=False)
    lo, hi = float(np.nanmin(PE)), float(np.nanmax(PE))
    fig.add_contour(
        x=x, y=y, z=PE.T, showscale=False,
        contours=dict(coloring="lines", start=lo, end=hi,
                      size=max((hi - lo) / n_contours, 1e-9)),
        line=dict(width=1.3, color=_CONTOUR))
    return fig


# ------------------------------------------- exported field-window viewer
def field_window_figure(data, meta, t_us, *, component="mag", plane="xy",
                        index=None, slice_mm=None, height=520, title=None):
    """Figure of an EXPORTED field window (field_cycle_io.save_field_window
    npz) at a lab time — the file-side twin of the live PE/|E| views, so a
    window handed to a collaborator can be inspected without the deck.

    data, meta : exactly what field_cycle_io.read_field_window returns.
    t_us       : requested lab time. The view SNAPS to the nearest STORED
                 sample (the file holds frames, not a continuum) and the
                 title states the sample actually shown. A time outside
                 the stored window [t0, t_end] REFUSES with the window
                 named — the file cannot answer for times it does not
                 hold.
    component  : 'mag' (|E|), or 'Ex' / 'Ey' / 'Ez' (a stored component).
    plane/index/slice_mm: for a 3-D window, the principal-plane slice,
                 as a node index or a position in mm on the STORED axis
                 (nearest node; a position outside the stored axis
                 refuses with its range). Default: mid. Give one of
                 index/slice_mm, not both. A 2-D window has only its
                 stored plane and refuses a slice request.

    Returns the plotly figure (saving/showing is the caller's decision).
    """
    times = np.asarray(data["t_us"], float)
    t0, t1 = float(times[0]), float(times[-1])
    t = float(t_us)
    if not np.isfinite(t) or t < t0 or t > t1:
        raise ValueError(
            f"t = {t_us!r} µs is outside the stored window "
            f"[{t0:g}, {t1:g}] µs ({len(times)} samples); this export "
            f"cannot answer for times it does not hold — re-export with "
            f"save_field_window(t_start_us=..., total_time_us=...) "
            f"covering the time you want")
    i = int(np.argmin(np.abs(times - t)))
    t_shown = float(times[i])
    comps = {"Ex": data["Ex"][i], "Ey": data["Ey"][i], "Ez": data["Ez"][i]}
    if component == "mag":
        F = np.sqrt(sum(np.asarray(c, float) ** 2 for c in comps.values()))
        qlabel = "|E| [V/mm]"
    elif component in comps:
        F = np.asarray(comps[component], float)
        qlabel = f"{component} [V/mm]"
    else:
        raise ValueError(f"component {component!r}: one of "
                         f"['mag', 'Ex', 'Ey', 'Ez']")
    # axes are POSITIONAL (x_mm/y_mm/z_mm are the stored array axes in
    # order); meta["axes"] carries the human names ("x = axial", ...)
    ax_arr = [np.asarray(data["x_mm"], float), np.asarray(data["y_mm"], float),
              np.asarray(data["z_mm"], float)]
    names = list(meta.get("axes", ("x", "y", "z")))
    while len(names) < 3:
        names.append("?")
    if index is not None and slice_mm is not None:
        raise ValueError(f"give index OR slice_mm, not both (index="
                         f"{index!r}, slice_mm={slice_mm!r})")
    if F.ndim == 3:
        _ax = {"xy": 2, "xz": 1, "yz": 0}
        if plane not in _ax:
            raise ValueError(f"plane {plane!r}: a 3-D window slices on "
                             f"['xy', 'xz', 'yz']")
        ax = _ax[plane]
        if slice_mm is not None:
            axv = ax_arr[ax]
            lo_mm, hi_mm = float(axv.min()), float(axv.max())
            if not (lo_mm <= float(slice_mm) <= hi_mm):
                raise ValueError(
                    f"slice {names[ax]} = {slice_mm!r} mm is outside the "
                    f"stored axis [{lo_mm:g}, {hi_mm:g}] mm")
            index = int(np.argmin(np.abs(axv - float(slice_mm))))
        k = F.shape[ax] // 2 if index is None else int(index)
        if not (0 <= k < F.shape[ax]):
            raise ValueError(f"index {index} outside the {names[ax]} axis "
                             f"(0..{F.shape[ax] - 1})")
        sl = [slice(None)] * 3
        sl[ax] = k
        F = F[tuple(sl)]
        in_plane = [j for j in range(3) if j != ax]
        a, b = ax_arr[in_plane[0]], ax_arr[in_plane[1]]
        an, bn = names[in_plane[0]], names[in_plane[1]]
        where = f" · {plane} @ {names[ax]} = {ax_arr[ax][k]:g} mm"
    elif F.ndim == 2:
        if index is not None or slice_mm is not None:
            raise ValueError("this window stores a 2-D plane; it has no "
                             "normal axis to slice")
        a, b = ax_arr[0], ax_arr[1]
        an, bn = names[0], names[1]
        where = ""
    else:
        raise ValueError(f"stored frames have {F.ndim} spatial dims; "
                         f"expected 2 or 3")
    fig = go.Figure(go.Heatmap(x=a, y=b, z=F.T, colorscale="Viridis",
                               colorbar=dict(title=qlabel)))
    dev = meta.get("deck", "exported window")
    held = list(meta.get("held_channels", []))
    fig.update_layout(height=height, xaxis_title=f"{an} [mm]",
                      yaxis_title=f"{bn} [mm]")
    V.layout_plotly_text(
        fig, title=title or f"{dev} · {qlabel}{where}",
        subtitles=[f"sample {i} at t = {t_shown:g} µs"
                   + (f" (requested {t:g})" if abs(t_shown - t) > 1e-12
                      else "")]
        + ([f"HELD over this window: {', '.join(held)}"] if held else []))
    fig.update_yaxes(scaleanchor=None)
    return fig


# ------------------------------------------------- time-series frame export
def save_pe_frames(model, times_us, out_dir, *, mz=None, charge=1,
                   plane=None, index=None, slice_mm=None,
                   component="effective", camera=None,
                   stem="pe", width=900, height=520, gif_path=None,
                   gif_frame_ms=120, fig_kw=None):
    """Render the PE view at each lab time in `times_us` and write one PNG
    per frame (optionally assembling them into an animated GIF).

    THE point of this exporter: a slow 'instant' drive (a travelling
    wave) is drawn where it actually is at each time, so the frame
    series shows the wave marching over the landscape. The vertical
    range and camera are FIXED across the series from the global
    min/max of every frame's surface — per-frame autoscaling makes an
    animation lie about amplitudes.

    model      : a built model (planar / r-z / 3-D) — the same object
                 every other pe_view entry point takes.
    times_us   : iterable of lab times (µs). One frame each, in order.
    out_dir    : directory for the frames (created if absent). Frames
                 are `{stem}_{i:04d}.png`; the time is stamped ON each
                 frame by pe_figure_3d, and the returned manifest pairs
                 every path with its time.
    plane/index/component/mz/charge : exactly compute_component's knobs;
                 slice_mm is the slice as a position in mm (slice_index;
                 give index OR slice_mm). Every frame is stamped with the
                 resolved slice.
    gif_path   : optional path for an animated GIF assembled from the
                 frames via Pillow (`gif_frame_ms` per frame, looping).
    fig_kw     : extra pe_figure_3d keyword arguments (title,
                 z_exaggerate, stride, ...), applied to every frame.

    Returns (frame_paths, gif_path_or_None). PNG rendering is plotly +
    kaleido, which needs a Chrome/Chromium binary; without one this
    RAISES with the remedy rather than writing nothing quietly (set
    BROWSER_PATH to an existing chromium binary — that is the variable
    kaleido honours — or install Chrome / `plotly_get_chrome`).
    """
    import os
    times = [float(t) for t in times_us]
    if not times:
        raise ValueError("times_us is empty — nothing to render")
    if index is not None and slice_mm is not None:
        raise ValueError(f"give index OR slice_mm, not both (index="
                         f"{index!r}, slice_mm={slice_mm!r})")
    if plane is None:
        plane = V.model_planes(model)[0]
    if slice_mm is not None:
        index = slice_index(model, plane, slice_mm)
    os.makedirs(out_dir, exist_ok=True)
    fig_kw = dict(fig_kw or {})
    surfaces = [compute_component(model, mz, charge, component,
                                  plane=plane, index=index, t_us=t)
                for t in times]
    lo = min(float(np.nanmin(PE)) for _x, _y, PE, _e in surfaces)
    hi = max(float(np.nanmax(PE)) for _x, _y, PE, _e in surfaces)
    mode = fig_kw.get("pe_scale_mode", "linear")
    exag = fig_kw.get("z_exaggerate", 1.0)
    zlo, _ = scale_pe(np.array([lo]), mode, exag)
    zhi, _ = scale_pe(np.array([hi]), mode, exag)
    pad = 0.05 * max(float(zhi[0]) - float(zlo[0]), 1e-12)
    paths = []
    for i, (t, surf) in enumerate(zip(times, surfaces)):
        fig = pe_figure_3d(model=model, mz=mz, charge=charge,
                           surface=surf, plane=plane, index=index,
                           component=component, camera=camera,
                           t_us=t, **fig_kw)
        fig.update_scenes(zaxis_range=[float(zlo[0]) - pad,
                                       float(zhi[0]) + pad])
        path = os.path.join(out_dir, f"{stem}_{i:04d}.png")
        try:
            fig.write_image(path, width=int(width), height=int(height))
        except Exception as e:
            raise RuntimeError(
                f"PNG export failed on frame {i} ({path}): plotly image "
                f"export needs kaleido AND a Chrome/Chromium binary. "
                f"Install Chrome (or `plotly_get_chrome`), or point "
                f"BROWSER_PATH at an existing chromium binary (that is "
                f"the variable kaleido honours).") from e
        paths.append(path)
    if gif_path is not None:
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                f"GIF assembly needs Pillow (`pip install pillow`); the "
                f"{len(paths)} PNG frames are already written in "
                f"{out_dir}") from e
        frames = [Image.open(fp).convert("P", palette=Image.ADAPTIVE)
                  for fp in paths]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=int(gif_frame_ms), loop=0)
    return paths, gif_path


# --------------------------------------------------------------- PE tab
def _surface_signature(spec, mz, exag, stride, drape, run_id, component,
                       plane="xy", metal="mask", slice_pos=None, t_us=0.0):
    """Hash of everything that changes the rendered surface — geometry,
    per-electrode DC + RF group, resolution, mass, scaling (mode+factor),
    view options, THE PLANE, the metal mode, and the draped run. Irrelevant
    spec fields (dt, gas) are deliberately excluded so the cache is not thrown
    away needlessly.

    The plane and metal mode MUST be in here: without them, switching to xz
    and pressing Compute returns the cached xy surface under an xz title.
    t_us likewise: an instant-drive surface at t=25 µs is a different
    surface from t=0 (checkbox off resolves to exactly 0.0, so the
    default cache keys are unchanged)."""
    import hashlib
    g = spec.geometry
    parts = [f"{g.mm_per_gu:.6g}", g.symmetry.coords, f"t{float(t_us):.9g}"]
    for e in g.electrodes:
        grps = ",".join(e.group_names())
        parts.append(f"{e.name}|{e.dc:.6g}|{grps}|{e.stl}|"
                     f"{bool(e.shapes)}")
    parts += [f"mz={mz:.6g}", f"scale={exag}", f"st={stride}",
              f"dr={int(drape)}", f"run={run_id}", f"comp={component}",
              f"plane={plane}", f"metal={metal}", f"slice={slice_pos}"]
    return hashlib.sha256("::".join(parts).encode()).hexdigest()


def _run_off_doc(build, apply_fn, on_error, join_timeout_s=120.0):
    """Run `build` on a worker thread; deliver its result to `apply_fn`
    on the DOCUMENT thread. Bokeh models may only be
    mutated from the doc thread, so heavy tab computes split into a
    widget-free build and a doc-thread apply, exactly like the Fly
    button's worker+poll pattern in sim_app (whose hard-won curdoc
    lessons this copies: probe pn.state.curdoc, TRY the periodic
    callback, and only fall back to a BOUNDED join when there is no
    schedulable server session — never an unbounded spin).
    `on_error(exc, tb_str)` also runs on the doc thread."""
    import threading as _th
    import traceback as _tb
    import panel as pn
    st = {"done": False, "res": None, "err": None, "tb": ""}

    def _work():
        try:
            st["res"] = build()
        except Exception as e:            # captured, re-raised on the doc
            st["err"] = e                 # thread via on_error — a worker
            st["tb"] = _tb.format_exc()   # thread cannot surface it itself
        st["done"] = True

    w = _th.Thread(target=_work, daemon=True, name="pe-view-compute")
    w.start()
    own = {"pcb": None}

    def _finish():
        if st["err"] is not None:
            on_error(st["err"], st["tb"])
        else:
            apply_fn(st["res"])
            st["res"] = None      # release once consumed; see sim_app 4594

    def _poll():
        if not st["done"]:
            return
        if own["pcb"] is not None:
            try:
                own["pcb"].stop()
            except (ValueError, RuntimeError):
                pass                      # already stopped: benign
            own["pcb"] = None
        _finish()

    in_server = False
    try:
        in_server = pn.state.curdoc is not None
    except (RuntimeError, AttributeError):
        in_server = False
    if in_server:
        try:
            own["pcb"] = pn.state.add_periodic_callback(_poll, 60)
            _poll()
            return
        except (RuntimeError, ValueError):
            pass                          # no schedulable doc after all
    # headless/notebook: bounded join, then finish synchronously
    w.join(timeout=join_timeout_s)
    if w.is_alive():
        on_error(TimeoutError(
            f"compute did not finish within {join_timeout_s:.0f} s; it may "
            f"still be running — check the console (headless path only)"),
            "")
        return
    _finish()


def compute_component(model, mz, charge, component, plane=None,
                      index=None, t_us=0.0):
    """Return (x, y, PE, ele) for the chosen component:
      'effective' : DC + RF Dehmelt pseudopotential (pe_surface) — the
                    surface the secular motion actually rides.
      'dc'        : DC potential energy only (q*phi_DC).
      'rf'        : RF pseudopotential only (effective - dc).
    For a DC-only device (einzel: no RF basis) all three coincide.
    t_us: lab time for 'instant' drives (slow TW) — both terms of the
    split are taken at the SAME time, so 'rf' stays the mass-dependent
    pseudopotential alone and the instant wave sits in the 'dc'
    (mass-independent) term."""
    # ASK the model which planes it has.  This used to CALL pe_surface(plane=)
    # and catch TypeError as a proxy for "2-D" -- a signature probe wearing a
    # capability probe's clothes.  It swallowed every genuine TypeError raised
    # INSIDE pe_surface and re-reported it as "this model has no 'xz' plane",
    # so a solver that threw was diagnosed as a geometry that didn't exist.
    planes = V.model_planes(model)
    if plane is None:
        plane = planes[0]        # not "xy": an r-z model's plane is "rz"
    if plane not in planes:
        raise ValueError(
            f"{type(model).__name__} has no {plane!r} plane; it has {planes}. "
            f"(The UI should not have offered it -- the plane selector is "
            f"driven by the model's declared PLANES.)")
    # only pass `index` when a specific slice was requested; 2-D models
    # (r-z, planar) have no perpendicular axis and their pe_surface takes no
    # index. A kwargs dict keeps the call signature-compatible for both.
    _ikw = {} if index is None else {"index": index}
    if component == "efield":
        # |E| on the SAME plane/slice machinery as PE, so the field can be
        # cut INTO the apparatus (a yz cross-section at a chosen x) rather
        # than only at the fixed mid-plane the transport views show.
        if not hasattr(model, "field_surface"):
            raise ValueError(
                f"{type(model).__name__} exposes no field_surface(); the "
                f"|E| component needs a 3-D model (scene3d/stl3d). Its PE "
                f"components are still available.")
        return model.field_surface(plane=plane, **_ikw)
    x, y, PE_eff, ele = model.pe_surface(mz=mz, charge=charge, plane=plane,
                                         t_us=t_us, **_ikw)
    if component == "effective":
        return x, y, PE_eff, ele
    # pure DC: potential_image without an RF phase is q*A for the r-z model;
    # for the planar model it adds peak RF, so subtract that back out via
    # pe_surface's own DC term. Simplest correct route: DC = q*phi_DC from
    # the model's A (mirrored consistently with pe_surface).
    # THE SPLIT NEEDS NO potential_image(). pe_surface() builds the effective
    # surface as (DC term) + (RF term) and only adds the RF term when an m/z is
    # given -- so pe_surface(mz=None) IS the DC term, on whatever plane was
    # asked for. The old route went through potential_image(), which is xy-only,
    # so component='dc' on an xz plane RAISED at the user. The refusal was
    # honest but unnecessary: the quantity was available the whole time.
    # THIRD copy of the sniff, and the most dangerous: it was
    #     try: ... except TypeError: pass   -> fall through to a legacy path.
    # It caught TypeErrors raised by the COMPONENT ARITHMETIC below it, not
    # just by the signature, and then silently answered from a DIFFERENT code
    # path -- so a bug in the subtraction would have returned a plausible
    # surface computed some other way, with no diagnostic at all.  Every model
    # now takes `plane`, so there is nothing to fall back FROM.  It is gone.
    _x, _y, PE_dc, _e = model.pe_surface(mz=None, charge=charge, plane=plane,
                                         t_us=t_us, **_ikw)
    if PE_dc.shape != PE_eff.shape:
        raise ValueError(
            f"{type(model).__name__}.pe_surface returned {PE_dc.shape} for the "
            f"DC term and {PE_eff.shape} for the effective term on plane "
            f"{plane!r}. These must agree -- the old code silently returned the "
            f"effective surface when they did not, i.e. it showed you 'dc' and "
            f"handed you 'effective'.")
    if component == "dc":
        return x, y, PE_dc, ele
    if component == "rf":
        return x, y, PE_eff - PE_dc, ele
    if component != "effective":
        raise ValueError(f"unknown component {component!r}; use 'effective', "
                         f"'dc' or 'rf'")
    return x, y, PE_eff, ele


class FieldSliceTab:
    """A 'Field Slice' tab: slice the model on any principal plane and show
    the field as a 2-D heatmap + contour overlay, with full display
    controls. Unlike the PE Surface tab (a 3-D landscape) this is
    a flat cut — the right tool for reading field VALUES on a plane.

    Controls: quantity (|E| / potential / PE), plane (xy/xz/yz) + slice
    position (like the PE tab), heatmap on/off + colormap + range + log
    scaling, and contours on/off + count + colour. Same accessor wire-up
    as PeSurfaceTab: construct with get_model/get_spec, add panel() to the
    Tabs, call refresh() on model/spec change.
    """

    def __init__(self, get_model, get_spec):
        import panel as pn
        self._pn = pn
        self.get_model = get_model
        self.get_spec = get_spec
        self._fig = None

        self.w_quantity = pn.widgets.Select(
            name="quantity", width=180,
            options={"|E| field [V/mm]": "efield",
                     "potential φ [V]": "phi",
                     "PE [eV]": "pe"},
            value="efield")
        self.w_mz = pn.widgets.FloatInput(name="ion m/z (Da)", value=100.0,
                                          width=120)
        self.w_plane = pn.widgets.Select(name="plane", width=90,
                                         options=["xy", "xz", "yz"],
                                         value="xy")
        self.w_slicepos = pn.widgets.FloatSlider(
            name="slice position (mm; auto if at min)", start=-1.0, end=0.0,
            value=-1.0, step=0.05, width=360)
        # numeric twin: typed exact slice position,
        # kept in sync with the slider both ways.
        self.w_slicenum = pn.widgets.FloatInput(
            name="slice (mm, exact)", value=-1.0, step=0.05, width=140)
        self.w_slicepos.link(self.w_slicenum, value="value")
        self.w_slicenum.link(self.w_slicepos, value="value")
        # heatmap controls
        self.w_heat_on = pn.widgets.Checkbox(name="heatmap", value=True)
        self.w_heat_cmap = pn.widgets.Select(
            name="heatmap colors", width=150,
            options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis",
                     "Turbo", "Hot", "Jet", "Greys", "RdBu"],
            value="Viridis")
        self.w_heat_log = pn.widgets.Checkbox(name="log scale", value=False)
        self.w_heat_lo = pn.widgets.FloatInput(name="range min (blank=auto)",
                                               value=None, width=140)
        self.w_heat_hi = pn.widgets.FloatInput(name="range max (blank=auto)",
                                               value=None, width=140)
        # contour controls
        self.w_cont_on = pn.widgets.Checkbox(name="contours", value=True)
        self.w_cont_n = pn.widgets.IntSlider(name="# contours", start=2,
                                             end=40, value=12, width=200)
        self.w_cont_color = pn.widgets.Select(
            name="contour colors", width=150,
            options=["white", "black", "red", "cyan", "yellow",
                     "match heatmap"],
            value="white")
        self.w_res = pn.widgets.IntSlider(
            name="resolution (pts/axis, ↓ = faster)", start=40, end=400,
            value=200, step=20, width=280)
        # aspect lock: scaleanchor forces equal mm/mm,
        # which pins the plot to a fixed landscape box and makes zoom
        # constrain one axis to the other. OFF by default so the figure
        # fills its pane and zoom/pan are free; ON when the user wants true
        # geometric proportions (undistorted mm).
        self.w_equal_aspect = pn.widgets.Checkbox(
            name="lock aspect (true mm)", value=False)
        self.w_go = pn.widgets.Button(name="Compute slice",
                                      button_type="primary", width=180)
        # L-58b: button -> async path; compute() stays synchronous
        self.w_go.on_click(self._on_go)
        # changing the PLANE changes which axis is sliced, so the slice-
        # position slider's range must follow it (the
        # slider didn't update to the new axis on a plane switch).
        self.w_plane.param.watch(lambda e: self._sync_slice_bounds(),
                                 "value")
        # CSV export of the computed slice: a filename
        # field + a download button that serialises the last computed slice
        # (grid + metadata header) so it round-trips to analysis offline.
        self.w_export_name = pn.widgets.TextInput(
            name="CSV filename", value="field_slice.csv", width=240)
        self.w_export = pn.widgets.FileDownload(
            label="⤓ export slice CSV", button_type="success", width=200,
            filename="field_slice.csv", callback=self._export_csv)
        self.w_export_name.param.watch(self._on_export_name, "value")
        # view-range windowing (the field slice on
        # the UI can be too big, so each field slice plot takes a
        # min/max, defaulted, user-settable, with a reset). Blank =
        # full domain; Reset clears all four.
        self.w_xlo = pn.widgets.FloatInput(name="x min (blank=full)",
                                           value=None, width=130)
        self.w_xhi = pn.widgets.FloatInput(name="x max", value=None,
                                           width=110)
        self.w_ylo = pn.widgets.FloatInput(name="y min (blank=full)",
                                           value=None, width=130)
        self.w_yhi = pn.widgets.FloatInput(name="y max", value=None,
                                           width=110)
        self.w_view_reset = pn.widgets.Button(name="reset view",
                                              width=110)
        def _reset_view(_=None):
            for w in (self.w_xlo, self.w_xhi, self.w_ylo, self.w_yhi):
                w.value = None
            if self._fig is not None:
                self.compute()
        self.w_view_reset.on_click(_reset_view)
        self._last_slice = None
        # sizing: stretch_both + min_height 520 rendered a
        # squat 2-D slice as an ENORMOUS pane. Width stretches; HEIGHT is
        # set per-compute from the slice's mm aspect, capped [300, 560].
        self.pane = pn.pane.Plotly(
            sizing_mode="stretch_width", height=430,
            config={"responsive": True})
        self.status = pn.pane.Markdown("_pick a plane and press Compute "
                                       "slice_")

    def panel(self):
        pn = self._pn
        return pn.Column(
            pn.Row(self.w_quantity, self.w_mz, self.w_plane),
            pn.Row(self.w_slicepos, self.w_slicenum),
            pn.pane.Markdown("**heatmap**"),
            pn.Row(self.w_heat_on, self.w_heat_cmap, self.w_heat_log),
            pn.Row(self.w_heat_lo, self.w_heat_hi),
            pn.pane.Markdown("**contours**"),
            pn.Row(self.w_cont_on, self.w_cont_n, self.w_cont_color),
            pn.Row(self.w_res, self.w_equal_aspect),
            pn.pane.Markdown("**view window** (mm; blank = full domain)"),
            pn.Row(self.w_xlo, self.w_xhi, self.w_ylo, self.w_yhi,
                   self.w_view_reset),
            self.w_go,
            pn.Row(self.w_export_name, self.w_export),
            self.status,
            self.pane,
            sizing_mode="stretch_width")

    def _sync_planes(self, model):
        # model_planes returns a TUPLE; Panel's Select.options rejects a
        # tuple (dict|list only), and comparing list != tuple is ALWAYS
        # unequal — which re-assigned options every periodic tick and threw
        # on the tuple, cycling forever. Coerce to a list
        # and compare like-with-like, matching PeSurfaceTab._sync_planes.
        planes = list(V.model_planes(model))
        if list(self.w_plane.options) != planes:
            self.w_plane.options = planes
            if self.w_plane.value not in planes:
                self.w_plane.value = planes[0]
        self.w_plane.disabled = (len(planes) == 1)

    def _slice_index(self, model, plane):
        """Slice node from the mm control, via the module's ONE rule
        (slice_index). The control at its minimum means AUTO (None) --
        the only UI-specific decision here. Any other value is mapped or
        REFUSED (off-grid position, a plane with no normal); the caller
        reports the refusal in the status line. The two tabs' private
        copies of this mapping disagreed on unknown planes (one silently
        sliced z) and let off-grid positions through to fail later on
        the worker thread; both are superseded by this."""
        if self.w_slicepos.value <= self.w_slicepos.start:
            return None
        return slice_index(model, plane, float(self.w_slicepos.value))

    def _surface(self, model, plane, idx, q=None, mz=None):
        """(a, b, values, ele) for the chosen quantity, or a diagnostic
        RuntimeError naming what the model cannot answer (never a silent
        empty plot). q/mz default from the widgets for direct (doc-thread)
        callers; the async path passes the _prep snapshot so the WORKER
        never touches a widget (L-58b)."""
        if q is None:
            q = self.w_quantity.value
        if mz is None:
            mz = float(self.w_mz.value)
        if plane == "rz":
            # r-z models (ion funnel, cylindrical builds) carry their own
            # accessor family: pe_surface(mz, charge, plane='rz') and
            # potential_image() both return (z, r_full, values, em) on the
            # ±r-mirrored grid, and efield_magnitude() gives the RF
            # cycle-peak |E| = rf_V·|∇B| in V/m on that same grid (the
            # confining field an ion-funnel user is asking about). Axes:
            # a = z (axial, horizontal per Convention A), b = r_full.
            if q == "pe":
                if not hasattr(model, "pe_surface"):
                    raise RuntimeError("this r-z model exposes no PE "
                                       "surface")
                return model.pe_surface(mz=mz,
                                        charge=1, plane="rz")
            if not hasattr(model, "potential_image"):
                raise RuntimeError("this r-z model exposes no potential "
                                   "image")
            z, r_full, img, em = model.potential_image()
            if q == "phi":
                return z, r_full, img, em
            # q == "efield"
            if not hasattr(model, "efield_magnitude"):
                raise RuntimeError(
                    "this r-z model exposes no |E| accessor "
                    "(efield_magnitude)")
            e_vmm = model.efield_magnitude() / 1.0e3     # V/m -> V/mm
            return z, r_full, e_vmm, em
        if q == "efield":
            if not hasattr(model, "field_surface"):
                raise RuntimeError(
                    "this model exposes no |E| surface (2-D planar/r-z "
                    "models carry PE and φ; pick those)")
            return model.field_surface(plane=plane, index=idx)
        # 2-D models (planar) take no slice index — the plane IS the
        # whole domain (the slice tab shows the plain 2-D
        # cross-section). Pass index only to models that
        # actually slice.
        _kw = ({"index": idx} if hasattr(model, "_plane_slice") else {})
        if q == "pe":
            if not hasattr(model, "pe_surface"):
                raise RuntimeError("this model exposes no PE surface")
            return model.pe_surface(mz=mz, charge=1,
                                    plane=plane, **_kw)
        # potential: a 2-D planar model answers phi DIRECTLY via
        # potential_image (the _has_phi_component probe
        # predates compute_component's vocabulary and asked for a
        # component it rejects); slicing 3-D models keep the component
        # path.
        if not hasattr(model, "_plane_slice") and hasattr(
                model, "potential_image"):
            return model.potential_image()
        return compute_component(model, mz, 1, "phi"
                                 if _has_phi_component(model) else "dc",
                                 plane=plane, **_kw)

    def _prep_slice(self):
        """Doc-thread part (L-58b): capability probing, option switching,
        and the widget snapshot the worker-safe _surface call needs.
        Returns None (status set) when there is nothing to compute."""
        import types
        model = self.get_model()
        if model is None:
            self.status.object = ("_no solved field yet — press Fly or "
                                  "Recompute, then Compute slice_")
            return None
        self._sync_planes(model)
        # QUANTITY follows model capability (a 2-D
        # slice tab used to refuse outright): planar/r-z 2-D models carry
        # phi and PE but no |E| accessor — offer what the model can
        # honestly answer instead of refusing the whole tab, and REPORT
        # the switch (never silent).
        # probe EXACTLY what _surface will call for the active plane
        # (planar carries efield_magnitude for the r-z convention but no
        # per-plane field_surface — the OR probe wrongly passed it)
        _pl_now = self.w_plane.value
        _has_E = (hasattr(model, "efield_magnitude") if _pl_now == "rz"
                  else hasattr(model, "field_surface"))
        _opts = dict(self.w_quantity.options) if isinstance(
            self.w_quantity.options, dict) else None
        full = {"|E| field [V/mm]": "efield", "potential φ [V]": "phi",
                "PE [eV]": "pe"}
        want = full if _has_E else {k: v for k, v in full.items()
                                    if v != "efield"}
        if _opts != want:
            self.w_quantity.options = want
        if not _has_E and self.w_quantity.value == "efield":
            self.w_quantity.value = "phi"
            self.status.object = ("_this 2-D model has no |E| accessor — "
                                  "showing φ (PE also available)_")
        p = types.SimpleNamespace(model=model)
        p.plane = self.w_plane.value
        try:
            p.idx = self._slice_index(model, p.plane)
        except ValueError as e:
            self.status.object = f"**refused** — {e}. Nothing was recomputed."
            return None
        p.q = self.w_quantity.value
        p.mz = float(self.w_mz.value)
        return p

    def compute(self):
        """Synchronous composition — harness/scripts keep the old
        contract (figure ready on return). The button goes through
        _on_go() instead (worker _surface, doc-thread finish)."""
        p = self._prep_slice()
        if p is None:
            return
        try:
            surf = self._surface(p.model, p.plane, p.idx, q=p.q, mz=p.mz)
        except RuntimeError as e:
            self.status.object = f"**cannot slice** — {e}"
            return
        self._finish_slice(p, surf)

    def _on_go(self, _event=None):
        """Button path (L-58b): the grid extraction/PE evaluation runs
        off the doc thread; everything that touches a widget or pane
        stays on it."""
        if getattr(self, "_busy", False):
            self.status.object = "_already computing — one at a time_"
            return
        p = self._prep_slice()
        if p is None:
            return
        self._busy = True
        self.pane.loading = True
        self.status.object = "_computing slice (off the UI thread)…_"

        def _fail(e, tb):
            self.pane.loading = False
            self._busy = False
            if isinstance(e, RuntimeError):
                self.status.object = f"**cannot slice** — {e}"
            else:
                self.status.object = (f"**field slice error** — "
                                      f"{type(e).__name__}: {e}")

        def _ok(surf):
            try:
                self._finish_slice(p, surf)
            finally:
                self._busy = False

        _run_off_doc(
            lambda: self._surface(p.model, p.plane, p.idx, q=p.q, mz=p.mz),
            _ok, _fail)

    def _finish_slice(self, p, surf):
        """Doc-thread part: decimation, figure assembly (cheap on the
        decimated arrays), pane/status writes. Body unchanged from the
        pre-split compute()."""
        import numpy as np
        import plotly.graph_objects as go
        self.pane.loading = True
        try:
            plane = p.plane
            idx = p.idx
            a, b, vals, ele = surf
            a = np.asarray(a, float)
            b = np.asarray(b, float)
            vals = np.asarray(vals, float)
            # decimate to the resolution cap (responsive render)
            stride = max(1, int(round(max(vals.shape) / self.w_res.value)))
            if stride > 1:
                a = a[::stride]
                b = b[::stride]
                vals = vals[::stride, ::stride]
            z = vals.T                          # plotly: z[row=b, col=a]
            # log scaling: guard non-positive (|E| and PE are >=0; φ may be
            # signed, so log is offered but clamped with a reported floor)
            zlog = False
            if self.w_heat_log.value:
                pos = z[z > 0]
                if pos.size:
                    floor = float(pos.min())
                    z = np.log10(np.clip(z, floor, None))
                    zlog = True
                else:
                    self.status.object = ("_log scale skipped — no positive "
                                          "values on this slice_")
            lo = self.w_heat_lo.value
            hi = self.w_heat_hi.value
            zmin = (np.log10(lo) if (zlog and lo and lo > 0)
                    else (lo if lo is not None else None))
            zmax = (np.log10(hi) if (zlog and hi and hi > 0)
                    else (hi if hi is not None else None))
            fig = go.Figure()
            qlabel = {"efield": "|E| [V/mm]", "phi": "φ [V]",
                      "pe": "PE [eV]"}[self.w_quantity.value]
            bar_t = ("log10 " + qlabel) if zlog else qlabel
            if self.w_heat_on.value:
                fig.add_heatmap(x=a, y=b, z=z, colorscale=self.w_heat_cmap.value,
                                zmin=zmin, zmax=zmax,
                                colorbar=dict(title=bar_t))
            if self.w_cont_on.value:
                cc = self.w_cont_color.value
                line_col = None if cc == "match heatmap" else cc
                fig.add_contour(
                    x=a, y=b, z=z, showscale=(not self.w_heat_on.value),
                    ncontours=int(self.w_cont_n.value),
                    colorscale=(self.w_heat_cmap.value if cc == "match heatmap"
                                else None),
                    line=dict(color=line_col, width=1),
                    contours=dict(coloring=("lines" if line_col
                                            else "heatmap")),
                    colorbar=(dict(title=bar_t)
                              if not self.w_heat_on.value else None))
            spec = self.get_spec()
            device = getattr(spec, "name", "device") if spec else "device"
            pos_txt = ("auto mid-plane" if idx is None
                       else f"slice idx {idx}")
            an, bn = plane_axis_names(plane)
            _rz_note = (" · RF cycle-peak"
                        if (plane == "rz"
                            and self.w_quantity.value == "efield") else "")
            fig.update_layout(autosize=True,
                              xaxis_title=f"{an} [mm]",
                              yaxis_title=f"{bn} [mm]")
            V.layout_plotly_text(
                fig, title=(f"{qlabel}{_rz_note} — {device} · {plane} "
                            f"plane ({pos_txt})"
                            f"{' · log10' if zlog else ''}"))
            # equal geometric aspect ONLY when locked — delivered via the
            # SANCTIONED mechanism (M-VIZ-Z / viz_core.pane_size_for):
            # size the PANE 1:1 in mm and leave the figure free. A
            # scaleanchor is the known anti-pattern (rigid box,
            # stunted zoom; test_viz_core V3 scans for it) and the
            # a gate run caught this block using it.
            if self.w_equal_aspect.value:
                from ion_gym.viz.viz_core import pane_size_for
                _ext = (float(a[0]), float(a[-1]),
                        float(b[0]), float(b[-1]))
                self.pane.sizing_mode = "fixed"
                for _k, _v in pane_size_for(_ext).items():
                    setattr(self.pane, _k, _v)
            else:
                self.pane.sizing_mode = "stretch_width"
                self.pane.width = None
            # user view window (blank = full); partial bounds allowed
            _xr = [self.w_xlo.value, self.w_xhi.value]
            _yr = [self.w_ylo.value, self.w_yhi.value]
            if any(v is not None for v in _xr):
                fig.update_xaxes(range=[
                    _xr[0] if _xr[0] is not None else float(a[0]),
                    _xr[1] if _xr[1] is not None else float(a[-1])])
            if any(v is not None for v in _yr):
                fig.update_yaxes(range=[
                    _yr[0] if _yr[0] is not None else float(b[0]),
                    _yr[1] if _yr[1] is not None else float(b[-1])])
            # height follows the DISPLAYED aspect (capped): a 12x3 mm
            # SLIM slice no longer fills the screen
            _aw = ((_xr[1] if _xr[1] is not None else float(a[-1]))
                   - (_xr[0] if _xr[0] is not None else float(a[0])))
            _ah = ((_yr[1] if _yr[1] is not None else float(b[-1]))
                   - (_yr[0] if _yr[0] is not None else float(b[0])))
            if not self.w_equal_aspect.value:
                _h = 160 + 900 * (abs(_ah) / max(abs(_aw), 1e-9))
                self.pane.height = int(min(560, max(300, _h)))
            self._fig = fig
            self.pane.object = fig
            # stash the RAW (un-logged, un-strided) slice for CSV export:
            # a, b are the in-plane coords (mm), vals is the physical
            # quantity on the a×b grid at its native units. Metadata carries
            # everything needed to reconstruct/label the slice offline.
            self._last_slice = dict(
                a=np.asarray(a, float), b=np.asarray(b, float),
                vals=np.asarray(vals, float),
                plane=plane, quantity=self.w_quantity.value,
                qlabel=qlabel, mz=float(self.w_mz.value),
                slice_idx=(None if idx is None else int(idx)),
                stride=int(stride))
            self.status.object = (
                f"**computed** — {qlabel}, {plane} plane, {pos_txt}, "
                f"{vals.size:,} pts (stride {stride})"
                f"{', log10' if zlog else ''}.")
        except RuntimeError as e:
            self.status.object = f"**cannot slice** — {e}"
        except Exception as e:
            self.status.object = f"**field slice error** — {type(e).__name__}: {e}"
        finally:
            self.pane.loading = False

    def _sync_slice_bounds(self):
        """Set the slice-position slider's range to the current plane's
        normal axis on the current model. Called both on model change
        (via refresh) and on a plane switch (watcher), so the slider always
        matches the axis being sliced. Resets the slider to its minimum
        (auto mid-plane) on a change, since a position from the OLD axis is
        meaningless on the new one."""
        model = self.get_model()
        if model is None:
            return
        try:
            h = float(getattr(model, "h_mm", 0.0)) or 0.0
            A = getattr(model, "A", None)
            ax = {"xy": 2, "xz": 1, "yz": 0}.get(self.w_plane.value)
            if (h > 0 and A is not None and getattr(A, "ndim", 0) == 3
                    and ax is not None):
                _mo = _world_off(model)   # world frame, not stored
                n = A.shape[ax]
                start = round(_mo[ax], 3)
                self.w_slicepos.start = start
                self.w_slicepos.end = round(_mo[ax] + (n - 1) * h, 3)
                self.w_slicepos.value = start        # reset to auto
        except Exception as e:
            # UI boundary (widget callback; nothing above catches).
            # AUDITED: was a SILENT pass — a failed range
            # update left stale slider bounds with no trace. Reports.
            self.status.object = (f"**slice-range update failed** — "
                                  f"{type(e).__name__}: {e}")

    def _on_export_name(self, event):
        """Keep the download widget's filename in sync with the text field,
        defaulting the extension to .csv."""
        name = (event.new or "field_slice").strip()
        if not name.lower().endswith(".csv"):
            name = name + ".csv"
        self.w_export.filename = name

    def _export_csv(self):
        """Build the CSV of the last computed slice as an in-memory buffer
        (Panel FileDownload callback contract). A commented metadata header
        (device, plane, slice position, quantity, units, extent, grid,
        provenance) precedes a long-format table: in-plane coords a_mm,
        b_mm and the value. Long format round-trips cleanly to pandas /
        Excel and needs no reshape knowledge. Returns None with a status
        message if nothing has been computed yet (no silent empty file)."""
        import io as _io
        import datetime as _dt
        s = self._last_slice
        if s is None:
            self.status.object = ("_nothing to export — press **Compute "
                                  "slice** first_")
            return None
        spec = self.get_spec()
        model = self.get_model()
        device = getattr(spec, "name", "device") if spec else "device"
        plane = s["plane"]
        a = s["a"]
        b = s["b"]
        vals = s["vals"]                        # shape (len(a), len(b))
        units = {"efield": "V/mm", "phi": "V", "pe": "eV"}[s["quantity"]]
        h = float(getattr(model, "h_mm", 0.0)) if model is not None else 0.0
        normal = {"xy": "z", "xz": "y", "yz": "x"}.get(plane,
                                                       "none (2-D)")
        buf = _io.StringIO()
        w = buf.write
        w("# ion_gym field-slice export\n")
        w("# generated: {0}\n".format(
            _dt.datetime.now().isoformat(timespec="seconds")))
        w("# device: {0}\n".format(device))
        w("# quantity: {0} (units: {1})\n".format(s["qlabel"], units))
        an, bn = plane_axis_names(plane)
        w("# plane: {0}  (in-plane axes: a={1}, b={2})\n".format(
            plane, an, bn))
        w("# slice: {0}  (normal axis: {1})\n".format(
            "auto mid-plane" if s["slice_idx"] is None
            else "index " + str(s["slice_idx"]), normal))
        w("# grid: {0} x {1} (a x b), display stride {2}\n".format(
            len(a), len(b), s["stride"]))
        w("# extent: {0} [{1:.4f}, {2:.4f}] mm x {3} [{4:.4f}, {5:.4f}] "
          "mm\n".format(an, float(a.min()), float(a.max()),
                        bn, float(b.min()), float(b.max())))
        w("# node pitch h: {0:g} mm\n".format(h))
        if s["quantity"] == "pe":
            w("# ion m/z: {0:g} Da\n".format(s["mz"]))
        w("# import_path: {0}\n".format(
            getattr(spec, "import_path", "") if spec else ""))
        w("#\n")
        # GRID format (the long format was abnormally
        # large): a rectilinear slice has SEPARABLE coordinates, so the
        # long form's one-row-per-cell repeated both a_mm and b_mm for
        # every value — ~2/3 of every line was redundant coordinate
        # text, and it grew with resolution. Grid form writes the `a`
        # axis once (header), each `b` value once (row label), then the
        # value matrix: identical information, no coordinate repetition
        # (~3x smaller and directly loadable as a labelled matrix).
        w("# format: grid — first row is {0}_mm axis; first column is "
          "{1}_mm; cells are {2} [{3}]\n".format(
              an, bn, s["quantity"], units))
        w("# to load: pandas.read_csv(f, comment='#', index_col=0) "
          "-> index={0}_mm, columns={1}_mm\n".format(bn, an))
        w("{0}_mm\\{1}_mm,".format(bn, an)
          + ",".join("{0:.6g}".format(ai) for ai in a) + "\n")
        for j in range(len(b)):
            w("{0:.6g},".format(b[j])
              + ",".join("{0:.6g}".format(vals[i, j])
                         for i in range(len(a))) + "\n")
        data = buf.getvalue().encode("utf-8")
        self.status.object = (
            "**exported** {0}x{1} grid ({2:,} cells) to `{3}` "
            "({4}, {5} plane).".format(
                len(a), len(b), len(a) * len(b), self.w_export.filename,
                s["qlabel"], plane))
        return _io.BytesIO(data)

    def refresh(self):
        """Sync the plane options + slider bounds to the current model.
        Does not auto-recompute (the slice is compute-on-press, like PE)."""
        model = self.get_model()
        if model is None:
            return
        self._sync_planes(model)
        self._sync_slice_bounds()


def _has_phi_component(model):
    """True when compute_component understands a bare 'phi' request for
    this model; falls back to the DC component otherwise."""
    return hasattr(model, "A")


class PeSurfaceTab:
    """A 'PE Surface (3-D)' tab: compute the node-centred landscape ONCE
    for the current view, cache it, and DESTROY the cache the moment any
    surface-affecting parameter changes (geometry, a voltage, the mass,
    resolution, the draped run). The user then re-presses Compute.

    Wire-up (sim_app): construct once with accessors, add panel() to the
    Tabs, and call `.refresh()` wherever the model/spec/active-run change
    (end of _redraw, _draw_background.done, _sync_spec)."""

    def __init__(self, get_model, get_spec, get_results, get_run_id):
        import panel as pn
        self._pn = pn
        self.get_model = get_model
        self.get_spec = get_spec
        self.get_results = get_results
        self.get_run_id = get_run_id
        self._sig = None
        self._fig = None

        self.w_mz = pn.widgets.FloatInput(name="ion m/z (Da)", value=100.0,
                                          width=130)
        self.w_comp = pn.widgets.Select(
            name="component", width=200,
            options={"effective (DC + RF pseudo)": "effective",
                     "DC only": "dc", "RF pseudopotential only": "rf",
                     "|E| field [V/mm]": "efield"})
        self.w_zaspect = pn.widgets.FloatSlider(
            name="height (z aspect)", start=0.1, end=2.5, step=0.05,
            value=0.55, width=230)
        self.w_exag = pn.widgets.FloatInput(
            name="PE transform factor / asinh knee (eV)", value=1.0,
            step=0.5, width=240)
        self.w_scalemode = pn.widgets.Select(
            name="PE transform (optional)", width=200,
            options={"none (linear ×1... factor)": "linear",
                     "asinh — tame deep wells": "asinh"})
        self.w_res = pn.widgets.IntSlider(
            name="resolution (pts/axis, ↓ = faster)", start=60, end=400,
            step=20, value=200, width=240)
        self.w_drape = pn.widgets.Checkbox(
            name="drape current trajectories", value=True)
        self.w_plane = pn.widgets.Select(
            name="plane", value="xy", options=["xy", "xz", "yz"], width=90)
        # slice position along the axis NOT in the plane (viewing xz,
        # let me see the PE at a chosen y). value in mm; -1 (default) means
        # the auto electrode-dense slice. Re-populated per plane/model in
        # refresh() so its range matches the perpendicular axis extent.
        self.w_slicepos = pn.widgets.FloatSlider(
            name="slice position (mm; auto if at min)", start=-1.0, end=0.0,
            step=0.05, value=-1.0, width=240)
        # numeric twin of the slice slider (a typed value
        # is precise where the slider is coarse). Bidirectional link keeps
        # the two in sync; either edits the same slice position.
        self.w_slicenum = pn.widgets.FloatInput(
            name="slice (mm, exact)", value=-1.0, step=0.05, width=140)
        self.w_slicepos.link(self.w_slicenum, value="value")
        self.w_slicenum.link(self.w_slicepos, value="value")
        # LAB-TIME for instant drives (slow TW): a CHECKBOX + a typed
        # value, deliberately not a slider — the surface only recomputes
        # on Compute, and a typed time commits once instead of streaming
        # invalidations. Checkbox off resolves to exactly t = 0.0, i.e.
        # the surface every deck showed before this control existed.
        self.w_time_on = pn.widgets.Checkbox(
            name="evaluate instant drives at a lab time", value=False)
        self.w_t_us = pn.widgets.FloatInput(
            name="lab time t (µs)", value=0.0, step=1.0, width=140,
            disabled=True)
        # VIEW ANGLE: checkbox + angle inputs. Off (the default) = the
        # historical camera exactly; on = camera_eye(azim, elev, dist,
        # ortho). Camera is pure LAYOUT: these widgets are excluded from
        # the surface signature and re-apply LIVE to the displayed
        # figure -- no recompute, and stale-cache invalidation never
        # fires for a view change.
        self.w_view_on = pn.widgets.Checkbox(
            name="set view angle / perspective", value=False)
        self.w_azim = pn.widgets.FloatInput(
            name="azimuth (°)", value=-48.6, step=5.0, width=110,
            disabled=True)
        self.w_elev = pn.widgets.FloatInput(
            name="elevation (°)", value=21.7, step=5.0, width=110,
            disabled=True)
        self.w_dist = pn.widgets.FloatInput(
            name="distance", value=2.44, step=0.2, width=100,
            disabled=True)
        self.w_ortho = pn.widgets.Checkbox(
            name="orthographic", value=False, disabled=True)
        self.reset_btn = pn.widgets.Button(name="Reset PE view",
                                           button_type="default", width=120)
        self.w_metal = pn.widgets.Select(
            name="electrodes on the PE scale", value="barrier", width=170,
            options={"mask (hole)": "mask",
                     "DC bias (q·V_DC)": "dc",
                     "barrier (wall height)": "barrier"})
        self.w_trust = pn.widgets.IntSlider(
            name="trust radius (cells from metal)", start=1, end=6, value=2,
            width=200)
        self.compute_btn = pn.widgets.Button(name="Compute surface",
                                             button_type="primary")
        self.status = pn.pane.Markdown("_press **Compute surface** "
                                       "(solve the field first via Fly / "
                                       "Recompute)_")
        self.pane = pn.pane.Plotly(height=560, sizing_mode="stretch_width",
                                   config={"scrollZoom": True})

        # L-58b: the button runs the split async path; direct
        # compute() calls (harness, scripts) stay synchronous.
        self.compute_btn.on_click(self._on_go)
        for w in (self.w_mz, self.w_comp, self.w_exag, self.w_scalemode,
                  self.w_res, self.w_drape, self.w_plane, self.w_metal,
                  self.w_trust, self.w_slicepos, self.w_t_us):
            w.param.watch(lambda e: self.refresh(), "value")
        self.w_time_on.param.watch(self._on_time_toggle, "value")
        for w in (self.w_azim, self.w_elev, self.w_dist, self.w_ortho):
            w.param.watch(lambda e: self._apply_camera(), "value")
        self.w_view_on.param.watch(self._on_view_toggle, "value")
        self.reset_btn.on_click(self._on_reset)
        # widget defaults captured ONCE, right here at construction, so
        # Reset restores exactly what a fresh tab would show -- no
        # second hand-maintained default table to drift (L-463)
        self._defaults = {w: w.value for w in (
            self.w_mz, self.w_comp, self.w_exag, self.w_scalemode,
            self.w_res, self.w_drape, self.w_plane, self.w_metal,
            self.w_trust, self.w_slicepos, self.w_slicenum,
            self.w_time_on, self.w_t_us, self.w_view_on, self.w_azim,
            self.w_elev, self.w_dist, self.w_ortho, self.w_zaspect)}
        # z-aspect is a pure VIEW property: restyle the cached figure in
        # place — no invalidation, no recompute.
        self.w_zaspect.param.watch(self._on_aspect, "value")

    def _on_aspect(self, event):
        if self._fig is not None:
            self._fig.layout.scene.aspectratio.z = max(float(event.new), 0.05)
            self.pane.object = self._fig

    # ---- cache logic
    def _stride_for(self, model):
        try:
            nx, ny = model.pe_surface(mz=self.w_mz.value)[2].shape
        except (AttributeError, TypeError):
            # model not ready yet -> full resolution. AUDITED
            # narrowed from Exception; a pe_surface bug on
            # a READY model now raises (and the compute boundary
            # reports it) instead of silently forcing stride 1.
            return 1
        return max(1, int(np.ceil(max(nx, ny) / max(self.w_res.value, 20))))

    def _camera_dict(self):
        """None with the checkbox off (= the historical default view);
        else the camera_eye kwargs the widgets spell."""
        if not self.w_view_on.value:
            return None
        return dict(azim_deg=float(self.w_azim.value),
                    elev_deg=float(self.w_elev.value),
                    dist=float(self.w_dist.value),
                    ortho=bool(self.w_ortho.value))

    def _apply_camera(self):
        """Re-apply the camera to the DISPLAYED figure, live. Layout
        only: the surface is untouched, nothing recomputes."""
        fig = getattr(self.pane, "object", None)
        if fig is None or not hasattr(fig, "update_scenes"):
            return
        cam = self._camera_dict()
        fig.update_scenes(camera=(camera_eye(**cam) if cam
                                  else _PE_CAMERA_DEFAULT))
        self.pane.param.trigger("object")

    def _on_view_toggle(self, _e=None):
        on = self.w_view_on.value
        for w in (self.w_azim, self.w_elev, self.w_dist, self.w_ortho):
            w.disabled = not on
        self._apply_camera()          # off -> back to the default view

    def _on_reset(self, _e=None):
        """Every PE control back to its constructed default (time off,
        slice auto, view off, default camera), then one refresh. The
        stale-cache contract applies as for any edit: press Compute."""
        for w, v in self._defaults.items():
            w.value = v
        self.status.object = ("_PE controls reset to defaults -- press "
                              "**Compute surface**_")

    def _on_time_toggle(self, _e=None):
        self.w_t_us.disabled = not self.w_time_on.value
        self.refresh()

    def _time_value(self):
        """Lab time the surface is computed at: exactly 0.0 with the
        checkbox off (the pre-existing default surface), else the typed
        value."""
        return float(self.w_t_us.value) if self.w_time_on.value else 0.0

    def _current_sig(self, model):
        spec = self.get_spec()
        return _surface_signature(
            spec, self.w_mz.value,
            (self.w_scalemode.value, self.w_exag.value),
            self._stride_for(model), self.w_drape.value,
            self.get_run_id(), self.w_comp.value,
            plane=self.w_plane.value,
            metal=(self.w_metal.value, int(self.w_trust.value)),
            slice_pos=round(float(self.w_slicepos.value), 4),
            t_us=self._time_value())

    def _sync_planes(self, model):
        """The selector offers exactly the planes the model DECLARES.

        This is the origin of a real crash: the widget was built with
        a hard-coded ["xy","xz","yz"] regardless of the model, so pressing
        'yz' on an r-z model reached the solver and blew up in the bokeh event
        loop.  A control that can request an impossible state is the bug; the
        exception was only the symptom.  Offer what exists.
        """
        planes = list(V.model_planes(model))
        if list(self.w_plane.options) != planes:
            self.w_plane.options = planes
            if self.w_plane.value not in planes:
                self.w_plane.value = planes[0]
        self.w_plane.disabled = (len(planes) == 1)
        return self.w_plane.value

    def _slice_index(self, model, plane):
        """Slice node from the mm control, via the module's ONE rule
        (slice_index). The control at its minimum means AUTO (None) --
        the only UI-specific decision here. Any other value is mapped or
        REFUSED (off-grid position, a plane with no normal); the caller
        reports the refusal in the status line. The two tabs' private
        copies of this mapping disagreed on unknown planes (one silently
        sliced z) and let off-grid positions through to fail later on
        the worker thread; both are superseded by this."""
        if self.w_slicepos.value <= self.w_slicepos.start:
            return None
        return slice_index(model, plane, float(self.w_slicepos.value))

    # compute is split so the HEAVY part (grid PE +
    # plotly figure) can run off the document thread. compute() stays the
    # synchronous composition (the headless harness calls it directly);
    # the button goes through _on_go(), which runs _build on a worker and
    # _apply on the doc thread via _run_off_doc. _prep touches widgets
    # and spec (doc thread only); _build touches NEITHER.
    def _prep(self):
        """Doc-thread part: read widgets, resolve DC, assemble every
        argument _build needs. Returns None (with status set) when there
        is nothing to compute."""
        import types
        model = self.get_model()
        if model is None or not hasattr(model, "pe_surface"):
            self.status.object = ("_no solved field yet — press Fly or "
                                  "Recompute, then Compute surface_")
            return None
        self._sync_planes(model)
        p = types.SimpleNamespace(model=model)
        p.t_us = self._time_value()
        p.camera = self._camera_dict()
        if self.w_time_on.value:
            t_max = float(self.get_spec().integration.t_max_us)
            if not np.isfinite(p.t_us) or p.t_us < 0.0 or p.t_us > t_max:
                # refuse-with-diagnostic, figure untouched: the valid
                # window is named and its source stated.
                self.status.object = (
                    f"**refused** — lab time t = {p.t_us!r} µs is outside "
                    f"the valid window [0, {t_max:g}] µs "
                    f"(integration.t_max_us; the flight this surface "
                    f"belongs to lives there). Nothing was recomputed.")
                return None
            if not model_instant_drives(model):
                self.status.object = (
                    "_note: this model has no instant drives — t has no "
                    "effect on the surface (sin drives are the averaged "
                    "pseudopotential)_")
        p.mz = float(self.w_mz.value)
        # spec charge rides the pack: the surface must show the ion the
        # spec flies (displayed == solver input), not a hard-coded z=1
        p.charge = int(getattr(self.get_spec().source, "charge", 1))
        p.stride = self._stride_for(model)
        p.comp = self.w_comp.value
        p.plane = self.w_plane.value
        # slice index along the perpendicular axis from the mm slider
        # (auto electrode-dense slice when the slider is at its minimum);
        # an off-grid or unsliceable request refuses HERE, on the doc
        # thread, with the valid range named
        try:
            p.idx = self._slice_index(model, p.plane)
        except ValueError as e:
            self.status.object = f"**refused** — {e}. Nothing was recomputed."
            return None
        p.results = self.get_results() if self.w_drape.value else None
        p.has_rf = model_has_rf(model)
        p.comp_label = {"effective": "effective (DC + RF pseudo)",
                        "dc": "DC only",
                        "rf": "RF pseudopotential"}.get(p.comp, str(p.comp))
        p.device = getattr(self.get_spec(), "name", "device")
        p.mmode = self.w_metal.value
        p.edc = None
        if p.mmode != "mask":
            sp = self.get_spec()
            # NOT a silent pass.  This resolves the DC ladder; if it
            # fails, `edc` below is built from UNRESOLVED voltages and the
            # picture shows numbers the SOLVER NEVER USED.  "Display must
            # equal solver input" is not a preference that degrades
            # gracefully -- a wrong voltage on a figure is a wrong figure.
            sp.resolve_dc_groups()      # ladder members: derived, live
            p.edc = {(e.basis if e.basis else i + 1): float(e.dc)
                     for i, e in enumerate(sp.geometry.electrodes)}

        _is_E = (p.comp == "efield")
        p.quant = ("|E|", "V/mm") if _is_E else ("PE", "eV")
        if _is_E:
            # a field surface has no PE plateau convention: q*V_DC is not
            # a field value, so the metal is MASKED regardless of the
            # metal selector, and that is stated rather than silently
            # applied.
            p.mmode = "mask"
        p.title = ((f"|E| surface — {p.device} · {p.plane} plane"
                    if _is_E else
                    f"PE surface — {p.device} · {p.comp_label} · m/z "
                    f"{p.mz:g} · {p.plane} plane"))
        if self.w_time_on.value:
            p.title += f" · t = {p.t_us:g} µs"
        if _is_E:
            p.title += ("<br><sup>|E| at the RF PEAK for an RF device "
                        "(full 3-D magnitude on the chosen slice); metal "
                        "masked</sup>")
        if p.mmode == "barrier":
            # the convention and its arbitrary parameter belong ON the
            # figure, not in a docstring nobody opens
            p.title += (f"<br><sup>electrodes: BARRIER convention "
                        f"(Ψ on trusted vacuum, {int(self.w_trust.value)} "
                        f"cells out) — a landmark, not a PE value; "
                        f"RF walls scale as 1/m</sup>")
        elif p.mmode == "dc":
            p.title += ("<br><sup>electrodes: DC BIAS (q·V_DC) — an RF rod "
                        "reads 0 V here, but |E_RF| is largest AT the rods: "
                        "on an RF device this draws the wall as a valley"
                        "</sup>")
        # RF caveat only when the device has RF AND the component shows it
        p.caveat = p.has_rf and p.comp in ("effective", "rf")
        # the slice position in the CANONICAL frame, for the drape's
        # slab filter (only ions near a SLICED plane get draped on it).
        # None on an auto/electrode-dense slice -> no slab restriction.
        p.norm_mm = None
        if p.plane in ("xz", "yz") and p.idx is not None:
            _h = float(getattr(model, "h_mm", 0.0)) or 0.0
            _mo = _world_off(model)   # world frame, not stored
            _axn = {"xz": 1, "yz": 0}[p.plane]
            if _h > 0:
                p.norm_mm = p.idx * _h + _mo[_axn]
        p.trust_cells = int(self.w_trust.value)
        p.z_exaggerate = self.w_exag.value
        p.pe_scale_mode = self.w_scalemode.value
        p.z_aspect = self.w_zaspect.value
        p.h_mm = (float(getattr(model, "h_mm", 0.1)) or 0.1)
        return p

    @staticmethod
    def _build(p):
        """WORKER-SAFE heavy part: grid component + plotly figure. Touches
        no widgets, no Panel objects, no shared spec (everything arrives
        in `p`, assembled on the doc thread by _prep)."""
        import numpy as np
        surface = compute_component(p.model, p.mz,
                                    getattr(p, "charge", 1), p.comp,
                                    plane=p.plane, index=p.idx,
                                    t_us=getattr(p, "t_us", 0.0))
        fig = pe_figure_3d(
            p.model, surface=surface, results=p.results,
            t_us=getattr(p, "t_us", 0.0),
            camera=getattr(p, "camera", None),
            z_exaggerate=p.z_exaggerate,
            pe_scale_mode=p.pe_scale_mode, stride=p.stride,
            title=p.title, show_adiabatic_caveat=p.caveat,
            z_aspect=p.z_aspect,
            metal_mode=p.mmode, electrode_dc=p.edc,
            trust_cells=p.trust_cells,
            quantity=p.quant, plane=p.plane, index=p.idx,
            normal_mm=p.norm_mm,
            h_mm=p.h_mm)
        npts = surface[2][::p.stride, ::p.stride].size
        # report the surface's actual EXTENT + grid shape, so a slice
        # that looks like it covers only part of the model can be read
        # against the true model bounds (PE/bounds
        # seem to cover half the model). a,b are the in-plane coords.
        _sa = np.asarray(surface[0], float)
        _sb = np.asarray(surface[1], float)
        fullshape = getattr(getattr(p.model, "A", None), "shape", None)
        extent = ("{0} [{1:.2f}, {2:.2f}] mm × {3} [{4:.2f}, {5:.2f}] mm"
                  .format(p.plane[0], float(_sa.min()), float(_sa.max()),
                          p.plane[1], float(_sb.min()), float(_sb.max()))
                  if _sa.size and _sb.size else "empty")
        return fig, npts, extent, fullshape

    def _apply(self, p, out):
        """Doc-thread part: pane/status/cache writes."""
        fig, npts, extent, fullshape = out
        self._fig = fig
        self._sig = self._current_sig(p.model)
        self.pane.object = fig
        note = ""
        if p.has_rf and p.comp == "dc":
            note = " ⚠️ DC-only on an RF device hides the pseudopotential "\
                   "confinement — use 'effective' or 'RF pseudopotential'"
        self.status.object = (
            f"**computed** — {p.comp_label}, m/z {p.mz:g}, "
            f"{npts:,} pts (stride {p.stride}); extent {extent}"
            f"{'; model grid ' + '×'.join(map(str, fullshape)) if fullshape else ''}"
            f". Cached until a parameter changes.{note}")
        self.pane.loading = False
        self._busy = False

    def compute(self):
        """Synchronous composition — the headless harness and any script
        caller get exactly the old contract (figure ready on return)."""
        p = self._prep()
        if p is None:
            return
        self.pane.loading = True
        try:
            self._apply(p, self._build(p))
        finally:
            self.pane.loading = False

    def _on_go(self, _event=None):
        """Button path: heavy _build off the doc thread (L-58b)."""
        if getattr(self, "_busy", False):
            self.status.object = "_already computing — one at a time_"
            return
        p = self._prep()
        if p is None:
            return
        self._busy = True
        self.pane.loading = True
        self.status.object = "_computing surface (off the UI thread)…_"

        def _fail(e, tb):
            self.pane.loading = False
            self._busy = False
            last = tb.strip().splitlines()[-6:] if tb else []
            self.status.object = (
                f"**compute failed** — {type(e).__name__}: {e}"
                + ("\n```\n" + "\n".join(last) + "\n```" if last else ""))

        _run_off_doc(lambda: self._build(p),
                     lambda out: self._apply(p, out), _fail)

    def refresh(self):
        """Cheap staleness check. Called on any model/spec/run change and
        on this tab's own control edits. If the cache no longer matches the
        current view, DESTROY it and prompt for recompute."""
        model = self.get_model()
        if model is None or not hasattr(model, "pe_surface"):
            if self._fig is not None:
                self._fig = None
                self._sig = None
                self.pane.object = self._pn.pane.Plotly().object
            self.status.object = ("_no solved field yet — press Fly or "
                                  "Recompute, then Compute surface_")
            return
        # |E| is offered ONLY when the model exposes field_surface() — the
        # same principle as the plane selector (never offer what the model
        # cannot answer). 2-D r-z/planar models keep their PE components.
        _base = {"effective (DC + RF pseudo)": "effective",
                 "DC only": "dc", "RF pseudopotential only": "rf"}
        if hasattr(model, "field_surface"):
            _base["|E| field [V/mm]"] = "efield"
        _have = self.w_comp.options
        _have = (set(_have.values()) if isinstance(_have, dict)
                 else set(_have))
        if _have != set(_base.values()):
            _cur = self.w_comp.value
            self.w_comp.options = _base
            if _cur in _base.values():
                self.w_comp.value = _cur

        # set the slice slider's range to the PERPENDICULAR axis extent for
        # the current plane (z for xy, y for xz, x for yz), in mm. Keeping
        # the minimum as the 'auto' sentinel: start one step below 0 so the
        # left end still means the electrode-dense auto slice.
        try:
            nx, ny, nz = model.A.shape
            h = float(getattr(model, "h_mm", 0.0)) or 0.0
            _axp = {"xy": 2, "xz": 1, "yz": 0}.get(self.w_plane.value, 2)
            perp = (nx, ny, nz)[_axp]
            _mo = _world_off(model)   # world frame, not stored
            if h > 0 and perp > 1:
                lo = _mo[_axp]                  # canonical low edge
                hi = _mo[_axp] + (perp - 1) * h
                step = max(h, (hi - lo) / 200.0)
                if abs(self.w_slicepos.end - hi) > 1e-9:
                    # AUTO must survive the retarget: a value sitting at
                    # the OLD sentinel moves to the NEW one. Without this
                    # the default -1.0 read as a user request on any axis
                    # extending below -1 mm (mirrored decks), and the
                    # "auto" view silently cut z = -1 (L-462; exposed by
                    # the slice stamp). A genuine request is kept as-is;
                    # if it is now off-grid, Compute REFUSES with the
                    # range (slice_index) -- never a quiet reset to auto.
                    _was_auto = (self.w_slicepos.value
                                 <= self.w_slicepos.start)
                    self.w_slicepos.start = lo - step   # sentinel = auto
                    self.w_slicepos.end = hi
                    self.w_slicepos.step = step
                    if _was_auto:
                        self.w_slicepos.value = lo - step
        except Exception as e:
            # UI boundary; AUDITED: silent pass -> reports.
            self.status.object = (f"**slice-range sync failed** — "
                                  f"{type(e).__name__}: {e}")
        if self._sig is None:
            self.status.object = ("_press **Compute surface** for this view_")
            return
        if self._current_sig(model) != self._sig:
            self._fig = None                       # destroy cached surface
            self._sig = None
            self.pane.object = self._pn.pane.Plotly().object
            self.status.object = ("**parameters changed** — cached surface "
                                  "discarded. Press **Compute surface** "
                                  "to rebuild.")

    def panel(self):
        pn = self._pn
        return pn.Column(
            pn.Row(self.w_mz, self.w_comp, self.w_plane),
            pn.Row(self.w_slicepos, self.w_slicenum),
            pn.Row(self.w_metal, self.w_trust),
            pn.Row(self.w_time_on, self.w_t_us),
            pn.Row(self.w_view_on, self.w_azim, self.w_elev, self.w_dist,
                   self.w_ortho),
            pn.Row(self.w_drape),
            pn.Row(self.w_zaspect),
            pn.Row(self.w_scalemode, self.w_exag),
            pn.Row(self.w_res),
            pn.Row(self.compute_btn, self.reset_btn),
            self.status,
            self.pane,
            sizing_mode="stretch_width")
