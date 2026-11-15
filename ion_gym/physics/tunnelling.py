"""
tunnelling.py -- can one integrator step carry an ion through metal?

The flight kernels test for metal where each step ENDS. An ion that moves
farther in one step than a wall is thick can land on the far side and fly
on: 20 keV ions at the default 1 ns step crossed a 0.5 mm plate 24 times
in 40, and none crossed it at 0.05 ns. Nothing said so.

After a build this module compares the longest step any ion can take with
the thinnest wall the route's hit test sees, and returns an advisory with
a step that would not tunnel. It warns and does not refuse, and it changes
no kernel.

  * Longest step: dt x the fastest possible speed. That is the fastest
    ion at birth (from the built births, so every source distribution and
    births file counts) given all the energy the deck's full voltage span
    can add, |z| x (highest - lowest potential), at the lightest mass. An
    upper bound: real ions rarely gain the full span next to the thinnest
    wall.
  * Thinnest wall: the Euclidean distance from each metal node to vacuum
    peaks along the middle of a wall, so the smallest of those peaks is
    half the thinnest wall (corners and voxel staircases fall away from
    the middle and do not register). Walls are measured on the model's
    electrode labels, r-z mirrored across the axis, and converted with the
    route's hit rule: planar and r-z stop an ion within n nodes of an
    n-node wall (nearest node), 3-D only inside all-metal cells, so
    (n - 1) nodes. Grid electrodes (transparent by declaration) and the
    domain's ground border are not walls.
"""
import math

import numpy as np

from ion_gym.physics.collision3d import E_CHG, KG_AMU

# a suggested step moves the fastest ion this fraction of the thinnest wall
SAFE_WALL_FRACTION = 0.5


def voltage_span(spec):
    """Highest minus lowest potential the deck can apply (V): each
    electrode's effective DC plus or minus the peak of every drive it is in
    (amplitude, or the largest table value, plus the offset); ground (0 V)
    is always present."""
    g = spec.geometry
    groups = {gr.name: gr for gr in g.rf_groups}
    hi = lo = 0.0
    for el in g.electrodes:
        v = float(spec.dc_effective(el))
        swing = 0.0
        for name in el.rf_groups or []:
            gr = groups.get(name)
            if gr is None:
                continue                    # validate() names a missing group
            if gr.waveform == "table" and gr.table_v:
                peak = max(abs(float(x)) for x in gr.table_v)
            else:
                peak = abs(float(gr.amplitude_v))
            swing += peak + abs(float(gr.offset_v or 0.0))
        hi, lo = max(hi, v + swing), min(lo, v - swing)
    return hi - lo


def fastest_speed_mm_per_ns(spec, births):
    """(speed mm/ns, details) of the fastest ion the deck can produce."""
    b = np.asarray(births, float)
    v0_mm_us = float(np.sqrt((b[:, 3:6] ** 2).sum(axis=1)).max()) if len(b) \
        else 0.0
    m_amu = min(float(m) for m in spec.source.mz_list)
    z = abs(int(spec.source.charge or 1))
    span = voltage_span(spec)
    v0 = v0_mm_us * 1e3                                 # m/s
    v = math.sqrt(v0 ** 2 + 2.0 * z * E_CHG * span / (m_amu * KG_AMU))
    ke_ev = 0.5 * m_amu * KG_AMU * v ** 2 / E_CHG
    return v * 1e-6, dict(mass_amu=m_amu, charge=z, span_v=span,
                          ke_ev=ke_ev)


def thinnest_wall(ele, h_mm, electrodes, *, rz, all_metal_cells):
    """(thickness mm, electrode index) of the thinnest wall among the deck's
    non-grid electrodes, as the route's hit test sees it, or None."""
    from scipy import ndimage
    lab = np.asarray(ele)
    if rz:                               # the stored half, axis at index 0
        lab = np.concatenate([lab[:, :0:-1], lab], axis=1)
    walls = [i for i, el in enumerate(electrodes, start=1)
             if not getattr(el, "is_grid", False)]
    metal = np.isin(lab, walls)
    if not metal.any():
        return None
    depth = ndimage.distance_transform_edt(metal)
    middle = metal & (depth >= ndimage.maximum_filter(depth, size=3))
    best = None
    for i in walls:
        sel = middle & (lab == i)
        if not sel.any():
            continue
        n_nodes = 2.0 * float(depth[sel].min()) - 1.0
        t = (n_nodes - 1.0 if all_metal_cells else n_nodes) * h_mm
        if t > 0 and (best is None or t < best[0]):
            best = (t, i)
    return best


def tunnelling_note(spec, model, births):
    """The advisory when the longest possible step exceeds the thinnest
    wall, else None."""
    h = getattr(model, "mm_per_gu", None) or getattr(model, "h_mm", None)
    ele = getattr(model, "ele", None)
    dt = spec.integration.dt_ns
    if h is None or ele is None or not dt:
        return None
    h, dt = float(h), float(dt)
    v, d = fastest_speed_mm_per_ns(spec, births)
    step = v * dt
    if step <= h:            # no step can cross even a one-node wall
        return None
    lab = np.asarray(ele)
    thin = thinnest_wall(lab, h, spec.geometry.electrodes,
                         rz=spec.geometry.symmetry.coords == "rz",
                         all_metal_cells=lab.ndim == 3)
    if thin is None or step <= thin[0]:
        return None
    t, i = thin
    name = spec.geometry.electrodes[i - 1].name
    dt_ok = SAFE_WALL_FRACTION * t / v
    return (f"integration: ions can step THROUGH metal -- the fastest "
            f"possible ion (m={d['mass_amu']:g} Da, z={d['charge']}, up to "
            f"{d['ke_ev']:.4g} eV with the deck's {d['span_v']:g} V span) "
            f"moves {step:.3g} mm per {dt:g} ns step, more than the "
            f"{t:.3g} mm wall of electrode {i} ({name!r}), the thinnest the "
            f"hit test sees at {h:g} mm/gu. The kernels test for metal only "
            f"where a step ends, so such an ion can land beyond the wall. "
            f"Use dt_ns <= {dt_ok:.3g} (half that wall per step) or thicken "
            f"the electrode. An upper bound: an ion rarely has the full span "
            f"beside the thinnest wall.")
