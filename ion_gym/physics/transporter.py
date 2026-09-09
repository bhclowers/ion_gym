"""transporter.py -- the device-agnostic validity gate for the periodic
transporter plane pair.

The transporter kernel hook is deliberately dumb; ITS entire physics
claim -- folded flight equals unfolded flight -- is the claim that every
solved drive basis in a slab around the emit plane is the accept-plane
slab translated by (emit - accept). This module CHECKS that claim from
the actual solve and refuses when it fails, naming the worst offender.
It knows nothing about any device: two planes, a tolerance, the field
pack.
"""
from __future__ import annotations
import numpy as np

_AX = {"x": 0, "y": 1, "z": 2}


def _axis_lerp(F, axis, idx0, frac):
    """F sampled at (idx0 + frac) along `axis` by linear interpolation
    along that axis only (the shift is 1-D by construction)."""
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = idx0
    sl1[axis] = idx0 + 1
    return (1.0 - frac) * F[tuple(sl0)] + frac * F[tuple(sl1)]


def check_field_equivalence(fields, *, axis, accept_mm, emit_mm,
                            tol_v_per_mm=1e-3, slab_gu=2):
    """W1: for the static field and EVERY drive channel, compare the
    gradient triplet on a slab of `slab_gu` planes around accept vs the
    same slab around emit (shift interpolated linearly along `axis`;
    the interpolation's own error is second-order in the sub-gu offset
    and reported). Returns a report dict; RAISES with the worst point
    named if the max deviation exceeds tol.

    Units: the packs store E in V/mm, so tol is V/mm.
    """
    h = float(fields["h_mm"])
    ax = _AX[str(axis)]
    d_gu = (emit_mm - accept_mm) / h
    triplets = [("EA", fields["EAx"], fields["EAy"], fields["EAz"])]
    ExK, EyK, EzK = (np.asarray(fields.get(k)) for k in ("ExK", "EyK", "EzK"))
    if ExK is not None and ExK.size:
        for k in range(ExK.shape[0]):
            triplets.append((f"ch{k}", ExK[k], EyK[k], EzK[k]))
    n_ax = triplets[0][1].shape[ax]
    ia = int(round(accept_mm / h))
    shift = int(np.floor(d_gu))
    frac = d_gu - shift
    worst = (0.0, "", 0, (0, 0, 0))
    for lab, Fx, Fy, Fz in triplets:
        for comp, F in (("Ex", Fx), ("Ey", Fy), ("Ez", Fz)):
            for off in range(-slab_gu, slab_gu + 1):
                i0 = ia + off
                i1 = i0 + shift
                if i0 < 0 or i0 >= n_ax or i1 < 0 or i1 + 1 >= n_ax:
                    raise ValueError(
                        f"W1: probe slab leaves the solved domain "
                        f"(axis {axis}, accept plane gu {ia}, offset {off}, "
                        f"shift {d_gu:.2f} gu) -- planes too close to the "
                        f"array edge for a {slab_gu}-gu slab")
                slA = [slice(None)] * 3
                slA[ax] = i0
                A = F[tuple(slA)]
                B = _axis_lerp(F, ax, i1, frac)
                dev = np.abs(A - B)
                m = float(dev.max())
                if m > worst[0]:
                    j = np.unravel_index(int(dev.argmax()), dev.shape)
                    worst = (m, f"{lab}.{comp}", off, j)
    report = dict(axis=axis, accept_mm=accept_mm, emit_mm=emit_mm,
                  shift_gu=d_gu, frac_gu=frac, slab_gu=slab_gu,
                  n_channels=len(triplets) - 1, worst_dev_v_mm=worst[0],
                  worst_channel=worst[1], worst_slab_offset_gu=worst[2],
                  worst_transverse_index=tuple(int(v) for v in worst[3]),
                  tol_v_per_mm=tol_v_per_mm)
    if worst[0] > tol_v_per_mm:
        raise ValueError(
            f"W1 FIELD EQUIVALENCE FAILED: max |E_accept - E_emit(shifted)| "
            f"= {worst[0]:.3e} V/mm > tol {tol_v_per_mm:.3e} on channel "
            f"{worst[1]}, slab offset {worst[2]} gu, transverse index "
            f"{worst[3]} (shift {d_gu:.3f} gu, lerp frac {frac:.3f}). The "
            f"planes are NOT one field period apart on this device; move "
            f"them or fix the geometry -- do not override silently."
            + ("" if frac < 1e-9 else
               f" NOTE: the shift is a NON-INTEGER {d_gu:.3f} gu, so the "
               f"comparison carries an O(h^2 f'') linear-interpolation error "
               f"that is largest near electrode surfaces; before concluding "
               f"the geometry is aperiodic, re-solve at a COMMENSURATE pitch "
               f"(period / integer, e.g. mm_per_gu = period_mm / "
               f"{int(round(d_gu))})."))
    return report
