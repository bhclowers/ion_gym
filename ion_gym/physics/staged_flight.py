"""Staged multi-region flight — fly an ion through SEPARATE field regions,
handing its state across the boundary between them (assemblies are
NOT all coaxial: an oa-TOF pushes ions 90 deg to the beam,
so the honest model is staged field regions with an ion handoff, not one
rotated grid).

WHY THIS AND NOT A COAXIAL FLATTEN
    io/assembly.py folds a COAXIAL stack into one grid and solves once —
    correct and efficient WHEN the units share an axis. It cannot express
    (nor should it fake) a region the beam enters at an angle: the oa
    pusher extracts orthogonally, the reflectron sits at the drift angle.
    Those are DISTINCT field regions the ion crosses between. This module
    flies that: region 1 to its exit plane, transform the ion's state into
    region 2's frame, continue — each region solved in its OWN grid at its
    OWN resolution, no giant rotated domain.

THE HANDOFF, AND THE DOCTRINE IT INHERITS
    A Region owns: a solved `fields` pack (tracer3d.fly3d input), a `pose`
    (translation + rotation of the region's local frame in the world/lab
    frame), and an `exit` surface (a plane in local coords) where the ion
    leaves for the next region. Flight in a region runs in LOCAL coords
    (the fields are local); the ion's exit state is mapped local->world by
    the pose, then world->local of the next region.

    SEAM DOCTRINE (inherited from the oaTOF bench seam work, gate M-2):
    a handoff belongs where the field is DEAD. The ion must cross the seam
    field-free, or the two regions' fields would both act at the boundary
    and double-count. We do NOT mesh/blend/resample fields between regions
    (seam doctrine: fields are never blended between field-array instances). Instead
    each region declares its field_extent; outside it the ion is field-free
    drift, and the seam must lie in that dead zone. check_seam() MEASURES
    |E| at the handoff from the actual solves and refuses a live seam
    rather than silently double-counting — the same refuse-with-diagnostic
    the bench uses.

NON-GOALS
    No field blending. No auto-solving (regions arrive solved). No implicit
    coaxial assumption — a region at any pose is expressible. Collisions
    within a region use that region's gas; the drift between regions is
    vacuum unless a region owns it.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ion_gym.physics.tracer3d import fly3d
from ion_gym.physics.collision3d import (E_CHG, KG_AMU,
                                        gas_kernel_scalars)
# Imported, not redefined. Three definitions of this constant already
# exist in the tree (stats, sim_build, and the oa_mrt study); a fourth
# would be a fourth place for it to drift, and a resolution computed with
# a stale conversion is wrong in a way no test would name.
from ion_gym.physics.stats import FWHM_PER_SIGMA


def _rot_matrix(rot_deg):
    """Intrinsic X->Y->Z rotation (deg) -> 3x3. Identity for None/zeros.
    Small, explicit; no scipy dependency."""
    if rot_deg is None:
        return np.eye(3)
    rx, ry, rz = (np.radians(a) for a in rot_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@dataclass
class Pose:
    """Rigid placement of a region's LOCAL frame in the world frame:
    world = R @ local + t. offset_mm is t; rot_deg is the intrinsic
    X-Y-Z Euler rotation. A coaxial unit has rot_deg=None (identity) —
    so this generalizes the old (axial, radial) offset without breaking
    it: a pure translation is just Pose(offset_mm=[dx,dy,dz])."""
    offset_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rot_deg: Optional[List[float]] = None

    def R(self):
        return _rot_matrix(self.rot_deg)

    def local_to_world(self, p_local, v_local):
        R = self.R()
        t = np.asarray(self.offset_mm, float)
        return R @ np.asarray(p_local, float) + t, R @ np.asarray(v_local,
                                                                  float)

    def world_to_local(self, p_world, v_world):
        R = self.R()
        t = np.asarray(self.offset_mm, float)
        return R.T @ (np.asarray(p_world, float) - t), R.T @ np.asarray(
            v_world, float)


@dataclass
class ExitPlane:
    """The surface where the ion leaves a region for the next, in the
    region's LOCAL coords. axis in {'x','y','z'}, at `value` mm; `sign`
    is the crossing direction (+1: exit when coord exceeds value). The
    region's own impact planes still terminate the ion on metal/aperture;
    this is the HANDOFF surface (seam), which must sit in the field-dead
    zone (checked)."""
    axis: str = "x"
    value_mm: float = 0.0
    sign: int = +1


@dataclass
class Region:
    """One solved field region in a staged instrument.

    name      : label (prefixes stats, like the coaxial flatten does).
    fields    : a solved tracer3d.fly3d fields pack (EAx.. + channels/ele).
    pose      : placement of this region's local frame in the world frame.
    exit      : the handoff surface (local coords) to the NEXT region;
                None for the final region (ion terminates on its own
                impact/timeout).
    gas       : collision spec for flight INSIDE this region (None=vacuum).
    field_extent_mm : optional [x0,x1,y0,y1,z0,z1] local crop; outside it
                the region's field is not applied (the seam must lie here).
    dt_ns, t_max_us : per-region integration controls.
    """
    name: str
    fields: dict
    pose: Pose = field(default_factory=Pose)
    exit: Optional[ExitPlane] = None
    gas: Optional[object] = None
    field_extent_mm: Optional[List[float]] = None
    dt_ns: float = 1.0
    t_max_us: float = 50.0
    stations: Optional[List[object]] = None   # the stage's own detect/
    #   monitor planes. Detection belongs to the DECK, not to analysis
    #   kwargs: a region that does not declare a detector does not have
    #   one, and staged flight must not invent it.
    rec_every: int = 1
    max_records: int = 100000
    metal_z_half_mm: Optional[float] = None   # half of the
    #   stage's DECLARED metal depth. Where declared, metal is finite in
    #   z, so an ion beyond it has LEFT THE HARDWARE and must terminate
    #   rather than fly on through plates that do not exist there. None /
    #   0 = undeclared, and then nothing is imposed: the planar model's
    #   z-invariance is the honest default when no claim was made.
    bounds: Optional[object] = None   # the stage's OWN declared
    #   BoundsSpec. The single-stage path has always honoured these
    #   (`make_planar_fly_fn` feeds `spec.bounds.as_tuple()` straight to
    #   the kernel); staged flight ignored them and built bounds from the
    #   exit plane alone. That parity gap is why an ion drifted to
    #   z = 324 mm through an analyzer whose deck DECLARES
    #   z_max = 226.85: the terminating bound existed, was written down,
    #   and was never read.


def _local_h(fields):
    return float(fields["h_mm"] if isinstance(fields, dict)
                 else fields.h_mm)


def _mirror_off(fields):
    """Canonical-frame offset carried by a solved fields pack (see
    build_stl3d: field mm + offset = canonical mm, mirror plane at 0 on
    mirrored axes). Zero vector when absent (non-mirrored packs and any
    pack built before the convention) — behaviour unchanged for those."""
    mo = fields.get("mirror_off_mm")
    return (np.zeros(3) if mo is None else np.asarray(mo, float))


def check_seam(region_a: Region, region_b: Region, mz_Da,
               samples=64, tol_v_per_mm=1.0):
    """Measure the WORST-CASE |E| at region_a.exit from a's ACTUAL solved
    field — DC plus EVERY drive channel at its amplitude bound, worst
    phase — and report a live seam as a QUANTIFIED DEFECT. Returns
    (ok, |E|worst_max, report); the report states DC and worst-case
    separately. Does NOT blend anything — it only measures.

    This measurement supersedes two prior defects in one move: the
    r-z branch refused ANY RF-on region categorically (over-conservative
    — RF decays spatially, and the funnel terminator ring measurably
    kills it in the bore), while the planar and 3-D branches sampled DC
    ONLY (under-conservative — a seam with 11.5 V/mm of RF amplitude
    and 0.5 of DC read as dead). Both are wrong in the same way: neither
    measured the quantity the seam doctrine cares about, which is the
    field an ion can actually see crossing the plane.

    THE BOUND: |E(t)| = |A + sum_k w_k(t) B_k| <= sqrt(sum_c (|A_c| +
    sum_k W_k |B_k,c|)^2) pointwise, with W_k the channel's amplitude
    bound (|amp|+|off| for sin/cos/square at |base|<=1; |amp|*max|tab|
    + |off| for tables; rf_V for the r-z single drive; gate channels at
    full). Rigorous for every phase, waveform and duty — an ion can
    never see more. A pass is therefore a real pass; a fail states the
    ceiling.
    """
    if region_a.exit is None:
        raise ValueError(f"region {region_a.name!r} has no exit plane to "
                         f"check a seam at")
    f = region_a.fields
    route = f.get("route")
    h = _local_h(f)
    if route == "planar":
        if region_a.exit.axis == "z":
            raise ValueError(
                f"seam for planar region {region_a.name!r} is on z, which "
                f"a planar solve does not model as a field axis. A seam "
                f"there is trivially 'dead' and the check would be "
                f"meaningless rather than reassuring.")
        ExA, EyA = np.asarray(f["ExA"]), np.asarray(f["EyA"])
        ax = {"x": 0, "y": 1}[region_a.exit.axis]
        anchor = f.get("anchor_mm", (0.0, 0.0))
        k_plane = int(round((region_a.exit.value_mm - float(anchor[ax])) / h))
        k_plane = max(0, min(ExA.shape[ax] - 1, k_plane))
        sl = [slice(None)] * 2
        sl[ax] = k_plane
        sl = tuple(sl)
        # UNITS: planar pack stores V/m (see the field cache manifest);
        # tol is V/mm. Scale once at the end.
        bx, by = np.abs(ExA[sl]), np.abs(EyA[sl])
        dc_mag = np.sqrt(ExA[sl] ** 2 + EyA[sl] ** 2) * 1e-3
        # planar channels: UNIT waveforms (build_planar._wave_eval:
        # |base| <= 1 for sin/cos/square; tables carry volts) over
        # amplitude-baked ExK/EyK bases.
        ExK = np.asarray(f.get("ExK", np.zeros((0,) + ExA.shape)))
        EyK = np.asarray(f.get("EyK", np.zeros((0,) + EyA.shape)))
        kinds = np.asarray(f.get("ch_kind", np.zeros(0, np.int64)))
        tab_v = np.asarray(f.get("tab_v", np.zeros(0)))
        tab_off = np.asarray(f.get("tab_off", np.zeros(1, np.int64)))
        for k in range(ExK.shape[0]):
            if int(kinds[k]) in (3, 4):
                o0, o1 = int(tab_off[k]), int(tab_off[k + 1])
                W = float(np.max(np.abs(tab_v[o0:o1]))) if o1 > o0 else 0.0
            else:
                W = 1.0
            bx = bx + W * np.abs(ExK[k][sl])
            by = by + W * np.abs(EyK[k][sl])
        mag = np.sqrt(bx ** 2 + by ** 2) * 1e-3
    elif route == "rz":
        # An exit on the axial coordinate indexes axis 0; a transverse
        # exit is a RADIUS, not a signed y, so it is refused rather than
        # silently indexed as one.
        if region_a.exit.axis != "x":
            raise ValueError(
                f"seam for r-z region {region_a.name!r} is on "
                f"{region_a.exit.axis!r}, but an r-z field is solved on "
                f"(axial, radial): the transverse coordinate is a radius, "
                f"not a signed axis. Put the handoff on the axial "
                f"coordinate (x).")
        EzA, EuA = np.asarray(f["EzA"]), np.asarray(f["EuA"])
        k_plane = int(round(region_a.exit.value_mm / h))
        k_plane = max(0, min(EzA.shape[0] - 1, k_plane))
        dc_mag = np.sqrt(EzA[k_plane] ** 2 + EuA[k_plane] ** 2) * 1e-3
        # single sinusoidal drive: s(t) = sin(wt) * rf_V on the per-volt
        # B basis (tracer_rz), plus the gate channel at g in {0,1} —
        # included at full whenever the gate can arm (tau_gate >= 0).
        rf = float(f.get("rf_V", 0.0))
        bz = np.abs(EzA[k_plane]) + abs(rf) * np.abs(f["EzB"][k_plane])
        bu = np.abs(EuA[k_plane]) + abs(rf) * np.abs(f["EuB"][k_plane])
        if float(f.get("tau_gate", -1.0)) >= 0.0:
            bz = bz + np.abs(f["EzG"][k_plane])
            bu = bu + np.abs(f["EuG"][k_plane])
        mag = np.sqrt(bz ** 2 + bu ** 2) * 1e-3
    elif route == "3d":
        # UNITS: the 3-D pack stores V/mm natively (tracer3d.fly_one) —
        # no scaling, and adding one to "match" the others would
        # understate every 3-D seam by 1000x.
        EAx, EAy, EAz = (np.asarray(f[k]) for k in ("EAx", "EAy", "EAz"))
        ax = {"x": 0, "y": 1, "z": 2}[region_a.exit.axis]
        _off = _mirror_off(f)[ax]
        k_plane = int(round((region_a.exit.value_mm - _off) / h))
        k_plane = max(0, min(EAx.shape[ax] - 1, k_plane))
        sl = [slice(None)] * 3
        sl[ax] = k_plane
        sl = tuple(sl)
        dc_mag = np.sqrt(EAx[sl] ** 2 + EAy[sl] ** 2 + EAz[sl] ** 2)
        bx, by, bz = np.abs(EAx[sl]), np.abs(EAy[sl]), np.abs(EAz[sl])
        # drive channels via the SAME normalization the tracer uses
        # (legacy single-sin packs expand to one channel): w(t) =
        # amp*base(t) + off, |base| <= 1 except tables (volts).
        from ion_gym.physics.tracer3d import _resolve_channels
        ch = _resolve_channels(f)
        ExK = np.asarray(ch.get("ExK", np.zeros((0,) + EAx.shape)))
        if ExK.shape[0]:
            EyK = np.asarray(ch["EyK"])
            EzK = np.asarray(ch["EzK"])
            kinds = np.asarray(ch["ch_kind"])
            amp = np.asarray(ch.get("ch_amp", np.ones(ExK.shape[0])))
            off = np.asarray(ch.get("ch_off", np.zeros(ExK.shape[0])))
            tab_v = np.asarray(ch.get("tab_v", np.zeros(0)))
            tab_off = np.asarray(ch.get("tab_off", np.zeros(1, np.int64)))
            for k in range(ExK.shape[0]):
                if int(kinds[k]) in (3, 4):
                    o0, o1 = int(tab_off[k]), int(tab_off[k + 1])
                    base = (float(np.max(np.abs(tab_v[o0:o1])))
                            if o1 > o0 else 0.0)
                else:
                    base = 1.0
                W = abs(float(amp[k])) * base + abs(float(off[k]))
                bx = bx + W * np.abs(ExK[k][sl])
                by = by + W * np.abs(EyK[k][sl])
                bz = bz + W * np.abs(EzK[k][sl])
        mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    else:
        # NO SILENT CATCH-ALL (see the untagged-pack history below the
        # route dispatch): an unknown route is refused by name.
        raise ValueError(
            f"region {region_a.name!r}: fly_fields declares route "
            f"{route!r}, which the seam gate has no field convention for. "
            f"Known: 'planar' and 'rz' (DC in V/m, transverse span "
            f"sampled) and '3d' (V/mm). A pack with no route tag is "
            f"hand-built rather than builder-produced and must declare "
            f"one. Refusing rather than guessing which channels to read.")
    emax = float(np.nanmax(mag)) if mag.size else 0.0
    dcmax = float(np.nanmax(dc_mag)) if np.size(dc_mag) else 0.0
    ok = emax <= tol_v_per_mm
    report = (f"seam {region_a.name!r}->{region_b.name!r} at "
              f"{region_a.exit.axis}={region_a.exit.value_mm:g}mm: "
              f"worst-case |E|max={emax:.3g} V/mm (DC {dcmax:.3g}) "
              f"({'DEAD, ok' if ok else f'LIVE > {tol_v_per_mm} tol'})")
    return ok, emax, report


def _fly_region(reg, mz_Da, *, r0_mm, v0_mm_us, tob_us, seed, planes):
    """Fly one region with the kernel its solve route calls for.

    Dispatch is on the pack's declared `route`, never on which keys are
    present. A missing key cannot distinguish "this is a different route"
    from "this solve is incomplete", and flying a region with the wrong
    kernel produces a trajectory that looks entirely plausible -- the
    failure would surface as a resolution that is merely disappointing,
    which is the hardest kind of defect to attribute.

    An unrecognised route REFUSES. There is no default kernel, because
    guessing here is exactly the mistake above.
    """
    route = reg.fields.get("route")
    if route == "3d":
        res = fly3d(reg.fields, mz_Da, r0_mm=r0_mm, v0_mm_us=v0_mm_us,
                    tob_us=tob_us, dt_ns=reg.dt_ns, t_max_us=reg.t_max_us,
                    collisions=reg.gas, seed=seed, planes=planes)
        # DECLARED HERE, not inferred from a missing key (this
        # function's own docstring says why). The 3-D route detects exit
        # planes by interpolated crossing and arms NO kernel bounds, so
        # a kind = 3 from it is an exit-plane crossing by construction.
        # -2 is that statement. NOTE the residual: declared BoundsSpec
        # is NOT ENFORCED on a 3-D staged region -- fly3d has no bounds
        # channel at all. That is a KNOWN remaining gap, recorded
        # rather than hidden behind this constant.
        res["bnd_face"] = -2
        return res
    if route == "planar":
        return _fly_region_planar(reg, mz_Da, r0_mm=r0_mm,
                                  v0_mm_us=v0_mm_us, tob_us=tob_us,
                                  seed=seed, planes=planes)
    if route == "rz":
        return _fly_region_rz(reg, mz_Da, r0_mm=r0_mm,
                              v0_mm_us=v0_mm_us, tob_us=tob_us,
                              seed=seed, planes=planes)
    raise ValueError(
        f"region {reg.name!r}: fields pack declares route {route!r}. "
        f"Staged flight dispatches on this tag and knows 'planar', 'rz' "
        f"and '3d'. A pack with no route tag came from a builder that "
        f"does not yet publish one (tw2d is the known case) "
        f"-- that builder must publish a tagged pack "
        f"rather than this call guessing a kernel.")


def _fly_region_planar(reg, mz_Da, *, r0_mm, v0_mm_us, tob_us, seed, planes):
    """Planar region flight, returned in the same shape fly3d emits.

    A planar solve is z-INVARIANT by construction, so the kernel carries
    z as an exact field-free drift (z = z0 + vz*t) rather than as a solved
    axis. That is not an approximation of a 3-D solve -- it is the
    closed-form statement of the same symmetry, and it is why a planar
    stage must NOT be extruded into a 3-D grid to join an assembly.

    Consequence worth stating: an exit plane on the z axis is a drift
    landmark, not a field boundary, and a planar region cannot terminate
    an ion on geometry in z because it has none.
    """
    from ion_gym.physics.build_planar import _fly_planar
    import numpy as _np

    f = reg.fields
    if planes:
        for ax, _val, _sgn in planes:
            if ax == "z":
                raise ValueError(
                    f"region {reg.name!r} is planar and its exit plane is "
                    f"on z, but a planar solve has no geometry in z -- the "
                    f"axis is a field-free drift. Put the handoff on x or "
                    f"y, or solve this stage on a route that models z.")
    anchor = f.get("anchor_mm", (0.0, 0.0))
    x0 = float(r0_mm[0]) - float(anchor[0])
    y0 = float(r0_mm[1]) - float(anchor[1])
    bnd_on, bnd_val, seam_faces = _planar_bounds(planes, anchor)
    src = [SRC_SEAM if k in seam_faces else SRC_NONE for k in range(6)]
    # The DECLARED metal envelope is a terminating aperture
    # in z, armed on every region. The seam can never own faces 4/5 on a
    # planar region (a z exit plane is refused upstream), so this arming
    # cannot collide with a seam face.
    _mz_half = getattr(reg, "metal_z_half_mm", None)
    if _mz_half and float(_mz_half) > 0:
        _h = float(_mz_half)
        for _k, _v in ((4, -_h), (5, _h)):
            if not bnd_on[_k]:
                bnd_on[_k] = True
                bnd_val[_k] = _v
                src[_k] = SRC_APERTURE
    # DECLARED TERMINATING BOUNDS, armed ON
    # EVERY REGION: the kernel reports the
    # terminating FACE, and fly_staged discriminates seam from bounds by
    # face identity, so the interim no-exit-plane restriction (which
    # existed only because a bare kind = 3 could not be attributed) is
    # lifted. Precedence and refusals live in _merge_declared_bounds.
    _off = (float(anchor[0]), float(anchor[1]), 0.0)   # z unanchored
    _merge_declared_bounds(reg, bnd_on, bnd_val, src, _off)
    # The kernel is numba-jitted: argument TYPES are part of its contract,
    # not just their values. nch_flags is unpacked into 13 channel
    # booleans, the bounds are typed arrays, and the SDS statistics table
    # is indexed even when SDS is off -- so none of these can be a scalar
    # placeholder or None. Marshalled exactly as make_planar_fly_fn does.
    # Gas comes from the REGION, uniformly with every other route --
    # not hardcoded here. This previously passed collisions OFF as eight
    # literal zeros, which silently flew a gas-filled planar stage in
    # vacuum: it completes, it transmits, and it reports a resolution
    # that is simply wrong.
    g = gas_kernel_scalars(reg.gas)
    nch = tuple([False] * 13)
    bnd_on = _np.array(bnd_on, _np.bool_)
    bnd_val = _np.array(bnd_val, _np.float64)
    # LENGTH IS A CONTRACT, NOT A DETAIL. `_fly_planar` indexes
    # bnd_on[5]/bnd_val[5] on every step with numba boundscheck OFF, so a
    # short array reads uninitialized heap and terminates ions at random
    # with a plausible fate and an implausible time. Checked here, at the
    # boundary between Python and the kernel, because it is the last point
    # where a wrong length is still a catchable error rather than a silent
    # wrong answer. Cheap: once per ion per region, against a 157 us
    # integration.
    if bnd_on.shape != (6,) or bnd_val.shape != (6,):
        raise ValueError(
            f"region {reg.name!r}: the planar kernel's bounds contract is "
            f"SIX flags and SIX values [x_min, x_max, y_min, y_max, "
            f"z_min, z_max], but this call built bnd_on{bnd_on.shape} and "
            f"bnd_val{bnd_val.shape}. The kernel indexes all six "
            f"unconditionally and does not bounds-check, so a short array "
            f"is read past its end. Fix the builder, not this check.")
    sds_stats = _np.zeros((5, 1002))
    ncol = 7
    max_records = int(reg.max_records)
    rec = _np.empty((max_records, ncol))
    acc = E_CHG / (mz_Da * KG_AMU)
    n, kind, _nc, bface = _fly_planar(
        x0, y0, float(v0_mm_us[0]), float(v0_mm_us[1]), float(tob_us),
        float(mz_Da), f["ExA"], f["EyA"], f["ExK"], f["EyK"],
        f["ch_kind"], f["ch_om"], f["ch_ph"],
        f["tab_t"], f["tab_v"], f["tab_off"],
        f["ele"].astype(_np.float64), float(f["h_mm"]), acc,
        float(reg.dt_ns) * 1e-3, float(reg.t_max_us),
        g["enabled"], g["T_k"], g["P_pa"], g["sigma_m2"],
        g["c_star"], g["c_bar"], g["sig1d"], g["m_gas"],
        rec, int(reg.rec_every), nch, bnd_on, bnd_val, int(seed),
        float(r0_mm[2]), float(v0_mm_us[2]),
        False, 0.0, 0.0, 0.0, 0.0, sds_stats)
    if n >= max_records:
        raise RuntimeError(
            f"region {reg.name!r}: the trajectory buffer filled "
            f"({max_records} records at rec_every={reg.rec_every}) before "
            f"the flight ended. The trace is TRUNCATED, so any station "
            f"crossing after that point is invisible and the ion would be "
            f"reported as never detected. Raise max_records or rec_every.")
    tr = rec[:n]
    _bval_user = (float(bnd_val[bface]) + _off[bface // 2]
                  if bface >= 0 else None)
    return dict(x=tr[:, 1] + float(anchor[0]), y=tr[:, 2] + float(anchor[1]),
                z=tr[:, 3], t_us=tr[:, 0], kind=kind,
                vx=tr[:, 4], vy=tr[:, 5], vz=tr[:, 6],
                tof_us=float(tr[-1, 0]),
                v_mm_us=(float(tr[-1, 4]), float(tr[-1, 5]),
                         float(tr[-1, 6])),
                bnd_face=int(bface), bnd_src=tuple(src),
                seam_faces=frozenset(seam_faces), bnd_user_val=_bval_user)


def _planar_bounds(planes, anchor):
    """Translate an exit plane into the planar kernel's bounds contract.

    SIX flags and SIX values, [x_min, x_max, y_min, y_max, z_min, z_max].
    This length is not a convention, it is the kernel's contract:
    `_fly_planar` indexes bnd_on[4], bnd_on[5], bnd_val[4] and bnd_val[5]
    UNCONDITIONALLY on every integration step. numba compiles with
    boundscheck off, so a shorter array is not an error -- it is a read of
    whatever heap follows the allocation, re-decided on every run.

    That was the defect here. This function returned
    FOUR, so every staged planar flight tested two garbage bounds per step.
    When the bytes past the end happened to be truthy and the adjacent
    float compared true, the ion terminated with kind=3 -- "crossed the
    exit plane" -- on its FIRST step, with a time of flight of one dt.
    That is exactly the banked sighting: 200/200 credited as arrived at
    T = 0.0011 us, R = 3. Because it is uninitialized memory, it was
    intermittent, it clustered, and the FIELDS were always bit-identical
    between a good run and a bad one -- which is why every hypothesis
    aimed at the solve or the basis cache came up empty.

    It held for ALL configurations, not just this deck: the read is
    unconditional, so any assembly with a planar stage was exposed. The
    single-stage path was never affected -- `make_planar_fly_fn` feeds
    `spec.bounds.as_tuple()`, which is six -- and the r-z staged path
    builds six directly. This was the one caller that did not.

    z is sized but never armed here: `_fly_region_planar` refuses a z exit
    plane upstream, because a planar solve has no geometry in z. The slots
    exist so the kernel reads defined memory, not so z is silently
    supported.

    An exit is a ONE-sided crossing, so only the matching side is armed;
    arming both would stop the ion at whichever it reached first and
    silently report the wrong seam.
    """
    bnd_on = [False] * 6
    bnd_val = [0.0] * 6
    seam_faces = set()
    if not planes:
        return bnd_on, bnd_val, seam_faces
    off = {"x": float(anchor[0]), "y": float(anchor[1])}
    idx = {("x", +1): 1, ("x", -1): 0, ("y", +1): 3, ("y", -1): 2}
    for ax, val, sgn in planes:
        k = idx.get((ax, int(sgn)))
        if k is None:
            raise ValueError(
                f"planar exit plane ({ax!r}, sign {sgn}) is not one of "
                f"x/y with sign +1 or -1.")
        bnd_on[k] = True
        bnd_val[k] = float(val) - off[ax]
        seam_faces.add(k)
    return bnd_on, bnd_val, seam_faces


FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

# bound-source labels, per face: who armed this face. Feeds the
# seam-vs-termination discrimination in fly_staged and the fate text.
SRC_NONE, SRC_SEAM, SRC_APERTURE, SRC_DECLARED = 0, 1, 2, 3
_SRC_TEXT = {SRC_APERTURE: "declared metal envelope",
             SRC_DECLARED: "declared bounds"}


def _merge_declared_bounds(reg, bnd_on, bnd_val, src, off):
    """Merge the stage's declared BoundsSpec into the kernel bounds, ON
    EVERY REGION (superseding an interim
    no-exit-plane restriction). Safe because the
    kernel reports WHICH face terminated the flight, so fly_staged
    discriminates a seam arrival from a bounds termination by face
    identity rather than being unable to tell them apart.

    Precedence, per face:
      * unarmed             -> arm the declared value (SRC_DECLARED).
      * armed by the seam   -> the seam keeps the face. A declared bound
        AT OR BEYOND the seam value is provably unreachable inside this
        region (the ion hands off at the seam first), so skipping it is
        semantically exact, and stated here rather than silent. A
        declared bound STRICTLY TIGHTER than the seam would terminate
        every ion before it could ever reach the seam -- a dead handoff
        is a mis-declared deck, and it REFUSES by name rather than
        flying an assembly whose seam is unreachable.
      * armed by the aperture -> both are terminations; the TIGHTER
        value is armed and the source label follows it, so the fate
        text names the surface the ion actually left through.

    `off` is the kernel-frame offset per axis (planar anchors x/y; z is
    never anchored), matching the interim fix's convention exactly.
    """
    b = getattr(reg, "bounds", None)
    if b is None:
        return
    bf, bv = b.as_tuple()
    for k in range(6):
        if not bf[k]:
            continue
        v = float(bv[k]) - off[k // 2]
        if not bnd_on[k]:
            bnd_on[k] = True
            bnd_val[k] = v
            src[k] = SRC_DECLARED
            continue
        if src[k] == SRC_SEAM:
            lo_side = (k % 2 == 0)
            tighter = (v > bnd_val[k]) if lo_side else (v < bnd_val[k])
            if tighter:
                raise ValueError(
                    f"region {reg.name!r}: declared bound "
                    f"{FACE_NAMES[k]} = {float(bv[k]):g} mm sits INSIDE "
                    f"the exit plane on the same face (seam at "
                    f"{bnd_val[k] + off[k // 2]:g} mm) -- every ion "
                    f"would terminate on the bound before reaching the "
                    f"seam, so the handoff is dead. Move the bound past "
                    f"the seam or move the seam.")
            # at-or-beyond the seam: unreachable within this region --
            # the seam statement is strictly stronger. Exact, not lossy.
            continue
        if src[k] == SRC_APERTURE:
            lo_side = (k % 2 == 0)
            if (v > bnd_val[k]) if lo_side else (v < bnd_val[k]):
                bnd_val[k] = v
                src[k] = SRC_DECLARED
            continue
        raise RuntimeError(
            f"region {reg.name!r}: face {FACE_NAMES[k]} armed by "
            f"unknown source {src[k]} -- the arming code and this "
            f"merge have diverged.")


def _fly_region_rz(reg, mz_Da, *, r0_mm, v0_mm_us, tob_us, seed, planes):
    """r-z region flight, returned in the shape fly3d emits.

    The r-z kernel carries FULL 3-D Cartesian ion state against a field
    solved on the (axial, radial) half-plane, so an ion in an r-z stage
    already has a real (x, y, z) trajectory and hands off to the next
    stage without any reconstruction. That is why an r-z stage can sit
    upstream of a planar one at all.

    Bounds are the kernel's own 6-axis contract here (x/y/z min and max),
    not the planar 4-tuple, so an exit plane on ANY axis is expressible
    -- including z, which a planar region cannot honour.
    """
    from ion_gym.physics.build_rz import _fly_rec_full
    import numpy as _np

    f = reg.fields
    bnd_on = _np.zeros(6, _np.bool_)
    bnd_val = _np.zeros(6, _np.float64)
    seam_faces = set()
    idx = {("x", -1): 0, ("x", +1): 1, ("y", -1): 2, ("y", +1): 3,
           ("z", -1): 4, ("z", +1): 5}
    for ax, val, sgn in (planes or []):
        k = idx.get((ax, int(sgn)))
        if k is None:
            raise ValueError(
                f"r-z exit plane ({ax!r}, sign {sgn}) is not x/y/z with "
                f"sign +1 or -1.")
        bnd_on[k] = True
        bnd_val[k] = float(val)
        seam_faces.add(k)
    # DECLARED TERMINATING BOUNDS: the r-z staged path never
    # merged the stage's own BoundsSpec at all -- the same parity gap
    # the planar path had, in its r-z form. Armed on every region, with
    # the same face-identity discrimination downstream. No anchor: the
    # r-z kernel flies in the user frame on all three axes. No aperture
    # source either -- an r-z stage has REAL geometry in z, so its metal
    # terminates ions by impact, not by a declared envelope.
    src = [SRC_SEAM if k in seam_faces else SRC_NONE for k in range(6)]
    _merge_declared_bounds(reg, bnd_on, bnd_val, src, (0.0, 0.0, 0.0))

    q_e = int(f.get("charge", 1))
    if q_e == 0:
        raise ValueError(
            f"region {reg.name!r}: the stage spec declares charge 0; an "
            f"uncharged ion has no electric acceleration.")
    acc = q_e * E_CHG / (mz_Da * KG_AMU)
    # Same single derivation the planar and 3-D routes use.
    g = gas_kernel_scalars(reg.gas)
    dt = float(reg.dt_ns) * 1e-3
    ncol = 7
    max_records = int(reg.max_records)
    rec = _np.empty((max_records, ncol))
    # 13 channel flags, all off: the assembly reads position and time off
    # the base columns. Built as a plain 13-tuple to match the kernel's
    # own flag arity -- the r-z registry is LOCAL to build_rz by design
    # (the planar route's global-registry flag bug is the cautionary
    # tale), so this must not be derived from OPTIONAL_CHANNELS.
    f13 = tuple([False] * 13)
    n, kind, _nc, bface = _fly_rec_full(
        float(r0_mm[0]), float(r0_mm[1]), float(r0_mm[2]),
        float(v0_mm_us[0]), float(v0_mm_us[1]), float(v0_mm_us[2]),
        float(tob_us), float(mz_Da),
        f["EzA"], f["EuA"], f["EzB"], f["EuB"], f["EzG"], f["EuG"],
        f["tau_gate"], f["ele"], f["u0"], f["h_mm"], acc,
        f["rf_V"], f["om_rad_us"], dt, float(reg.t_max_us),
        g["T_k"], g["P_pa"], g["sigma_m2"], g["c_star"], g["c_bar"],
        g["sig1d"], g["m_gas"], rec, int(reg.rec_every), int(seed),
        *f13, bnd_on, bnd_val)
    if n >= max_records:
        raise RuntimeError(
            f"region {reg.name!r}: the trajectory buffer filled "
            f"({max_records} records at rec_every={reg.rec_every}) before "
            f"the flight ended. The trace is TRUNCATED, so any station "
            f"crossing after that point is invisible.")
    tr = rec[:n]
    _bval_user = float(bnd_val[bface]) if bface >= 0 else None
    return dict(x=tr[:, 1], y=tr[:, 2], z=tr[:, 3], t_us=tr[:, 0],
                vx=tr[:, 4], vy=tr[:, 5], vz=tr[:, 6], kind=kind,
                tof_us=float(tr[-1, 0]),
                v_mm_us=(float(tr[-1, 4]), float(tr[-1, 5]),
                         float(tr[-1, 6])),
                bnd_face=int(bface), bnd_src=tuple(src),
                seam_faces=frozenset(seam_faces), bnd_user_val=_bval_user)


def _detect_hit(reg, res):
    """First IN-WINDOW crossing of this region's detect station, or None.

    Reuses physics.stations.station_hits rather than reimplementing plane
    crossing: that function already interpolates the crossing state
    between records, counts the crossing ordinal k (which is the fold
    order for a multi-reflecting analyzer), and applies the station's
    window. A second implementation here would be a second place for the
    interpolation convention to drift, and the arrival TIME is the whole
    measurement.

    Detect stations are NON-DESTRUCTIVE by design: the ion flies
    on through the plane, and the arrival is the FIRST in-window crossing
    read back off the completed trace. So this reports, it does not
    terminate.
    """
    import numpy as _np
    from ion_gym.physics.stations import station_hits

    dets = [s for s in (reg.stations or [])
            if getattr(s, "kind", None) == "detect"]
    if not dets:
        return None
    if len(dets) > 1:
        raise ValueError(
            f"region {reg.name!r} declares {len(dets)} detect stations "
            f"({[d.name for d in dets]}); staged flight cannot choose "
            f"which one defines arrival. Declare one, or name it in the "
            f"stage entry.")
    traj = _np.column_stack([res["t_us"], res["x"], res["y"], res["z"],
                             res["vx"], res["vy"], res["vz"]])
    cols = ["t", "x", "y", "z", "vx", "vy", "vz"]
    for h in station_hits(traj, cols, dets[0]):
        if h["in_window"]:
            return h
    return None

def fly_staged(regions: List[Region], mz_Da, p0_world, v0_world_mm_us,
               tob_us=0.0, seed=1, check_seams=True, seam_tol_v_per_mm=1.0):
    """Fly ONE ion through the regions in order, handing its state across
    each seam. Positions/velocities in WORLD mm / mm-per-us. Returns a
    dict: per-region traces (world coords), the handoff states, cumulative
    tof, the final fate, and any seam advisories (a live seam is a loudly
    stated, never-blocking, quantified defect).

    Contract: regions arrive SOLVED. The ion enters region i in that
    region's local frame (world->local via its pose), flies to its exit
    plane (or terminates), and its exit state maps back to world for the
    next region. Between regions the ion is on the seam (field-dead), so
    no drift model is imposed here — adjacent poses place the next entry
    face at the same world point as the exit.
    """
    if not regions:
        raise ValueError("fly_staged: no regions")
    # seam preflight: MEASURE every non-final region's seam. A live seam
    # is a QUANTIFIED DEFECT, stated LOUDLY and never blocking (this
    # supersedes an earlier hard refusal): the flight proceeds,
    # and the user is told exactly how much field an ion can see at the
    # crossing while the upstream region's contribution is dropped there
    # — that magnitude bounds the trajectory error introduced at the
    # seam. Advisory contract: magnitude-stating, never-blocking, once
    # per flight (fly_packet runs the preflight on ion 0 only).
    seam_advisories = []
    if check_seams:
        for a, b in zip(regions[:-1], regions[1:]):
            if a.exit is not None:
                ok, emax, rep = check_seam(a, b, mz_Da,
                                           tol_v_per_mm=seam_tol_v_per_mm)
                if not ok:
                    msg = (f"[SEAM ADVISORY] {rep} — the handoff is NOT "
                           f"field-dead: an ion crossing at worst phase "
                           f"can see up to {emax:.3g} V/mm while "
                           f"{a.name!r}'s field is dropped at the plane, "
                           f"so trajectories near this seam carry a "
                           f"defect bounded by that magnitude. Stated, "
                           f"not refused (PI ruling 2026-09-05); reduce "
                           f"it with a DC terminator electrode or a "
                           f"longer field-free gap.")
                    print(msg)
                    seam_advisories.append(msg)

    p_world = np.asarray(p0_world, float)
    v_world = np.asarray(v0_world_mm_us, float)
    tof = float(tob_us)
    out = {"regions": [], "handoffs": [], "mz": mz_Da,
           "seam_advisories": seam_advisories}
    fate = "completed"

    for i, reg in enumerate(regions):
        # world -> this region's local frame
        p_loc, v_loc = reg.pose.world_to_local(p_world, v_world)
        # the region's LOCAL frame is CANONICAL (mirror plane at 0); the
        # tracer flies in the field's array frame. Convert on the way in
        # (positions and plane values; velocities are frame-invariant) and
        # back on the way out. _moff is zero for non-mirrored packs.
        _moff = _mirror_off(reg.fields)
        _axi = {"x": 0, "y": 1, "z": 2}
        planes = None
        if reg.exit is not None:
            # the exit plane is a boundary crossing in local coords: the
            # tracer stops the ion at the plane (kind 1 = left the box),
            # and we read its interpolated state there. Contract is a list
            # of (axis, value_mm, sign) tuples.
            planes = [(reg.exit.axis,
                       reg.exit.value_mm - _moff[_axi[reg.exit.axis]],
                       reg.exit.sign)]
        res = _fly_region(reg, mz_Da,
                          r0_mm=tuple(np.asarray(p_loc, float) - _moff),
                          v0_mm_us=tuple(v_loc),
                          tob_us=tof, seed=seed, planes=planes)
        # map the whole trace back to world for a coherent path
        R = reg.pose.R()
        t = np.asarray(reg.pose.offset_mm, float)
        P = np.stack([res["x"], res["y"], res["z"]], axis=1) + _moff
        Pw = (R @ P.T).T + t                  # canonical local -> world
        out["regions"].append(dict(name=reg.name, x=Pw[:, 0], y=Pw[:, 1],
                                   z=Pw[:, 2], t_us=res["t_us"],
                                   kind=res["kind"]))
        # ARRIVAL. A region that declares a detector defines the ion's
        # arrival by the FIRST in-window crossing of it, not by where the
        # integration happened to stop. Without this the ion runs to
        # t_max and the run reports a timeout with a plausible partial
        # trajectory -- which reads as a transmission failure rather than
        # as a detector that was never consulted.
        _hit = _detect_hit(reg, res)
        if _hit is not None:
            out["detection"] = dict(region=reg.name, t_us=float(_hit["t_us"]),
                                    k=int(_hit["k"]),
                                    x=float(_hit["x"]), y=float(_hit["y"]),
                                    z=float(_hit["z"]),
                                    # station_hits interpolates the FULL
                                    # crossing state; narrowing it to
                                    # position here is what made speed and
                                    # ke_ev underivable downstream.
                                    # Same interpolated
                                    # sample, no second convention.
                                    vx=float(_hit["vx"]),
                                    vy=float(_hit["vy"]),
                                    vz=float(_hit["vz"]))
            out["regions"][-1]["detected_t_us"] = float(_hit["t_us"])
            tof = float(_hit["t_us"])
            fate = "detected"
            break
        tof = float(res["tof_us"])
        # exit state -> world for the next region. res['v_mm_us'] is in
        # mm/us; the pose rotation is unit-agnostic (a rotation), so map
        # in mm/us and keep mm/us in world.
        p_exit_loc = (np.array([res["x"][-1], res["y"][-1], res["z"][-1]])
                      + _moff)                # canonical local frame
        v_exit_loc = np.asarray(res["v_mm_us"])
        p_world, v_world = reg.pose.local_to_world(p_exit_loc, v_exit_loc)
        out["handoffs"].append(dict(after=reg.name, p_world=p_world.copy(),
                                    v_world=v_world.copy(), tof_us=tof,
                                    kind=res["kind"]))
        # kind: 0 impact metal, 1 left box, 2 timeout, 3 crossed the exit
        # plane. Only kind 3 (reached the declared seam) continues the
        # chain; anything else terminates the ion here.
        # DID IT ACTUALLY REACH THE SEAM? The kernel reports every bound
        # crossing as kind = 3, so "kind 3" alone cannot tell a seam
        # arrival from an ion leaving through the declared metal
        # aperture. Handing the latter across the seam would report a
        # transmitted ion that in reality struck hardware -- a wrong
        # answer with a plausible trajectory. Discriminate by FACE
        # IDENTITY: the kernel reports WHICH
        # armed face terminated the flight, and the flyer labels each
        # face's source, so a seam arrival is a crossing of a seam-armed
        # face and everything else is a termination. This supersedes the
        # interim position test (an aperture hit at |z| >= half-depth),
        # which could only discriminate the one source that lived on an
        # axis the seam can never occupy -- face identity discriminates
        # ALL sources on ALL axes with no position tolerance at all.
        # bnd_face -2: the route (3-D) arms no kernel bounds, so its
        # kind = 3 is an exit-plane crossing by construction. bnd_face
        # -1 with kind = 3 is impossible by the kernel's own contract
        # (kind 3 is only ever set together with a face) -- if it
        # arrives, the contract broke, and that is surfaced, not padded
        # over.
        _crossed_exit = res["kind"] == 3
        if _crossed_exit:
            _face = int(res["bnd_face"])
            if _face >= 0:
                _crossed_exit = _face in res["seam_faces"]
            elif _face == -1:
                raise RuntimeError(
                    f"region {reg.name!r}: kernel returned kind = 3 "
                    f"with no terminating face -- the kernel's "
                    f"face-reporting contract (L-165) is broken.")
        if reg.exit is None or not _crossed_exit:
            if reg.exit is None:
                fate = "completed"
            elif res["kind"] == 3:
                _face = int(res["bnd_face"])
                _src = _SRC_TEXT.get(res["bnd_src"][_face],
                                     f"source {res['bnd_src'][_face]}")
                fate = (f"terminated in {reg.name}: crossed "
                        f"{FACE_NAMES[_face]} at "
                        f"{res['bnd_user_val']:.4f} mm ({_src}) "
                        f"without reaching the seam")
            else:
                # WORDS, not kind numbers ("(kind 2)" in
                # the loss tally read as died-on-hardware; kind 2 is a
                # TIMEOUT). The number rides in parentheses for grep.
                # The timeout names the CLOCK THAT EXPIRED (raising
                # "total time" in the UI once moved
                # nothing, because each region flies on ITS OWN stage's
                # t_max — the edit landed on a different spec and the
                # 30 us hexapole clock cut every run at the same z).
                _kind_words = {0: "hit metal", 1: "left the solve box",
                               2: (f"TIMED OUT at THIS STAGE'S OWN "
                                   f"t_max = {reg.t_max_us:g} us — "
                                   f"still in flight when it expired; "
                                   f"raise this stage's integration "
                                   f"t_max_us (or the stage's override "
                                   f"in the instrument JSON), not "
                                   f"another stage's"),
                               3: "crossed a bound"}
                fate = (f"terminated in {reg.name}: "
                        f"{_kind_words.get(int(res['kind']), 'unknown')}"
                        f" (kind {res['kind']})")
            break

    out["fate"] = fate
    out["tof_us"] = tof
    return out


# =====================================================================
# STAGED-ASSEMBLY JSON: load_assembly SOLVES each stage's fields via
# build_run — "JSON doc -> solved Regions" is builder-layer work, and
# this module already owns Region / Pose / ExitPlane / fly_staged
# (it once lived under an io name; no re-exports remain).
# Section doc:
# """Declarative assembly JSON -> staged flight regions.
# 
# The front end for physics/staged_flight: an instrument.json declares the
# stages of a (possibly non-coaxial) instrument -- which spec each stage
# solves, where it sits (pose: translation + rotation), and the seam to the
# next stage -- and load_assembly() builds the solved Regions ready for
# fly_staged.
# 
# WHY A SEPARATE FILE FROM THE UNITS
#     A Region needs a SOLVED field pack, which cannot live in JSON. So the
#     assembly JSON references each stage's OWN spec (by relative path, the
#     same portable-folder convention as STLs: instrument.json sits beside
#     the stage folders), and this loader solves each and assembles them.
#     The instrument is therefore a DIRECTORY:
# 
#         my_instrument/
#           instrument.json          <- this schema
#           cooler/    spec.json (+ stls/)
#           pusher/    spec.json (+ stls/)
#           reflectron/spec.json (+ stls/)
# 
#     Zipping that directory is transport; the structure is the folder +
#     this JSON, human-readable, no opaque archive.
# 
# SCHEMA (instrument.json)
#     {
#       "schema": "ion_gym.assembly/1",
#       "name": "oa-TOF bench",
#       "stages": [
#         {
#           "name": "beamline",
#           "spec": "cooler/spec.json",        # relative to THIS file
#           "pose": {"offset_mm": [0,0,0]},    # rot_deg optional
#           "exit": {"axis":"x","value_mm":37.8,"sign":1},
#           "gas":  "spec",                    # 'spec'|'vacuum' (default spec)
#           "dt_ns": 2.0, "t_max_us": 40.0
#         },
#         {
#           "name": "pusher",
#           "spec": "pusher/spec.json",
#           "pose": {"offset_mm": [37.8,0,0], "rot_deg": [0,0,90]},
#           "exit": null                       # final stage: no seam
#         }
#       ],
#       "beam": {                              # initial ion state (world)
#         "mz": 622.0, "p0_mm": [1,0,0], "v0_mm_us": [5,0,0], "tob_us": 0.0
#       }
#     }
# 
# DOCTRINE
#     Seam checks run by default (fly_staged): a live handoff is loudly
#     quantified, never silently double-counted. Poses generalize the
#     plain coaxial offset (a
#     stage with no rot_deg is a pure translation). Nothing is auto-placed
#     or auto-rotated by inference -- every pose is declared.
# """
# =====================================================================
import json
import math
from pathlib import Path
from typing import Optional

from ion_gym.io.spec_io import load_any_spec
from ion_gym.physics.sim_build import build_run

SCHEMA = "ion_gym.assembly/1"

# TRACE VIEW POLICY. How many ions' paths a
# multi-FA view DRAWS. Named here with documented defaults rather than
# buried as literals, and declarable per instrument (see `trace_policy`).
#
# THIS IS A VIEW QUANTITY AND NOTHING ELSE. Every reported statistic --
# arrivals, losses, T, FWHM, R -- is computed from the WHOLE packet
# regardless of what is drawn. A resolution that quietly depended on how
# many paths were on screen would be the worst kind of defect, so the
# subset is applied only when collecting traces.
#
# Why draw a subset at all, measured rather than assumed on a
# reference MRT packet: 1,137 trace points per ion. Retaining
# all 200 costs 6.9 MB, which is nothing -- MEMORY IS NOT THE REASON.
# The reason is rendering: 200 ions is 227,400 polyline points handed to
# plotly in a browser, on top of the geometry's own traces. At the floor
# it is 28,425.
TRACE_FLOOR = 25          # draw every ion up to this many: a small packet
                          # is fully drawn, because sampling 25% of 10
                          # ions would hide most of a packet the user can
                          # already see in full.
TRACE_FRACTION = 0.25     # above the floor, draw this share of the packet.


def trace_keep_count(n_ions, fraction=TRACE_FRACTION, floor=TRACE_FLOOR):
    """How many ion paths to DRAW for a packet of n_ions.

    All of them up to `floor`; above it, `fraction` of the packet but
    never fewer than `floor`. So 10 -> 10, 25 -> 25, 200 -> 50,
    1000 -> 250: a small packet is shown whole and a large one stays
    legible without the count collapsing as the packet grows.

    REFUSES a fraction outside (0, 1] or a floor below 1 rather than
    clamping: a silently corrected view parameter would draw a different
    number of ions than the instrument declares, and the whole point of
    declaring it is that the picture is checkable against the document.
    """
    n = int(n_ions)
    if n < 0:
        raise ValueError(f"n_ions must be >= 0, got {n}")
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError(
            f"trace fraction must be in (0, 1], got {fraction!r}. It is a "
            f"share of the packet, not a percentage or a count.")
    if int(floor) < 1:
        raise ValueError(f"trace floor must be >= 1, got {floor!r}")
    floor = int(floor)
    if n <= floor:
        return n
    return max(floor, int(math.ceil(float(fraction) * n)))


def trace_policy(doc):
    """(fraction, floor) for an instrument document, defaults if undeclared.

    Read from an optional top-level `view` block:

        "view": {"trace_fraction": 0.25, "trace_floor": 25}

    Kept OUT of `beam` deliberately: beam is what is flown and is physics,
    this is what is drawn and is not. Mixing them would make a display
    preference look like part of the packet definition.
    """
    v = (doc or {}).get("view") or {}
    unknown = [k for k in v if k not in ("trace_fraction", "trace_floor")]
    if unknown:
        raise ValueError(
            f"assembly 'view' declares unknown key(s) {unknown}; it takes "
            f"'trace_fraction' and 'trace_floor'. A view key read under a "
            f"name nothing consumes is a setting the user believes is "
            f"applied and is not (L-159).")
    return (float(v.get("trace_fraction", TRACE_FRACTION)),
            int(v.get("trace_floor", TRACE_FLOOR)))


_POSE_KEYS = ("offset_mm", "rot_deg")


def _pose_from(d) -> Pose:
    """Build a stage Pose from its declared JSON block.

    `offset_mm` IS THE CANONICAL KEY.
    It is the `Pose` dataclass field and the name the flight has always
    actually honoured.

    Why this refuses instead of defaulting: the assembly document was
    being read under TWO key names -- the app's overview drew from
    `origin_mm` while this function flew from `offset_mm` -- and because
    `.get(..., [0,0,0])` silently supplies a default, a stage declared
    only under the other name was posed at the world origin with no
    complaint. Every reference-assembly pose was [0,0,0], so the two agreed by
    coincidence and the split stayed invisible. Pose a stage off-origin
    and the instrument would DRAW displaced and FLY coaxial: a wrong
    answer with a plausible picture.

    Note the name it must not be confused with: `geometry.origin_mm` is
    the SOLVE GRID's anchor, an entirely different quantity from
    a stage's rigid placement. That collision is precisely why the wrong
    key looked right to two different readers.

    A pose key this function does not know is therefore an ERROR, not
    something to skip -- silently ignoring it is the defect being fixed.
    """
    if not d:
        return Pose()
    unknown = [k for k in d if k not in _POSE_KEYS]
    if unknown:
        hint = ""
        if "origin_mm" in unknown:
            hint = (" `origin_mm` was an older spelling of `offset_mm` and "
                    "is no longer accepted, because a pose silently read "
                    "under one name and written under another places the "
                    "stage at the world origin without saying so (L-159). "
                    "Note that `geometry.origin_mm` is a DIFFERENT thing -- "
                    "the solve grid's anchor -- and is unaffected.")
        raise ValueError(
            f"stage pose declares unknown key(s) {unknown}; a pose is "
            f"{list(_POSE_KEYS)}.{hint}")
    return Pose(offset_mm=list(d.get("offset_mm", [0.0, 0.0, 0.0])),
                rot_deg=(list(d["rot_deg"]) if d.get("rot_deg") else None))



def _exit_from(d) -> Optional[ExitPlane]:
    if not d:
        return None
    if "axis" not in d or "value_mm" not in d:
        raise ValueError(f"assembly exit plane needs 'axis' and 'value_mm'; "
                         f"got {d}")
    return ExitPlane(axis=str(d["axis"]), value_mm=float(d["value_mm"]),
                     sign=int(d.get("sign", +1)))


def exit_world_to_local(ex: ExitPlane, pose: Pose) -> ExitPlane:
    """Map a WORLD-authored exit plane into a region's LOCAL frame,
    honouring the FULL pose — rotation included.

    Why this exists (mixed-geometry assemblies): routes
    carry different axial conventions — the r-z kernel's axial axis is
    local x (r = sqrt(y^2+z^2)), the 3-D route's instruments run along
    local z — so joining them coaxially REQUIRES a rotated pose, and the
    previous conversion (subtract the offset component, keep the axis)
    silently produced the wrong local plane for any rotated stage. Every
    zero-rotation deck is bit-identical under this function: R = I gives
    n = e_axis, value - t_axis, sign unchanged — exactly the old
    subtraction.

    Math: world plane  e_a . w = V  with  w = R l + t  becomes
    (R^T e_a) . l = V - e_a . t. The tracers arm AXIS-ALIGNED bounds
    only, so if R^T e_a is not (+/-) a coordinate axis the plane is not
    expressible and this REFUSES with the numbers rather than arming a
    wrong plane.
    """
    e = np.zeros(3)
    axi = {"x": 0, "y": 1, "z": 2}
    e[axi[ex.axis]] = 1.0
    n = pose.R().T @ e
    rhs = ex.value_mm - float(e @ np.asarray(pose.offset_mm, float))
    k = int(np.argmax(np.abs(n)))
    if abs(abs(n[k]) - 1.0) > 1e-9 or np.sum(np.abs(n) > 1e-9) != 1:
        raise ValueError(
            f"exit plane on world {ex.axis!r} maps through this pose "
            f"(rot_deg={pose.rot_deg}) to local normal {np.round(n, 6)}, "
            f"which is not a coordinate axis. The region tracers arm "
            f"axis-aligned bounds only, so a non-axis-aligned seam cannot "
            f"be honoured — use 90-degree pose rotations, or re-author "
            f"the seam on an axis this pose preserves.")
    s = 1.0 if n[k] > 0 else -1.0
    return ExitPlane(axis="xyz"[k], value_mm=rhs * s,
                     sign=int(ex.sign * s))


def _metal_z_half(spec):
    """Half of a stage's DECLARED metal depth, or None when undeclared.

    Takes the LARGEST declared extent among the stage default and any
    per-electrode override: the aperture is the
    hardware's outer envelope, and a stage whose widest plate is 30 mm
    does not stop ions at 10 mm because a narrower plate exists
    elsewhere. Undeclared (0/None everywhere) returns None, imposing
    nothing.
    """
    g = spec.geometry
    d = [float(getattr(g, "metal_depth_mm", 0.0) or 0.0)]
    for el in g.electrodes:
        v = getattr(el, "metal_depth_mm", None)
        if v is not None:
            d.append(float(v))
    m = max(d) if d else 0.0
    return (0.5 * m) if m > 0 else None


def stage_shape_boxes_world(spec, pose):
    """World-frame boxes of every individual rect shape in a stage.

    PER SHAPE, and the granularity is the finding.
    Per-STAGE boxes span mirror-to-mirror and overlap on
    everything (91/91 electrode pairs on a reference MRT). Per-ELECTRODE is
    also wrong, non-obviously: an MRT electrode holds a MIRROR-SYMMETRIC
    PAIR of shapes (cap plates at symmetric +/-x), so unioning an
    electrode's shapes invents a phantom slab spanning the whole analyzer
    through the middle of which nothing exists. Per shape: 672 pairs ->
    26, all of them the one REAL case (pusher plates vs the drift liner).

    Each shape's z extent comes from its electrode's `metal_depth_mm`
    override, else the stage's declarations. Rotation
    Returns [(label, x0, x1, y0, y1, z0, z1), ...].

    ROTATION: supported by boxing
    the ROTATED corners. The earlier objection was to an axis-aligned box
    taken in the LOCAL frame and merely translated — that reports
    clearance for the wrong volume. The world-frame AABB of the rotated
    corner set is exact for the box's own extent claim, uses the SAME
    rotation matrix the flight handoff applies (display == flight), and
    unlocks the non-coaxial assemblies this route exists for.
    """
    import numpy as _np
    R = _rot_matrix(pose.rot_deg) if pose.rot_deg and any(
        abs(float(v)) > 1e-9 for v in pose.rot_deg) else None
    ox, oy, oz = (list(pose.offset_mm) + [0.0, 0.0, 0.0])[:3]
    g = spec.geometry
    # ROUTE FRAME (mixed-geometry assemblies): an r-z
    # stage's rect shapes live on the (axial, radial) half-plane — axial
    # is the r-z kernel's LOCAL X, and the shape revolves about that
    # axis. Boxing them as planar world x/y rects put the funnel's rings
    # on the wrong axes entirely (a wrong volume passes or refuses
    # clearance with a plausible label). The revolved solid's exact AABB
    # in the local frame is [x0,x1] x [-r_out,+r_out]^2, same extent
    # doctrine as the rotated-corner AABB below.
    _is_rz = (getattr(getattr(g, "symmetry", None), "coords", None) == "rz")
    out = []
    for el in g.electrodes:
        d = (el.metal_depth_mm
             if getattr(el, "metal_depth_mm", None) is not None
             else float(getattr(g, "metal_depth_mm", 0.0) or 0.0))
        hz = 0.5 * float(d)
        for k, sh in enumerate(el.shapes or []):
            if sh.type != "rect":
                continue
            pr = sh.params if hasattr(sh, "params") else {}
            x0 = float(pr.get("x_mm", 0.0)); y0 = float(pr.get("y_mm", 0.0))
            x1 = x0 + float(pr.get("width_mm", 0.0))
            y1 = y0 + float(pr.get("height_mm", 0.0))
            if _is_rz:
                # axial span [x0,x1] on local x; outer radius y1 bounds
                # the revolution on local y AND z. Inner radius y0 is
                # real but inexpressible in an AABB — the box is the
                # solid's exact axis-aligned extent, exactly as the
                # rotated-corner path claims for its own boxes.
                lo3 = (x0, -y1, -y1)
                hi3 = (x1, +y1, +y1)
            else:
                lo3 = (x0, y0, -hz)
                hi3 = (x1, y1, +hz)
            if R is None:
                out.append((f"{el.name}#{k}",
                            lo3[0] + ox, hi3[0] + ox,
                            lo3[1] + oy, hi3[1] + oy,
                            lo3[2] + oz, hi3[2] + oz))
            else:
                corners = _np.array([[xx, yy, zz]
                                     for xx in (lo3[0], hi3[0])
                                     for yy in (lo3[1], hi3[1])
                                     for zz in (lo3[2], hi3[2])], float)
                w = corners @ _np.asarray(R, float).T
                w += _np.array([ox, oy, oz], float)
                lo, hi = w.min(axis=0), w.max(axis=0)
                out.append((f"{el.name}#{k}",
                            float(lo[0]), float(hi[0]),
                            float(lo[1]), float(hi[1]),
                            float(lo[2]), float(hi[2])))
    return out


def check_stage_clearance(named_stage_boxes):
    """REFUSE interpenetrating declared metal — ONCE, naming EVERY pair.

    (Refusal not warning; per shape; report
    all overlapping pairs in one refusal.) One refusal listing the whole
    picture beats one-per-pair: the user fixing an injection geometry
    needs to see that all 26 hits are OA-plate x LINER — the same
    physical fact — not to fix them one reload at a time.

    Pairs where either shape's declared z thickness is zero are SKIPPED:
    z undeclared means the overlap question is unanswerable, and refusing
    on a claim nobody made would block every legacy planar deck.

    `named_stage_boxes`: [(stage_name, [shape boxes...]), ...].
    """
    import itertools
    hits = []
    for (na, A), (nb, B) in itertools.combinations(named_stage_boxes, 2):
        for a in (A or []):
            for b in (B or []):
                if (a[6] - a[5]) <= 0 or (b[6] - b[5]) <= 0:
                    continue
                if all(a[1 + 2 * k] < b[2 + 2 * k]
                       and b[1 + 2 * k] < a[2 + 2 * k] for k in range(3)):
                    hits.append((na, a[0], nb, b[0]))
    if hits:
        shown = ", ".join(f"{na}.{sa} x {nb}.{sb}"
                          for na, sa, nb, sb in hits[:12])
        more = f" (+{len(hits) - 12} more)" if len(hits) > 12 else ""
        raise ValueError(
            f"stage clearance: {len(hits)} pair(s) of DECLARED metal "
            f"interpenetrate: {shown}{more}. An assembly with overlapping "
            f"stages is not an instrument (PI, 2026-08-24). Move a "
            f"pose offset, correct a metal_depth_mm — or, if a beam is "
            f"meant to fly THROUGH one of these (an injection aperture in "
            f"a liner), that aperture is not yet declarable and is "
            f"Tier-3 geometry.")


def load_assembly(path):
    """Load an instrument.json, SOLVE each stage's INLINE spec, and return
    (regions, beam). regions is the list for fly_staged; beam is the
    initial ion state dict {mz, p0_mm, v0_mm_us, tob_us}.

    GEOMETRY IS INLINE, NEVER A PATH. One file is the
    instrument. A stage that names an external spec file makes the
    assembly a manifest rather than an artifact: the pair can drift, the
    zip can carry one without the other, and a reader cannot tell what was
    flown from what is in front of them. This module therefore REFUSES a
    path-valued `spec` rather than silently supporting both.
    """
    apath = Path(path)
    doc = json.loads(apath.read_text())
    schema = doc.get("schema", "")
    if schema != SCHEMA:
        raise ValueError(f"assembly schema {schema!r} != {SCHEMA!r} "
                         f"(file {apath})")
    stages = doc.get("stages") or []
    if not stages:
        raise ValueError(f"assembly {apath} has no stages")

    regions = []
    for st in stages:
        if "name" not in st or "spec" not in st:
            raise ValueError(f"stage needs 'name' and 'spec': {st}")
        raw = st["spec"]
        if isinstance(raw, str):
            raise TypeError(
                f"stage {st['name']!r}: 'spec' is a path ({raw!r}), but "
                f"stage geometry must be INLINE. A single file is the "
                f"instrument; a manifest of paths can drift from what it "
                f"names and cannot be zipped as one artifact. Inline the "
                f"spec dict, e.g. with "
                f"`json.loads(Path(x).read_text())`.")
        if not isinstance(raw, dict):
            raise TypeError(
                f"stage {st['name']!r}: 'spec' must be an inline spec "
                f"object, got {type(raw).__name__}.")
        spec = load_any_spec(json.dumps(raw))
        # An inline spec has no file of its own, so anything it resolves
        # relative to one (STL meshes) has to resolve against the assembly
        # instead. Stated rather than left to fail deep in the builder.
        spec._loaded_from = str(apath)
        model, _fly, _cols, _births = build_run(spec)
        fields = getattr(model, "fly_fields", None)
        if fields is None:
            raise ValueError(
                f"stage {st['name']!r}: built model exposes no fly_fields "
                f"pack (builder {spec.builder!r} is not staged-flight "
                f"ready). Every current route publishes one — planar and "
                f"r-z included (route-tagged packs, L-113/L-134) — so a "
                f"missing pack means this builder predates the contract "
                f"or its solve did not complete; it does NOT mean the "
                f"stage's geometry class is unsupported.")
        gas_mode = st.get("gas", "spec")
        if gas_mode == "vacuum":
            gas = None
        elif gas_mode == "spec":
            gas = spec.collisions if getattr(spec.collisions, "enabled",
                                             False) else None
        else:
            raise ValueError(f"stage {st['name']!r}: gas must be 'spec' or "
                             f"'vacuum', got {gas_mode!r}")
        # Integration defaults come from the STAGE'S OWN SPEC, which
        # already declares them, and a stage entry may override. The
        # previous fixed defaults (1 ns / 50 us) silently truncated any
        # instrument whose flight is longer than 50 us -- and a truncated
        # flight reports a timeout fate with a perfectly plausible partial
        # trajectory, which reads as a transmission failure rather than as
        # a setting that was never consulted.
        _integ = getattr(spec, "integration", None)
        _dt = float(st.get("dt_ns", getattr(_integ, "dt_ns", 1.0)))
        _tmax = float(st.get("t_max_us", getattr(_integ, "t_max_us", 50.0)))
        # EXITS ARE AUTHORED IN WORLD COORDINATES; the Region holds them
        # LOCAL. The pose is how a stage lands in the world, and its exit
        # plane is an assembly-level fact (a seam at "x = 12.562" is at
        # x = 12.562 regardless of any stage's internal frame), so the document
        # states it in world and the conversion happens HERE, once —
        # through the FULL pose (exit_world_to_local), rotation included,
        # because mixed-geometry assemblies pose stages with 90-degree
        # rotations to align their differing axial conventions (r-z axial
        # = local x; 3-D axial = local z). Zero-rotation decks are
        # bit-identical to the previous offset-only subtraction.
        _ex = _exit_from(st.get("exit"))
        _po = _pose_from(st.get("pose"))
        if _ex is not None:
            _ex = exit_world_to_local(_ex, _po)
        regions.append(Region(
            name=st["name"], fields=fields, pose=_po,
            exit=_ex, gas=gas,
            field_extent_mm=st.get("field_extent_mm"),
            dt_ns=_dt, t_max_us=_tmax,
            stations=list(getattr(spec, "stations", []) or []),
            rec_every=int(st.get("rec_every",
                                 getattr(_integ, "rec_every", 1))),
            max_records=int(st.get("max_records",
                                   getattr(_integ, "max_records", 100000))),
            bounds=getattr(spec, "bounds", None),
            metal_z_half_mm=_metal_z_half(spec)))

    # Refuse interpenetrating DECLARED metal at load, so an
    # unphysical assembly never reaches a solver. Skips undeclared-z
    # stages by design — see check_stage_clearance.
    _boxes = []
    for _st, _reg in zip(stages, regions):
        _sp = load_any_spec(json.dumps(_st["spec"]))
        _boxes.append((_st["name"], stage_shape_boxes_world(
            _sp, _pose_from(_st.get("pose")))))
    check_stage_clearance(_boxes)

    beam = doc.get("beam") or {}
    if "mz" not in beam:
        raise ValueError(f"assembly 'beam' needs 'mz': {beam}")
    # A beam is EITHER one ion (p0_mm/v0_mm_us) or a packet (ions table).
    # Both present is ambiguous about what was actually flown, so it
    # refuses rather than picking one -- a resolution reported for the
    # wrong ensemble is indistinguishable from a real one.
    has_single = "p0_mm" in beam or "v0_mm_us" in beam
    has_packet = "ions" in beam
    has_stage_src = "from_stage" in beam
    if has_stage_src and (has_single or has_packet):
        raise ValueError(
            "assembly 'beam' declares 'from_stage' AND a literal beam "
            "(ions or p0_mm). Declare one: 'from_stage' means the packet "
            "is PRODUCED by flying that stage from its own source, and a "
            "literal beam beside it makes it undecidable which was flown.")
    if has_stage_src:
        # BIRTHS COME FROM A STAGE'S OWN DECLARED SOURCE. This exists
        # because the alternative -- a
        # literal packet frozen at a seam -- silently DECOUPLES the
        # stages: the upstream stage's optics act on nothing, its knobs
        # move nothing, and every downstream number is conditional on one
        # fossilised realisation. A staged assembly once ran exactly that
        # way ("assembly flown" over a 2-record upstream leg); the
        # contract is that beam parameters from one field region are
        # transmitted faithfully into the next -- ALL of them,
        # produced by flight, not asserted.
        #
        # Generated at LOAD, through the stage's own build (same births
        # path as flying that stage alone), then mapped through the
        # stage's pose into the world frame -- one transform, the same
        # one the flight uses (one authority).
        _nm = beam["from_stage"]
        _cand = [st for st in stages if st.get("name") == _nm]
        if not _cand:
            raise ValueError(
                f"beam.from_stage = {_nm!r}, but the assembly's stages "
                f"are {[st.get('name') for st in stages]}")
        _spec = load_any_spec(json.dumps(_cand[0]["spec"]))
        if int(beam.get("n") or 0) > 0:
            _spec.source.n_ions = int(beam["n"])
        _model, _fly_fn, _cols, _births = build_run(_spec)
        # BIRTHS ARE POSITIONAL [x,y,z,vx,vy,vz,tob] ROWS. `_cols` names
        # the TRAJECTORY record columns (t first), not the births --
        # mapping births by those names shifted every quantity by one
        # column, and the packet came out centred at the y-box instead of
        # the storage gap. Caught by the smoke run's birth extents, not
        # by reading.
        import numpy as _np
        _ba = _np.asarray(_births, dtype=float)
        if _ba.ndim != 2 or _ba.shape[1] < 6:
            raise ValueError(
                f"stage {_nm!r} births have shape {_ba.shape}; expected "
                f"rows of [x,y,z,vx,vy,vz,(tob)]")
        _pose = _pose_from(_cand[0].get("pose"))
        _ox, _oy, _oz = (list(_pose.offset_mm) + [0.0, 0.0, 0.0])[:3]
        # ROTATION IMPLEMENTED (mixed-geometry assemblies; supersedes
        # the earlier refusal). Births are generated in the stage's
        # LOCAL frame, and a posed stage's local frame lands in the world
        # as world = R @ local + t — positions AND velocities, the exact
        # transform the flight applies at every handoff (Pose.
        # local_to_world), so the packet is born where the stage is.
        # Time-of-birth is frame-invariant. Zero-rotation stages take
        # R = I and reproduce the previous offset-only arithmetic
        # bit-for-bit.
        _R = _pose.R()
        beam = dict(beam)
        _rows = []
        for b in _ba:
            _p = _R @ np.array([float(b[0]), float(b[1]), float(b[2])])
            _v = _R @ np.array([float(b[3]), float(b[4]), float(b[5])])
            _rows.append([_p[0] + _ox, _p[1] + _oy, _p[2] + _oz,
                          _v[0], _v[1], _v[2],
                          float(b[6]) if _ba.shape[1] > 6 else 0.0])
        beam["ions"] = _rows
        from ion_gym.physics.sim_build import mz_of
        beam["mz_per_ion"] = [float(mz_of(_spec, _i))
                              for _i in range(len(_rows))]
        beam["ions_provenance"] = {
            "from_stage": _nm, "n": len(beam["ions"]),
            "note": "generated at load from the stage's declared source; "
                    "posed through the stage's full pose (rotation "
                    "included); not a literal packet"}
        has_packet = True
    if has_single and has_packet:
        raise ValueError(
            "assembly 'beam' declares BOTH a single ion (p0_mm/v0_mm_us) "
            "and a packet ('ions'). Declare one. Which was flown decides "
            "every ensemble number the run reports.")
    if not has_single and not has_packet:
        raise ValueError(
            f"assembly 'beam' declares neither a single ion (p0_mm and "
            f"v0_mm_us) nor a packet ('ions': a table of "
            f"[x,y,z,vx,vy,vz,tob] rows): {beam}")
    if has_packet:
        rows = beam["ions"]
        if isinstance(rows, str):
            raise TypeError(
                f"assembly 'beam.ions' is a path ({rows!r}). The packet is "
                f"part of the instrument and must be INLINE, for the same "
                f"reason stage geometry is: one file is the artifact. A "
                f"births CSV can be inlined as a list of "
                f"[x,y,z,vx,vy,vz,tob] rows.")
        if not rows:
            raise ValueError("assembly 'beam.ions' is empty.")
        widths = {len(r) for r in rows}
        if widths != {7}:
            raise ValueError(
                f"assembly 'beam.ions' rows must each be 7 values "
                f"[x,y,z,vx,vy,vz,tob]; found row widths {sorted(widths)}. "
                f"A short row would silently default a velocity or a birth "
                f"time, and the packet's internal time structure is the "
                f"quantity this instrument exists to preserve.")
    else:
        for k in ("p0_mm", "v0_mm_us"):
            if k not in beam:
                raise ValueError(f"assembly 'beam' needs {k!r}: {beam}")
    return regions, beam


def fly_assembly(path, check_seams=True):
    """Convenience: load an assembly and fly its beam through it. Returns
    the fly_staged result dict."""
    from ion_gym.physics.staged_flight import fly_staged
    regions, beam = load_assembly(path)
    if "ions" in beam:
        return fly_packet(regions, beam, check_seams=check_seams)
    return fly_staged(regions, beam["mz"], beam["p0_mm"], beam["v0_mm_us"],
                      tob_us=float(beam.get("tob_us", 0.0)),
                      check_seams=check_seams)


def fly_packet(regions, beam, *, check_seams=True, tracker=None,
               keep_traces=0):
    """Fly an ENSEMBLE through a staged assembly and report its statistics.

    Resolution is a property of a packet, not of an ion. A single-ion
    staged flight can prove the seams are sane and the geometry connects,
    but it cannot produce an R at all: R = T / (2 dT) needs a spread, and
    one ion has none. This is the entry point that makes a staged assembly
    comparable against a known answer.

    IONS THAT DO NOT ARRIVE ARE REPORTED, NOT DROPPED. A packet that loses
    thirty ions and reports a beautiful FWHM on the survivors is the
    classic way an instrument looks better than it is -- the lost ions are
    usually the ones with the extreme initial conditions, so discarding
    them silently narrows exactly the distribution being measured. Losses
    are counted, their reasons tallied, and the caller gets both.

    The seam preflight runs ONCE, not per ion: seams are a property of the
    geometry and the solve, and re-checking them 200 times would only
    reprint the same verdict.
    """
    import numpy as _np

    mz = float(beam["mz"])
    rows = [[float(v) for v in r] for r in beam["ions"]]
    # PER-ION m/z (a 3-mass Ion-tab packet once flew every row
    # at the beam's single scalar mass). A from_stage beam carries
    # mz_per_ion generated by the SAME i//n_ions blocks convention the
    # single-stage flight uses (sim_build.mz_of — one authority); absent
    # (legacy docs, literal packets) every row flies at beam["mz"],
    # bit-identical to before. A wrong-length list is refused, not
    # truncated: a packet whose masses are misaligned by one block
    # mislabels every ion after the first.
    _mzs = beam.get("mz_per_ion")
    if _mzs is not None and len(_mzs) != len(rows):
        raise ValueError(
            f"beam.mz_per_ion has {len(_mzs)} entries for {len(rows)} "
            f"ions — the per-ion mass table must align with the packet "
            f"row for row.")
    def _mz_i(i):
        return float(_mzs[i]) if _mzs is not None else mz
    # Does ANY region declare a detector? If so, arrival means crossing
    # it, and a flight that ends without one has not measured anything.
    _has_detector = any(
        getattr(s, "kind", None) == "detect"
        for rg in regions for s in (rg.stations or []))
    arrivals, losses, per_ion = [], {}, []
    # WHICH ions get their path kept, for DRAWING only (see trace_policy).
    # Evenly spaced across the packet, not the first N: the births table
    # is ordered, so the head of it is one corner of the distribution and
    # drawing only that would show a systematically unrepresentative
    # bundle -- narrow where the real packet is wide. A stride spans the
    # whole table. Statistics below are unaffected either way; they are
    # computed from every ion.
    _keep = max(0, int(keep_traces))
    n_rows = len(rows)
    if _keep >= n_rows:
        _keep_idx = set(range(n_rows))
    elif _keep > 0:
        _keep_idx = {int(round(i * (n_rows - 1) / (_keep - 1))) if _keep > 1
                     else 0 for i in range(_keep)}
    else:
        _keep_idx = set()
    traces = []
    it = rows if tracker is None else tracker(rows)
    for i, r in enumerate(it):
        x, y, z, vx, vy, vz, tob = r
        try:
            res = fly_staged(regions, _mz_i(i), (x, y, z), (vx, vy, vz),
                             tob_us=tob, seed=i + 1,
                             check_seams=(check_seams and i == 0))
        except ValueError as e:
            # A refusal on ion i is information about ion i, not a reason
            # to abandon the packet -- but it is never silent.
            losses[str(e)[:80]] = losses.get(str(e)[:80], 0) + 1
            per_ion.append(dict(i=i, arrived=False, why=str(e)[:120]))
            continue
        fate = res.get("fate", "completed")
        # Kept BEFORE the fate branches below, and regardless of fate: a
        # lost ion's path is the most informative one to look at, and
        # collecting only survivors would draw a picture in which nothing
        # ever goes wrong. Coordinates are already world-frame.
        if i in _keep_idx:
            # TRUNCATED AT DETECTION. Detect stations are non-destructive
            # by design, so the kernel keeps integrating to
            # t_max and the raw trace runs well past the detector --
            # measured on a reference MRT, ~100 mm past the end of the
            # drift window. Drawing that tail invites the
            # reader to interpret travel that happened after the
            # measurement as part of it. The flight ENDS, for every
            # purpose this view serves, at the plane it was measured to.
            _rg_out = []
            for rg in res["regions"]:
                _t = np.asarray(rg["t_us"], float)
                _cut = rg.get("detected_t_us")
                _m = (_t <= float(_cut)) if _cut is not None else slice(None)
                _rg_out.append(dict(
                    name=rg["name"], x=np.asarray(rg["x"], float)[_m],
                    y=np.asarray(rg["y"], float)[_m],
                    z=np.asarray(rg["z"], float)[_m], t_us=_t[_m]))
            traces.append(dict(i=i, fate=fate, mz=_mz_i(i),
                               regions=_rg_out))
        # ARRIVED means the ion reached the end of the instrument: it
        # crossed a declared detector ("detected"), or the final region
        # ran to its natural end with no detector to cross ("completed").
        # Enumerated rather than written as `fate != "completed"`, because
        # that formulation silently reclassifies any fate added LATER as a
        # loss. That is not hypothetical: introducing "detected" did
        # exactly this, and 200 of 200 ions arrived while all 200 were
        # counted lost. The only symptom was an undefined R.
        if fate not in ("detected", "completed"):
            losses[fate] = losses.get(fate, 0) + 1
            per_ion.append(dict(i=i, arrived=False, why=fate,
                                mz=_mz_i(i)))
            continue
        det = res.get("detection")
        # A DECLARED DETECTOR THAT WAS NEVER CROSSED IS A LOSS, not an
        # arrival at whatever time the last region happened to stop.
        # Without this the run reports the region's EXIT time as a time of
        # flight: observed once as "200/200 arrived, T = 0.0011 us,
        # R = 3" -- every ion counted as transmitted, with a number that
        # is not a flight time at all. A resolution is only meaningful
        # between the source and the plane it was measured to.
        if det is None and _has_detector:
            losses["declared detector never crossed"] = losses.get(
                "declared detector never crossed", 0) + 1
            per_ion.append(dict(i=i, arrived=False,
                                why="declared detector never crossed",
                                mz=_mz_i(i)))
            continue
        tof = float(res["tof_us"])
        arrivals.append(tof)
        per_ion.append(dict(i=i, arrived=True, tof_us=tof,
                            k=(int(det["k"]) if det else None),
                            # WHERE it landed, not just when. The impact
                            # cross-section is the detector face, and the
                            # crossing state is already interpolated by
                            # station_hits -- dropping it here and
                            # re-deriving it later would be a second
                            # interpolation convention.
                            station=(det.get("region") if det else None),
                            x=(det.get("x") if det else None),
                            y=(det.get("y") if det else None),
                            z=(det.get("z") if det else None),
                            # ...AND HOW FAST. station_hits
                            # interpolates vx/vy/vz at the crossing and
                            # this record threw them away, so speed and
                            # ke_ev were underivable for an assembly
                            # flight -- while the Analysis tab advertised
                            # both and, when they came back empty, blamed
                            # "lost ions" on a flight where every ion
                            # arrived. Keeping them costs three floats and
                            # is the SAME interpolated state the position
                            # above is taken from, by the argument already
                            # made there. mz travels too: ke_ev is
                            # meaningless without it, and re-reading it
                            # from the beam at analysis time would be a
                            # second source for one fact.
                            vx=(det.get("vx") if det else None),
                            vy=(det.get("vy") if det else None),
                            vz=(det.get("vz") if det else None),
                            mz=_mz_i(i),
                            t_us=tof))

    n = len(rows)
    out = dict(n=n, n_arrived=len(arrivals), n_lost=n - len(arrivals),
               losses=losses, per_ion=per_ion,
               # A DRAWN subset, never the measured set. n_traced is
               # reported alongside n so a reader of this dict can never
               # mistake "50 paths shown" for "50 ions flown".
               traces=traces, n_traced=len(traces))
    if len(arrivals) < 2:
        # Refusing to compute is the honest outcome: an FWHM over fewer
        # than two arrivals is not a small sample, it is undefined.
        out.update(T_us=float(arrivals[0]) if arrivals else float("nan"),
                   fwhm_ns=float("nan"), R=float("nan"),
                   note=(f"only {len(arrivals)} of {n} ions arrived; a "
                         f"time spread is undefined below two."))
        return out
    if _mzs is not None and len(set(float(v) for v in _mzs)) > 1:
        # A time spread pooled ACROSS masses is not a resolution — the
        # separation between masses would masquerade as peak width. The
        # per-record mz travels on per_ion, so the stats card computes
        # honest per-m/z numbers; this headline refuses.
        out.update(T_us=float("nan"), fwhm_ns=float("nan"),
                   R=float("nan"),
                   note=(f"multi-m/z packet "
                         f"({len(set(float(v) for v in _mzs))} masses) — "
                         f"per-m/z statistics on the Stats card; a pooled "
                         f"R across masses is undefined."))
        return out
    a = _np.asarray(arrivals)
    fwhm_ns = float(FWHM_PER_SIGMA * a.std() * 1e3)
    T_us = float(a.mean())
    out.update(T_us=T_us, fwhm_ns=fwhm_ns,
               R=(T_us * 1e3 / (2.0 * fwhm_ns)) if fwhm_ns > 0
                 else float("inf"))
    return out
