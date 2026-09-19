"""
build_stl3d.py — FULL 3-D solve + flight for imported STL electrodes.

This is the SimSpec->3D wiring milestone: it composes three individually
validated legs with no new physics —
  * voxelize.voxelize_meshes  (Rung-2 voxelizer, reference-matched ±5.5 µm)
  * solver3d.solve_bases      (native 3-D Laplace, gate-validated)
  * tracer3d.fly3d            (field-aware 3-D tracer from the
                               quad_monolithic validation, absolute clock)

Drive model (matches the validated tracer): phi(t) = A + sin(w t)·B, i.e.
DC everywhere plus ONE RF sinusoid. RF groups are folded into B with
signed amplitudes: phase 0 -> +amp, phase 180 -> -amp. All RF groups must
share one frequency and use phases in {0, 180}; anything else RAISES
(refuse-unwired-paths — the 3-D tracer has one sinusoid, and silently
mis-phasing a drive is worse than refusing).

Fates: tracer3d kinds (0 impact / 1 left box / 2 timeout) match the
ion_gym convention directly; bounding planes are applied post-hoc here
(-> fate 3) because the 3-D tracer predates the bounds feature.
"""
from __future__ import annotations
import math
import time

import numpy as np

from ion_gym.io.sim_spec import SimSpec, BASE_CHANNELS, OPTIONAL_CHANNELS
from ion_gym.physics.sim_build import generate_births
from ion_gym.physics.collision3d import KG_AMU, E_CHG
from ion_gym.physics.solver3d import solve_bases
from ion_gym.physics.tracer3d import fly3d

# tracer3d kinds (0 impact / 1 left box / 2 timeout) are IDENTICAL to the
# ion_gym fate convention — no remap. (An earlier remap here was wrong;
# the gate now pins each fate explicitly.)


class Stl3DModel:

    """Minimal model surface the app expects: extent, potential_image
    (mid-z slice of the DC potential for the 2-D background), pe_surface
    (mid-z effective potential: DC + RF Dehmelt from the 3-D |grad B|)."""
    # Which principal planes this model HAS.  Declared, never sniffed.
    # A true 3-D solve: all three principal planes are real.
    PLANES = ('xy', 'xz', 'yz')

    def __init__(self, A, B, ele, h_mm, rf_V, om_rad_us, off_mm,
                 anchor_mm=(0.0, 0.0)):
        self.A, self.B, self.ele = A, B, ele
        self.h_mm = h_mm
        self.rf_V, self.om_rad_us = rf_V, om_rad_us
        self.off_mm = np.asarray(off_mm, float)   # build-frame offset
        # SIGNED FRAME: the deck's declared
        # origin_mm, the SAME attribute name PlanarModel carries so
        # viz_core's Scene composes it identically on both routes. Node
        # (0,0) of the stored grid sits AT this world coordinate.
        # (0,0) on every legacy deck -- byte-identical output.
        self.anchor_mm = (float(anchor_mm[0]), float(anchor_mm[1]))
        self.mm = h_mm

    @property
    def world_off_mm(self):
        """Stored-frame mm + this = WORLD (deck-frame) mm. Composes the
        mirror canonical offset with the deck origin -- the two offsets
        the cross-session patch says must coexist, never replace each
        other. mirror_off_mm keeps its pure-mirror meaning (staged
        assemblies read it from fly_fields); every DISPLAY consumer
        reads this composition instead."""
        m = getattr(self, "mirror_off_mm", (0.0, 0.0, 0.0))
        return (m[0] + self.anchor_mm[0], m[1] + self.anchor_mm[1], m[2])

    @property
    def extent(self):
        nx, ny, nz = self.A.shape
        return (nx * self.h_mm, ny * self.h_mm, nz * self.h_mm)

    def _mid(self):
        return self.A.shape[2] // 2

    def potential_image(self, rf_phase=None):
        k = self._mid()
        s = math.sin(rf_phase) if rf_phase is not None else 0.0
        img = self.A[:, :, k] + s * self.rf_V * self.B[:, :, k]
        nx, ny = img.shape
        # canonical frame: mirrored axes read [-H,+H], mirror plane at 0
        # (mirror_off_mm zero on non-mirrored axes — unchanged output)
        _moff = self.world_off_mm
        x = np.arange(nx) * self.h_mm + _moff[0]
        y = np.arange(ny) * self.h_mm + _moff[1]
        return x, y, img, self.ele[:, :, k]

    def efield_magnitude(self):
        """|E| at the mid-z slice. For an RF device the DC field alone is ~0
        (pure-RF quad), so include the RF at PEAK phase — this shows the real
        quadrupole saddle (strong near rods, null on axis) instead of a blank
        map. Matches PlanarModel's (nx,ny) return shape."""
        k = self._mid()
        phi = self.A[:, :, k] + self.rf_V * self.B[:, :, k]   # RF peak
        gx, gy = np.gradient(phi, self.h_mm)
        return np.hypot(gx, gy)

    def _plane_slice(self, plane, index=None):
        """(normal-axis, node index) for a principal plane. ONE rule, shared
        by pe_surface and field_surface: index=None picks the ELECTRODE-DENSE
        plane (most metal on the normal axis), not the geometric middle,
        falling back to the middle only when there is no metal."""
        nx, ny, nz = self.A.shape
        ax = {"xy": 2, "xz": 1, "yz": 0}.get(plane)
        if ax is None:
            raise ValueError(f"plane must be 'xy'|'xz'|'yz', got {plane!r}")
        n_norm = (nx, ny, nz)[ax]
        if index is None:
            metal_by_k = [int((self.ele.take(kk, axis=ax) > 0).sum())
                          for kk in range(n_norm)]
            k = (int(np.argmax(metal_by_k)) if max(metal_by_k) > 0
                 else n_norm // 2)
        else:
            k = int(index)
        if not (0 <= k < n_norm):
            raise ValueError(f"index {k} outside the {'xyz'[ax]} axis "
                             f"(0..{n_norm - 1})")
        return ax, k

    def field_surface(self, plane="xy", index=None, rf_phase=None):
        """|E| in V/mm on ANY principal plane at ANY slice — the field
        counterpart of pe_surface, so the field can be inspected on a cut
        INTO the apparatus (e.g. a yz cross-section at a chosen x along the
        transport axis) instead of only the fixed mid-plane the transport
        views show (you cannot judge the field from the
        ends alone).

        The magnitude is the FULL 3-D |E| (all three components), not the
        in-plane part: the normal derivative is taken from a 3-node slab
        around the slice, so no full-domain gradient is ever allocated.
        rf_phase=None uses the RF PEAK for an RF device (the same snapshot
        potential_image/field_slice_3d show); pass a phase in radians for
        an instant, or 0.0 for DC only.

        Returns (a, b, |E|, ele) with a,b the in-plane coordinates in mm in
        the canonical frame — same contract as pe_surface.
        """
        ax, k = self._plane_slice(plane, index)
        h = float(self.h_mm)
        s = (math.sin(rf_phase) if rf_phase is not None else 1.0)
        phi = self.A if not self.rf_V else self.A + s * self.rf_V * self.B
        n_norm = phi.shape[ax]
        # in-plane derivatives on the slice itself
        sl = [slice(None)] * 3
        sl[ax] = k
        face = phi[tuple(sl)]
        ga, gb = np.gradient(face, h)
        # normal derivative from a 3-node slab (2 at a domain edge, where
        # np.gradient falls back to a one-sided difference — correct there)
        k0, k1 = max(0, k - 1), min(n_norm, k + 2)
        sl[ax] = slice(k0, k1)
        slab = phi[tuple(sl)]
        gn = np.gradient(slab, h, axis=ax)[
            (slice(None),) * ax + (min(k - k0, slab.shape[ax] - 1),)]
        emag = np.sqrt(ga ** 2 + gb ** 2 + gn ** 2)
        _moff = self.world_off_mm
        coords = [np.arange(n) * h + _moff[i]
                  for i, n in enumerate(self.A.shape)]
        a, b = [c for i, c in enumerate(coords) if i != ax]
        return a, b, emag, self.ele[tuple([slice(None)] * ax + [k])]

    def pe_surface(self, mz=None, charge=1, plane="xy", index=None,
                   t_us=0.0):
        """PE on ANY principal plane, composed from EVERY drive channel.

        plane is 'xy' | 'xz' | 'yz'; index is the node along the normal
        (default: the electrode-dense slice via _plane_slice).

        Per channel, by its resolved pe_mode (same reduction as the
        planar and r-z models, L-456):
          'pseudo' sin    : Dehmelt from the pack's OWN E stacks (the
                            fields the tracer flies, so field_method is
                            honoured here too), quadrature-composed per
                            frequency: |E0|^2 = |Es|^2 + |Ec|^2 with
                            Es/Ec the amp*cos/sin(phase)-weighted sums.
          'pseudo' square : digital-trap harmonic factor pi^2/6, per
                            channel.
          'instant'       : the REAL potential amp*base(t_us)*phi_k --
                            the slow travelling wave the ions surf, at
                            LAB time t_us. This is the time knob: step
                            t_us to watch the wave march.
        A channel's offset_v is a static shift and is ALWAYS included
        (charge * off * phi_k), matching the 2-D routes where it folds
        into A. Pseudo terms need an m/z; with mz=None the surface is
        the mass-independent part (static + offsets + instant drives).

        SUPERSEDED here: the single representative RF pair (rf_V * B)
        this method used to reduce -- it dropped every square/table
        drive and every sin group beyond the largest pair.
        potential_image/field_surface still draw the representative
        pair, unchanged.

        Returns (a, b, PE, ele) where a,b are the in-plane coordinates in mm
        and ele is the electrode LABEL array on that plane (0 = vacuum), so a
        caller can either mask the metal or drape over it.
        """
        from ion_gym.physics.collision3d import KG_AMU, E_CHG
        nx, ny, nz = self.A.shape
        ax, k = self._plane_slice(plane, index)
        sl = [slice(None)] * 3
        sl[ax] = k
        sl = tuple(sl)

        # in-plane coordinates in the CANONICAL frame: mirrored axes read
        # [-H,+H] with the mirror plane at 0 (mirror_off_mm is 0 on
        # non-mirrored axes, so unmirrored output is byte-identical).
        _moff = self.world_off_mm
        coords = [np.arange(n) * self.h_mm + _moff[i]
                  for i, n in enumerate((nx, ny, nz))]
        a, b = [c for i, c in enumerate(coords) if i != ax]

        pe = charge * self.A[sl].copy()
        f = getattr(self, "fly_fields", None)
        phis = getattr(self, "chan_phi", None)
        grps = getattr(self, "chan_groups", None)
        if f is None or phis is None or grps is None:
            raise ValueError(
                "Stl3DModel.pe_surface needs the channel surfaces the "
                "runner attaches (fly_fields, chan_phi, chan_groups); "
                "this model was constructed without them — build it "
                "through build_stl3d_run, not by hand")
        from ion_gym.physics.tracer3d import _wave_eval
        kinds = np.asarray(f["ch_kind"])
        # |E| terms use the pack's OWN field stacks (the fields the
        # tracer flies, electrode-aware when the deck says so), the FULL
        # 3-D magnitude sliced — an in-plane gradient of a pre-sliced
        # plane drops the normal component (across a SLIM gap it points
        # along x) and collapses the pseudopotential into a flat skirt.
        # Pack fields are V/mm; Dehmelt below wants V/m (the 1e3).
        m_kg = (mz * KG_AMU) if mz else None
        quad = {}                      # om_rad_us -> [Es_x..Ec_z] slices
        for kk, gr in enumerate(grps):
            amp = float(f["ch_amp"][kk])
            off = float(f["ch_off"][kk])
            ph = float(f["ch_ph"][kk])
            om_us = float(f["ch_om"][kk])
            if off:
                # offset_v is a STATIC shift of the group's members —
                # always present, exactly as the 2-D routes fold it
                pe = pe + charge * off * phis[kk][sl]
            if amp == 0.0:
                continue
            mode = gr.resolved_pe_mode()
            if mode == "instant":
                o0, o1 = int(f["tab_off"][kk]), int(f["tab_off"][kk + 1])
                w = _wave_eval(int(kinds[kk]), om_us, ph, amp, off,
                               float(f["ch_duty"][kk]), f["tab_t"],
                               f["tab_v"], o0, o1, float(t_us)) - off
                pe = pe + charge * w * phis[kk][sl]
                continue
            if not (mz and om_us > 0.0):
                continue               # pseudo needs a mass and a period
            if int(kinds[kk]) == 0:    # sin: quadrature-compose per freq
                ent = quad.setdefault(round(om_us, 12), [0.0] * 6)
                c, sph = math.cos(ph), math.sin(ph)
                for i2, EK in enumerate((f["ExK"], f["EyK"], f["EzK"])):
                    ent[i2] = ent[i2] + amp * c * EK[kk][sl]
                    ent[3 + i2] = ent[3 + i2] + amp * sph * EK[kk][sl]
            else:                      # square/cos: per-channel Dehmelt
                e0_sq = (f["ExK"][kk][sl] ** 2 + f["EyK"][kk][sl] ** 2
                         + f["EzK"][kk][sl] ** 2) * (amp * 1e3) ** 2
                fac = (math.pi ** 2 / 6.0) if int(kinds[kk]) == 2 else 1.0
                om = om_us * 1e6
                pe = pe + charge * ((charge * E_CHG) * e0_sq * fac
                                    / (4 * m_kg * om ** 2))
        for om_us, ent in quad.items():
            e0_sq = sum(np.asarray(c2) ** 2 for c2 in ent) * 1e6
            om = om_us * 1e6
            pe = pe + charge * ((charge * E_CHG) * e0_sq
                                / (4 * m_kg * om ** 2))
        return a, b, pe, self.ele[sl]


def _grid_from_spec(spec):
    """Node counts of the 3-D stored grid, through THE counting
    function. The per-route `floor(mm/h) + 1` this replaces silently
    absorbed any non-integer remainder — a declared 20.0 mm at 0.35
    quietly became a real 19.95 and displaced a quarter-symmetric
    deck's symmetry plane by 0.86 cell in z. Now a non-conforming
    extent is REFUSED with the two nearest conforming extents; for a
    conforming extent floor+1 == cells+1, so every conforming deck's
    grid (and basis cache key) is byte-identical."""
    from ion_gym.io.lattice import gu_nodes
    g = spec.geometry
    h = g.mm_per_gu
    nx = gu_nodes(g.width_mm, h, axis="x", what="width_mm domain extent")
    ny = gu_nodes(g.height_mm, h, axis="y", what="height_mm domain extent")
    nz = gu_nodes(g.depth_mm, h, axis="z", what="depth_mm domain extent")
    return nx, ny, nz, h


def stl_masks_3d(spec: SimSpec, verbose=False):
    """Voxelize each STL electrode on the FULL 3-D grid (no slicing).
    Same calling convention as the Rung-2-validated stl_masks_2d: meshes
    scaled to grid units, dict keyed by electrode index, plain grid obj."""
    from ion_gym.physics.build_stl import _require_trimesh
    _require_trimesh()      # availability probe; module unused here
    from ion_gym.physics.voxelize import voxelize_meshes

    g = spec.geometry
    nx, ny, nz, h = _grid_from_spec(spec)

    # GridSpec IS the named owner of (nx, ny, nz, mm_per_gu); the ad-hoc
    # _Grid attribute bag this replaced was copy-pasted between the two
    # STL builders.
    from ion_gym.physics.scene3d import GridSpec
    grid = GridSpec(nx=nx, ny=ny, nz=nz, mm_per_gu=h)

    from ion_gym.io.stl_resolve import require_stls, load_mesh
    stl_dir = require_stls(spec)      # preflight: refuse-with-diagnostic
    meshes = {}
    for idx, el in enumerate(g.electrodes, start=1):
        if not el.stl:
            raise ValueError(f"stl_masks_3d: electrode {el.name!r} has no "
                             "stl reference")
        # SINGLE MESH-INGEST POINT: declared frame applied there.
        m = load_mesh(spec, el, stl_dir=stl_dir)
        m.apply_scale(1.0 / h)
        meshes[idx] = m
        if verbose:
            print(f"  loaded {el.stl} -> {len(m.faces)} faces")
    lab = voxelize_meshes(meshes, grid)
    return {i: (lab == i) for i in meshes}


_WAVE_KIND = {"sin": 0, "cos": 1, "square": 2, "table": 3}


def _display_rf_pair(spec, bases, shape):
    """(group, B) for the 2-D PE background only: the dominant sinusoidal
    group's rail difference. Display convenience — the FLIGHT uses every
    channel from compose_drive_channels, not this."""
    import numpy as np
    g = spec.geometry
    sin_groups = [gr for gr in g.rf_groups
                  if gr.amplitude_v and getattr(gr, "waveform", "sin")
                  in ("sin", "cos")]
    if not sin_groups:
        return None, np.zeros(shape, np.float32)
    # display background: the dominant group vs its 180-deg PARTNER group
    # (rail pair). Members are summed plain; the rail split is now between
    # GROUPS, so B = phi(top) - phi(nearest 180-deg-partner group).
    top = max(sin_groups, key=lambda gr: gr.amplitude_v)
    partner = None
    for gr in sin_groups:
        if gr is top:
            continue
        if abs((gr.phase_deg - top.phase_deg) % 360 - 180) < 1e-6:
            partner = gr
            break
    B = np.zeros(shape, np.float32)
    for i, el in enumerate(g.electrodes, start=1):
        names = el.group_names()
        if top.name in names:
            B += bases[i]
        elif partner and partner.name in names:
            B -= bases[i]
    return top, B


def _drive_reweight_key(spec, bases):
    """Key for the EXPENSIVE, drive-INDEPENDENT part of the compose: the
    per-group E-field GRADIENTS. These depend only on geometry, the
    electrode->group MEMBERSHIP, the metal mask, and the field method —
    NOT on any voltage, amplitude, phase, or frequency (those enter as
    the scalar ch_* arrays). So when only the drive changes, the group
    gradients are reused and the compose is a scalar re-pack — EXACT,
    because E = -grad is linear in phi and the electrode-aware one-sided
    rule is a fixed linear operator once the metal mask is set.

    The key includes a strided CONTENT fingerprint of the bases (every
    4th node per axis): a geometry edit that happened to preserve shape/
    pitch/membership would otherwise falsely HIT. The stride cannot miss
    a real change — Laplace solutions have global support, so ANY
    boundary edit moves the potential at every sampled node.
    """
    import hashlib
    import json
    import numpy as np
    g = spec.geometry
    membership = {}
    for i, el in enumerate(g.electrodes, start=1):
        membership[str(i)] = sorted(el.group_names())
    shape = next(iter(bases.values())).shape
    fp = hashlib.sha256()
    for i in sorted(bases):
        b = np.asarray(bases[i])
        fp.update(str(i).encode())
        fp.update(np.ascontiguousarray(b[::4, ::4, ::4]).tobytes())
    payload = {
        "shape": list(shape),
        "pitch": float(g.mm_per_gu),
        "field_method": getattr(g, "field_method", "electrode_aware"),
        "channel_dtype": getattr(g, "channel_dtype", "float64"),
        "membership": membership,
        "bases_fp": fp.hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


# module-level cache: key -> the drive-independent gradient bundle. Small
# (one geometry in flight in the app), so a single-slot dict is enough;
# holding per-electrode DC gradients lets a DC re-tune reweight too.
_COMPOSE_GRAD_CACHE = {}


def clear_memory_cache():
    """Drop the composed-gradient cache. Self-clearing on a key change,
    so it normally holds one geometry — but that one entry is every
    electrode's DC gradient triple and is not small. Returns entries
    dropped, matching build_planar/build_rz/build_stl."""
    n = len(_COMPOSE_GRAD_CACHE)
    _COMPOSE_GRAD_CACHE.clear()
    return n


def unsupported_drive_features(spec):
    """Declared drive features the 3-D channel pack does NOT apply:
    NONE, as of 2026-09-16 (L-455). A group with amplitude_v == 0 but a
    non-zero offset_v is packed (the offset is its drive), and a table
    group's interp is packed as declared (hold -> kind 3, linear ->
    kind 4). Kept as the route's contract statement for the field
    export door."""
    return []




def compose_drive_channels(spec, bases, verbose=False):
    """(EA, channels) for the 3-D flight from DECLARED drive groups.

    Supersedes _compose_AB (which handled ONE frequency and phases 0/180
    only). Now every RFGroupSpec becomes ONE affine drive CHANNEL over
    its own composed basis field, so an electrode may belong to SEVERAL
    groups at once (ElectrodeSpec.group_names) in addition to its DC:

      EA           = sum_i dc_i * b_i           (static, incl. DC ladders)
      channel[g]   = basis  E_g = sum_{i in g} phase_sign_i * b_i,
                     waveform (sin/cos/square/table), om=2pi f, amp, phase

    The channel pack is exactly tracer3d's (ExK/EyK/EzK + ch_* + table);
    the tracer already sums channels with per-channel waveforms and is
    frozen-anchored (test_multidrive3d Gate 1). So this composer is the
    only new logic: spec -> channels. DISPLAY == COMPUTATION — the fields
    are the same bases the solve produced, never re-derived.

    A group's amplitude/phase live on the GROUP; an electrode's
    membership contributes its basis with a per-electrode phase offset
    only through the legacy single-electrode path (kept for back-compat).
    Table waveforms carry (table_t_us, table_v) from the group.
    """
    import numpy as np
    from ion_gym.physics.tracer3d import build_field_aware_3d

    g = spec.geometry
    groups = {gr.name: gr for gr in g.rf_groups}
    shape = next(iter(bases.values())).shape
    h = g.mm_per_gu
    # DECLARED field-build options (GeometrySpec): electrode-aware vs plain
    # gradient, and stored channel precision. Defaults preserve current
    # behavior (electrode_aware, float64).
    _eaware = getattr(g, "field_method", "electrode_aware") == "electrode_aware"
    _cdtype = np.float32 if getattr(g, "channel_dtype",
                                    "float64") == "float32" else np.float64

    # ---- static A: DC on every electrode (ladders already resolved) ----
    A = np.zeros(shape, np.float32)
    for i, el in enumerate(g.electrodes, start=1):
        if el.dc:
            A += np.float32(el.dc) * bases[i]

    # ---- per-group potential bases (which electrodes drive each) -------
    # Phase lives on the GROUP (clean drive model). A rod pair is two
    # groups 180 deg apart, each with its own members — NOT one group
    # with signed electrodes. So a group's basis is the plain sum of its
    # members' bases; the phase is applied in the waveform.
    group_phi = {name: np.zeros(shape, np.float32) for name in groups}
    used = set()
    for i, el in enumerate(g.electrodes, start=1):
        for gname in el.group_names():
            if gname not in groups:
                raise ValueError(
                    f"electrode {el.name!r} names drive group "
                    f"{gname!r}, which is not defined in geometry."
                    f"rf_groups ({sorted(groups)})")
            group_phi[gname] += bases[i]
            used.add(gname)

    # a defined-but-unused group is a spec error surfaced, not ignored
    unused = [n for n in groups if n not in used
              and (groups[n].amplitude_v
                   or getattr(groups[n], "offset_v", 0.0))]
    if unused:
        raise ValueError(
            f"drive groups {unused} are defined and non-zero but no "
            f"electrode belongs to them (nothing to drive) — assign "
            f"members or remove the group")

    # ---- pack each group as one affine channel over its E basis --------
    ExK, EyK, EzK = [], [], []
    ch_kind, ch_om, ch_ph, ch_amp, ch_off, ch_duty = ([], [], [],
                                                        [], [], [])
    tab_t, tab_v, tab_off = [], [], [0]
    ch_name = []            # group name per channel, for consumers that
                            # must say which drive a channel is (exports)
    ele = _ele_from_bases(spec, bases)     # metal mask for edge-aware grad
    # PROGRESS: the per-group gradient passes below are the work that runs
    # AFTER the last [vmg] line (the bases solve) — a full-grid gradient
    # each, ~as costly as a solve cycle, with NO logging before this. On a
    # big multi-group scene (SLIM: 10 groups + static) that is many silent
    # seconds and reads as a hang. Announce each pass so the phase is
    # visible.
    # a group with amplitude 0 but a non-zero offset_v still drives its
    # members at that DC shift (w = amp*base + off), so it stays a live
    # channel — dropping it dropped the offset (L-455)
    _active = [(nm, gr) for nm, gr in groups.items()
               if nm in used and (gr.amplitude_v
                                  or getattr(gr, "offset_v", 0.0))]
    _ntot = len(_active) + 1               # +1 for the static field A below
    import time as _t
    # ---- DRIVE-ONLY REWEIGHT ------------------------------------
    # Per-group gradients are drive-independent (see _drive_reweight_key),
    # so on a drive-only change every group gradient is reused from the
    # cache and the compose reduces to the scalar ch_* re-pack plus ONE
    # gradient (static DC). Single-slot cache: one geometry in flight per
    # app; a new key clears the old slot (frees the old grids). Lazily
    # filled per group, so a group toggled from amplitude 0 to non-zero
    # just computes its gradient on first need.
    _rw_key = _drive_reweight_key(spec, bases)
    _rw = _COMPOSE_GRAD_CACHE.get(_rw_key)
    if _rw is None:
        _COMPOSE_GRAD_CACHE.clear()
        _rw = {"grads": {}}
        _COMPOSE_GRAD_CACHE[_rw_key] = _rw
    elif verbose and _rw["grads"]:
        print(f"[compose] reweight cache HIT ({_rw_key[:8]}) — group "
              f"gradients reused; only the static-DC gradient recomputes",
              flush=True)
    for _gi, (name, gr) in enumerate(_active, 1):
        if verbose:
            _t0 = _t.time()
        kind = _WAVE_KIND.get(getattr(gr, "waveform", "sin"))
        if kind is None:
            raise ValueError(
                f"group {name!r} waveform {gr.waveform!r} unsupported; "
                f"one of {sorted(_WAVE_KIND)}")
        if kind == 3 and getattr(gr, "interp", "hold") != "hold":
            kind = 4          # table-linear: interp as DECLARED (L-455)
        _cg = _rw["grads"].get(name)
        if _cg is None:
            Ex, Ey, Ez = build_field_aware_3d(group_phi[name], ele, h,
                                              electrode_aware=_eaware)
            _rw["grads"][name] = (Ex, Ey, Ez)
            if verbose:
                print(f"[compose] channel {_gi}/{_ntot} ({name}): "
                      f"gradient {_t.time() - _t0:.1f}s", flush=True)
        else:
            Ex, Ey, Ez = _cg
            if verbose:
                print(f"[compose] channel {_gi}/{_ntot} ({name}): "
                      f"REWEIGHT — cached gradient reused (drive-only "
                      f"change)", flush=True)
        ExK.append(Ex)
        EyK.append(Ey)
        EzK.append(Ez)
        ch_name.append(str(name))
        ch_kind.append(kind)
        ch_om.append(2.0 * math.pi * float(gr.frequency_hz) * 1e-6)
        ch_ph.append(math.radians(float(gr.phase_deg)))
        ch_amp.append(float(gr.amplitude_v))
        ch_off.append(float(getattr(gr, 'offset_v', 0.0)))
        d = float(getattr(gr, 'duty', 0.5))
        if not (0.0 < d < 1.0):
            raise ValueError(f"group {name!r}: duty {d} outside (0, 1)")
        ch_duty.append(d)  # per-drive DC offset (unipolar squares); static DC stays in EA
        if kind == 3:                      # table
            t = list(getattr(gr, "table_t_us", []) or [])
            v = list(getattr(gr, "table_v", []) or [])
            if len(t) != len(v) or len(t) < 2:
                raise ValueError(f"group {name!r} table waveform needs "
                                 f"matching table_t_us/table_v (>=2 pts)")
            tab_t.extend(t)
            tab_v.extend(v)
        tab_off.append(len(tab_t))

    if verbose:
        _t0 = _t.time()
    # static DC is RECOMPUTED each compose: EA depends on the DC VALUES,
    # and reweighting it from cached per-electrode gradients would cost
    # 3 full-grid arrays PER ELECTRODE (GBs on the big grids) to save one
    # gradient pass. One gradient is the floor of a drive-only compose.
    EAx, EAy, EAz = build_field_aware_3d(A, ele, h,
                                        electrode_aware=_eaware)
    if verbose:
        print(f"[compose] channel {_ntot}/{_ntot} (static DC): "
              f"gradient {_t.time() - _t0:.1f}s", flush=True)
    # PER-GROUP POTENTIALS retained for the instant PE view (L-456): one
    # float32 volume per driven group, in channel order — the same arrays
    # this compose already allocated. The runner POPS them onto the
    # display model; they never enter the staged-flight pack.
    chan_phi_list = [group_phi[name] for name, _gr in _active]
    K = len(ch_kind)
    # empty-channel fallback: allocate ONLY when there are no channels
    # (K==0). Allocating z=zeros((K,)+shape) unconditionally wasted a full
    # (K,nx,ny,nz) array (~1.3 GB for SLIM) that was immediately discarded
    # whenever K>0 — pure memory pressure on the exact big-grid case that
    # is already tight.
    def _empty():
        return np.zeros((0,) + shape, _cdtype)
    def _stack(L):
        return np.ascontiguousarray(np.stack(L)).astype(_cdtype, copy=False)
    channels = dict(
        EAx=EAx.astype(_cdtype, copy=False),
        EAy=EAy.astype(_cdtype, copy=False),
        EAz=EAz.astype(_cdtype, copy=False),
        ExK=(_stack(ExK) if K else _empty()),
        EyK=(_stack(EyK) if K else _empty()),
        EzK=(_stack(EzK) if K else _empty()),
        ch_name=list(ch_name),
        chan_phi=chan_phi_list,
        ch_kind=np.array(ch_kind, np.int64),
        ch_om=np.array(ch_om, np.float64),
        ch_ph=np.array(ch_ph, np.float64),
        ch_amp=np.array(ch_amp, np.float64),
        ch_off=np.array(ch_off, np.float64),
        ch_duty=np.array(ch_duty, np.float64),
        tab_t=np.array(tab_t, np.float64),
        tab_v=np.array(tab_v, np.float64),
        tab_off=np.array(tab_off, np.int64))
    return channels


def _ele_from_bases(spec, bases):
    """Metal label grid for edge-aware gradients: reuse the masks the
    solve rastered (never a second geometry). Falls back to nonzero-basis
    union only if a builder didn't stash them."""
    import numpy as np
    m = getattr(bases, "ele", None)
    if m is not None:
        return m
    shape = next(iter(bases.values())).shape
    ele = np.zeros(shape, np.int16)
    for i in sorted(bases):
        # a solved basis is ~1.0 on its own electrode; label that region
        ele[bases[i] > 0.5] = i
    return ele



def _declared_mirror_axes(spec):
    """THE ONE authority for declared mirror planes on the 3-D routes.
    A scene-bearing spec declares its mirror on scene.grid.mirror (the
    JSON the user wrote); specs without a scene declare it via
    geometry.symmetry.planes. The app loads example JSONs
    directly (no simspec_from_scene), so symmetry.planes is 'none' there
    while the scene says 'z' — reading only planes silently disabled the
    mirror fix on the app path (one-board views, and a mid-build
    symmetry mutation changed the cache key -> the solve-then-fly
    re-solve)."""
    sc = getattr(spec, "scene", None)
    if sc:
        m = (sc.get("grid", {}) or {}).get("mirror", "") or ""
        return tuple(a in m for a in ("x", "y", "z"))
    pl = getattr(spec.geometry.symmetry, "planes", {}) or {}
    return tuple(pl.get(a) == "mirror" for a in ("x", "y", "z"))


def _bases_cache_key(spec, geom_bytes=None, tag="stl3d-v1"):
    """Geometry-ONLY cache key: geometry bytes, pitch, dims, solver version.
    Deliberately excludes voltages/RF/source — bases are drive-independent
    (that's the whole point of fast-adjust), so retuning must be a cache
    HIT, not a re-solve.

    geom_bytes: callable(spec) -> bytes describing the geometry. Defaults to
    the STL file contents. A GeomScene supplies its own JSON instead, so the same
    cache serves both without either knowing about the other."""
    import hashlib
    g = spec.geometry
    h = hashlib.sha256()
    # SOLVE-BC FIX MARKER: declared mirrors now reach the solver
    # (they previously did not — older fields were solved under a
    # wrong mid-plane BC). Version marker + the mirror tuple invalidate
    # every stale entry.
    h.update(b"mirror-bc-fix-2026-07-19")
    h.update(repr(_declared_mirror_axes(spec)).encode())
    if geom_bytes is None:
        from ion_gym.io.stl_resolve import resolve_stl_dir
        sdir = resolve_stl_dir(spec)
        for el in g.electrodes:
            h.update(el.name.encode())
            h.update((sdir / el.stl).read_bytes())
    else:
        h.update(geom_bytes(spec))
    # DECLARED PLACEMENT: frame_offset_mm moves metal at the mesh-
    # ingest point, so it is geometry identity on THIS route too -- two
    # decks with identical STL bytes but different declared frames must
    # never share a basis (the bytes-only hash would have served the
    # first deck's field for the second's geometry). Hashed ONLY when
    # declared, through the one interpretation authority, so every basis
    # banked by an undeclaring spec keeps its key and keeps hitting.
    if getattr(g, "frame_offset_mm", None) is not None:
        from ion_gym.io.stl_resolve import placement_offset
        h.update(("frame|" + "|".join(
            f"{v:.9g}" for v in placement_offset(spec))).encode())
    nx, ny, nz, pitch = _grid_from_spec(spec)
    # CACHE KEY LITERAL. The trailing tag names the stencil this basis
    # was solved with. It once carried an external tool's name;
    # renaming the
    # STRING deliberately invalidates every basis banked under the old key,
    # which is correct rather than merely tidy -- a cache entry should not
    # outlive the vocabulary that describes it. First run after this change
    # re-solves; subsequent runs hit as before.
    h.update(f"{pitch:.9g}|{nx}|{ny}|{nz}|{tag}|ghost_linear|1e4".encode())
    return {f"{tag}_bases": h.hexdigest()}


def build_stl3d_run(spec: SimSpec, verbose=False, masks_fn=None,
                    geom_bytes=None, tag="stl3d-v1"):
    """SimSpec (xyz, depth>0) -> (model, fly_fn, col_names, births).

    The 3-D VOXEL-MASK builder. It needs exactly one thing from the geometry:
    masks {electrode index -> (nx,ny,nz) bool}. Where those masks come from is
    a PROVIDER (masks_fn), not a hard-coded assumption:

      * STL electrodes      -> stl_masks_3d   (voxelize, Rung-2)
      * scene3d.GeomScene CSG   -> scene_masks_3d (rasterize3d, ANALYTIC)

    Both land in the same solver, the same cache, and the same flyer, so
    neither path is privileged and a fix to one cannot quietly special-case
    the other."""
    from ion_gym.io import fa_cache
    # DECLARED mirror planes (e.g. a half-gap SLIM scene): the solve MUST
    # honor them and the flight/display must see the UNFOLDED whole.
    # The defect this fixed (wrong trajectories, one-board views): the
    # mirror was recorded in metadata but the solve ran (F,F,F) — the
    # mid-gap plane got the wrong boundary and the partner board never
    # existed in the physics.
    mirror_axes = _declared_mirror_axes(spec)
    key = _bases_cache_key(spec, geom_bytes=geom_bytes, tag=tag)
    t0 = time.time()
    cached = None
    try:
        cached, _meta = fa_cache.load(key)
    except (OSError, ValueError, KeyError) as e:
        # A cache that is MISSING is a miss.  A cache that is CORRUPT is a bug,
        # and it must say so out loud before we quietly recompute over it
        # (doctrine C: a loader misses LOUDLY, it never coerces).
        print(f"[stl3d] WARNING: cache at {key} unreadable ({e}); recomputing")
        cached = None
    if cached is not None:
        bases = {int(k.split("_")[1]): v for k, v in cached.items()
                 if k.startswith("basis_")}
        lab = cached["labels"]
        masks = {i: (lab == i) for i in bases}
        t_vox = t_solve = 0.0
        # UNCONDITIONAL (an unexplained re-solve is a diagnosis nobody
        # should have to guess at):
        # every build says hit/miss + key head, so a surprise re-solve
        # self-diagnoses (a key change means the GEOMETRY changed).
        print(f"[stl3d] bases: CACHE HIT (key "
              f"{next(iter(key.values()))[:8]}, {time.time()-t0:.2f}s load)")
    else:
        print(f"[stl3d] bases: cache MISS (key "
              f"{next(iter(key.values()))[:8]}) — first solve "
              f"for this geometry (a re-solve after an edit means the "
              f"geometry/grid changed)")
        masks = (masks_fn or stl_masks_3d)(spec, verbose=verbose)
        t_vox = time.time() - t0

        t0 = time.time()
        V_BASIS = 1e4          # solver headroom; normalize to unit bases
        # Corrected multigrid (this session's fix: honest full-cycle
        # convergence + verified symmetric-subspace projection). A full 3-D
        # user STL scene is exactly the large, open problem SOR crawls on
        # (the einzel needed ~16k SOR sweeps); multigrid solves it in ~100.
        # Falls back to SOR if unavailable.
        try:
            from ion_gym.physics.multigrid3d import solve_bases_mg
        except ImportError as e:
            # ImportError ONLY -- the SAME silent solver swap as build_planar.
            # `except Exception` also caught multigrid THROWING and recomputed
            # the field with a DIFFERENT SOLVER, unannounced.
            print(f"[stl3d] multigrid unavailable ({e}); solving by SOR")
            bases = solve_bases(masks, mirror=mirror_axes,
                                v_basis=V_BASIS, tol=1e-4, stencil="ghost_linear",
                                verbose=verbose, omega=1.9)
        else:
            bases = solve_bases_mg(masks, mirror=mirror_axes,
                                   v_basis=V_BASIS, tol=1e-4,
                                   stencil="ghost_linear", verbose=verbose)
        bases = {i: (b / V_BASIS).astype(np.float32)
                 for i, b in bases.items()}
        t_solve = time.time() - t0
        try:
            lab = np.zeros(next(iter(masks.values())).shape, np.uint8)
            for i, m in masks.items():
                lab[m] = i
            arrays = {f"basis_{i}": b for i, b in bases.items()}
            arrays["labels"] = lab
            fa_cache.store(key, arrays)
            # descriptor + spec provenance: the 3-D
            # route stored bare-keyed entries with no identity — the
            # gap that made cache entries unselectable. Same annotate
            # contract as basis_cache.store; descriptive only.
            from ion_gym.io.basis_cache import describe_spec
            fa_cache.annotate(key, getattr(spec, "name", ""),
                              label=describe_spec(spec),
                              spec_json=spec.to_dict())
        except (OSError, ValueError) as e:
            # Accelerator, not authority: a failed STORE changes no number.  But
            # it is announced unconditionally -- a store that fails every run
            # means every run silently recomputes, and `verbose` hid that.
            print(f"[stl3d] WARNING: cache store failed ({e}); "
                  f"this solve will not be reused")

    # UNFOLD declared mirrors: solve stored the half (cheap, correct BC);
    # flight/display/PE run in the FULL domain (slim3d pattern). Potentials
    # are EVEN across a mirror. Births on a mirrored axis shift by the
    # stored half-extent so spec coordinates keep meaning "distance from
    # the mirror plane".
    _shift_gu = [0, 0, 0]
    for ax in (0, 1, 2):
        if mirror_axes[ax]:
            n_ax = next(iter(bases.values())).shape[ax]
            sl = [slice(None)] * 3
            sl[ax] = slice(None, 0, -1)
            bases = {i: np.ascontiguousarray(
                        np.concatenate([b[tuple(sl)], b], axis=ax))
                     for i, b in bases.items()}
            masks = {i: np.concatenate([m[tuple(sl)], m], axis=ax)
                     for i, m in masks.items()}
            _shift_gu[ax] = n_ax - 1
    # CANONICAL [-H,+H] FRAME: every mirrored
    # axis is REPORTED and DISPLAYED with the mirror plane at 0, spanning
    # [-H,+H] — the r-z radial convention generalised, and the frame field array
    # merging needs (each field array's symmetry centre at its local 0). The FIELD
    # arrays stay in [0, 2H] index space (the kernel is untouched); this
    # offset converts field-frame mm -> canonical mm by ADDITION. It equals
    # minus the birth shift: spec births are already authored relative to
    # the mirror plane, so in the canonical frame spec input == recorded
    # output == display. Zero on non-mirrored axes (no behaviour change).
    _mirror_off_mm = [-sg * spec.geometry.mm_per_gu for sg in _shift_gu]
    # SIGNED FRAME: origin_mm composes WITH the mirror
    # offset -- world = stored + mirror_off + origin. origin_mm is 2-D
    # ([x_lo, y_lo]); z decks author [0, depth] so its origin term is 0.
    _org2 = spec.geometry.origin_mm or (0.0, 0.0)
    _origin3 = (float(_org2[0]), float(_org2[1]), 0.0)
    _world_off_mm = [m + o for m, o in zip(_mirror_off_mm, _origin3)]
    # ele carries the electrode INDEX per voxel (0 = vacuum), not a boolean.
    ele = np.zeros(next(iter(bases.values())).shape, np.int16)
    for idx in sorted(masks):
        ele[masks[idx]] = int(idx)
    metal = ele > 0
    h = spec.geometry.mm_per_gu
    # expose the label grid to the composer's edge-aware gradients
    try:
        bases.ele = ele
    except AttributeError:
        pass

    # MULTIDRIVE: one channel per declared group; an electrode may join
    # several (ElectrodeSpec.group_names) plus DC. Supersedes the single
    # A/B sin pair. The tracer sums channels with per-channel waveforms.
    channels = compose_drive_channels(spec, bases, verbose=verbose)
    # A7 SYMMETRY GATE: after EVERY mirrored build, prove —
    # never assume — that the unfolded potentials are bit-symmetric
    # about each declared plane and that the normal E component is
    # identically zero on every plane row. Both hold exactly when the
    # mirror machinery is right; two real symmetry defects hid for a
    # long time because nothing checked the plane the machinery
    # claimed to use.
    if any(mirror_axes):
        from ion_gym.physics.symmetry import assert_mirror_field_symmetry
        for _ax in (0, 1, 2):
            if not mirror_axes[_ax]:
                continue
            _c = "xyz"[_ax]
            _sl = [slice(None)] * 3
            _sl[_ax] = _shift_gu[_ax]          # the plane row/slab
            _sl = tuple(_sl)
            _normal = [(f"EA{_c} plane slab", channels[f"EA{_c}"][_sl])]
            _K = channels[f"E{_c}K"]
            for _k in range(_K.shape[0]):
                _normal.append((f"E{_c}K[{_k}] plane slab", _K[_k][_sl]))
            assert_mirror_field_symmetry(
                axis_index=_ax, plane_node=_shift_gu[_ax],
                potentials=[(f"basis_{_i}", _b)
                            for _i, _b in sorted(bases.items())],
                normal_E=_normal,
                context=("stl3d mirrored build "
                         f"({getattr(spec, 'name', None) or 'unnamed spec'})"))
            print(f"[stl3d] A7 symmetry gate PASS: {_c}-mirror plane on "
                  f"node {_shift_gu[_ax]} — potentials bit-symmetric, "
                  f"E_{_c} identically zero on the plane row")
    fields = dict(ele=metal, h_mm=h, **channels)
    # per-group potentials ride the MODEL (PE view), not the flight pack
    _chan_phi = fields.pop("chan_phi")
    # Route tag. A staged assembly dispatches on this rather than
    # sniffing which keys happen to be present: an absent key is ambiguous
    # between "different route" and "solve incomplete", and guessing
    # between those is how a region gets flown by the wrong kernel.
    fields["route"] = "3d"
    # canonical-frame offset rides with the fields pack so a staged
    # assembly Region knows its local frame is mirror-centred
    fields["mirror_off_mm"] = np.asarray(_mirror_off_mm, float)
    # composed field->deck offset for DIAGNOSTIC reporting (the birth-in-
    # metal refusal); flight conversions elsewhere use _world_off_mm.
    fields["world_off_mm"] = np.asarray(_world_off_mm, float)
    # expose the fly3d-ready pack so a staged assembly (physics/
    # staged_flight.Region) can reuse this SOLVED region without
    # re-solving: the region loader reads model.fly_fields directly.
    _fly_fields = fields

    # display model: the static A + a representative RF pair for the 2-D
    # PE background. Pick the largest-amplitude sinusoidal group as the
    # "B" the mid-plane pseudopotential draws (display only; the flight
    # uses ALL channels above).
    A = np.zeros(ele.shape, np.float32)
    for i, el in enumerate(spec.geometry.electrodes, start=1):
        if el.dc:
            A += np.float32(el.dc) * bases[i]
    disp_group, B = _display_rf_pair(spec, bases, ele.shape)
    rf_V = float(disp_group.amplitude_v) if disp_group else 0.0
    om = (2 * math.pi * float(disp_group.frequency_hz) * 1e-6
          if disp_group else 0.0)
    model = Stl3DModel(A, B, ele, h, rf_V, om, off_mm=[0, 0, 0],
                       anchor_mm=_org2)
    # canonical [-H,+H] display frame: field mm + mirror_off_mm
    model.mirror_off_mm = tuple(float(v) for v in _mirror_off_mm)
    model.el_masks = masks         # per-electrode masks -> labeled display

    births = generate_births(spec)
    if any(_shift_gu):
        # THE RETURNED ARRAY STAYS CANONICAL. This block used to shift
        # `births` into the field's [0,2H] index frame, which made the
        # births the ONE external quantity not in the canonical frame --
        # spec input, bounds, display and recorded output are all
        # canonical (see the "ONE conversion point" note in fly_fn).
        # The leak was invisible single-stage, because the same shifted
        # array was handed straight back to fly_fn. It broke the moment
        # another consumer read it: load_assembly's `from_stage` beam
        # maps generated births into world with the stage pose, and for
        # a MIRRORED 3-D stage it received field-frame rows, so
        # fly_staged applied mirror_off a second time and parked the
        # whole packet on the domain face (measured on the oa3d quarter:
        # births at y 20.175 / z 21.700 instead of ~0 / 0, 0/200 arrived
        # as kind 1 at tof 1e-4 us).
        # The shift now happens where the field frame is actually
        # entered -- in fly_fn, per ion -- so every consumer of the
        # returned array sees the frame the rest of the API documents.
        print(f"[stl3d] mirrored axes unfolded to the full domain; spec "
              f"birth coordinates are mirror-plane-relative and are "
              f"converted to the field frame at flight time (the mirror "
              f"plane sits at {[round(sg * h, 3) for sg in _shift_gu]} mm "
              f"in the unfolded field frame, so e.g. spec z0=0 IS the "
              f"mirror plane / midgap -- births are not being displaced)")
    integ = spec.integration
    bnd = spec.bounds

    def _plane_list():
        """Stations + enabled bounding planes for the KERNEL.

        Delegates to physics.stations.kernel_planes — the ONE
        implementation of the station contract, shared with every other
        route (extracted from this closure 2026-09-12; it had been
        stl3d-private, which is why planar/r-z/stl2d silently ignored
        stations). This route unfolds mirrored axes, so it passes its
        world offset: the deck is authored in the canonical [-H,+H]
        frame and the kernel flies the field's [0,2H] frame.
        """
        from ion_gym.physics.stations import kernel_planes
        return kernel_planes(spec, _world_off_mm)

    def _apply_bounds(tr, summ):
        """FALLBACK ONLY. The kernel now finds the crossing per step and
        interpolates the exit state within the step (fate 3). This post-hoc
        scan is kept for trajectories that arrived with no in-kernel planes,
        and it does NOT run when the kernel already terminated on a plane --
        it works on the DECIMATED trajectory, so its 'exit' point is the first
        RECORDED sample past the plane (up to rec_every*dt of overshoot, ~2 mm
        in the Q3 sweep). That is bigger than the transverse spread being
        measured, so it must never be the source of exit statistics."""
        if int(summ.get("kind", -1)) == 3:
            return tr, summ                # kernel already did it, exactly
        planes = []
        for ax, col in (("x", 1), ("y", 2), ("z", 3)):
            if getattr(bnd, f"{ax}_min_on", False):
                planes.append((col, getattr(bnd, f"{ax}_min"), -1))
            if getattr(bnd, f"{ax}_max_on", False):
                planes.append((col, getattr(bnd, f"{ax}_max"), +1))
        if not planes or tr is None or not len(tr):
            return tr, summ
        hit = None
        for col, v, sgn in planes:
            c = tr[:, col]
            idx = np.where((c - v) * sgn >= 0)[0]
            if idx.size and (hit is None or idx[0] < hit):
                hit = int(idx[0])
        if hit is not None:
            tr = tr[:max(hit, 1) + 1]
            summ = dict(summ, kind=3, tof=float(tr[-1, 0]),
                        x_end=float(tr[-1, 1]), y_end=float(tr[-1, 2]),
                        z_end=float(tr[-1, 3]),
                        r_end=float(math.hypot(tr[-1, 1], tr[-1, 2])))
        return tr, summ

    # COLLISIONS. tracer3d.fly3d defaults collisions=None -> VACUUM. This
    # builder never passed them, so the entire 3-D path (STL and scene3d
    # alike) silently flew in vacuum: the spec's gas/pressure/sigma were
    # parsed, shown in the UI, and thrown away. Changing the pressure did
    # nothing, because nothing was reading it.
    coll = spec.collisions if getattr(spec.collisions, "enabled", False) else None
    # Columns follow the SPEC's recording selection (same contract as the
    # planar route): BASE + the requested optionals, in OPTIONAL_CHANNELS
    # order. _opt_3d names the optional slots; _ncol_3d is the row width.
    _opt_3d = [c for c in OPTIONAL_CHANNELS if c in spec.integration.record_channels]
    _ncol_3d = len(BASE_CHANNELS) + len(_opt_3d)
    if coll is not None and verbose:
        print(f"[3d] collisions ON: {coll.model} {coll.gas} "
              f"P={coll.P_pa:.4g} Pa  T={coll.T_k:g} K  "
              f"sigma={coll.sigma_m2:.3g} m^2")
    elif verbose:
        print("[3d] collisions OFF (vacuum)")

    # seed policy resolved ONCE per run (random/fixed per source.seed)
    from ion_gym.physics.ion_envelope import run_seed_base
    _seed_base = run_seed_base(spec)
    # LOUD capacity check (the old silent 100k cap hid mid-flight
    # truncation): announce BEFORE flying if the recording cannot cover
    # t_max, with the numbers and the levers.
    _mrec = int(getattr(integ, "max_records", 100000))
    _need = int(integ.t_max_us * 1e3 / (integ.dt_ns * max(int(integ.rec_every), 1))) + 2
    if _need > _mrec:
        _t_cover = _mrec * integ.dt_ns * max(int(integ.rec_every), 1) * 1e-3
        print(f"[stl3d] WARNING: recording covers only the first "
              f"{_t_cover:.0f} us of a {integ.t_max_us:g} us flight "
              f"({_need} records needed, max_records={_mrec}). Traces will "
              f"truncate mid-flight (announced per ion). Raise "
              f"integration.max_records or rec_every to cover the flight.")

    def fly_fn(i):
        b = births[i]
        # CANONICAL -> FIELD frame, the mirror image of the one
        # conversion point below (which takes recorded output back the
        # other way). The kernel flies in the field's [0,2H] index
        # frame; everything the API hands out or takes in is canonical.
        # Done per ion at the point of entry rather than on the shared
        # births array, so the array a caller reads is the frame the
        # caller expects. Non-mirrored axes have _shift_gu == 0 and are
        # untouched, so unmirrored decks are bit-identical.
        if any(_world_off_mm):
            b = np.array(b, float)
            for _ax in (0, 1, 2):
                if _world_off_mm[_ax]:
                    b[_ax] -= _world_off_mm[_ax]
        # Per-ion mass + seed from THE envelope:
        # CRN policy preserved — the envelope seed IS spec.source.seed+i,
        # so a rerun of ion i is bit-identical and two operating points
        # are compared on the SAME collision draws.
        from ion_gym.physics.ion_envelope import per_ion
        env = per_ion(spec, i, seed_base=_seed_base)
        # MODEL-AWARE GAS DISPATCH. This route once called fly3d
        # unconditionally, so a spec declaring collisions.model="sds" was
        # SILENTLY flown under HS — same trajectories, wrong physics (SDS
        # is mobility drift + positional diffusion and carries no thermal
        # velocity content by design; Appelhans & Dahl 2005). Caught when
        # V06 was re-rooted off the retired import route, which DID dispatch on
        # the model: the native route returned an Einstein ratio of 0.89
        # (HS's signature) where SDS is characterized at ~2.2-2.4, and
        # model="sds" and model="hs" gave bit-identical paths. The
        # dispatch belongs HERE, in the module that owns the native 3-D
        # flight, so it survives the removal of the import route.
        _model = str(getattr(coll, "model", "hs")).lower() if coll else ""
        if coll is not None and _model == "sds":
            from ion_gym.physics.tracer3d import fly3d_sds
            out = fly3d_sds(fields, env.mz, spec.source.charge,
                            (b[0], b[1], b[2]),
                            (b[3], b[4], b[5]),
                            b[6], coll,
                            dt_ns=integ.dt_ns, t_max_us=integ.t_max_us,
                            record_every=max(int(integ.rec_every), 1),
                            max_records=_mrec, seed=env.seed,
                            ion_label=f"ion {i}",
                            # stations + declared bounds, same list the
                            # HS path gets: the model choice must not
                            # change which declarations a run obeys.
                            planes=_plane_list())
        elif coll is not None and _model not in ("hs", ""):
            raise ValueError(
                f"build_stl3d: unknown collisions.model {_model!r} on the "
                f"native 3-D route; known models are 'hs' and 'sds'. "
                f"Refusing to fly a different model than the spec declares.")
        else:
            out = fly3d(fields, env.mz,
                        (b[0], b[1], b[2]),
                        (b[3], b[4], b[5]),
                        b[6],
                        dt_ns=integ.dt_ns, t_max_us=integ.t_max_us,
                        record_every=max(int(integ.rec_every), 1),
                        max_records=_mrec,
                        collisions=coll,
                        seed=env.seed,
                        planes=_plane_list(), ion_label=f"ion {i}",
                        transporter=spec.transporter,
                        charge=spec.source.charge)      # q/m = charge*e/m
        n = len(out["x"])
        # HONOR spec.integration.record_channels (the
        # 3-D route once hardcoded [t..vz,e_x,e_y,e_z] and ignored the recording
        # selection, so a run that asked for speed/ke_ev got neither -- the
        # columns were silently the field components instead). Build the
        # column set from the spec, then fill each optional channel from what
        # fly3d ALREADY returns (velocity, field) -- speed and ke_ev are
        # derived, not re-simulated.
        tr = np.zeros((max(n, 1), _ncol_3d))
        tr[:n, 0] = out["t_us"]
        tr[:n, 1] = out["x"]
        tr[:n, 2] = out["y"]
        tr[:n, 3] = out["z"]
        tr[:n, 4] = out["vx"]
        tr[:n, 5] = out["vy"]
        tr[:n, 6] = out["vz"]
        _vx = out["vx"][:n]
        _vy = out["vy"][:n]
        _vz = out["vz"][:n]
        _sp = np.sqrt(_vx * _vx + _vy * _vy + _vz * _vz)      # mm/us
        _has_e = "ex" in out and "ey" in out and "ez" in out
        _ex_a = out["ex"][:n] if _has_e else np.zeros(n)
        _ey_a = out["ey"][:n] if _has_e else np.zeros(n)
        _ez_a = out["ez"][:n] if _has_e else np.zeros(n)
        _mz_i = env.mz            # Tier 3: envelope, not a re-derivation
        _fill = {
            "speed": _sp,
            "ke_ev": 0.5 * (_mz_i * KG_AMU) * (_sp * 1e3) ** 2 / E_CHG,
            # per-axis KE (eV): 1/2 m v_i^2 / e per component. Sums to ke_ev
            # by construction (the channel test asserts it).
            "ke_x": 0.5 * (_mz_i * KG_AMU) * (_vx * 1e3) ** 2 / E_CHG,
            "ke_y": 0.5 * (_mz_i * KG_AMU) * (_vy * 1e3) ** 2 / E_CHG,
            "ke_z": 0.5 * (_mz_i * KG_AMU) * (_vz * 1e3) ** 2 / E_CHG,
            # Field channels exist only if the kernel sampled the field per
            # record; None (-> refusal above) rather than a zero field.
            "e_field": (np.sqrt(_ex_a ** 2 + _ey_a ** 2 + _ez_a ** 2)
                        if _has_e else None),
            "e_axial": _ez_a,          # transport axis is z in the 3-D frame
            "e_radial": np.sqrt(_ex_a ** 2 + _ey_a ** 2),
            "e_x": _ex_a, "e_y": _ey_a, "e_z": _ez_a,
            "radius": np.sqrt(out["x"][:n] ** 2 + out["y"][:n] ** 2),
            # KERNEL-DEPENDENT channels. The SDS kernel is a continuum
            # mobility+diffusion model: it has no discrete collision count
            # and no wrap bookkeeping, so these are ABSENT rather than zero.
            # None here means "this kernel cannot produce it"; requesting it
            # REFUSES below instead of writing a lie -- a silent 0
            # collisions would read as a collisionless flight.
            "n_col": (np.full(n, float(out["ncol"])) if "ncol" in out
                      else None),
            "wrap_passes": (out["wrap_passes"][:n].astype(float)
                            if "wrap_passes" in out else None),
        }
        for _slot, _name in enumerate(_opt_3d, start=7):
            _chan = _fill[_name]
            if _chan is None:
                raise ValueError(
                    f"build_stl3d: record channel {_name!r} was requested "
                    f"but the collisions.model="
                    f"{str(getattr(coll, 'model', 'hs')).lower()!r} kernel "
                    f"does not produce it. Refusing to record a placeholder "
                    f"value; drop the channel or change the model.")
            tr[:n, _slot] = _chan
        # KE at termination. The flyer knows the final velocity; nothing was
        # writing it into the summary, so the landing energy -- the thing you
        # need to know whether an ion arrives thermalised or hot -- was simply
        # unavailable downstream.
        _mz = env.mz
        # The TERMINAL state, taken from the flyer's own exact final values
        # (out["v_mm_us"], out["KE_eV"]) -- NOT from the last recorded sample,
        # which is decimated. Velocity is a BASE channel, so no new recording
        # is needed for exit statistics: the flyer has always known this and
        # simply never wrote it down.
        _v = out["v_mm_us"]
        # CANONICAL [-H,+H] FRAME: recorded positions leave the
        # field's [0,2H] index frame here — the ONE conversion point — so
        # recorded output == display == spec input frame (spec births are
        # already mirror-relative). Velocities/field components are frame-
        # invariant. Applied BEFORE _apply_bounds so the fallback bounds
        # scan compares like frames (user bounds are canonical too).
        _off = _world_off_mm
        if any(_off):
            tr[:n, 1] += _off[0]
            tr[:n, 2] += _off[1]
            tr[:n, 3] += _off[2]
        # TRUE final state (the point of stoppage is
        # shown, not the last recorded sample -- for a truncated record
        # they differ by the whole unrecorded tail of the flight).
        _fx, _fy, _fz = out["final_xyz_mm"]
        _ex = float(_fx) + _off[0]
        _ey = float(_fy) + _off[1]
        _ez = float(_fz) + _off[2]
        from ion_gym.physics.ion_envelope import make_summary
        summ = make_summary(kind=int(out["kind"]),
                    tof=float(out["tof_us"]), mz=env.mz,
                    x_end=_ex, y_end=_ey, z_end=_ez,
                    ke_end=float(out["KE_eV"]),
                    vx_end=float(_v[0]), vy_end=float(_v[1]),
                    vz_end=float(_v[2]),
                    r_end=float(math.hypot(_ex, _ey)) if n else 0.0,
                    # The kernel has ALWAYS counted collisions (ncol) and the
                    # wrapper has always returned it; this summary hard-coded
                    # n_col=0 regardless -- absence presented as a fact,
                    # and it actively misdirected a real stl3d
                    # diagnosis (a buffer-gas-thermalised ion reported ZERO
                    # collisions). Direct access: a provider that omits ncol
                    # now raises, it does not default.
                    # ncol ABSENT means the kernel does not count discrete
                    # collisions (SDS is a continuum model) -- pass None so
                    # downstream shows "not tracked" rather than a false 0.
                    # A kernel that DOES count still must provide it: no
                    # silent default.
                    n_col=(int(out["ncol"]) if "ncol" in out else None),
                    truncated=bool(out.get("truncated", False)),
                    t_end_us=float(out.get("t_end_us",
                                           out["tof_us"])))
        tr, summ = _apply_bounds(tr, summ)
        return tr, summ

    col_names = list(BASE_CHANNELS) + _opt_3d
    # DESYNC GUARD (see build_planar): the recorded width and the promised
    # names must agree, or a channel was added to _fill/_opt_3d but not to
    # column_names() (or vice versa).
    assert col_names == spec.column_names(), (
        f"3-D column desync: {col_names} != {spec.column_names()}")
    assert _ncol_3d == len(col_names), "3-D row width != column count"
    model.fly_fields = _fly_fields
    # channel-aligned display surfaces for the instant PE view (L-456):
    # potentials per group plus the RFGroupSpec each channel came from
    # (pe_mode/waveform resolution needs the group object, and the model
    # deliberately does not hold the whole spec)
    model.chan_phi = _chan_phi
    _gm = {gr.name: gr for gr in spec.geometry.rf_groups}
    model.chan_groups = [_gm[nm] for nm in _fly_fields["ch_name"]]
    if verbose:
        nx, ny, nz = A.shape
        print(f"[stl3d] {nx}x{ny}x{nz}: voxelize {t_vox:.1f}s, "
              f"solve {t_solve:.1f}s, rf_V={rf_V} om={om:.4g} rad/us")
    return model, fly_fn, col_names, births
